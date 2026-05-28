#!/usr/bin/env python
"""
fit.py – Gaussian Mixture Model membership fitting for a Gaia field.

Usage
-----
    conda run -n pymc_new python fit.py Fornax_dSph
    conda run -n pymc_new python fit.py NGC_55 --draws 3000 --chains 4
    conda run -n pymc_new python fit.py Fornax_dSph --spatial-draws 500 --spatial-tune 200

The script writes all output (trace + PNG plots) to:
    ./gal_fitting_results/<field>/
"""

# XLA/JAX environment flags must be set before any JAX import.
# On Linux CPU, JAX exposes only 1 device by default; numpyro needs one device
# per chain to run chains in parallel.
# The device count defaults to 4 but can be overridden via POP_FIT_N_DEVICES
# or XLA_FLAGS in the caller's environment, e.g.:
#   POP_FIT_N_DEVICES=16 python fit.py Leo_I --chains 16
import os
_n_devices = os.environ.get("POP_FIT_N_DEVICES", "4")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={_n_devices}",
)

# JAX/XLA compilation runs in C++ threads that ignore Python's KeyboardInterrupt.
# Install a SIGINT handler that calls os._exit() to force an immediate OS-level
# exit, which terminates C threads too — making Ctrl+C work reliably.
import signal
def _sigint_handler(sig, frame):
    print('\nInterrupted — exiting.', flush=True)
    os._exit(1)
signal.signal(signal.SIGINT, _sigint_handler)

import argparse
import csv
import json
import sys
import numpy as np
import xarray as xr
from datetime import datetime, timezone

# ── local modules ──────────────────────────────────────────────────────────
from config import N_CLUSTERS, GAIA_BG_SPATIAL_THRESHOLD, GAIA_BG_MIN_STARS, MILLIQUAS_PATH
from data import (load_lvd_catalog, load_gaia_data, get_radec_center,
                  load_field_priors, compute_kinematics,
                  apply_field_spatial_cutoff, compute_background_stats,
                  fit_background_gmm, galactic_pm_rotation_matrices,
                  resolve_field_name,
                  query_gaia_qso_candidates, crossmatch_milliquas,
                  measure_pm_systematic, build_qso_obs_arrays,
                  compute_qso_surface_density,
                  load_bp3m_data, load_bp3m_latent_data)
from photometry import (med_mags, compute_photometric_prior,
                        compute_photometric_prior_kde,
                        build_log_prior_weights,
                        compute_qso_photometric_prior)
from utils import compute_elliptical_r
from models import (build_spatial_model, run_spatial_model,
                    extract_spatial_posterior,
                    build_gmm_model, run_gmm_model, extract_gmm_posterior)
from membership import compute_membership_probs
from plots import (plot_initial_selection, plot_density_profile,
                   plot_pre_gmm, plot_all_diagnostics, plot_final_membership,
                   plot_cmd_prior_diagnostics, plot_background_gmm,
                   plot_spatial_diagnostics, plot_qso_cleaning,
                   plot_qso_membership_diagnostics,
                   plot_qso_cmd_prior_diagnostics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('field', help='Field name (e.g. Fornax_dSph, NGC_55)')

    g = p.add_argument_group('MCMC — full GMM')
    g.add_argument('--draws',  type=int, default=2000, help='Posterior draws (default 2000)')
    g.add_argument('--tune',   type=int, default=2000, help='Tuning steps   (default 2000)')
    g.add_argument('--chains', type=int, default=4,    help='MCMC chains    (default 4)')

    s = p.add_argument_group('MCMC — spatial model')
    s.add_argument('--spatial-draws', type=int, default=1000,
                   help='Spatial model posterior draws (default 1000)')
    s.add_argument('--spatial-tune',  type=int, default=100,
                   help='Spatial model tuning steps   (default 100)')

    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--n-member-samples', type=int, default=1000,
                   help='Posterior draws used to compute membership probs (default 1000)')
    p.add_argument('--membership-threshold', type=float, default=0.95,
                   help='P(dwarf) cut for "member" label in final plots (default 0.95)')
    p.add_argument('--n-stars-max', type=int, default=100000,
                   help='Max stars passed to the GMM (default 100000)')
    p.add_argument('--stop-after', type=int, default=9, metavar='N',
                   help='Stop pipeline after step N (1–9, default 9)')
    p.add_argument('--from-trace', action='store_true',
                   help='Load an existing trace and re-run steps 8–9 (diagnostic plots '
                        'and membership) without re-running MCMC. Steps 1–4 are re-run '
                        'quickly to reconstruct kinematic arrays; all model arrays are '
                        'loaded from constant_data in the saved trace file.')
    p.add_argument('--binned-prior', action='store_true',
                   help='Use binned colour profiles instead of 2-D error-weighted KDE')
    p.add_argument('--spatial-profile', choices=['plummer', 'sersic'],
                   default='sersic',
                   help='Spatial density profile for the dwarf component (default: sersic)')
    p.add_argument('--bg-components', type=int, default=5,
                   help='sklearn GMM components for pre-fit MW background (default: 5)')
    p.add_argument('--search-radius', type=float, default=None, metavar='DEG',
                   help='Gaia cone-search radius in degrees (default: auto from 5×rhalf)')
    p.add_argument('--redownload', action='store_true',
                   help='Force re-download of Gaia data even if a cache exists')
    p.add_argument('--n-init-density-fit', type=int, default=1, metavar='N',
                   help='Iterations of initial spatial density-profile fitting (default: 1). '
                        'For N>1 each iteration updates background GMM and re-centers '
                        'the spatial prior on the previous posterior median.')
    p.add_argument('--galactic-coords', action='store_true',
                   help='Fit in Galactic PM coordinates (pm_l_cosb, pm_b, parallax) '
                        'instead of ICRS (pmra*, pmdec, parallax). '
                        'Rotation is per-star and exact; Mahalanobis distances are preserved.')
    p.add_argument('--qso-correction', action='store_true',
                   help='Measure and subtract a field-level PM zero-point using '
                        'Gaia DR3 QSO candidates (in_qso_candidates flag) within '
                        '--qso-radius degrees of the field centre. Off by default.')
    p.add_argument('--qso-radius', type=float, default=5.0, metavar='DEG',
                   help='Search radius in degrees for the QSO candidate query (default: 5.0)')
    p.add_argument('--bp3m-dir', type=str, default=None, metavar='PATH',
                   help='Path to BP3M output directory containing stellar_astrometry.csv. '
                        'Enables HST PM incorporation: Group A stars (G < 20.7 in both '
                        'Gaia and BP3M) have their PMs replaced at step 7; Group B '
                        'stars (G > 20.7, BP3M-only) are added as a separate population '
                        'with a frozen spatial prior from step 5.')
    p.add_argument('--bp3m-latent', action='store_true',
                   help='Enable z-latent image-transformation model for BP3M stars (v2). '
                        'Requires K_matrices.npz and *_cond columns in stellar_astrometry.csv '
                        '(produced by current run_bp3m.py). Adds z~Normal(0,I) latent '
                        'variables for image alignment modes; uses conditional covariances '
                        'C_vT instead of marginal C_obs. Falls back to v1 if K_matrices.npz '
                        'is absent. Only meaningful with --bp3m-dir.')
    p.add_argument('--bp3m-latent-sigma-eff', type=float, default=0.015, metavar='MAS_YR',
                   help='Minimum per-mode RMS PM sensitivity threshold for z-latent mode '
                        'selection (mas/yr, default 0.015). Lower values keep more modes.')
    p.add_argument('--bp3m-a-chi2-threshold', type=float, default=16.0, metavar='CHI2',
                   help='Chi-squared threshold for Group A BP3M–Gaia PM consistency filter '
                        '(default 16). Stars where chi2 = (y_bp3m - y_gaia)^T C_gaia^{-1} '
                        '(y_bp3m - y_gaia) exceeds this value revert to Gaia-only '
                        'observations. Uses only Gaia-valid dimensions; stars with no '
                        'Gaia PM are never filtered. Set to inf to disable.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _background_mask(r_ell, has_pms, good_mags,
                      threshold=None, min_stars=None):
    """
    Build a pure-spatial background mask (r_ell >= threshold).
    Progressively lowers the threshold if too few stars are found.
    Returns (mask, threshold_used).
    """
    if threshold is None:
        threshold = GAIA_BG_SPATIAL_THRESHOLD
    if min_stars is None:
        min_stars = GAIA_BG_MIN_STARS

    base = has_pms & good_mags & np.isfinite(r_ell)
    bad_rhalf = not base.any() or not np.any(np.isfinite(r_ell))
    if bad_rhalf:
        return has_pms & good_mags, None

    used = threshold
    mask = (r_ell >= used) & base
    for fallback in np.arange(GAIA_BG_SPATIAL_THRESHOLD,1-1e-10,-0.5).astype(float):
        if mask.sum() >= min_stars:
            break
        if fallback < used:
            used = fallback
            mask = (r_ell >= used) & base
            print(f'  WARNING: fewer than {min_stars} background stars at '
                  f'r_ell≥{threshold:.0f}; using r_ell≥{used:.0f}')
    if mask.sum() < min_stars:
        print(f'  WARNING: only {mask.sum()} background stars available.')
    return mask, used


def make_result_path(field):
    path = os.path.join('gal_fitting_results', field)
    os.makedirs(path, exist_ok=True)
    return path


def _bp3m_gaia_chi2(y_gaia, y_bp3m, C_gaia):
    """Chi-squared distance between BP3M and Gaia using Gaia covariance.

    Only uses dimensions where C_gaia diagonal is positive and finite.
    Returns np.nan if no valid dimensions exist or matrix inversion fails.
    This deliberately uses only C_gaia (not C_bp3m + C_gaia) because
    C_bp3m itself depends on C_gaia, making the combined denominator
    hard to interpret independently.
    """
    diag = np.diag(C_gaia)
    valid = np.isfinite(diag) & (diag > 0)
    if not valid.any():
        return np.nan
    sub_C = C_gaia[np.ix_(valid, valid)]
    diff  = (y_bp3m - y_gaia)[valid]
    try:
        C_inv = np.linalg.inv(sub_C)
    except np.linalg.LinAlgError:
        return np.nan
    return float(diff @ C_inv @ diff)


def initial_selection(kin, priors):
    """
    Build the initial keep mask: has valid PMs, within 3-sigma of the
    expected PM+parallax, and (if bad_rhalf) within a fraction of the field.
    Returns keep, r_ell (normalized to rhalf).
    """
    bad_rhalf    = priors['bad_rhalf']
    rhalf_mean   = priors['rhalf_mean']
    ellipticity_mean = priors['ellipticity_mean']
    pa_mean      = priors['pa_mean']

    pm_and_para_dists = kin['pm_and_para_dists']
    has_pms           = kin['has_pms']
    radec_offsets     = kin['radec_offsets']
    radec_dists       = kin['radec_dists']
    survey_mean_width = kin['survey_mean_width']

    keep = has_pms & (pm_and_para_dists < 2)

    if bad_rhalf:
        keep &= (radec_dists < 0.25 * survey_mean_width)
        r_ell = np.full(len(keep), np.nan)
    else:
        r_ell = compute_elliptical_r(
            radec_offsets, 0.0, 0.0,
            pa_mean, ellipticity_mean,
            scale_deg=rhalf_mean / 60.0,
        )
        keep &= (r_ell < 3)

    return keep, r_ell


def prepare_obs_arrays(kin, priors, prior_log_probs, _unused,
                        keep, bad_rhalf, r_ell, n_stars_max, seed, max_r_ell=np.inf):
    """
    Select the subset of stars passed to the GMM and build observation arrays.
    Returns keep_inds, pos_obs, y_obs, S_obs, log_prior_ws.

    Note: good_backgrounds is used only to label stars as "likely background"
    for the prior weights; the GMM receives ALL stars with valid PMs and
    photometry so that galaxy members are not accidentally excluded.
    """
    rng = np.random.default_rng(seed)
    pms = kin['pms']

    test_keep = (
        (r_ell < 2) & (kin['pm_and_para_dists'] <= 1)
        & np.isfinite(prior_log_probs)
    )

    # Include all stars that have valid PMs and photometry.
    # good_backgrounds is NOT used as an inclusion filter here — it only
    # informs the prior weights so background-region stars get low dwarf weight.
    good_to_keep = np.isfinite(pms[:, 0]) & np.isfinite(prior_log_probs)
    # Fallback for bad_rhalf where prior_log_probs may be all-NaN
    if not good_to_keep.any():
        good_to_keep = np.isfinite(pms[:, 0])

    if np.isfinite(max_r_ell):
        # Stars with NaN r_ell (bad_rhalf) are treated as within bounds.
        good_to_keep &= np.where(np.isfinite(r_ell), r_ell <= max_r_ell, True)
    if good_to_keep.sum() <= n_stars_max:
        keep_inds = np.where(good_to_keep)[0]
    else:
        keep_inds = rng.choice(np.where(good_to_keep)[0], n_stars_max, replace=False)

    pos_obs = kin['radec_offsets'][keep_inds]
    y_obs   = kin['pm_and_paras'][keep_inds]
    S_obs   = kin['pm_and_para_covs'][keep_inds]

    log_prior_ws = build_log_prior_weights(
        prior_log_probs, keep, keep_inds, bad_rhalf, test_keep, N_CLUSTERS
    )

    print(f'  Using N={len(keep_inds):,} stars for model fitting')
    return keep_inds, pos_obs, y_obs, S_obs, log_prior_ws, good_to_keep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arrays_to_dataset(d):
    """Convert a dict of numpy arrays to an xarray Dataset with named dims.

    ArviZ interprets plain numpy arrays passed via add_groups() as chain×draw
    samples and warns when the first axis is much larger than the second.
    Wrapping in an xarray Dataset with explicit per-key dimension names bypasses
    that inference entirely.
    """
    return xr.Dataset({
        k: xr.DataArray(np.asarray(v), dims=[f'{k}_dim{i}' for i in range(np.asarray(v).ndim)])
        for k, v in d.items()
    })


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_run_outputs(trace, priors, pm_labels, kin,
                      clean_sample, clean_background, result_path,
                      spatial_profile='sersic'):
    """Write posterior summary CSV and ancillary reference files.

    Files created
    -------------
    posterior_parameter_summaries.csv  – prior/posterior stats for all GMM params
    run_metadata.json                  – G_ref, center priors, labels, timestamp
    photometry_kde_training.npz        – G/BP/RP training photometry for KDE
                                         (skipped when clean_sample is None)
    """
    # Robustly extract posterior and prior as plain xr.Datasets regardless of
    # whether trace is an ArviZ InferenceData or newer xarray DataTree.
    _post_raw = trace.posterior
    post = _post_raw.ds if hasattr(_post_raw, 'ds') else _post_raw
    try:
        _prior_raw = trace.prior
        prior = _prior_raw.ds if hasattr(_prior_raw, 'ds') else _prior_raw
    except AttributeError:
        prior = None
    try:
        pri_avail = set(prior.data_vars) if prior is not None else set()
    except Exception:
        pri_avail = set()

    def _flat(group, var):
        arr = np.array(group[var])
        return arr.reshape(-1, *arr.shape[2:])

    def _stats(vals):
        qs = np.nanpercentile(vals, [2.5, 16, 50, 84, 97.5])
        return dict(zip(['q2.5', 'q16', 'median', 'q84', 'q97.5'], qs))

    rows = []
    avail = set(post.data_vars)

    def _add(name, post_arr, prior_arr, units, description):
        ps = _stats(post_arr)
        prs = _stats(prior_arr) if prior_arr is not None else {}
        rows.append({
            'parameter':       name,
            'description':     description,
            'units':           units,
            'posterior_median': ps['median'],
            'posterior_q16':   ps['q16'],
            'posterior_q84':   ps['q84'],
            'posterior_q2.5':  ps['q2.5'],
            'posterior_q97.5': ps['q97.5'],
            'prior_median':    prs.get('median', np.nan),
            'prior_q16':       prs.get('q16',    np.nan),
            'prior_q84':       prs.get('q84',    np.nan),
            'prior_q2.5':      prs.get('q2.5',   np.nan),
            'prior_q97.5':     prs.get('q97.5',  np.nan),
        })

    ra_c  = float(priors['radec_center'][0])
    dec_c = float(priors['radec_center'][1])
    cos_dec = np.cos(np.deg2rad(dec_c))

    # Center coordinates → convert from delta (arcmin, RA*-corrected) to absolute
    post_ra  = ra_c  + _flat(post,  'delta_ra_center')  / (60.0 * cos_dec)
    post_dec = dec_c + _flat(post,  'delta_dec_center') / 60.0
    pri_ra  = (ra_c  + _flat(prior, 'delta_ra_center')  / (60.0 * cos_dec)
               if 'delta_ra_center'  in pri_avail else None)
    pri_dec = (dec_c + _flat(prior, 'delta_dec_center') / 60.0
               if 'delta_dec_center' in pri_avail else None)
    _add('ra_center',  post_ra,  pri_ra,  'deg', 'Galaxy centre RA (J2000)')
    _add('dec_center', post_dec, pri_dec, 'deg', 'Galaxy centre Dec (J2000)')

    # Scalar spatial / structural parameters
    for var, unit, desc in [
        ('a_plummer',   'arcmin', 'Plummer/Sérsic scale radius'),
        ('ellipticity', '',       'Ellipticity (1 − b/a)'),
        ('pa_deg',      'deg',    'Position angle (N through E, 0–180°)'),
    ]:
        if var not in avail:
            continue
        _add(var, _flat(post, var),
             _flat(prior, var) if var in pri_avail else None,
             unit, desc)

    # Error-model parameters
    for var, unit, desc in [
        ('k',   '',      'Mean Gaia error inflation factor'),
        ('k_1', '1/mag', 'Log-slope of error inflation vs G magnitude'),
    ]:
        if var not in avail:
            continue
        _add(var, _flat(post, var),
             _flat(prior, var) if var in pri_avail else None,
             unit, desc)

    # Dwarf mean proper motion + parallax (vector [pmra*, pmdec, parallax])
    if 'mu_dwarf' in avail:
        post_v = _flat(post,  'mu_dwarf')
        pri_v  = _flat(prior, 'mu_dwarf') if 'mu_dwarf' in pri_avail else None
        dim_info = [
            (pm_labels[0], 'mas/yr', f'Dwarf mean proper motion {pm_labels[0]}'),
            (pm_labels[1], 'mas/yr', f'Dwarf mean proper motion {pm_labels[1]}'),
            ('parallax',   'mas',    'Dwarf mean parallax'),
        ]
        for i, (lbl, unit, desc) in enumerate(dim_info):
            _add(f'mu_dwarf[{i}]',
                 post_v[:, i], pri_v[:, i] if pri_v is not None else None,
                 unit, desc)

    # Intrinsic dispersion + floor vectors (pmra*, pmdec, parallax)
    for var, desc_base in [
        ('sigma_intrinsic_dwarf', 'Dwarf intrinsic dispersion'),
        ('sigma_floor',           'Systematic error floor dispersion'),
    ]:
        if var not in avail:
            continue
        post_v = _flat(post,  var)
        pri_v  = _flat(prior, var) if var in pri_avail else None
        dim_info = [
            (pm_labels[0], 'mas/yr'),
            (pm_labels[1], 'mas/yr'),
            ('parallax',   'mas'),
        ]
        for i, (lbl, unit) in enumerate(dim_info):
            _add(f'{var}[{i}]',
                 post_v[:, i], pri_v[:, i] if pri_v is not None else None,
                 unit, f'{desc_base} ({lbl})')

    # Correlation coefficients
    for var, desc in [
        ('rho_pm_intrinsic_dwarf',              f'Intrinsic PM corr (dwarf, {pm_labels[0]}–{pm_labels[1]})'),
        ('rho_pmra_parallax_intrinsic_dwarf',   f'Intrinsic corr (dwarf, {pm_labels[0]}–parallax)'),
        ('rho_pmdec_parallax_intrinsic_dwarf',  f'Intrinsic corr (dwarf, {pm_labels[1]}–parallax)'),
        ('rho_pm_floor',                        f'Systematic floor PM corr ({pm_labels[0]}–{pm_labels[1]})'),
        ('rho_pmra_parallax_floor',             f'Systematic floor corr ({pm_labels[0]}–parallax)'),
        ('rho_pmdec_parallax_floor',            f'Systematic floor corr ({pm_labels[1]}–parallax)'),
    ]:
        if var not in avail:
            continue
        _add(var, _flat(post, var),
             _flat(prior, var) if var in pri_avail else None,
             '', desc)

    # Mixture fractions
    for var, desc in [
        ('f_dwarf', 'P(dwarf | star)'),
        ('f_star',  'P(source is a star) [3-class model]'),
    ]:
        if var not in avail:
            continue
        _add(var, _flat(post, var),
             _flat(prior, var) if var in pri_avail else None,
             '', desc)

    # delta_pm_sys (vector) — 3-class only
    if 'delta_pm_sys' in avail:
        post_v = _flat(post,  'delta_pm_sys')
        pri_v  = _flat(prior, 'delta_pm_sys') if 'delta_pm_sys' in pri_avail else None
        for i, lbl in enumerate(pm_labels):
            _add(f'delta_pm_sys[{i}]',
                 post_v[:, i], pri_v[:, i] if pri_v is not None else None,
                 'mas/yr', f'Field PM zero-point offset ({lbl}) [3-class]')

    # HST error-model parameters (only when --bp3m-dir is used)
    if 'k_hst' in avail:
        _add('k_hst',
             _flat(post, 'k_hst'),
             _flat(prior, 'k_hst') if 'k_hst' in pri_avail else None,
             '', 'HST/BP3M mean error inflation factor')
    if 'delta_pm_sys_hst' in avail:
        post_v = _flat(post,  'delta_pm_sys_hst')
        pri_v  = _flat(prior, 'delta_pm_sys_hst') if 'delta_pm_sys_hst' in pri_avail else None
        for i, lbl in enumerate(pm_labels):
            _add(f'delta_pm_sys_hst[{i}]',
                 post_v[:, i], pri_v[:, i] if pri_v is not None else None,
                 'mas/yr', f'HST/BP3M PM zero-point offset ({lbl})')
    if 'sigma_floor_hst' in avail:
        post_v = _flat(post,  'sigma_floor_hst')
        pri_v  = _flat(prior, 'sigma_floor_hst') if 'sigma_floor_hst' in pri_avail else None
        for i, (lbl, unit) in enumerate([
            (pm_labels[0], 'mas/yr'),
            (pm_labels[1], 'mas/yr'),
            ('parallax',   'mas'),
        ]):
            _add(f'sigma_floor_hst[{i}]',
                 post_v[:, i], pri_v[:, i] if pri_v is not None else None,
                 unit, f'HST systematic floor dispersion ({lbl})')
    for var, desc in [
        ('rho_pm_floor_hst',             f'HST systematic floor PM corr ({pm_labels[0]}–{pm_labels[1]})'),
        ('rho_pmra_parallax_floor_hst',  f'HST systematic floor corr ({pm_labels[0]}–parallax)'),
        ('rho_pmdec_parallax_floor_hst', f'HST systematic floor corr ({pm_labels[1]}–parallax)'),
    ]:
        if var not in avail:
            continue
        _add(var, _flat(post, var),
             _flat(prior, var) if var in pri_avail else None,
             '', desc)

    # Write CSV
    csv_path = os.path.join(result_path, 'posterior_parameter_summaries.csv')
    fieldnames = [
        'parameter', 'description', 'units',
        'posterior_median', 'posterior_q16', 'posterior_q84',
        'posterior_q2.5',   'posterior_q97.5',
        'prior_median',     'prior_q16',     'prior_q84',
        'prior_q2.5',       'prior_q97.5',
    ]
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f'{v:.8g}' if isinstance(v, (float, np.floating)) else v)
                for k, v in row.items()
            })
    print(f'  Parameter summary → {csv_path}')

    # run_metadata.json — G_ref + other scalar context
    # trace.constant_data may be a DataTree node; use item access for safety.
    try:
        _cdata = trace['constant_data']
        _cdata = _cdata.ds if hasattr(_cdata, 'ds') else _cdata
        gmags_const = np.array(_cdata['gmags_obs'])
    except Exception:
        gmags_const = np.array([])
    G_ref = float(np.nanmedian(gmags_const))
    meta = {
        'G_ref':            G_ref,
        'pm_labels':        list(pm_labels),
        'spatial_profile':  spatial_profile,
        'ra_center_prior':  ra_c,
        'dec_center_prior': dec_c,
        'generated_utc':    datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(result_path, 'run_metadata.json')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'  Run metadata → {meta_path}')

    # photometry_kde_training.npz — training magnitudes + errors for each KDE
    # Skipped in --from-trace mode (clean_sample is None).
    if clean_sample is not None and clean_background is not None:
        g  = kin['gmags']
        bp = kin['bpmags']
        rp = kin['rpmags']
        np.savez(
            os.path.join(result_path, 'photometry_kde_training.npz'),
            member_gmags=g[clean_sample, 0],   member_gmag_errs=g[clean_sample, 1],
            member_bpmags=bp[clean_sample, 0], member_bpmag_errs=bp[clean_sample, 1],
            member_rpmags=rp[clean_sample, 0], member_rpmag_errs=rp[clean_sample, 1],
            bg_gmags=g[clean_background, 0],   bg_gmag_errs=g[clean_background, 1],
            bg_bpmags=bp[clean_background, 0], bg_bpmag_errs=bp[clean_background, 1],
            bg_rpmags=rp[clean_background, 0], bg_rpmag_errs=rp[clean_background, 1],
        )
        print(f'  Photometry KDE training data → {os.path.join(result_path, "photometry_kde_training.npz")}')


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    field = args.field

    # Resolve field name before creating the output directory so the path
    # always uses the canonical CLI key regardless of how the user typed it.
    lvd_df = load_lvd_catalog()
    field  = resolve_field_name(field, lvd_df)

    result_path = make_result_path(field)
    print(f'\n{"="*60}')
    print(f' Field : {field}')
    print(f' Output: {result_path}')
    print(f'{"="*60}')

    # ── 1. Load data ──────────────────────────────────────────────────────
    print('\n[1/9] Loading data...')
    gaia_df  = load_gaia_data(field, lvd_df,
                               search_radius_deg=args.search_radius,
                               redownload=args.redownload)
    priors   = load_field_priors(field, lvd_df)

    # Compute initial offsets for spatial cutoff (some fields only)
    radec_center   = priors['radec_center']
    radecs         = gaia_df[['ra', 'dec']].to_numpy()
    radec_offsets0 = radecs - radec_center
    radec_offsets0[:, 0] = (radec_offsets0[:, 0] + 180.0)%360.0 - 180.0
    radec_offsets0[:, 0] *= np.cos(np.deg2rad(radec_center[1]))
    radec_dists0   = np.sqrt(np.sum(radec_offsets0**2, axis=1))

    gaia_df, radec_offsets0, radec_dists0 = apply_field_spatial_cutoff(
        field, gaia_df, radec_offsets0, radec_dists0
    )

    kin = compute_kinematics(gaia_df, priors)
    print(f'  {len(gaia_df):,} stars loaded')

    # BP3M data (loaded here so it is available throughout the pipeline)
    bp3m_df    = None
    C_pm_bp3m  = None
    # Latent-mode data (v2: z ~ Normal(0,I) for image-transformation modes)
    S_pm_bp3m     = None   # (N_bp3m, 3, M_modes) sensitivity matrices
    C_vT_pm_bp3m  = None   # (N_bp3m, 3, 3) conditional covariances
    cond_means_bp3m = None # (N_bp3m, 3) conditional PM+plx means
    latent_gaia_ids = None # (N_bp3m,) Gaia source IDs matching above arrays
    if args.bp3m_dir is not None:
        bp3m_df, C_pm_bp3m = load_bp3m_data(args.bp3m_dir)
        if getattr(args, 'bp3m_latent', False):
            try:
                S_pm_bp3m, C_vT_pm_bp3m, cond_means_bp3m, latent_gaia_ids = \
                    load_bp3m_latent_data(
                        args.bp3m_dir,
                        sigma_eff_threshold=getattr(args, 'bp3m_latent_sigma_eff', 0.015),
                    )
            except FileNotFoundError as exc:
                print(f'  WARNING: --bp3m-latent ignored — {exc}')
                print('  Falling back to v1 (marginal covariance) mode.')
            except ValueError as exc:
                print(f'  WARNING: --bp3m-latent ignored — {exc}')
                print('  Falling back to v1 (marginal covariance) mode.')

    # Defaults — always defined so downstream code is unconditional.
    is_qso_main       = np.zeros(len(gaia_df), dtype=bool)
    is_milliquas_main = np.zeros(len(gaia_df), dtype=bool)
    is_gaia_only_main = np.zeros(len(gaia_df), dtype=bool)
    log_prior_qso_src = np.full(len(gaia_df), np.log(0.01))  # 1% base QSO prior
    qso_clean_df      = None
    y_obs_qso = S_obs_qso = gmags_qso = None
    bpmags_qso_clean = rpmags_qso_clean = None
    gmag_errs_qso = bpmag_errs_qso = rpmag_errs_qso = None
    log_prior_qso_train_arr = None   # per-source catalog prior for training set
    _pos_obs_qso_train      = None   # sky offsets of cleaned QSOs (for membership)
    n_gaia_raw        = 0
    delta_pm_sys_init = np.zeros(2)   # ICRS init; rotated to Galactic frame below if needed

    # Download wide-field QSO candidates, cross-match against MILLIQUAS for purity,
    # then run two-stage cleaning (parallax cut + iterative PM sigma-clip).
    # Cleaned QSOs constrain delta_pm_sys via wide-field Potential.
    # Per-source soft priors (log_prior_qso_src) replace hard GMM exclusion.
    if args.qso_correction:
        ra_c, dec_c = float(priors['radec_center'][0]), float(priors['radec_center'][1])
        qso_cache   = os.path.join(result_path,
                                    f'gaia_dr3_qso_{args.qso_radius:.1f}deg.csv.gz')
        qso_df_gaia = query_gaia_qso_candidates(
            ra_c, dec_c, args.qso_radius, qso_cache, redownload=args.redownload,
        )
        n_gaia_raw = len(qso_df_gaia)

        # MILLIQUAS cross-match: higher-purity confirmed QSO list
        if os.path.exists(MILLIQUAS_PATH):
            mq_mask = crossmatch_milliquas(
                qso_df_gaia['ra'].to_numpy(), qso_df_gaia['dec'].to_numpy(),
            )
            qso_df = qso_df_gaia[mq_mask].reset_index(drop=True)
            print(f'  MILLIQUAS cross-match: {len(qso_df):,} / {n_gaia_raw:,} '
                  f'Gaia QSOs confirmed  (match radius 1.5 arcsec)')
        else:
            qso_df = qso_df_gaia.copy()
            print(f'  MILLIQUAS not found at {MILLIQUAS_PATH}; skipping cross-match')

        qso_res = measure_pm_systematic(qso_df)
        plot_qso_cleaning(qso_df, qso_res, result_path, n_gaia_raw=n_gaia_raw)

        if qso_res['delta_pmra'] is not None:
            delta_pm_sys_init = np.array([qso_res['delta_pmra'], qso_res['delta_pmdec']])
            print(f'  QSO PM systematic (delta_pm_sys initval): '
                  f'Δμ_α* = {qso_res["delta_pmra"]:+.4f} ± {qso_res["sigma_pmra"]:.4f}  '
                  f'Δμ_δ = {qso_res["delta_pmdec"]:+.4f} ± {qso_res["sigma_pmdec"]:.4f} mas/yr  '
                  f'({qso_res["n_kept"]:,} survived all cleaning, '
                  f'{n_gaia_raw:,} Gaia raw)')
        else:
            print(f'  WARNING: only {qso_res["n_kept"]:,} QSOs survived cleaning '
                  f'(from {n_gaia_raw:,} Gaia raw, {len(qso_df):,} MILLIQUAS-matched); '
                  f'fewer than 20 — delta_pm_sys may be poorly constrained.')

        # Cleaned QSOs used for wide-field PM systematic constraint
        qso_clean_df = qso_df[qso_res['mask_kept']].reset_index(drop=True)
        print(f'  {len(qso_clean_df):,} cleaned QSOs for wide-field PM systematic')

        # Build per-source catalog classification masks in the main Gaia catalog.
        # These drive the soft QSO prior: MILLIQUAS=99%, Gaia-only=50%, other=1%.
        if 'source_id' in gaia_df.columns:
            if 'source_id' in qso_df.columns:
                qso_ids_mq       = set(qso_df['source_id'].to_numpy().tolist())
                is_milliquas_main = gaia_df['source_id'].isin(qso_ids_mq).to_numpy(dtype=bool)
            if 'source_id' in qso_df_gaia.columns:
                qso_ids_gaia_all = set(qso_df_gaia['source_id'].to_numpy().tolist())
                is_gaia_only_main = (
                    gaia_df['source_id'].isin(qso_ids_gaia_all).to_numpy(dtype=bool)
                    & ~is_milliquas_main
                )
            print(f'  Main-catalog QSOs: {is_milliquas_main.sum():,} MILLIQUAS-confirmed, '
                  f'{is_gaia_only_main.sum():,} Gaia-only candidates')
        else:
            print('  source_id not available (old cache?) — use --redownload to refresh')

        is_qso_main = is_milliquas_main | is_gaia_only_main
        log_prior_qso_src[is_gaia_only_main] = np.log(0.50)
        log_prior_qso_src[is_milliquas_main] = np.log(0.99)

    # Optionally rotate all PM observables to Galactic coordinates.
    # The Mahalanobis distances (pm_and_para_dists) are invariant under this
    # orthogonal transform and do not need recomputing.
    if args.galactic_coords:
        ra_all  = gaia_df['ra'].to_numpy()
        dec_all = gaia_df['dec'].to_numpy()
        R_all   = galactic_pm_rotation_matrices(ra_all, dec_all)  # (N, 2, 2)

        # Rotate 3D kinematic arrays in-place
        kin['pm_and_paras'][:, :2] = np.einsum(
            'nij,nj->ni', R_all, kin['pm_and_paras'][:, :2])
        S3 = kin['pm_and_para_covs']
        pm_block  = S3[:, :2, :2].copy()
        cross     = S3[:, :2,  2].copy()
        S3[:, :2, :2] = np.einsum('nij,njk,nlk->nil', R_all, pm_block, R_all)
        S3[:, :2,  2] = np.einsum('nij,nj->ni',       R_all, cross)
        S3[:,  2, :2] = S3[:, :2, 2]

        # Rotate 2D PM arrays (used in plots)
        kin['pms'][:] = np.einsum('nij,nj->ni', R_all, kin['pms'])
        S2 = kin['pm_covs']
        S2[:] = np.einsum('nij,njk,nlk->nil', R_all, S2.copy(), R_all)

        # Rotate the LVD prior mean using the field-centre rotation
        ra_c, dec_c = float(priors['radec_center'][0]), float(priors['radec_center'][1])
        R_cen = galactic_pm_rotation_matrices(np.array([ra_c]), np.array([dec_c]))[0]
        priors['pm_and_para_mean'] = priors['pm_and_para_mean'].copy()
        priors['pm_and_para_mean'][:2] = R_cen @ priors['pm_and_para_mean'][:2]
        priors['mean_pm'] = R_cen @ priors['mean_pm']
        delta_pm_sys_init = R_cen @ delta_pm_sys_init

        pm_labels = (r'$\mu_l \cos b$', r'$\mu_b$')
        print('  Rotated to Galactic PM frame (pm_l_cosb, pm_b, parallax)')
    else:
        pm_labels = (r'$\mu_{\alpha*}$', r'$\mu_\delta$')

    # Build QSO observation arrays now that any Galactic rotation has been applied.
    if qso_clean_df is not None and len(qso_clean_df) > 0:
        (y_obs_qso, S_obs_qso, gmags_qso, bpmags_qso_clean, rpmags_qso_clean,
         gmag_errs_qso, bpmag_errs_qso, rpmag_errs_qso) = build_qso_obs_arrays(qso_clean_df)
        if args.galactic_coords:
            R_qso = galactic_pm_rotation_matrices(
                qso_clean_df['ra'].to_numpy(), qso_clean_df['dec'].to_numpy()
            )
            y_obs_qso[:, :2] = np.einsum('nij,nj->ni', R_qso, y_obs_qso[:, :2])
            _S2  = S_obs_qso[:, :2, :2].copy()
            _crx = S_obs_qso[:, :2, 2].copy()
            S_obs_qso[:, :2, :2] = np.einsum('nij,njk,nlk->nil', R_qso, _S2, R_qso)
            S_obs_qso[:, :2,  2] = np.einsum('nij,nj->ni', R_qso, _crx)
            S_obs_qso[:,  2, :2] = S_obs_qso[:, :2, 2]
        print(f'  QSO obs arrays: {len(y_obs_qso):,} sources '
              f'{"(Galactic frame)" if args.galactic_coords else "(ICRS frame)"}')

        # Per-source catalog prior for the training set: MILLIQUAS-confirmed
        # sources get log(0.99); Gaia-only candidates get log(0.50).
        # Used in both the GMM model (soft mixture likelihood) and membership step.
        _p_train = np.log(0.99) if os.path.exists(MILLIQUAS_PATH) else np.log(0.50)
        log_prior_qso_train_arr = np.full(len(y_obs_qso), _p_train)

        # Sky offsets for QSO training sources (used in membership step).
        # Computed here so the result can be saved in constant_data for --from-trace.
        _qso_ok = (np.isfinite(qso_clean_df['pmra'].to_numpy(dtype=float)) &
                   np.isfinite(qso_clean_df['pmdec'].to_numpy(dtype=float)) &
                   np.isfinite(qso_clean_df['parallax'].to_numpy(dtype=float)))
        _ra_c_q  = float(priors['radec_center'][0])
        _dec_c_q = float(priors['radec_center'][1])
        _qra = qso_clean_df.loc[_qso_ok, 'ra'].to_numpy(dtype=float)
        _qdc = qso_clean_df.loc[_qso_ok, 'dec'].to_numpy(dtype=float)
        _dx_q = ((_qra - _ra_c_q + 180) % 360 - 180) * np.cos(np.deg2rad(_dec_c_q))
        _pos_obs_qso_train = np.column_stack([_dx_q, _qdc - _dec_c_q])

    # ── 2. Initial selection ──────────────────────────────────────────────
    print('\n[2/9] Initial star selection...')
    keep, r_ell = initial_selection(kin, priors)
    keep &= ~is_qso_main   # confirmed QSOs are never treated as stellar members
    print(f'  {keep.sum():,} stars in initial keep mask')

    # ── 3. Initial photometric membership prior ───────────────────────────
    print('\n[3/9] Computing initial photometric membership prior...')
    bad_rhalf = priors['bad_rhalf']

    clean_background_init, bg_thresh_init = _background_mask(
        r_ell, kin['has_pms'], kin['good_mags'],
    )
    clean_background_init &= ~is_qso_main
    print(f'  Initial background: {clean_background_init.sum():,} stars '
          + (f'(r_ell ≥ {bg_thresh_init:.0f} × rhalf)'
             if bg_thresh_init is not None else '(bad_rhalf fallback)'))

    if not args.binned_prior:
        prior_log_probs_init = compute_photometric_prior_kde(
            kin['gmags'], kin['rpmags'], kin['bpmags'],
            keep, clean_background_init, priors['gmag_limit'],
        )
        color_profiles_init = {}
    else:
        prior_log_probs_init, color_profiles_init = compute_photometric_prior(
            kin['gmags'], kin['rpmags'], kin['bpmags'],
            keep, clean_background_init, priors['gmag_limit'],
        )
    # Stars very far from expected mean are treated as certain non-members.
    prior_log_probs_init[kin['pm_and_para_dists'] >= 5] = -1e10

    plot_cmd_prior_diagnostics(
        kin['gmags'], kin['rpmags'], kin['bpmags'],
        keep, clean_background_init,
        prior_log_probs_init, color_profiles_init,
        result_path, tag='initial',
    )
    plot_initial_selection(
        kin['radec_offsets'], kin['pms'],
        kin['colors'], kin['gmags'],
        keep, priors['mean_pm'], result_path,
        pm_labels=pm_labels,
    )

    if args.stop_after <= 3:
        print(f'Stopping after step 3.')
        return

    # ── 4. Initial background statistics (using LVD-prior morphology) ────────
    # Pure spatial cut: no PM-space filter so zero-PM sources are included.
    print('\n[4/9] Computing initial background (MW) statistics...')
    *_, good_backgrounds_init = compute_background_stats(
        kin['pm_and_paras'], r_ell, kin['has_pms'],
    )
    good_backgrounds_init &= ~is_qso_main
    print(f'  Background sample: {good_backgrounds_init.sum():,} stars '
          f'(r_ell ≥ {GAIA_BG_SPATIAL_THRESHOLD:.0f} × rhalf, LVD morphology)')
    bg_means, bg_covs, bg_weights = fit_background_gmm(
        kin['pm_and_paras'], good_backgrounds_init, n_components=args.bg_components,
    )
    print(f'  Pre-fit MW background GMM: K={len(bg_weights)} components')
    plot_background_gmm(
        kin['pm_and_paras'], good_backgrounds_init,
        bg_means, bg_covs, bg_weights,
        kin['radec_offsets'], r_ell,
        result_path, tag='initial',
        gmags=kin['gmags'], colors=kin['colors'],
        pm_labels=pm_labels,
    )

    if args.stop_after <= 4:
        print(f'Stopping after step 4.')
        return

    # mcmc_path is used in steps 5 and 8; create it unconditionally here.
    mcmc_path = os.path.join(result_path, 'mcmc_summaries')
    os.makedirs(mcmc_path, exist_ok=True)

    # ── --from-trace: load saved trace and skip steps 5–7 ────────────────
    if args.from_trace:
        trace_path = os.path.join(result_path, f'{field}_trace.nc')
        if not os.path.exists(trace_path):
            raise FileNotFoundError(
                f'--from-trace: no trace found at {trace_path}. '
                f'Run the full pipeline first.')
        print(f'\n[--from-trace] Loading trace from {trace_path}...')
        import arviz as az
        gmm_trace = az.from_netcdf(trace_path)

        # Restore pm_labels and spatial_profile from saved metadata.
        _meta_path = os.path.join(result_path, 'run_metadata.json')
        if os.path.exists(_meta_path):
            with open(_meta_path) as _fh:
                _meta = json.load(_fh)
            pm_labels = tuple(_meta.get('pm_labels',
                                         [r'$\mu_{\alpha*}$', r'$\mu_\delta$']))
            _spatial_profile_saved = _meta.get('spatial_profile', args.spatial_profile)
        else:
            pm_labels = (r'$\mu_l \cos b$', r'$\mu_b$') if args.galactic_coords \
                        else (r'$\mu_{\alpha*}$', r'$\mu_\delta$')
            _spatial_profile_saved = args.spatial_profile
        # Use the saved spatial profile so membership is consistent with the trace.
        # Allow CLI to override only if the user explicitly passed the flag.
        spatial_profile_ft = _spatial_profile_saved
        args.spatial_profile = spatial_profile_ft  # propagate into membership call

        # Retrieve constant_data (may be DataTree node or plain Dataset).
        # sp / r_ell_updated / clean_sample / prior_log_probs are produced by
        # steps 5–6 which still run; only step 7 (GMM MCMC) is skipped.
        _cdt_raw = gmm_trace['constant_data']
        _cdt = _cdt_raw.ds if hasattr(_cdt_raw, 'ds') else _cdt_raw

        def _cd(name, default=None):
            try:
                return np.array(_cdt[name])
            except (KeyError, Exception):
                return default

        y_obs        = _cd('y_obs')
        pos_obs      = _cd('pos_obs')
        S_obs        = _cd('S_obs')
        log_prior_ws = _cd('log_prior_ws')
        gmags_obs    = _cd('gmags_obs')
        bg_means     = _cd('bg_means')
        bg_covs      = _cd('bg_covs')
        bg_weights   = _cd('bg_weights')
        keep_inds    = _cd('keep_inds')
        # Backward compatibility: traces saved before this change lack keep_inds.
        # Reconstruct by nearest-neighbour matching of pos_obs against radec_offsets.
        if keep_inds is None and pos_obs is not None:
            print('  keep_inds not in trace (old format) — reconstructing via '
                  'nearest-neighbour match of pos_obs...')
            from scipy.spatial import cKDTree
            _kd = cKDTree(kin['radec_offsets'])
            _, keep_inds = _kd.query(pos_obs)
            keep_inds = np.asarray(keep_inds, dtype=int)
            print(f'  Reconstructed keep_inds: {len(keep_inds):,} stars')

        _sa          = _cd('gmm_survey_area')
        gmm_survey_area = float(_sa[0]) if _sa is not None else kin['survey_area']
        _iqm = _cd('is_qso_main')
        is_qso_main  = _iqm.astype(bool) if _iqm is not None \
                       else np.zeros(len(gaia_df), dtype=bool)
        _gtk = _cd('good_to_keep')
        good_to_keep = _gtk.astype(bool) if _gtk is not None \
                       else np.isfinite(kin['pms'][:, 0])

        # QSO arrays (only present when --qso-correction was active).
        log_prior_qso_gmm      = _cd('log_prior_qso_gmm')
        y_obs_qso              = _cd('y_obs_qso')
        S_obs_qso              = _cd('S_obs_qso')
        gmags_qso              = _cd('gmags_qso')
        bpmags_qso_clean       = _cd('bpmags_qso_clean')
        rpmags_qso_clean       = _cd('rpmags_qso_clean')
        _pos_obs_qso_train     = _cd('pos_obs_qso_train')
        log_prior_qso_train_arr = _cd('log_prior_qso_train')
        _imm = _cd('is_milliquas_main')
        is_milliquas_main = _imm.astype(bool) if _imm is not None \
                            else np.zeros(len(gaia_df), dtype=bool)
        _igom = _cd('is_gaia_only_main')
        is_gaia_only_main = _igom.astype(bool) if _igom is not None \
                            else np.zeros(len(gaia_df), dtype=bool)
        qso_clean_df = None   # not needed past this point

        # HST/BP3M arrays (only present when --bp3m-dir was active).
        _ihm = _cd('is_hst_main_gmm')
        is_hst_main_gmm = _ihm.astype(bool) if _ihm is not None else None
        y_obs_hst_b_gmm     = _cd('y_obs_hst_b')
        S_obs_hst_b_gmm     = _cd('S_obs_hst_b')
        log_spatial_b_gmm   = _cd('log_spatial_hst_b')
        _ha = _cd('hst_area')
        hst_area_gmm        = float(_ha[0]) if _ha is not None else None
        log_prior_ws_b_gmm  = _cd('log_prior_ws_hst_b')
        S_latent_hst_a_gmm  = _cd('S_latent_hst_a')
        S_latent_hst_b_gmm  = _cd('S_latent_hst_b')

        # Group B plot/CSV data saved as separate arrays.
        _hst_b_pos    = _cd('hst_b_pos')
        _hst_b_pms    = _cd('hst_b_pms')
        _hst_b_colors = _cd('hst_b_colors')
        _hst_b_gmags  = _cd('hst_b_gmags')
        _hst_b_probs  = _cd('hst_b_prior_probs')
        hst_b_pos_gmm         = _hst_b_pos
        hst_b_prior_probs_gmm = _hst_b_probs
        _hst_b_data = None
        if _hst_b_pos is not None:
            _hst_b_data = {
                'pos':    _hst_b_pos,    'pms':    _hst_b_pms,
                'colors': _hst_b_colors, 'gmags':  _hst_b_gmags,
                'probs':  _hst_b_probs,
            }

        # Reconstruct a minimal Group B dataframe for CSV output.
        bp3m_b_df_gmm = None
        _hst_b_gaia_id  = _cd('hst_b_gaia_id')
        _hst_b_ra       = _cd('hst_b_ra')
        _hst_b_dec      = _cd('hst_b_dec')
        _hst_b_pmra_bp3m  = _cd('hst_b_pmra_bp3m')
        _hst_b_pmdec_bp3m = _cd('hst_b_pmdec_bp3m')
        _hst_b_plx        = _cd('hst_b_parallax_bp3m')
        _hst_b_nhst       = _cd('hst_b_n_hst_used')
        if (_hst_b_gaia_id is not None and y_obs_hst_b_gmm is not None):
            import pandas as pd
            bp3m_b_df_gmm = pd.DataFrame({
                'Gaia_id':        _hst_b_gaia_id,
                'ra':             _hst_b_ra   if _hst_b_ra   is not None else np.zeros(len(_hst_b_gaia_id)),
                'dec':            _hst_b_dec  if _hst_b_dec  is not None else np.zeros(len(_hst_b_gaia_id)),
                'pmra_bp3m':      _hst_b_pmra_bp3m  if _hst_b_pmra_bp3m  is not None else y_obs_hst_b_gmm[:, 0],
                'pmdec_bp3m':     _hst_b_pmdec_bp3m if _hst_b_pmdec_bp3m is not None else y_obs_hst_b_gmm[:, 1],
                'parallax_bp3m':  _hst_b_plx        if _hst_b_plx        is not None else y_obs_hst_b_gmm[:, 2],
                'gmag':           _hst_b_gmags if _hst_b_gmags is not None else np.zeros(len(_hst_b_gaia_id)),
                'n_hst_used':     _hst_b_nhst  if _hst_b_nhst  is not None else np.zeros(len(_hst_b_gaia_id)),
            })

        n_gmm = len(y_obs) if y_obs is not None else 0
        print(f'  Restored: {n_gmm:,} GMM stars, '
              f'survey area = {gmm_survey_area:.5f} deg²')
        if keep_inds is not None:
            print(f'  keep_inds: {len(keep_inds):,}  '
                  f'is_qso_main: {is_qso_main.sum():,}  '
                  f'good_to_keep: {good_to_keep.sum():,}')
        if log_prior_qso_gmm is not None:
            print(f'  QSO correction active ({len(y_obs_qso):,} training QSOs)')
        if is_hst_main_gmm is not None:
            print(f'  HST/BP3M active ({is_hst_main_gmm.sum():,} Group A, '
                  f'{len(y_obs_hst_b_gmm) if y_obs_hst_b_gmm is not None else 0:,} Group B)')
        print('  Steps 5–6 will re-run to get the refined spatial model and photometric prior.')

    # ── 5–7. Spatial model, photometric prior, GMM (step 7 skipped with --from-trace) ──
    # Steps 5 and 6 are fast and refine the spatial model / photometric prior used
    # in membership; only step 7 (GMM MCMC) is expensive and skipped via --from-trace.
    class _SkipToStep8(Exception):
        pass
    try:

        # ── 5. Spatial model (optionally iterated) ────────────────────────

        n_iter = max(1, args.n_init_density_fit)
        print(f'\n[5/9] Running spatial (density-profile) model '
              f'({n_iter} iteration{"s" if n_iter > 1 else ""})...')

        # Raw sky positions needed to re-center offsets between iterations
        radecs_all = gaia_df[['ra', 'dec']].to_numpy()

        # Anchor the hard upper bound for a_plummer to the ORIGINAL catalog value.
        # This prevents iterated updates from gradually ratcheting the prior to
        # unphysical scales (e.g. Draco finding 100-arcmin radii).
        catalog_rhalf    = float(priors['rhalf_mean'])
        max_a_plummer_sp = catalog_rhalf * 4.0

        # Snapshot catalog spatial priors so the red "Prior" ellipses in density-
        # profile plots always reflect the original catalog values, not the
        # iteratively-absorbed updates.
        original_spatial_priors = {k: priors[k]
                                    for k in ('pa_mean', 'ellipticity_mean', 'rhalf_mean')}
        # Also snapshot the original centre so we can place the Prior ellipse at
        # the original LVD position even after the frame has been shifted.
        original_radec_center = priors['radec_center'].copy()

        # r_ell and photometric prior used for spatial model; updated between iterations
        r_ell_spatial        = r_ell.copy()
        prior_log_probs_iter = prior_log_probs_init.copy()

        for iter_idx in range(n_iter):
            is_last = (iter_idx == n_iter - 1)
            if n_iter > 1:
                print(f'\n  -- Spatial iteration {iter_idx + 1}/{n_iter} --')

            (keep_inds_sp, pos_obs_sp, _, _, log_prior_ws_sp,
             _) = prepare_obs_arrays(
                kin, priors, prior_log_probs_iter, None,
                keep, bad_rhalf, r_ell_spatial, args.n_stars_max, args.seed,
            )

            # Only fit stars with kinematics consistent with the expected dwarf PM.
            # This removes most MW contamination that would otherwise inflate a_plummer
            # in dense / contaminated fields (e.g. Draco).
            keep_subset = kin['pm_and_para_dists'][keep_inds_sp] < 2
            keep_inds_sp    = keep_inds_sp[keep_subset]
            pos_obs_sp      = pos_obs_sp[keep_subset]
            log_prior_ws_sp = log_prior_ws_sp[keep_subset]

            # Estimate f_dwarf for the spatial model from the photometric prior:
            # sum of exp(log_prior_ws[:,0]) gives the expected number of dwarf
            # members in the PM-cut sample.  This is self-consistent with the
            # sample composition without needing to extrapolate background density.
            _N_eff_sp       = float(np.sum(np.exp(log_prior_ws_sp[:, 0])))
            f_dwarf_sp_prior = float(np.clip(_N_eff_sp / max(len(log_prior_ws_sp), 1),
                                              0.001, 0.999))
            print(f'  Spatial f_dwarf prior: {f_dwarf_sp_prior:.4f}  '
                  f'(~{_N_eff_sp:.0f} / {len(log_prior_ws_sp):,} stars PM-selected)')

            spatial_model = build_spatial_model(
                pos_obs_sp, log_prior_ws_sp, kin['survey_area'], priors, bad_rhalf,
                spatial_profile=args.spatial_profile,
                max_a_plummer=max_a_plummer_sp,
                f_dwarf_prior=f_dwarf_sp_prior,
            )
            spatial_trace = run_spatial_model(
                spatial_model,
                draws=args.spatial_draws, tune=args.spatial_tune,
                chains=args.chains, seed=args.seed,
            )
            sp = extract_spatial_posterior(spatial_trace)
            print(f'  Posterior a_plummer   = {sp["new_a_plummer"]:.2f} ± {sp["new_a_plummer_err"]:.2f} arcmin')
            print(f'  Posterior ellipticity = {sp["new_ellipticity"]:.3f} ± {sp["new_ellipticity_err"]:.3f}')
            print(f'  Posterior PA          = {sp["new_pa_deg"]:.1f} ± {sp["new_pa_deg_err"]:.1f} deg')

            iter_tag = f'iter_{iter_idx + 1}' if n_iter > 1 else ''
            plot_spatial_diagnostics(spatial_trace, field, mcmc_path, tag=iter_tag)

            # Density profile plot per iteration: always compare against the original
            # catalog priors (red ellipses) so the reference is stable across iterations.
            sp_plot_tag = f'update_{iter_tag}' if iter_tag else 'update'
            _r_ell_sp = (
                compute_elliptical_r(
                    kin['radec_offsets'],
                    sp['new_delta_ra_center'], sp['new_delta_dec_center'],
                    sp['new_pa_deg'], sp['new_ellipticity'],
                    scale_deg=original_spatial_priors['rhalf_mean'] / 60.0,
                ) if not bad_rhalf
                else np.full(len(kin['radec_offsets']), np.nan)
            )
            # Prior ellipse centre in the current offset frame (non-zero only
            # when iterated centre updates have shifted the frame origin).
            _cos_dec_sp = np.cos(np.deg2rad(priors['radec_center'][1]))
            _prior_cen_sp = (
                (((original_radec_center[0] - priors['radec_center'][0])+180.0)%360.0-180.0) * _cos_dec_sp,
                (original_radec_center[1] - priors['radec_center'][1]),
            )
            plot_density_profile(
                kin['radec_offsets'], _r_ell_sp,
                original_spatial_priors, sp, result_path, tag=sp_plot_tag,
                prior_center=_prior_cen_sp,
            )

            if not is_last:
                # Absorb centre offset into priors and recompute kin positions.
                # Prior WIDTHS (rhalf_err, ellipticity_err, pa_err) are kept fixed.
                cos_dec = np.cos(np.deg2rad(priors['radec_center'][1]))
                priors['radec_center'] = np.array([
                    priors['radec_center'][0]
                        + sp['new_delta_ra_center'] / 60.0 / cos_dec,
                    priors['radec_center'][1]
                        + sp['new_delta_dec_center'] / 60.0,
                ])
                # Cap the scale radius update: don't let iterative re-centering
                # ratchet rhalf_mean beyond 3× the original catalog value.
                priors['rhalf_mean']       = min(float(sp['new_a_plummer']),
                                                 3.0 * catalog_rhalf)
                priors['ellipticity_mean'] = sp['new_ellipticity']
                priors['pa_mean']          = sp['new_pa_deg']

                # Recompute offsets from the new centre
                new_offsets = radecs_all - priors['radec_center']
                new_offsets[:, 0] = (new_offsets[:, 0] + 180.0)%360.0 - 180.0
                new_offsets[:, 0] *= np.cos(np.deg2rad(priors['radec_center'][1]))
                kin['radec_offsets'] = new_offsets

                # Recompute r_ell for the next iteration (delta already absorbed)
                r_ell_spatial = (
                    compute_elliptical_r(
                        kin['radec_offsets'], 0.0, 0.0,
                        priors['pa_mean'], priors['ellipticity_mean'],
                        scale_deg=priors['rhalf_mean'] / 60.0,
                    ) if not bad_rhalf
                    else np.full(len(kin['radec_offsets']), np.nan)
                )

                # Refit background GMM with the updated elliptical radii
                print(f'  Refitting background GMM ({iter_tag})...')
                *_, good_bgs_iter = compute_background_stats(
                    kin['pm_and_paras'], r_ell_spatial, kin['has_pms'],
                )
                good_bgs_iter &= ~is_qso_main
                bg_means, bg_covs, bg_weights = fit_background_gmm(
                    kin['pm_and_paras'], good_bgs_iter,
                    n_components=args.bg_components,
                )
                print(f'  Background GMM: K={len(bg_weights)}, '
                      f'{good_bgs_iter.sum():,} background stars')
                plot_background_gmm(
                    kin['pm_and_paras'], good_bgs_iter,
                    bg_means, bg_covs, bg_weights,
                    kin['radec_offsets'], r_ell_spatial,
                    result_path, tag=iter_tag,
                    gmags=kin['gmags'], colors=kin['colors'],
                    pm_labels=pm_labels,
                )

                # Recompute photometric prior with updated background definition
                clean_sample_iter = (
                    (r_ell_spatial <= 2) & kin['has_pms']
                    & (kin['pm_and_para_dists'] <= 2) & kin['good_mags']
                    & ~is_qso_main
                )
                clean_bg_iter, _ = _background_mask(
                    r_ell_spatial, kin['has_pms'], kin['good_mags'],
                )
                clean_bg_iter &= ~is_qso_main
                print(f'  Recomputing photometric prior ({iter_tag})...')
                if not args.binned_prior:
                    prior_log_probs_iter = compute_photometric_prior_kde(
                        kin['gmags'], kin['rpmags'], kin['bpmags'],
                        clean_sample_iter, clean_bg_iter, priors['gmag_limit'],
                    )
                    color_profiles_iter = {}
                else:
                    prior_log_probs_iter, color_profiles_iter = compute_photometric_prior(
                        kin['gmags'], kin['rpmags'], kin['bpmags'],
                        clean_sample_iter, clean_bg_iter, priors['gmag_limit'],
                    )
                prior_log_probs_iter[kin['pm_and_para_dists'] >= 5] = -1e10
                plot_cmd_prior_diagnostics(
                    kin['gmags'], kin['rpmags'], kin['bpmags'],
                    clean_sample_iter, clean_bg_iter,
                    prior_log_probs_iter, color_profiles_iter,
                    result_path, tag=iter_tag,
                )

        if args.stop_after <= 5:
            print(f'Stopping after step 5.')
            return

        # ── 6. Recompute morphology and photometric prior ─────────────────────
        print('\n[6/9] Recomputing morphology and photometric prior...')

        if bad_rhalf:
            # Estimate rhalf from the confident members found with initial prior
            confident = keep & (np.exp(prior_log_probs_init) > 0.5)
            r_ell_deg = compute_elliptical_r(
                kin['radec_offsets'], 0.0, 0.0,
                priors['pa_mean'], priors['ellipticity_mean'],
                scale_deg=1.0,  # unnormalised; result is in degrees
            )
            if confident.sum() > 5:
                priors['rhalf_mean'] = float(np.nanmedian(r_ell_deg[confident]) * 60)
                priors['rhalf_err']  = 0.5 * priors['rhalf_mean']
            print(f'  Estimated rhalf_mean = {priors["rhalf_mean"]:.2f} arcmin')

        # r_ell normalized by CATALOG rhalf (for clean_sample thresholds)
        r_ell_updated = compute_elliptical_r(
            kin['radec_offsets'],
            sp['new_delta_ra_center'], sp['new_delta_dec_center'],
            sp['new_pa_deg'], sp['new_ellipticity'],
            scale_deg=priors['rhalf_mean'] / 60.0,
        )

        clean_sample = ((r_ell_updated <= 2) & kin['has_pms']
                        & (kin['pm_and_para_dists'] <= 2) & kin['good_mags']
                        & ~is_qso_main)
        clean_background, bg_thresh = _background_mask(
            r_ell_updated, kin['has_pms'], kin['good_mags'],
        )
        clean_background &= ~is_qso_main
        print(f'  clean_sample={clean_sample.sum():,}  '
              f'clean_background={clean_background.sum():,} '
              + (f'(r_ell ≥ {bg_thresh:.0f} × rhalf)'
                 if bg_thresh is not None else '(bad_rhalf fallback)'))

        # Refit background GMM with updated morphology — the spatial model may
        # have substantially changed the centre/shape estimate (e.g. NGC_300 where
        # the LVD prior is poor), meaning the initial r_ell cut might have included
        # real galaxy members in the background training set.
        print('  Refitting background GMM with updated morphology...')
        *_, good_backgrounds = compute_background_stats(
            kin['pm_and_paras'], r_ell_updated, kin['has_pms'],
        )
        good_backgrounds &= ~is_qso_main
        print(f'  Refined background: {good_backgrounds.sum():,} stars '
              f'(r_ell ≥ {GAIA_BG_SPATIAL_THRESHOLD:.0f} × rhalf, spatial posterior)')
        bg_means, bg_covs, bg_weights = fit_background_gmm(
            kin['pm_and_paras'], good_backgrounds, n_components=args.bg_components,
        )
        print(f'  Refined background GMM: K={len(bg_weights)} components')
        plot_background_gmm(
            kin['pm_and_paras'], good_backgrounds,
            bg_means, bg_covs, bg_weights,
            kin['radec_offsets'], r_ell_updated,
            result_path, tag='updated',
            gmags=kin['gmags'], colors=kin['colors'],
            pm_labels=pm_labels,
        )

        if not args.binned_prior:
            prior_log_probs = compute_photometric_prior_kde(
                kin['gmags'], kin['rpmags'], kin['bpmags'],
                clean_sample, clean_background, priors['gmag_limit'],
            )
            color_profiles = {}
        else:
            prior_log_probs, color_profiles = compute_photometric_prior(
                kin['gmags'], kin['rpmags'], kin['bpmags'],
                clean_sample, clean_background, priors['gmag_limit'],
            )
        prior_log_probs[kin['pm_and_para_dists'] >= 5] = -1e10

        plot_cmd_prior_diagnostics(
            kin['gmags'], kin['rpmags'], kin['bpmags'],
            clean_sample, clean_background,
            prior_log_probs, color_profiles,
            result_path, tag='updated',
        )

        if args.stop_after <= 6:
            print(f'Stopping after step 6.')
            return

        # ── 7. Prepare GMM data and run ───────────────────────────────────────
        if args.from_trace:
            raise _SkipToStep8()
        print('\n[7/9] Running full GMM...')
        GMM_MAX_R_ELL = 7.0   # selection radius in units of rhalf
        (keep_inds, pos_obs, y_obs, S_obs,
         log_prior_ws, good_to_keep) = prepare_obs_arrays(
            kin, priors, prior_log_probs, None,
            keep, bad_rhalf, r_ell_updated, args.n_stars_max, args.seed,
            max_r_ell=GMM_MAX_R_ELL,
        )

        # --- BP3M injection at step 7 ---
        # Group A: Gaia stars also in BP3M → substitute y_obs/S_obs with BP3M values.
        # Group B: BP3M-only stars (G > 20.7, not in gaia_df) → separate dataset.
        # In latent mode (v2): use conditional means + C_vT; build S_latent padding.
        is_hst_main_gmm       = None
        y_obs_hst_b_gmm       = None
        S_obs_hst_b_gmm       = None
        log_spatial_b_gmm     = None
        log_prior_ws_b_gmm    = None
        hst_area_gmm          = None
        hst_pm_sys_init       = None
        bp3m_b_df_gmm         = None
        hst_b_pos_gmm         = None   # sky offsets for Group B (for plots)
        hst_b_prior_probs_gmm = None   # P(member|CMD) for Group B (for plots)
        S_latent_hst_a_gmm    = None   # (N_gmm, 3, M) padded sensitivity (latent v2)
        S_latent_hst_b_gmm    = None   # (N_B, 3, M) Group B sensitivity (latent v2)

        _use_latent = S_pm_bp3m is not None  # True only when latent mode succeeded

        if bp3m_df is not None:
            # Build ID lookup: Gaia source_id → row index in gaia_df
            gaia_source_ids = gaia_df['source_id'].values if 'source_id' in gaia_df.columns else None
            bp3m_id_to_row  = {int(gid): i for i, gid in enumerate(bp3m_df['Gaia_id'].values)}
            if _use_latent:
                latent_id_to_row = {int(gid): i
                                    for i, gid in enumerate(latent_gaia_ids)}
            else:
                latent_id_to_row = {}

            if gaia_source_ids is not None:
                # Find which GMM stars are in BP3M (Group A mask over keep_inds)
                gmm_gaia_ids  = gaia_source_ids[keep_inds]
                hst_a_in_gmm  = np.array([int(gid) in bp3m_id_to_row
                                           for gid in gmm_gaia_ids], dtype=bool)
                is_hst_main_gmm = hst_a_in_gmm
                N_gmm = len(y_obs)

                # In latent mode, allocate padded sensitivity matrix for Group A
                if _use_latent:
                    _N_modes_a = S_pm_bp3m.shape[2]
                    S_latent_hst_a_gmm = np.zeros((N_gmm, 3, _N_modes_a))

                # Substitute y_obs / S_obs for Group A stars.
                # Latent (v2): use conditional means + C_vT_pm.
                # V1:          use marginal means + C_pm (C_obs).
                n_subst = 0
                n_chi2_rejected = 0
                _chi2_thr = args.bp3m_a_chi2_threshold
                # When galactic coords are active, y_obs is already Galactic-rotated;
                # precompute per-star rotation matrices to bring BP3M ICRS values into
                # the same frame before chi-squared comparison.
                if args.galactic_coords and _chi2_thr < np.inf:
                    _ra_gmm_chi2  = gaia_df['ra'].values[keep_inds]
                    _dec_gmm_chi2 = gaia_df['dec'].values[keep_inds]
                    _R_gmm_chi2   = galactic_pm_rotation_matrices(_ra_gmm_chi2, _dec_gmm_chi2)
                else:
                    _R_gmm_chi2 = None
                for i_gmm, gid in enumerate(gmm_gaia_ids):
                    bp3m_idx = bp3m_id_to_row.get(int(gid))
                    if bp3m_idx is None:
                        continue
                    row = bp3m_df.iloc[bp3m_idx]
                    # Chi-squared pre-filter: revert to Gaia-only if BP3M PM is
                    # inconsistent with Gaia PM within Gaia uncertainties.
                    # Uses C_gaia only (not C_bp3m) since C_bp3m depends on C_gaia.
                    # Stars with no valid Gaia PM (zero/non-finite C_gaia diagonal)
                    # always pass (nan chi2 → no rejection).
                    if _chi2_thr < np.inf:
                        _y_cmp = np.array([float(row['pmra_bp3m']),
                                           float(row['pmdec_bp3m']),
                                           float(row['parallax_bp3m'])])
                        if _R_gmm_chi2 is not None:
                            _y_cmp[:2] = _R_gmm_chi2[i_gmm] @ _y_cmp[:2]
                        chi2_i = _bp3m_gaia_chi2(y_obs[i_gmm], _y_cmp, S_obs[i_gmm])
                        if np.isfinite(chi2_i) and chi2_i > _chi2_thr:
                            is_hst_main_gmm[i_gmm] = False
                            n_chi2_rejected += 1
                            continue
                    if _use_latent:
                        lat_idx = latent_id_to_row.get(int(gid))
                        if lat_idx is not None:
                            y_obs[i_gmm] = cond_means_bp3m[lat_idx]
                            S_obs[i_gmm] = C_vT_pm_bp3m[lat_idx]
                            S_latent_hst_a_gmm[i_gmm] = S_pm_bp3m[lat_idx]
                        else:
                            # Fallback to v1 for this star if not in latent set
                            y_obs[i_gmm] = [float(row['pmra_bp3m']),
                                             float(row['pmdec_bp3m']),
                                             float(row['parallax_bp3m'])]
                            S_obs[i_gmm] = C_pm_bp3m[bp3m_idx]
                    else:
                        y_obs[i_gmm] = [float(row['pmra_bp3m']),
                                         float(row['pmdec_bp3m']),
                                         float(row['parallax_bp3m'])]
                        S_obs[i_gmm] = C_pm_bp3m[bp3m_idx]
                    n_subst += 1
                _mode_label = 'latent v2' if _use_latent else 'v1'
                _chi2_msg = (f'; {n_chi2_rejected:,} reverted to Gaia (chi2>{_chi2_thr})'
                             if n_chi2_rejected > 0 else '')
                print(f'  Group A: substituted BP3M obs for {n_subst:,} stars '
                      f'in GMM dataset ({_mode_label}){_chi2_msg}')

                # Rotate Group A BP3M values to Galactic frame if needed.
                if args.galactic_coords and np.any(is_hst_main_gmm):
                    _a_inds = np.where(is_hst_main_gmm)[0]
                    _ra_a   = gaia_df['ra'].values[keep_inds][_a_inds]
                    _dec_a  = gaia_df['dec'].values[keep_inds][_a_inds]
                    R_a     = galactic_pm_rotation_matrices(_ra_a, _dec_a)
                    y_obs[_a_inds, :2] = np.einsum('nij,nj->ni', R_a, y_obs[_a_inds, :2])
                    _S3  = S_obs[_a_inds].copy()
                    _pm  = _S3[:, :2, :2].copy()
                    _crx = _S3[:, :2,  2].copy()
                    _S3[:, :2, :2] = np.einsum('nij,njk,nlk->nil', R_a, _pm, R_a)
                    _S3[:, :2,  2] = np.einsum('nij,nj->ni',       R_a, _crx)
                    _S3[:,  2, :2] = _S3[:, :2, 2]
                    S_obs[_a_inds] = _S3
                    # Rotate latent sensitivity matrices for Group A (ICRS → Galactic)
                    if _use_latent and S_latent_hst_a_gmm is not None:
                        # R_a: (n_a, 2, 2); S[:, :2, :] rotated as R_a @ S[:, :2, :]
                        _Sa = S_latent_hst_a_gmm[_a_inds].copy()
                        _Sa[:, :2, :] = np.einsum('nij,njk->nik', R_a, _Sa[:, :2, :])
                        S_latent_hst_a_gmm[_a_inds] = _Sa

                # Group B: BP3M stars NOT in the Gaia PM catalog
                gaia_id_set = set(int(g) for g in gaia_source_ids)
            else:
                is_hst_main_gmm = None
                gaia_id_set     = set()
                print('  WARNING: source_id not in gaia_df; BP3M Group A substitution skipped '
                      '(use --redownload to refresh the Gaia cache)')

            # Build Group B from BP3M stars absent from gaia_df
            bp3m_in_gaia = np.array([int(gid) in gaia_id_set
                                      for gid in bp3m_df['Gaia_id'].values], dtype=bool)
            bp3m_b_df_gmm = bp3m_df[~bp3m_in_gaia].reset_index(drop=True)
            C_pm_b         = C_pm_bp3m[~bp3m_in_gaia]
            N_B = len(bp3m_b_df_gmm)

            if N_B > 0:
                b_ra  = bp3m_b_df_gmm['ra'].to_numpy(dtype=float)
                b_dec = bp3m_b_df_gmm['dec'].to_numpy(dtype=float)
                _cos  = np.cos(np.deg2rad(radec_center[1]))
                b_dx  = ((b_ra - radec_center[0] + 180.0) % 360.0 - 180.0) * _cos
                b_dy  = b_dec - radec_center[1]

                # Latent mode (v2): use conditional means + C_vT_pm for Group B.
                # Group B stars are a subset of bp3m_df (same filter as latent_data),
                # so nearly all will be in latent_id_to_row; fall back per-star if not.
                if _use_latent:
                    _N_modes_b = S_pm_bp3m.shape[2]
                    y_obs_hst_b_gmm = np.zeros((N_B, 3))
                    S_obs_hst_b_gmm = np.zeros((N_B, 3, 3))
                    S_latent_hst_b_gmm = np.zeros((N_B, 3, _N_modes_b))
                    for _ib, _gid in enumerate(bp3m_b_df_gmm['Gaia_id'].values):
                        _lr = latent_id_to_row.get(int(_gid), -1)
                        if _lr >= 0:
                            y_obs_hst_b_gmm[_ib] = cond_means_bp3m[_lr]
                            S_obs_hst_b_gmm[_ib] = C_vT_pm_bp3m[_lr]
                            S_latent_hst_b_gmm[_ib] = S_pm_bp3m[_lr]
                        else:
                            _brow = bp3m_id_to_row.get(int(_gid))
                            y_obs_hst_b_gmm[_ib] = [
                                float(bp3m_df.iloc[_brow]['pmra_bp3m']),
                                float(bp3m_df.iloc[_brow]['pmdec_bp3m']),
                                float(bp3m_df.iloc[_brow]['parallax_bp3m'])]
                            S_obs_hst_b_gmm[_ib] = C_pm_bp3m[_brow]
                else:
                    y_obs_hst_b_gmm = np.column_stack([
                        bp3m_b_df_gmm['pmra_bp3m'].to_numpy(dtype=float),
                        bp3m_b_df_gmm['pmdec_bp3m'].to_numpy(dtype=float),
                        bp3m_b_df_gmm['parallax_bp3m'].to_numpy(dtype=float),
                    ])
                    S_obs_hst_b_gmm = C_pm_b.copy()

                hst_b_pos_gmm   = np.column_stack([b_dx, b_dy])

                # Rotate Group B BP3M values to Galactic frame if needed.
                if args.galactic_coords:
                    R_b = galactic_pm_rotation_matrices(b_ra, b_dec)
                    y_obs_hst_b_gmm[:, :2] = np.einsum('nij,nj->ni', R_b, y_obs_hst_b_gmm[:, :2])
                    _pm  = S_obs_hst_b_gmm[:, :2, :2].copy()
                    _crx = S_obs_hst_b_gmm[:, :2,  2].copy()
                    S_obs_hst_b_gmm[:, :2, :2] = np.einsum('nij,njk,nlk->nil', R_b, _pm, R_b)
                    S_obs_hst_b_gmm[:, :2,  2] = np.einsum('nij,nj->ni',       R_b, _crx)
                    S_obs_hst_b_gmm[:,  2, :2] = S_obs_hst_b_gmm[:, :2, 2]
                    # Rotate latent sensitivity matrices for Group B
                    if _use_latent and S_latent_hst_b_gmm is not None:
                        S_latent_hst_b_gmm[:, :2, :] = np.einsum(
                            'nij,njk->nik', R_b, S_latent_hst_b_gmm[:, :2, :])

                # Fixed log spatial density from step-5 posterior medians
                b_offsets  = np.column_stack([b_dx, b_dy])
                a_deg_sp   = float(sp['new_a_plummer']) / 60.0
                q_sp       = 1.0 - float(sp['new_ellipticity'])
                r_ell_b    = compute_elliptical_r(
                    b_offsets,
                    sp['new_delta_ra_center'], sp['new_delta_dec_center'],
                    sp['new_pa_deg'], sp['new_ellipticity'],
                    scale_deg=priors['rhalf_mean'] / 60.0,
                )
                if args.spatial_profile == 'sersic':
                    log_spatial_b_gmm = (- np.log(2 * np.pi)
                                          - 2 * np.log(a_deg_sp)
                                          - np.log(q_sp)
                                          - r_ell_b / a_deg_sp)
                else:
                    log_spatial_b_gmm = (np.log(1.0 / (np.pi * a_deg_sp**2 * q_sp))
                                          - 2.0 * np.log(1.0 + r_ell_b**2 / a_deg_sp**2))

                # Photometric prior for Group B via the same KDE/profile used for
                # Gaia stars.  Concatenate Group B magnitudes with the Gaia arrays
                # so the training-set masks align; Group B rows are excluded from
                # training (all-False extension) and evaluated as query points only.
                # The KDE naturally becomes uninformative past G~20.7 where training
                # data thins out, so no special casing is needed.
                _b_gmags  = bp3m_b_df_gmm[['gmag',  'gmag_error' ]].to_numpy(dtype=float)
                _b_bpmags = bp3m_b_df_gmm[['bpmag', 'bpmag_error']].to_numpy(dtype=float)
                _b_rpmags = bp3m_b_df_gmm[['rpmag', 'rpmag_error']].to_numpy(dtype=float)
                _N_gaia   = len(kin['gmags'])
                _gcat  = np.vstack([kin['gmags'],  _b_gmags])
                _bpcat = np.vstack([kin['bpmags'], _b_bpmags])
                _rpcat = np.vstack([kin['rpmags'], _b_rpmags])
                _mcat  = np.concatenate([clean_sample,     np.zeros(N_B, dtype=bool)])
                _bgcat = np.concatenate([clean_background, np.zeros(N_B, dtype=bool)])
                if not args.binned_prior:
                    _plp_cat = compute_photometric_prior_kde(
                        _gcat, _rpcat, _bpcat, _mcat, _bgcat, priors['gmag_limit'],
                    )
                else:
                    _plp_cat, _ = compute_photometric_prior(
                        _gcat, _rpcat, _bpcat, _mcat, _bgcat, priors['gmag_limit'],
                    )
                _prior_log_probs_b = _plp_cat[_N_gaia:]
                log_prior_ws_b_gmm = build_log_prior_weights(
                    _prior_log_probs_b, np.ones(N_B, dtype=bool), np.arange(N_B),
                    bad_rhalf, np.zeros(N_B, dtype=bool), N_CLUSTERS,
                )
                hst_b_prior_probs_gmm = np.exp(log_prior_ws_b_gmm[:, 0])
                print(f'  Group B photometric prior: median P(member|CMD) = '
                      f'{float(np.median(hst_b_prior_probs_gmm)):.3f}')

                # HST footprint area: bounding box of all BP3M star offsets
                all_dx = ((bp3m_df['ra'].to_numpy(dtype=float) - radec_center[0] + 180.0)
                          % 360.0 - 180.0) * _cos
                all_dy = bp3m_df['dec'].to_numpy(dtype=float) - radec_center[1]
                hst_area_gmm = float(np.ptp(all_dx) * np.ptp(all_dy))
                if hst_area_gmm <= 0:
                    hst_area_gmm = float(gmm_survey_area)  # fallback
                print(f'  Group B: {N_B:,} BP3M-only stars; '
                      f'HST footprint ≈ {hst_area_gmm:.4f} deg²')
            else:
                print('  Group B: no BP3M-only stars (all BP3M stars have Gaia PMs)')

        # Survey area for the GMM = area of the elliptical selection region.
        # r_ell_updated was normalised by priors['rhalf_mean']/60 deg, so the
        # selection ellipse has semi-major axis = GMM_MAX_R_ELL * rhalf_deg and
        # semi-minor axis = GMM_MAX_R_ELL * rhalf_deg * q.
        # For bad_rhalf the spatial cutoff was not applied, so use the full field.
        if bad_rhalf:
            gmm_survey_area = kin['survey_area']
        else:
            _rhalf_deg      = priors['rhalf_mean'] / 60.0
            _q_gmm          = 1.0 - sp['new_ellipticity']
            gmm_survey_area = np.pi * (GMM_MAX_R_ELL * _rhalf_deg)**2 * _q_gmm
        print(f'  GMM survey area: {gmm_survey_area:.5f} deg²  '
              f'(full field: {kin["survey_area"]:.4f} deg²)')

        # Estimate the field-level membership fraction from the background surface
        # density measured in the outer annulus (r_ell >= threshold).
        # annulus_area ≈ full_field − selection_ellipse; floored so it stays positive.
        _annulus_area   = max(kin['survey_area'] - gmm_survey_area, gmm_survey_area * 0.01)
        _bg_density     = good_backgrounds.sum() / _annulus_area          # stars / deg²
        _n_bg_expected  = _bg_density * gmm_survey_area
        f_dwarf_prior   = float(np.clip(
            1.0 - _n_bg_expected / max(len(y_obs), 1), 0.001, 0.999))
        print(f'  f_dwarf prior:   {f_dwarf_prior:.4f}  '
              f'(~{f_dwarf_prior * len(y_obs):.0f} / {len(y_obs):,} stars expected as members)')

        # Per-source QSO prior for the GMM stars (3-class model).
        log_prior_qso_gmm = None
        f_qso_prior_gmm   = None
        log_qso_photo_all = None
        if args.qso_correction and bpmags_qso_clean is not None:
            log_qso_photo_all = compute_qso_photometric_prior(
                kin['gmags'], kin['rpmags'], kin['bpmags'],
                gmags_qso, rpmags_qso_clean, bpmags_qso_clean,
                qso_gmag_errs=gmag_errs_qso,
                qso_bpmag_errs=bpmag_errs_qso,
                qso_rpmag_errs=rpmag_errs_qso,
            )
            log_prior_qso_gmm = (log_qso_photo_all + log_prior_qso_src)[keep_inds]

            qso_surf_dens   = compute_qso_surface_density(len(qso_clean_df),
                                                           n_gaia_raw, args.qso_radius)
            n_qso_expected  = qso_surf_dens * gmm_survey_area
            f_qso_prior_gmm = float(np.clip(n_qso_expected / max(len(y_obs), 1), 0.001, 0.5))
            print(f'  QSO surface density: {qso_surf_dens:.2f} /deg²  →  '
                  f'f_qso prior = {f_qso_prior_gmm:.4f}')

        # QSO candidates in main catalog and wide-field photometry for plot overlay.
        _qso_pos    = kin['radec_offsets'][is_qso_main] if is_qso_main.any() else None
        _qso_pms    = kin['pms'][is_qso_main]           if is_qso_main.any() else None
        _qso_priors = (np.exp(log_prior_qso_src[is_qso_main])
                       if is_qso_main.any() else None)
        _qso_cmd_c  = ((bpmags_qso_clean - rpmags_qso_clean)
                       if bpmags_qso_clean is not None and rpmags_qso_clean is not None
                       else None)

        # Build Group-B overlay dict for population summary figures.
        # y_obs_hst_b_gmm already contains BP3M PMs (Galactic-rotated if needed).
        _hst_b_data = None
        if hst_b_pos_gmm is not None and bp3m_b_df_gmm is not None:
            _hst_b_data = {
                'pos':    hst_b_pos_gmm,
                'pms':    y_obs_hst_b_gmm[:, :2],
                'colors': bp3m_b_df_gmm['bp_rp'].to_numpy(dtype=float),
                'gmags':  bp3m_b_df_gmm['gmag'].to_numpy(dtype=float),
                'probs':  hst_b_prior_probs_gmm,   # P(member|CMD) from KDE
            }

        plot_pre_gmm(
            pos_obs, y_obs,
            kin['colors'], kin['gmags'], keep_inds,
            log_prior_ws, priors['mean_pm'], result_path,
            pm_labels=pm_labels,
            qso_pos=_qso_pos, qso_pms=_qso_pms,
            qso_prior_probs=_qso_priors,
            qso_cmd_colors=_qso_cmd_c, qso_cmd_gmags=gmags_qso,
            hst_b_data=_hst_b_data,
        )

        if log_qso_photo_all is not None and gmags_qso is not None:
            plot_qso_cmd_prior_diagnostics(
                kin['gmags'], kin['rpmags'], kin['bpmags'],
                log_qso_photo_all, log_prior_qso_src,
                gmags_qso, rpmags_qso_clean, bpmags_qso_clean,
                result_path,
            )

        gmags_obs = kin['gmags'][keep_inds, 0]
        if y_obs_qso is not None:
            print(f'  Passing {len(y_obs_qso):,} wide-field QSOs to GMM (delta_pm_sys constraint)')
        if log_prior_qso_gmm is not None:
            print(f'  3-class GMM: per-source QSO prior enabled')
        if _use_latent:
            print(f'  Z-latent mode: {S_pm_bp3m.shape[2]} image-transformation modes')
        gmm_model = build_gmm_model(
            pos_obs, y_obs, S_obs, log_prior_ws, gmm_survey_area,
            priors, sp,
            bg_means=bg_means, bg_covs=bg_covs, bg_weights=bg_weights,
            gmags_obs=gmags_obs,
            spatial_profile=args.spatial_profile,
            f_dwarf_prior=f_dwarf_prior,
            y_obs_qso=y_obs_qso, S_obs_qso=S_obs_qso, gmags_qso=gmags_qso,
            delta_pm_sys_init=delta_pm_sys_init,
            log_prior_qso=log_prior_qso_gmm, f_qso_prior=f_qso_prior_gmm,
            log_prior_qso_train=log_prior_qso_train_arr,
            is_hst_main=is_hst_main_gmm,
            y_obs_hst_b=y_obs_hst_b_gmm,
            S_obs_hst_b=S_obs_hst_b_gmm,
            log_spatial_hst_b=log_spatial_b_gmm,
            hst_area=hst_area_gmm,
            log_prior_ws_hst_b=log_prior_ws_b_gmm,
            hst_pm_sys_init=hst_pm_sys_init,
            S_latent_hst_a=S_latent_hst_a_gmm,
            S_latent_hst_b=S_latent_hst_b_gmm,
        )
        gmm_trace = run_gmm_model(
            gmm_model,
            draws=args.draws, tune=args.tune,
            chains=args.chains, seed=args.seed,
        )

        # Compute good_to_keep for plots — needs ~is_qso_main so done here
        # after all QSO/membership masks are finalised.  Saved in constant_data
        # so --from-trace can reconstruct it without re-running steps 5–6.
        good_to_keep = (np.isfinite(kin['pms'][:, 0])
                        & np.isfinite(prior_log_probs)
                        & ~is_qso_main)

        # Store observed data and pre-fit background GMM for portability.
        # Extra arrays allow --from-trace to skip steps 5–7 entirely.
        _const = {
            'y_obs':        y_obs,
            'pos_obs':      pos_obs,
            'S_obs':        S_obs,
            'log_prior_ws': log_prior_ws,
            'gmags_obs':    gmags_obs,
            'bg_means':     bg_means,
            'bg_covs':      bg_covs,
            'bg_weights':   bg_weights,
            # arrays needed by --from-trace to reconstruct steps 8–9 state
            'keep_inds':       keep_inds,
            'gmm_survey_area': np.array([gmm_survey_area]),
            'is_qso_main':     is_qso_main.astype(np.int8),
            'good_to_keep':    good_to_keep.astype(np.int8),
        }
        # QSO arrays (only when --qso-correction)
        if y_obs_qso is not None:
            _const['y_obs_qso']           = y_obs_qso
            _const['S_obs_qso']           = S_obs_qso
            _const['gmags_qso']           = gmags_qso
        if log_prior_qso_gmm is not None:
            _const['log_prior_qso_gmm']   = log_prior_qso_gmm
        if bpmags_qso_clean is not None:
            _const['bpmags_qso_clean']    = bpmags_qso_clean
            _const['rpmags_qso_clean']    = rpmags_qso_clean
        if _pos_obs_qso_train is not None:
            _const['pos_obs_qso_train']   = _pos_obs_qso_train
        if log_prior_qso_train_arr is not None:
            _const['log_prior_qso_train'] = log_prior_qso_train_arr
        if is_milliquas_main.any() or is_gaia_only_main.any():
            _const['is_milliquas_main']   = is_milliquas_main.astype(np.int8)
            _const['is_gaia_only_main']   = is_gaia_only_main.astype(np.int8)
        # HST/BP3M arrays (only when --bp3m-dir)
        if is_hst_main_gmm is not None:
            _const['is_hst_main_gmm']     = is_hst_main_gmm.astype(np.int8)
        if y_obs_hst_b_gmm is not None:
            _const['y_obs_hst_b']         = y_obs_hst_b_gmm
            _const['S_obs_hst_b']         = S_obs_hst_b_gmm
            _const['log_spatial_hst_b']   = log_spatial_b_gmm
            _const['hst_area']            = np.array([hst_area_gmm if hst_area_gmm else 0.0])
            _const['log_prior_ws_hst_b']  = log_prior_ws_b_gmm
            # Group B plot arrays (sky offset, PMs, colours, photometry, prior)
            _const['hst_b_pos']           = hst_b_pos_gmm
            _const['hst_b_pms']           = y_obs_hst_b_gmm[:, :2]
            if bp3m_b_df_gmm is not None:
                _const['hst_b_colors']    = bp3m_b_df_gmm['bp_rp'].to_numpy(dtype=float)
                _const['hst_b_gmags']     = bp3m_b_df_gmm['gmag'].to_numpy(dtype=float)
            if hst_b_prior_probs_gmm is not None:
                _const['hst_b_prior_probs'] = hst_b_prior_probs_gmm
            # Group B CSV columns
            if bp3m_b_df_gmm is not None:
                _const['hst_b_gaia_id']       = bp3m_b_df_gmm['Gaia_id'].to_numpy()
                _const['hst_b_ra']            = bp3m_b_df_gmm['ra'].to_numpy(dtype=float)
                _const['hst_b_dec']           = bp3m_b_df_gmm['dec'].to_numpy(dtype=float)
                _const['hst_b_pmra_bp3m']     = bp3m_b_df_gmm['pmra_bp3m'].to_numpy(dtype=float)
                _const['hst_b_pmdec_bp3m']    = bp3m_b_df_gmm['pmdec_bp3m'].to_numpy(dtype=float)
                _const['hst_b_parallax_bp3m'] = bp3m_b_df_gmm['parallax_bp3m'].to_numpy(dtype=float)
                _const['hst_b_n_hst_used']    = bp3m_b_df_gmm['n_hst_used'].to_numpy(dtype=float)
        if S_latent_hst_a_gmm is not None:
            _const['S_latent_hst_a']      = S_latent_hst_a_gmm
        if S_latent_hst_b_gmm is not None:
            _const['S_latent_hst_b']      = S_latent_hst_b_gmm
        _const_ds = _arrays_to_dataset(_const)
        try:
            gmm_trace.add_groups({'constant_data': _const_ds})
        except AttributeError:
            # Newer ArviZ returns xarray.DataTree which uses item assignment
            gmm_trace['constant_data'] = _const_ds

        trace_path = os.path.join(result_path, f'{field}_trace.nc')
        _trace_saved = False
        # Try engines in order of preference; DataTree needs netCDF4 or h5netcdf
        # for NETCDF4 format (required for multi-group files).
        for _engine in [None, 'h5netcdf', 'netcdf4']:
            try:
                _kw = {} if _engine is None else {'engine': _engine}
                gmm_trace.to_netcdf(trace_path, **_kw)
                _trace_saved = True
                break
            except Exception:
                pass
        if not _trace_saved:
            # Last resort: save as zarr (no external library needed)
            _zarr_path = trace_path.replace('.nc', '.zarr')
            try:
                gmm_trace.to_zarr(_zarr_path)
                print(f'  WARNING: NetCDF save failed (install h5netcdf or netCDF4). '
                      f'Trace saved as zarr → {_zarr_path}')
                trace_path = _zarr_path
                _trace_saved = True
            except Exception as e:
                print(f'  ERROR: could not save trace ({e}). '
                      f'Install h5netcdf: conda install h5netcdf')
        if _trace_saved and trace_path.endswith('.nc'):
            print(f'  Trace saved → {trace_path}')

        if args.stop_after <= 7:
            print(f'Stopping after step 7.')
            return


    except _SkipToStep8:
        pass

    # ── 8. Posterior summary and diagnostic plots ─────────────────────────
    print('\n[8/9] Saving diagnostic plots...')
    gmm_sp = extract_gmm_posterior(gmm_trace)

    # Density profile: prior vs GMM posterior
    r_ell_gmm = compute_elliptical_r(
        kin['radec_offsets'],
        gmm_sp['new_delta_ra_center'], gmm_sp['new_delta_dec_center'],
        gmm_sp['new_pa_deg'], gmm_sp['new_ellipticity'],
        scale_deg=priors['rhalf_mean'] / 60.0,
    )
    _cos_dec_gmm = np.cos(np.deg2rad(priors['radec_center'][1]))
    _prior_cen_gmm = (
        (original_radec_center[0] - priors['radec_center'][0]) * _cos_dec_gmm,
        (original_radec_center[1] - priors['radec_center'][1]),
    )
    plot_density_profile(
        kin['radec_offsets'], r_ell_gmm,
        priors, gmm_sp, result_path, tag='posterior',
        prior_center=_prior_cen_gmm,
    )

    plot_all_diagnostics(gmm_trace, field, mcmc_path)

    if args.stop_after <= 8:
        print(f'Stopping after step 8.')
        return

    # Print posterior PM summary
    _post9 = gmm_trace.posterior
    if hasattr(_post9, 'ds'):
        _post9 = _post9.ds
    mu_d_post = np.nanmedian(
        np.array(_post9['mu_dwarf']).reshape(-1, 3), axis=0
    )
    mu_d_cov  = np.cov(
        np.array(_post9['mu_dwarf']).reshape(-1, 3), rowvar=False
    )
    mu_d_err  = np.sqrt(np.diag(mu_d_cov))
    lbl0, lbl1 = pm_labels
    print(f'  Posterior PM (dwarf): '
          f'{lbl0} = {mu_d_post[0]:.4f} ± {mu_d_err[0]:.4f}  '
          f'{lbl1} = {mu_d_post[1]:.4f} ± {mu_d_err[1]:.4f}  '
          f'ϖ = {mu_d_post[2]:.5f} ± {mu_d_err[2]:.5f}  mas(/yr)')

    # ── 9. Membership probabilities and final plots ───────────────────────
    print('\n[9/9] Computing per-star membership probabilities...')
    # _pos_obs_qso_train was computed early (step-1 QSO block) and saved to
    # constant_data.  In --from-trace mode it is restored from constant_data.
    # No recomputation needed here.

    mem_result = compute_membership_probs(
        gmm_trace, y_obs, pos_obs, S_obs, log_prior_ws,
        gmm_survey_area,
        bg_means=bg_means, bg_covs=bg_covs, bg_weights=bg_weights,
        gmags_obs=gmags_obs,
        log_prior_qso=log_prior_qso_gmm,
        spatial_profile=getattr(args, 'spatial_profile', 'sersic'),
        n_samples=args.n_member_samples,
        seed=args.seed,
        y_obs_qso_train=y_obs_qso,
        S_obs_qso_train=S_obs_qso,
        pos_obs_qso_train=_pos_obs_qso_train,
        log_prior_qso_train=log_prior_qso_train_arr,
        gmags_qso_train=gmags_qso,
        is_hst_main=is_hst_main_gmm,
        y_obs_hst_b=y_obs_hst_b_gmm,
        S_obs_hst_b=S_obs_hst_b_gmm,
        log_spatial_hst_b=log_spatial_b_gmm,
        hst_area=hst_area_gmm,
        log_prior_ws_hst_b=log_prior_ws_b_gmm,
        S_latent_hst_a=S_latent_hst_a_gmm,
        S_latent_hst_b=S_latent_hst_b_gmm,
    )
    # The last element is always p_dwarf_hst_b_final (or None) when --bp3m-dir is set.
    has_hst_b_mem = (y_obs_hst_b_gmm is not None and len(y_obs_hst_b_gmm) > 0)
    if has_hst_b_mem:
        p_dwarf_hst_b_final = mem_result[-1]
        mem_result = mem_result[:-1]
    else:
        p_dwarf_hst_b_final = None

    has_qso_mem       = len(mem_result) >= 4
    has_qso_train_mem = len(mem_result) == 5
    if has_qso_mem:
        final_probs, p_dwarf_samps, p_background_samps, p_qso_samps = mem_result[:4]
        final_qso_probs       = np.median(p_qso_samps, axis=0)
        qso_train_final_probs = mem_result[4] if has_qso_train_mem else None
    else:
        final_probs, p_dwarf_samps, p_background_samps = mem_result
        final_qso_probs       = None
        p_qso_samps           = None
        qso_train_final_probs = None

    n_members = (final_probs > args.membership_threshold).sum()
    print(f'  {n_members:,} members at P > {args.membership_threshold}')
    if final_qso_probs is not None:
        n_qso_members = (final_qso_probs > args.membership_threshold).sum()
        print(f'  {n_qso_members:,} sources classified as QSOs at P > {args.membership_threshold}')

    # good_to_keep was computed in step 7 (normal mode) or loaded from
    # constant_data (--from-trace).  Either way it is already set correctly.

    # For Group A stars: replace Gaia PMs with BP3M PMs in the VPD.
    pms_for_plot = kin['pms'].copy()
    if is_hst_main_gmm is not None and np.any(is_hst_main_gmm):
        _a_inds = np.where(is_hst_main_gmm)[0]
        pms_for_plot[keep_inds[_a_inds]] = y_obs[_a_inds, :2]

    # Attach posterior membership probabilities to the Group B overlay.
    _hst_b_final_data = None
    if _hst_b_data is not None and p_dwarf_hst_b_final is not None:
        _hst_b_final_data = dict(_hst_b_data, probs=p_dwarf_hst_b_final)

    plot_final_membership(
        kin['radec_offsets'], pms_for_plot,
        kin['colors'], kin['gmags'],
        keep_inds, good_to_keep,
        final_probs, mu_d_post,
        result_path,
        pm_labels=pm_labels,
        final_qso_probs=final_qso_probs,
        hst_b_data=_hst_b_final_data,
    )

    if has_qso_mem:
        plot_qso_membership_diagnostics(
            kin['radec_offsets'], kin['pms'],
            kin['colors'], kin['gmags'],
            keep_inds,
            final_qso_probs,
            is_milliquas_main, is_gaia_only_main,
            result_path,
            pm_labels=pm_labels,
            qso_train_radec=_pos_obs_qso_train,
            y_obs_qso=y_obs_qso,
            qso_train_gmags=gmags_qso,
            qso_train_bpmags=bpmags_qso_clean,
            qso_train_rpmags=rpmags_qso_clean,
            qso_train_probs=qso_train_final_probs,
        )

    # Save Group B membership to a separate CSV
    if p_dwarf_hst_b_final is not None and bp3m_b_df_gmm is not None:
        import pandas as pd
        n_b_members = (p_dwarf_hst_b_final > args.membership_threshold).sum()
        print(f'  Group B: {n_b_members:,} / {len(p_dwarf_hst_b_final):,} members '
              f'at P > {args.membership_threshold}')
        bp3m_b_out = bp3m_b_df_gmm[['Gaia_id', 'ra', 'dec',
                                      'pmra_bp3m', 'pmdec_bp3m', 'parallax_bp3m',
                                      'gmag', 'n_hst_used']].copy()
        bp3m_b_out['p_member'] = p_dwarf_hst_b_final
        bp3m_b_out['is_member'] = p_dwarf_hst_b_final > args.membership_threshold
        bp3m_b_path = os.path.join(result_path, 'bp3m_group_b_membership.csv')
        bp3m_b_out.to_csv(bp3m_b_path, index=False)
        print(f'  Group B membership → {bp3m_b_path}')

    _save_run_outputs(gmm_trace, priors, pm_labels, kin,
                      clean_sample, clean_background, result_path,
                      spatial_profile=getattr(args, 'spatial_profile', 'sersic'))

    print(f'\nDone.  All outputs in {result_path}/')


if __name__ == '__main__':
    main()
