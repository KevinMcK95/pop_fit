"""
PyMC models: spatial (density-profile) model and full Gaussian Mixture Model.
"""

import os
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from utils import get_packed_init, get_packed_init_from_cov
from config import N_CLUSTERS, DEFAULT_BG_COMPONENTS

# Sérsic n=1 (exponential) normalisation constant: γ(2, b₁) = Γ(2)/2
_B1 = 1.678


def _setup_jax_cache():
    """Enable JAX persistent compilation cache in ~/.cache/pop_fit_jax."""
    try:
        import jax
        cache_dir = os.path.expanduser("~/.cache/pop_fit_jax")
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
    except Exception:
        pass


def _log_spatial_dwarf_pt(r_ell, a_deg, q, profile):
    """
    PyTensor log surface-density at elliptical radius *r_ell* (in degrees).

    profile : 'plummer'  – Plummer (1911) projected profile
              'sersic'   – Sérsic n=1 (exponential) profile
    """
    if profile == 'sersic':
        return ( - np.log(2 * np.pi)
                - 2 * pt.log(a_deg) - pt.log(q)
                - r_ell / a_deg)
    else:
        return (pt.log(1.0 / (np.pi * a_deg**2 * q))
                - 2.0 * pt.log(1.0 + r_ell**2 / a_deg**2))


# ---------------------------------------------------------------------------
# Spatial / density-profile model
# ---------------------------------------------------------------------------

def build_spatial_model(pos_obs, log_prior_ws, survey_area,
                         priors, bad_rhalf, spatial_profile='plummer',
                         max_a_plummer=None, f_dwarf_prior=None):
    """
    Fit the galaxy's spatial profile (centre, Plummer/Sérsic scale, ellipticity,
    PA) using photometric membership weights as a soft prior.

    Parameters
    ----------
    pos_obs        : (N, 2)  sky offsets in degrees
    log_prior_ws   : (N, N_clusters)  per-star, per-component log-weights
    survey_area    : float   field area in deg²
    priors         : dict    from data.load_field_priors()
    bad_rhalf      : bool
    spatial_profile: 'plummer' or 'sersic'
    max_a_plummer  : float or None
        Hard upper bound (arcmin) on a_plummer.  Passed from fit.py as
        4× the original catalog half-light radius so iterated spatial
        models can't run away to unphysical scales (e.g. Draco).
    f_dwarf_prior  : float or None
        Initial estimate of the field-level membership fraction.  Used to
        set Beta hyperparameters with concentration ≈ 10 (more diffuse than
        the GMM so the spatial-only data can move the posterior freely).

    Returns
    -------
    model : pm.Model
    """
    rhalf_mean       = priors['rhalf_mean']
    rhalf_err        = priors['rhalf_err']
    ellipticity_mean = priors['ellipticity_mean']
    ellipticity_err  = priors['ellipticity_err']
    pa_mean          = priors['pa_mean']
    pa_err           = priors['pa_err']

    with pm.Model() as model:

        # Centre offsets in arcmin
        delta_ra  = pm.Normal("delta_ra_center",  mu=0.0, sigma=1.0, initval=0.0)
        delta_dec = pm.Normal("delta_dec_center",  mu=0.0, sigma=1.0, initval=0.0)

        # Scale radius in arcmin.
        # For Sérsic, a_plummer is the scale length h = r_e / _B1, so the prior
        # must be centred at rhalf_mean / _B1 (not at the half-light radius itself).
        _rh = rhalf_mean / _B1 if spatial_profile == 'sersic' else rhalf_mean
        _rh_err = rhalf_err / _B1 if spatial_profile == 'sersic' else rhalf_err
        if bad_rhalf:
            a_plummer = pm.LogNormal("a_plummer", mu=np.log(_rh), sigma=0.5)
        else:
            # Use catalog uncertainty as the prior width (floored at 10% of _rh).
            # Switch to TruncatedNormal so we can set an explicit ceiling that
            # stops the model from finding unphysical large-radius solutions in
            # dense / contaminated fields.
            _a_sigma = max(float(_rh_err) if np.isfinite(_rh_err) else _rh * 0.3,
                           _rh * 0.1)
            _a_upper = (float(max_a_plummer) / _B1
                        if spatial_profile == 'sersic' and max_a_plummer is not None
                        else float(max_a_plummer) if max_a_plummer is not None
                        else _rh * 4.0)
            _a_lower = max(0.01, _rh / 10.0)
            a_plummer = pm.TruncatedNormal(
                "a_plummer",
                mu=_rh, sigma=_a_sigma,
                lower=_a_lower, upper=_a_upper,
                initval=float(np.clip(_rh, _a_lower, _a_upper)),
            )

        # Ellipticity
        ell_sigma = ellipticity_err * 5 if np.isfinite(ellipticity_err) else 0.3
        ellipticity = pm.TruncatedNormal(
            "ellipticity",
            mu=ellipticity_mean, sigma=ell_sigma,
            lower=0, upper=0.99,
            initval=min(max(ellipticity_mean, 0.01),0.95),
        )
        q = 1.0 - ellipticity

        # Position angle
        pa_sigma = pa_err * 5 if np.isfinite(pa_err) else 90.0
        #use VonMises (instead of normal) for cyclical PA definition 
        #forcing it to go from 0 to 180 degrees (to prevent multimodality), 
        #but also making sure that the mean is not right at the boundary
        # pa_rad = pm.VonMises("pa_rad", 
        #                         mu=np.radians((pa_mean%180) * 2), 
        #                         kappa=1.0 / (np.radians(pa_sigma*2) ** 2),
        #                         initval=np.radians((pa_mean%180) * 2))
        # pa_deg = 0.5 * pa_rad * 180.0/np.pi
        dpa_rad = pm.VonMises("dpa_rad", 
                                mu=0, 
                                kappa=1.0 / (np.radians(pa_sigma*2) ** 2),
                                initval=0)
        pa_deg = pa_mean%180 + 0.5 * dpa_rad * 180.0/np.pi


        # pa_deg = pm.TruncatedNormal("pa_deg", mu=pa_mean%360, sigma=pa_sigma,
        #                             initval=pa_mean%360)

        # Elliptical radius for each observed star
        theta   = (pa_deg + 90.0) * (np.pi / 180.0)
        x_sh    = pos_obs[:, 0] - delta_ra  / 60.0
        y_sh    = pos_obs[:, 1] - delta_dec / 60.0
        X_major = -x_sh * pt.cos(theta) + y_sh * pt.sin(theta)
        Y_minor =  x_sh * pt.sin(theta) + y_sh * pt.cos(theta)
        r_ell   = pt.sqrt(X_major**2 + (Y_minor / q)**2)

        a_deg = a_plummer / 60.0
        log_spatial_dwarf = _log_spatial_dwarf_pt(r_ell, a_deg, q, spatial_profile)
        log_spatial_mw    = pt.log(1.0 / survey_area)

        # Field-level membership fraction (Beta prior, concentration ≈ 10)
        if f_dwarf_prior is not None:
            _f0   = float(np.clip(f_dwarf_prior, 0.001, 0.999))
            # _conc = 10.0
            _conc = 5.0
            f_dwarf_sp = pm.Beta("f_dwarf",
                                  alpha=max(1.0, _f0 * _conc),
                                  beta =max(1.0, (1.0 - _f0) * _conc),
                                  initval=_f0)
        else:
            f_dwarf_sp = pm.Beta("f_dwarf", alpha=1.0, beta=1.0, initval=0.5)

        # Pre-compute background log-weight: logsumexp of all non-dwarf columns
        log_pw_bg = np.copy(log_prior_ws[:, 1])
        for j in range(2, log_prior_ws.shape[1]):
            log_pw_bg = np.logaddexp(log_pw_bg, log_prior_ws[:, j])

        log_p_dw    = pt.log(f_dwarf_sp) + log_spatial_dwarf + log_prior_ws[:, 0]
        log_p_bg    = pt.log(1.0 - f_dwarf_sp) + log_spatial_mw + log_pw_bg
        log_p_floor = pt.full_like(log_p_dw, np.log(1e-300))

        # Normalise by the per-star CMD marginal so that f_dwarf is identified
        # purely by the *spatial* distribution.  Without this correction,
        # p(pos, cmd) = f_dw×Σ(r)×P_photo + (1-f_dw)×(1/A)×(1-P_photo) drives
        # f_dw → 1 whenever P_photo is large (double-suppression of the
        # background term), inflating the Plummer radius.  Dividing by
        # p(cmd) = f_dw×P_photo + (1-f_dw)×(1-P_photo) gives p(pos | cmd),
        # where high-P_photo stars contribute only Σ(r) to the likelihood and
        # have no leverage on f_dwarf.
        log_norm_cmd = pm.math.logsumexp(
            pt.stack([pt.log(f_dwarf_sp)       + log_prior_ws[:, 0],
                      pt.log(1.0 - f_dwarf_sp) + log_pw_bg]),
            axis=0,
        )
        pm.Potential("likelihood",
                     (pm.math.logsumexp(
                          pt.stack([log_p_dw, log_p_bg, log_p_floor]),
                          axis=0)
                      - log_norm_cmd).sum())
        pm.Deterministic("pa_deg",     pa_deg)

    return model


def run_spatial_model(model, draws=1000, tune=100, chains=4, seed=42):
    """Sample the spatial model and return the trace (with prior predictive)."""
    _setup_jax_cache()
    with model:
        trace = pm.sample(
            nuts_sampler="numpyro",
            chains=chains,
            tune=tune,
            draws=draws,
            target_accept=0.95,
            random_seed=seed,
        )
        prior_samples = pm.sample_prior_predictive(draws=10000)
    trace.extend(prior_samples)
    return trace


def extract_spatial_posterior(trace):
    """Return posterior medians and 1-sigma half-widths for the spatial model."""
    def _med_err(var):
        vals = np.array(trace.posterior[var])
        med = np.nanmedian(vals)
        err = 0.5 * np.diff(np.nanpercentile(vals, [16, 84]))[0]
        return float(med), float(err)

    keys = ['a_plummer', 'delta_ra_center', 'delta_dec_center',
            'ellipticity', 'pa_deg']
    result = {}
    for k in keys:
        result[f'new_{k}'], result[f'new_{k}_err'] = _med_err(k)
    return result


# ---------------------------------------------------------------------------
# Full Gaussian Mixture Model
# ---------------------------------------------------------------------------

def build_gmm_model(pos_obs, y_obs, S_obs, log_prior_ws, survey_area,
                     priors, spatial_posterior,
                     bg_means, bg_covs, bg_weights,
                     gmags_obs=None, spatial_profile='plummer',
                     f_dwarf_prior=None,
                     y_obs_qso=None, S_obs_qso=None, gmags_qso=None,
                     delta_pm_sys_init=None,
                     log_prior_qso=None, f_qso_prior=None,
                     log_prior_qso_train=None,
                     is_hst_main=None,
                     y_obs_hst_b=None, S_obs_hst_b=None,
                     log_spatial_hst_b=None, hst_area=None,
                     log_prior_ws_hst_b=None,
                     hst_pm_sys_init=None,
                     S_latent_hst_a=None,
                     S_latent_hst_b=None):
    """
    Build the GMM (dwarf + K pre-fit background components) with:
      - Elliptical Plummer or Sérsic spatial prior for the dwarf
      - G-magnitude-dependent error inflation k(G) = k₀ · exp(k₁·(G–G_ref))
      - Optional delta_pm_sys (field-level PM zero-point), constrained only
        by QSO likelihoods when y_obs_qso is provided
      - Per-star error inflation and systematic floor
      - LKJ Cholesky covariance for the dwarf only
      - K fixed MW background Gaussians (sklearn pre-fit, numpy constants)

    Parameters
    ----------
    pos_obs, y_obs, S_obs : (N,2), (N,3), (N,3,3)
    log_prior_ws   : (N, N_clusters) photometric + spatial prior weights
    survey_area    : float
    priors         : dict from load_field_priors
    spatial_posterior : dict from extract_spatial_posterior
    bg_means  : (K, 3)   pre-fit MW background component means (numpy)
    bg_covs   : (K, 3, 3) pre-fit MW background component covariances (numpy)
    bg_weights: (K,)     pre-fit MW background mixing weights (numpy)
    gmags_obs       : (N,) G magnitudes (for k(G)); uses k(G)=k₀ if None
    spatial_profile : 'plummer' or 'sersic'
    f_dwarf_prior   : float or None
    y_obs_qso  : (M, 3) or None  — cleaned QSO [pmra, pmdec, parallax] observations
    S_obs_qso  : (M, 3, 3) or None — QSO measurement covariances
    gmags_qso  : (M,) or None — QSO G magnitudes
    delta_pm_sys_init : (2,) or None — initial value for delta_pm_sys from
        the pre-fit QSO systematic estimate; defaults to zeros if None
    log_prior_qso_train : (M,) or None — per-source log catalog prior for the
        wide-field QSO training sample (e.g. log 0.99 for MILLIQUAS-confirmed,
        log 0.50 for Gaia-only).  When supplied, the training likelihood is a
        soft mixture of QSO and MW background components rather than treating
        all training sources as certain QSOs.
    is_hst_main : (N,) bool ndarray or None
        Mask indicating which of the N main-dataset stars have their y_obs/S_obs
        already set to BP3M values (Group A: G < 20.7, in both Gaia and BP3M).
        These stars use k_hst/chol_floor_hst/delta_pm_sys_hst instead of the
        Gaia error model.  None means no HST Group A stars.
    y_obs_hst_b : (N_B, 3) or None
        BP3M PM observations for Group B stars (G > 20.7, no Gaia PM).
    S_obs_hst_b : (N_B, 3, 3) or None
        BP3M PM covariances for Group B stars.
    log_spatial_hst_b : (N_B,) or None
        Fixed log spatial surface density for Group B (precomputed from step-5
        posterior medians; these stars are excluded from spatial fitting).
    hst_area : float or None
        HST footprint area in deg² used as the background normalisation for
        Group B stars.  Replaces the Gaia survey_area for this subset.
    log_prior_ws_hst_b : (N_B, N_clusters) or None
        Per-star photometric prior weights for Group B stars.
    hst_pm_sys_init : (2,) or None
        Initial value for delta_pm_sys_hst; defaults to zeros.
    S_latent_hst_a : (N, 3, M) ndarray or None
        Z-latent sensitivity matrices for BP3M Group A (v2 mode).  Must be
        padded to length N (zero rows for Gaia-only stars).  When provided,
        the model adds z ~ Normal(0,1,shape=M) and computes per-star
        y_eff += S_latent_hst_a @ z for all stars (zero-padding means only
        Group A stars are affected).  y_obs and S_obs must then use the
        BP3M conditional (not marginal) means and covariances.
    S_latent_hst_b : (N_B, 3, M) ndarray or None
        Z-latent sensitivity matrices for BP3M Group B (v2 mode).  Shares
        the same z variable as Group A.  y_obs_hst_b must use conditional
        means and S_obs_hst_b must use conditional covariances.

    Returns
    -------
    model : pm.Model
    """
    pm_and_para_mean          = priors['pm_and_para_mean']
    pm_and_para_mean_scatters = priors['pm_and_para_mean_scatters']
    vlos_implied_pm_sig       = priors['vlos_implied_pm_sig']
    mean_parallax_err         = priors['mean_parallax_err']
    pm_sys_errs               = np.array([0.026, 0.026, 0.011])
    gaia_mean_inflate         = 1.20

    sp           = spatial_posterior
    new_a        = sp['new_a_plummer']
    new_a_err    = sp['new_a_plummer_err']
    new_dra      = sp['new_delta_ra_center']
    new_dra_err  = sp['new_delta_ra_center_err']
    new_ddec     = sp['new_delta_dec_center']
    new_ddec_err = sp['new_delta_dec_center_err']
    new_ell      = sp['new_ellipticity']
    new_ell_err  = sp['new_ellipticity_err']
    new_pa       = sp['new_pa_deg']
    new_pa_err   = sp['new_pa_deg_err']

    # G_ref for k(G): use median of the sample (fixed constant)
    use_kg = gmags_obs is not None
    G_ref  = float(np.nanmedian(gmags_obs)) if use_kg else 0.0

    # Background prior weight (per star): logsumexp over all background columns
    # = log(1 - p_dwarf).  There are N_clusters-1 equal background columns.
    log_pw_bg = np.copy(log_prior_ws[:, 1])  # (N,)
    if len(log_prior_ws[0]) > 2:
        for j in range(2,len(log_prior_ws[0])):
            log_pw_bg = np.logaddexp(log_pw_bg, log_prior_ws[:,j])

    K             = len(bg_weights)
    log_bg_w      = np.log(np.maximum(bg_weights, 1e-300))            # (K,)

    with pm.Model() as model:

        # --- Spatial parameters (tightened by spatial model posterior) ---
        # Minimum sigmas guard against degenerate (zero-spread) spatial posteriors.
        _sig_dra  = max(float(new_dra_err)  * 5, 0.1)   # arcmin
        _sig_ddec = max(float(new_ddec_err) * 5, 0.1)
        _sig_a    = max(float(new_a_err)    * 5, float(new_a) * 0.2, 0.1)
        _sig_ell  = max(float(new_ell_err)  * 5, 0.02)
        _sig_pa   = max(float(new_pa_err)   * 5, 5.0)   # degrees

        delta_ra  = pm.Normal("delta_ra_center",
                              mu=new_dra,  sigma=_sig_dra,
                              initval=float(new_dra))
        delta_dec = pm.Normal("delta_dec_center",
                              mu=new_ddec, sigma=_sig_ddec,
                              initval=float(new_ddec))
        _a_lower_gmm = max(0.01, float(new_a) / 10.0)
        _a_upper_gmm = float(new_a) * 4.0
        a_plummer = pm.TruncatedNormal("a_plummer",
                                        mu=new_a, sigma=_sig_a,
                                        lower=_a_lower_gmm,
                                        upper=_a_upper_gmm,
                                        initval=float(np.clip(new_a, _a_lower_gmm, _a_upper_gmm)))
        ellipticity = pm.TruncatedNormal("ellipticity",
                                          mu=new_ell, sigma=_sig_ell,
                                          lower=0, upper=0.99,
                                          initval=float(np.clip(new_ell, 0.02, 0.95)))
        q      = 1.0 - ellipticity
        #use VonMises (instead of normal) for cyclical PA definition 
        #and also ensure it is restricted to 0 to 180 degrees
        #but also making sure that the mean is not right at the boundary
        # pa_rad = pm.VonMises("pa_rad", 
        #                         mu=np.radians((new_pa%180) * 2), 
        #                         kappa=1.0 / (np.radians(_sig_pa*2) ** 2), 
        #                         initval=np.radians((new_pa%180) * 2))
        # pa_deg = 0.5 * pa_rad * 180.0/np.pi
        dpa_rad = pm.VonMises("dpa_rad", 
                                mu=0, 
                                kappa=1.0 / (np.radians(_sig_pa*2) ** 2), 
                                initval=0)
        pa_deg = new_pa%180 + 0.5 * dpa_rad * 180.0/np.pi

        # pa_deg = pm.Normal("pa_deg", mu=new_pa, sigma=_sig_pa,
        #                    initval=float(new_pa))

        # Elliptical radius
        theta   = (pa_deg + 90.0) * (np.pi / 180.0)
        x_sh    = pos_obs[:, 0] - delta_ra  / 60.0
        y_sh    = pos_obs[:, 1] - delta_dec / 60.0
        X_major = -x_sh * pt.cos(theta) + y_sh * pt.sin(theta)
        Y_minor =  x_sh * pt.sin(theta) + y_sh * pt.cos(theta)
        r_ell   = pt.sqrt(X_major**2 + (Y_minor / q)**2)

        a_deg = a_plummer / 60.0
        log_spatial_dwarf = _log_spatial_dwarf_pt(r_ell, a_deg, q, spatial_profile)
        log_spatial_mw    = pt.log(1.0 / survey_area)

        # --- Error model: k(G) = k₀ · exp(k₁ · (G - G_ref)) ---
        # initval must be the parameter value itself (not log-transformed).
        k   = pm.LogNormal("k",   mu=np.log(gaia_mean_inflate), sigma=0.5,
                           initval=float(gaia_mean_inflate))
        k_1 = pm.Normal("k_1", mu=0.0, sigma=0.2, initval=0.0)
        if use_kg:
            # Replace NaN G magnitudes with G_ref so missing photometry doesn't
            # propagate NaN into the pytensor graph.
            gmags_safe = np.where(np.isfinite(gmags_obs), gmags_obs, G_ref)
            dG = pt.as_tensor_variable(gmags_safe.astype(np.float64)) - G_ref
            # Clip k₁·ΔG to [-6, 6] to prevent float32/float64 overflow when
            # NUTS warm-up explores k₁ values far from the prior mean.
            k_per_star = pt.maximum(k * pt.exp(pt.clip(k_1 * dG, -6.0, 6.0)), 0.1)
        else:
            k_per_star = k * pt.ones(len(y_obs))

        L_floor, corr_floor, stds_floor = pm.LKJCholeskyCov(
            "chol_floor", n=3, eta=2.0,
            sd_dist=pm.LogNormal.dist(mu=np.log(pm_sys_errs), sigma=0.3),
            initval=get_packed_init(pm_sys_errs, [0, 0, 0]),
        )
        S_floor = L_floor @ L_floor.T

        # --- HST error model (Group A + Group B) ---
        # HST/BP3M floor prior is smaller than Gaia's: BP3M already marginalises
        # over image transforms, so residual systematics should be sub-Gaia level.
        hst_sys_errs = np.array([0.015, 0.015, 0.010])  # mas/yr, mas/yr, mas

        _has_hst_main = (is_hst_main is not None and np.any(is_hst_main))
        _has_hst_b    = (y_obs_hst_b is not None and len(y_obs_hst_b) > 0)
        _has_hst      = _has_hst_main or _has_hst_b
        if _has_hst:
            k_hst = pm.LogNormal("k_hst", mu=0.0, sigma=0.3, initval=1.0)
            L_floor_hst, corr_floor_hst, stds_floor_hst = pm.LKJCholeskyCov(
                "chol_floor_hst", n=3, eta=2.0,
                sd_dist=pm.LogNormal.dist(mu=np.log(hst_sys_errs), sigma=0.3),
                initval=get_packed_init(hst_sys_errs, [0, 0, 0]),
            )
            S_floor_hst = L_floor_hst @ L_floor_hst.T
            _dpm_hst_init = (np.asarray(hst_pm_sys_init, dtype=float)
                             if hst_pm_sys_init is not None else np.zeros(2))
            delta_pm_sys_hst = pm.Normal("delta_pm_sys_hst", mu=0.0, sigma=0.5,
                                          shape=2, initval=_dpm_hst_init)
            pm_shift_hst = pt.concatenate(
                [delta_pm_sys_hst,
                 pt.as_tensor_variable(np.zeros(1, dtype=np.float64))])

        # --- Z-latent image-transformation modes (BP3M v2 only) ---
        # z ~ Normal(0, I, M) shared across ALL BP3M stars; encodes image
        # alignment uncertainty explicitly instead of inflating C_obs.
        # S_latent_hst_a is padded to length N (zeros for Gaia-only stars).
        _has_latent = (S_latent_hst_a is not None)
        if _has_latent:
            _N_modes = S_latent_hst_a.shape[2]
            z_latent = pm.Normal("z_latent", mu=0.0, sigma=1.0,
                                  shape=_N_modes, initval=np.zeros(_N_modes))
            S_lat_a_pt = pt.as_tensor_variable(
                S_latent_hst_a.astype(np.float64))   # (N, 3, M)
            # y_z_corr_a: (N, 3) — zero for Gaia stars (S=0 there)
            y_z_corr_a = (S_lat_a_pt * z_latent[None, None, :]).sum(axis=-1)

        # --- Total per-star measurement covariance (N, 3, 3) ---
        # For Group A HST stars: k_hst² · S_obs + S_floor_hst.
        # For Gaia-only stars:   k(G)² · S_obs + S_floor.
        # k is the *error* inflation factor, so covariance inflation is k².
        if _has_hst_main:
            S_total_gaia_part = k_per_star[:, None, None]**2 * S_obs + S_floor
            S_total_hst_part  = k_hst**2 * S_obs + S_floor_hst
            _m33 = pt.as_tensor_variable(
                is_hst_main.astype(np.float64)[:, None, None])
            S_obs_total = (1.0 - _m33) * S_total_gaia_part + _m33 * S_total_hst_part
        else:
            S_obs_total = k_per_star[:, None, None]**2 * S_obs + S_floor

        # --- PM zero-point systematic (Gaia; only when QSO data provided) ---
        has_qso = y_obs_qso is not None and len(y_obs_qso) > 0
        if has_qso:
            _dpm_init = (np.asarray(delta_pm_sys_init, dtype=float)
                         if delta_pm_sys_init is not None else np.zeros(2))
            delta_pm_sys = pm.Normal("delta_pm_sys", mu=0.0, sigma=0.1,
                                      shape=2, initval=_dpm_init)
            pm_shift_gaia = pt.concatenate(
                [delta_pm_sys, pt.as_tensor_variable(np.zeros(1, dtype=np.float64))]
            )
        else:
            pm_shift_gaia = pt.as_tensor_variable(np.zeros(3, dtype=np.float64))

        # y_obs_eff: observations shifted by the appropriate PM zero-point.
        # Gaia stars use pm_shift_gaia; HST Group A stars use pm_shift_hst.
        # In z-latent mode y_obs already contains conditional means for Group A;
        # y_z_corr_a adds the alignment correction (zero for Gaia-only stars).
        if _has_hst_main:
            y_eff_gaia = y_obs - pm_shift_gaia[None, :]
            y_eff_hst  = y_obs - pm_shift_hst[None, :]
            _m1 = pt.as_tensor_variable(
                is_hst_main.astype(np.float64)[:, None])
            y_obs_eff = (1.0 - _m1) * y_eff_gaia + _m1 * y_eff_hst
        elif has_qso:
            y_obs_eff = y_obs - pm_shift_gaia[None, :]
        else:
            y_obs_eff = y_obs
        if _has_latent:
            y_obs_eff = y_obs_eff + y_z_corr_a

        # --- Dwarf component ---
        mu_dwarf = pm.Normal("mu_dwarf",
                              mu=pm_and_para_mean,
                              sigma=pm_and_para_mean_scatters,
                              shape=3,
                              initval=np.array(pm_and_para_mean))
        _sig_pm  = max(float(vlos_implied_pm_sig),  1e-3)
        _sig_plx = max(float(mean_parallax_err),    1e-3)
        L_dwarf, corr_dwarf, stds_dwarf = pm.LKJCholeskyCov(
            "chol_intrinsic_dwarf", n=3, eta=2.0,
            sd_dist=pm.LogNormal.dist(
                mu=[np.log(_sig_pm), np.log(_sig_pm), np.log(_sig_plx)],
                sigma=[0.3, 0.3, 0.3],
            ),
            initval=get_packed_init([_sig_pm, _sig_pm, _sig_plx], [0, 0, 0]),
        )
        intrinsic_cov_dwarf = L_dwarf @ L_dwarf.T

        # --- 3-class flag: dwarf + MW background + QSO ---
        # True when per-source QSO priors are provided (--qso-correction).
        has_3class = log_prior_qso is not None

        # --- Field-level membership fraction ---
        # f_dwarf = P(dwarf | star).  In the 2-class model this equals P(dwarf).
        # In the 3-class model f_star = P(star) is a separate Beta variable and
        # P(dwarf) = f_star × f_dwarf.  Concentration ≈ 5 keeps the prior weak.
        if f_dwarf_prior is not None:
            _f0   = float(np.clip(f_dwarf_prior, 0.001, 0.999))
            _conc = 5.0
            f_dwarf = pm.Beta("f_dwarf",
                               alpha=max(1.0, _f0 * _conc),
                               beta =max(1.0, (1.0 - _f0) * _conc),
                               initval=_f0)
        else:
            f_dwarf = pm.Beta("f_dwarf", alpha=1.0, beta=1.0, initval=0.5)

        # f_star = P(source is a star)  [only in 3-class model]
        if has_3class:
            _f_qso   = float(np.clip(f_qso_prior if f_qso_prior is not None
                                     else 0.05, 0.001, 0.999))
            _conc_fs = 5.0
            f_star   = pm.Beta("f_star",
                                alpha=max(1.0, (1.0 - _f_qso) * _conc_fs),
                                beta =max(1.0, _f_qso           * _conc_fs),
                                initval=float(np.clip(1.0 - _f_qso, 0.01, 0.99)))
            log_fstar = pt.log(f_star)
        else:
            log_fstar = pt.as_tensor_variable(np.float64(0.0))  # log(1) — no QSO class

        # --- Dwarf kinematic likelihood ---
        comp_dwarf = pm.MvNormal.dist(mu=mu_dwarf,
                                       cov=intrinsic_cov_dwarf + S_obs_total)
        logL_dwarf = pm.logp(comp_dwarf, y_obs_eff)              # (N,)
        log_p_dwarf = (log_fstar + pt.log(f_dwarf) + logL_dwarf
                       + log_spatial_dwarf + log_prior_ws[:, 0])

        # --- K fixed background components (sklearn pre-fit, no new MCMC params) ---
        # bg_covs[k] is the sklearn total observed covariance (intrinsic + average
        # measurement noise).  We do NOT add S_obs_total to avoid double-counting.
        # Background uses raw y_obs (not shifted): the PM zero-point shift is small
        # relative to the spread of each background component, so decoupling the
        # background from delta_pm_sys simplifies the posterior and speeds sampling.
        logL_bg_parts = []
        for k_idx in range(K):
            comp_bg   = pm.MvNormal.dist(mu=bg_means[k_idx],
                                          cov=bg_covs[k_idx])
            logL_bg_k = pm.logp(comp_bg, y_obs)                 # (N,) raw y_obs
            logL_bg_parts.append(log_bg_w[k_idx] + logL_bg_k)

        # logsumexp over K background components, then add spatial + photometric weight
        logL_bg_mix = pm.math.logsumexp(pt.stack(logL_bg_parts), axis=0)  # (N,)
        log_p_bg    = (log_fstar + pt.log(1.0 - f_dwarf) + logL_bg_mix
                       + log_spatial_mw + log_pw_bg)

        # --- QSO kinematic component (3-class only) ---
        # QSO true PM = 0; in the shifted frame (y_obs_eff) the expected PM is also
        # 0.  Only measurement errors + floor contribute to the covariance.
        if has_3class:
            log_prior_qso_pt = pt.as_tensor_variable(log_prior_qso.astype(np.float64))
            comp_qso_main    = pm.MvNormal.dist(mu=pt.zeros(3), cov=S_obs_total)
            logL_qso_main    = pm.logp(comp_qso_main, y_obs_eff)      # (N,)
            log_p_qso        = (pt.log(1.0 - f_star) + logL_qso_main
                                + log_spatial_mw + log_prior_qso_pt)

        # Uniform floor prevents -inf from extreme PM outliers that fall
        # outside all model components (e.g. nearby high-PM stars).
        log_p_floor = pt.full_like(log_p_dwarf, np.log(1e-300))

        # Normalise by the per-star CMD marginal so that f_dwarf (and f_star) are
        # identified by kinematic + spatial evidence only (same reasoning as
        # spatial model).  In the 3-class case the QSO photometric prior also
        # enters so that the model fraction is not driven by CMD information alone.
        if has_3class:
            log_norm_cmd = pm.math.logsumexp(
                pt.stack([
                    log_fstar + pt.log(f_dwarf)       + log_prior_ws[:, 0],
                    log_fstar + pt.log(1.0 - f_dwarf) + log_pw_bg,
                    pt.log(1.0 - f_star)              + log_prior_qso_pt,
                ]),
                axis=0,
            )
            logL_all = pm.math.logsumexp(
                pt.stack([log_p_dwarf, log_p_bg, log_p_qso, log_p_floor]),
                axis=0,
            )
        else:
            log_norm_cmd = pm.math.logsumexp(
                pt.stack([pt.log(f_dwarf)       + log_prior_ws[:, 0],
                          pt.log(1.0 - f_dwarf) + log_pw_bg]),
                axis=0,
            )
            logL_all = pm.math.logsumexp(
                pt.stack([log_p_dwarf, log_p_bg, log_p_floor]),
                axis=0,
            )
        pm.Potential("likelihood", (logL_all - log_norm_cmd).sum())

        # --- Soft ordering constraint: dwarf intrinsic dispersion << background ---
        mean_bg_var = float(
            bg_weights @ np.array([np.trace(c) for c in bg_covs])
        )
        var_dwarf = pt.sum(pt.diag(intrinsic_cov_dwarf))
        excess = pt.maximum(0.0, var_dwarf - mean_bg_var)
        pm.Potential("dwarf_compact_constraint",
                     -0.5 * pt.square(excess / (0.5 * mean_bg_var)))

        # --- QSO training likelihood (constrains delta_pm_sys, k, k_1, chol_floor) ---
        # QSO true PM = 0; observed PM ≈ pm_shift_gaia + measurement noise.
        # When log_prior_qso_train is provided, each training source enters as a
        # soft mixture of the QSO and MW background components, weighted by the
        # per-source catalog prior (0.99 MILLIQUAS, 0.50 Gaia-only).  This
        # down-weights MW contaminants that survived the cleaning pipeline.
        # Without log_prior_qso_train all sources are treated as certain QSOs.
        if has_qso:
            M_qso = len(y_obs_qso)
            S_obs_qso_pt = pt.as_tensor_variable(S_obs_qso.astype(np.float64))
            y_obs_qso_pt = pt.as_tensor_variable(y_obs_qso.astype(np.float64))

            if use_kg and gmags_qso is not None:
                gmags_qso_safe = np.where(np.isfinite(gmags_qso), gmags_qso, G_ref)
                dG_qso = (pt.as_tensor_variable(gmags_qso_safe.astype(np.float64))
                          - G_ref)
                k_per_qso = pt.maximum(
                    k * pt.exp(pt.clip(k_1 * dG_qso, -6.0, 6.0)), 0.1)
            else:
                k_per_qso = k * pt.ones(M_qso)

            S_total_qso = (k_per_qso[:, None, None]**2 * S_obs_qso_pt
                           + S_floor[None, :, :])

            # QSO kinematic likelihood: zero-mean MvNormal in shifted Gaia frame (M,)
            comp_qso_train = pm.MvNormal.dist(mu=pm_shift_gaia, cov=S_total_qso)
            logL_qso_train = pm.logp(comp_qso_train, y_obs_qso_pt)

            if log_prior_qso_train is not None:
                # Soft mixture: P(QSO|catalog) * L_qso + P(star|catalog) * L_bg
                log_pq_t  = pt.as_tensor_variable(
                    log_prior_qso_train.astype(np.float64))        # (M,)
                log_ps_t  = pt.log(1.0 - pt.exp(log_pq_t))        # log P(star)

                # Background likelihood for training sources: raw y_obs_qso (no
                # shift), same K-component pre-fit GMM as the main field.
                bg_logL_train = []
                for k_idx in range(K):
                    comp_bg_t = pm.MvNormal.dist(mu=bg_means[k_idx],
                                                  cov=bg_covs[k_idx])
                    bg_logL_train.append(log_bg_w[k_idx]
                                         + pm.logp(comp_bg_t, y_obs_qso_pt))
                logL_bg_train = pm.math.logsumexp(
                    pt.stack(bg_logL_train), axis=0)               # (M,)

                logL_qso_soft = pm.math.logsumexp(
                    pt.stack([log_pq_t + logL_qso_train,
                               log_ps_t + logL_bg_train]),
                    axis=0)                                         # (M,)
                pm.Potential("qso_likelihood", logL_qso_soft.sum())
            else:
                pm.Potential("qso_likelihood", logL_qso_train.sum())

        # --- Group B likelihood (HST-only stars, G > 20.7, frozen spatial prior) ---
        # These stars are not in the main Gaia dataset.  Their spatial density is
        # precomputed from the step-5 posterior and passed as a numpy constant.
        # They share f_dwarf with the main dataset but use the HST error model.
        if _has_hst_b:
            N_B = len(y_obs_hst_b)
            S_obs_b_pt  = pt.as_tensor_variable(S_obs_hst_b.astype(np.float64))
            y_obs_b_pt  = pt.as_tensor_variable(y_obs_hst_b.astype(np.float64))
            log_sp_b_pt = pt.as_tensor_variable(log_spatial_hst_b.astype(np.float64))
            log_sp_b_mw = pt.log(pt.as_tensor_variable(
                np.float64(1.0 / float(hst_area))))

            # Photometric prior weights for Group B
            _lpws_b = log_prior_ws_hst_b  # (N_B, N_clusters) numpy
            _lpw_bg_b = np.copy(_lpws_b[:, 1])
            for _j in range(2, _lpws_b.shape[1]):
                _lpw_bg_b = np.logaddexp(_lpw_bg_b, _lpws_b[:, _j])

            S_total_b = k_hst**2 * S_obs_b_pt + S_floor_hst[None, :, :]
            y_eff_b   = y_obs_b_pt - pm_shift_hst[None, :]
            if _has_latent and S_latent_hst_b is not None:
                S_lat_b_pt  = pt.as_tensor_variable(
                    S_latent_hst_b.astype(np.float64))  # (N_B, 3, M)
                y_z_corr_b  = (S_lat_b_pt * z_latent[None, None, :]).sum(axis=-1)
                y_eff_b     = y_eff_b + y_z_corr_b

            comp_b_dw   = pm.MvNormal.dist(
                mu=mu_dwarf, cov=intrinsic_cov_dwarf + S_total_b)
            logL_b_dw   = pm.logp(comp_b_dw, y_eff_b)               # (N_B,)
            log_p_b_dw  = (pt.log(f_dwarf) + logL_b_dw
                           + log_sp_b_pt + _lpws_b[:, 0])

            logL_b_bg_parts = []
            for k_idx in range(K):
                comp_b_bg_k = pm.MvNormal.dist(mu=bg_means[k_idx],
                                                cov=bg_covs[k_idx])
                logL_b_bg_parts.append(
                    log_bg_w[k_idx] + pm.logp(comp_b_bg_k, y_obs_b_pt))
            logL_b_bg_mix = pm.math.logsumexp(
                pt.stack(logL_b_bg_parts), axis=0)                   # (N_B,)
            log_p_b_bg  = (pt.log(1.0 - f_dwarf) + logL_b_bg_mix
                           + log_sp_b_mw + _lpw_bg_b)

            log_p_b_floor = pt.full_like(log_p_b_dw, np.log(1e-300))
            log_norm_b    = pm.math.logsumexp(
                pt.stack([pt.log(f_dwarf)       + _lpws_b[:, 0],
                          pt.log(1.0 - f_dwarf) + _lpw_bg_b]),
                axis=0,
            )
            logL_b_all = pm.math.logsumexp(
                pt.stack([log_p_b_dw, log_p_b_bg, log_p_b_floor]), axis=0)
            pm.Potential("likelihood_hst_b", (logL_b_all - log_norm_b).sum())

        # --- Derived quantities for easy posterior access ---
        pm.Deterministic("pa_deg",     pa_deg)
        pm.Deterministic("sigma_intrinsic_dwarf",     stds_dwarf)
        pm.Deterministic("rho_pm_intrinsic_dwarf",    corr_dwarf[0, 1])
        pm.Deterministic("rho_pmra_parallax_intrinsic_dwarf",  corr_dwarf[0, 2])
        pm.Deterministic("rho_pmdec_parallax_intrinsic_dwarf", corr_dwarf[1, 2])

        pm.Deterministic("sigma_floor",              stds_floor)
        pm.Deterministic("rho_pm_floor",             corr_floor[0, 1])
        pm.Deterministic("rho_pmra_parallax_floor",  corr_floor[0, 2])
        pm.Deterministic("rho_pmdec_parallax_floor", corr_floor[1, 2])

        if _has_hst:
            pm.Deterministic("sigma_floor_hst",              stds_floor_hst)
            pm.Deterministic("rho_pm_floor_hst",             corr_floor_hst[0, 1])
            pm.Deterministic("rho_pmra_parallax_floor_hst",  corr_floor_hst[0, 2])
            pm.Deterministic("rho_pmdec_parallax_floor_hst", corr_floor_hst[1, 2])

    return model


def run_gmm_model(model, draws=2000, tune=2000, chains=4, seed=42):
    """Sample the GMM and return the trace (with prior predictive)."""
    _setup_jax_cache()
    with model:
        trace = pm.sample(
            nuts_sampler="numpyro",
            chains=chains,
            tune=tune,
            draws=draws,
            target_accept=0.95,
            random_seed=seed,
        )
        prior_samples = pm.sample_prior_predictive(draws=10000)
    trace.extend(prior_samples)
    return trace


def extract_gmm_posterior(trace):
    """Return posterior medians and errors for the spatial parameters in the GMM."""
    def _med_err(var):
        vals = np.array(trace.posterior[var])
        med  = float(np.nanmedian(vals))
        err  = float(0.5 * np.diff(np.nanpercentile(vals, [16, 84]))[0])
        return med, err

    result = {}
    for k in ['a_plummer', 'delta_ra_center', 'delta_dec_center',
              'ellipticity', 'pa_deg']:
        result[f'new_{k}'], result[f'new_{k}_err'] = _med_err(k)
    return result
