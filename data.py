"""
Data loading and kinematic preprocessing.
"""

import os
import warnings
import numpy as np
import pandas as pd

from config import (
    LVD_CATALOG_PATH, GAIA_DATA_PATH, MILLIQUAS_PATH,
    LVD_TRANSLATION, GMAG_LIMITS, FIELD_SPATIAL_CUTOFFS, PROP_DICT,
    SAGITTARIUS_RADEC_CENTER,
    PARALLAX_SYS_ERR, PM_SYS_ERR, GAIA_MEAN_INFLATE,
    PM_TO_VEL_FACT, PRIOR_MEAN_UNCERT,
    GAIA_BG_RADIUS_FACTOR, GAIA_MIN_RADIUS_DEG, GAIA_MAX_RADIUS_DEG,
    GAIA_BG_SPATIAL_THRESHOLD, GAIA_BG_MIN_STARS,
)


# Columns that must be present in a valid main-catalog cache file.
# The cache is written after _standardise_gaia_df(), so column names are
# the standardised forms (gmag, bpmag, etc.), not the raw TAP names.
_GAIA_REQUIRED_COLS = frozenset({
    'source_id', 'ra', 'dec',
    'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'pmra_pmdec_corr',
    'parallax', 'parallax_error', 'parallax_pmra_corr', 'parallax_pmdec_corr',
    'gmag', 'gmag_error', 'bpmag', 'bpmag_error', 'rpmag', 'rpmag_error',
    'bp_rp', 'bp_rp_error', 'ruwe',
})

# Columns that must be present in a valid QSO candidate cache file.
# The QSO cache stores the raw TAP result (not standardised).
_QSO_REQUIRED_COLS = frozenset({
    'source_id', 'ra', 'dec',
    'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr',
    'parallax', 'parallax_error', 'parallax_pmra_corr', 'parallax_pmdec_corr',
    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ruwe',
    'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error',
})


def load_lvd_catalog():
    return pd.read_csv(LVD_CATALOG_PATH)


def resolve_field_name(field, lvd_df=None):
    """
    Resolve *field* to the canonical CLI key used in LVD_TRANSLATION.

    Matching order:
    1. Exact match in LVD_TRANSLATION — return as-is.
    2. Case/separator-insensitive match on LVD_TRANSLATION keys
       (strips spaces, underscores, hyphens, lowercases).
    3. Case-insensitive match against the LVD catalog 'key' and 'name'
       columns, then back-maps the matched row's 'key' to a
       LVD_TRANSLATION entry.

    Raises ValueError (listing available fields) if nothing matches.
    """
    def _norm(s):
        return s.lower().replace(' ', '').replace('_', '').replace('-', '')

    # 1. Exact
    if field in LVD_TRANSLATION:
        return field

    # 2. Case/separator-insensitive against LVD_TRANSLATION keys
    norm_in = _norm(field)
    for canonical in LVD_TRANSLATION:
        if _norm(canonical) == norm_in:
            print(f'  Note: resolved field "{field}" → "{canonical}"')
            return canonical

    # 3. Match against LVD catalog columns
    if lvd_df is not None:
        # Build reverse map: lvd_key → cli_key
        rev = {v: k for k, v in LVD_TRANSLATION.items()}
        for col in ['key', 'name']:
            if col not in lvd_df.columns:
                continue
            for val in lvd_df[col].dropna().astype(str):
                if _norm(val) == norm_in:
                    # val matched; find the LVD 'key' for this row
                    matched_rows = lvd_df[lvd_df[col].astype(str) == val]
                    if matched_rows.empty:
                        continue
                    lvd_key = str(matched_rows['key'].values[0])
                    if lvd_key in rev:
                        canonical = rev[lvd_key]
                        print(f'  Note: resolved field "{field}" → "{canonical}" '
                              f'(matched LVD {col}="{val}")')
                        return canonical

    available = '\n  '.join(sorted(LVD_TRANSLATION.keys()))
    raise ValueError(
        f'Field "{field}" not recognised.\n'
        f'Available fields:\n  {available}\n'
        f'To add a new field, add an entry to LVD_TRANSLATION in config.py.'
    )


def _auto_search_radius_deg(field, lvd_df):
    """
    Search radius = GAIA_BG_RADIUS_FACTOR × rhalf, clamped to
    [GAIA_MIN_RADIUS_DEG, GAIA_MAX_RADIUS_DEG].
    """
    lvd_key = LVD_TRANSLATION.get(field)
    rhalf_arcmin = np.nan
    if lvd_key and lvd_df is not None:
        row = lvd_df.loc[lvd_df['key'] == lvd_key]
        if not row.empty:
            for col in ['rhalf', 'rcore', 'rad_sersic']:
                if col in row.columns:
                    v = float(row[col].values[0])
                    if np.isfinite(v) and v > 0:
                        rhalf_arcmin = v
                        break
    if not np.isfinite(rhalf_arcmin):
        rhalf_arcmin = 6.0
    radius = GAIA_BG_RADIUS_FACTOR * rhalf_arcmin / 60.0
    return float(np.clip(radius, GAIA_MIN_RADIUS_DEG, GAIA_MAX_RADIUS_DEG))


def _get_field_center(field, lvd_df):
    """Return (ra, dec) in degrees for the field centre."""
    if field == 'Sagittarius_dSph':
        return tuple(float(x) for x in SAGITTARIUS_RADEC_CENTER)
    pd_entry = PROP_DICT.get(field, {})
    if pd_entry.get('ra') is not None:
        return float(pd_entry['ra']), float(pd_entry['dec'])
    lvd_key = LVD_TRANSLATION.get(field)
    if lvd_key and lvd_df is not None:
        row = lvd_df.loc[lvd_df['key'] == lvd_key]
        if not row.empty:
            return float(row['ra'].values[0]), float(row['dec'].values[0])
    raise ValueError(f'Cannot determine centre coordinates for field "{field}"')


def _query_gaia_dr3(ra_deg, dec_deg, radius_deg):
    """
    TAP cone search on gaiadr3.gaia_source.  Returns a raw astropy Table.
    Stars without a 5- or 6-parameter astrometric solution are excluded so
    the result is limited to sources with proper motions and parallax.
    """
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
    Gaia.ROW_LIMIT = -1

    query = f"""
    SELECT source_id, ra, dec,
           pmra, pmra_error, pmdec, pmdec_error, pmra_pmdec_corr,
           parallax, parallax_error, parallax_pmra_corr, parallax_pmdec_corr,
           phot_g_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error,
           phot_bp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error,
           phot_rp_mean_mag, phot_rp_mean_flux, phot_rp_mean_flux_error,
           bp_rp, ruwe
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, {radius_deg:.6f})
    ) = 1
    AND astrometric_params_solved >= 31
    """

    print(f'  Querying Gaia DR3: centre ({ra_deg:.4f}, {dec_deg:.4f}), '
          f'r = {radius_deg * 60:.1f} arcmin...')
    job = Gaia.launch_job_async(query, dump_to_file=False, verbose=False)
    return job.get_results()


def _standardise_gaia_df(tbl):
    """
    Convert a raw Gaia TAP Table to the DataFrame column layout that
    compute_kinematics() expects.  Computes magnitude errors from fluxes.
    """
    import math
    df = tbl.to_pandas()
    _c = 2.5 / math.log(10)

    for flux_col, ferr_col, err_col in [
        ('phot_g_mean_flux',  'phot_g_mean_flux_error',  'gmag_error'),
        ('phot_bp_mean_flux', 'phot_bp_mean_flux_error', 'bpmag_error'),
        ('phot_rp_mean_flux', 'phot_rp_mean_flux_error', 'rpmag_error'),
    ]:
        flux = df[flux_col].to_numpy(dtype=float)
        ferr = df[ferr_col].to_numpy(dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            df[err_col] = np.where(flux > 0, _c * ferr / flux, np.nan)

    df['bp_rp_error'] = np.sqrt(df['bpmag_error']**2 + df['rpmag_error']**2)
    df = df.rename(columns={
        'phot_g_mean_mag':  'gmag',
        'phot_bp_mean_mag': 'bpmag',
        'phot_rp_mean_mag': 'rpmag',
    })
    drop_cols = [c for c in df.columns if 'flux' in c]
    return df.drop(columns=drop_cols, errors='ignore')


def query_gaia_qso_candidates(ra_deg, dec_deg, radius_deg, cache_path, redownload=False):
    """
    Download Gaia DR3 QSO candidates within radius_deg of (ra_deg, dec_deg).

    Filters on the native in_qso_candidates flag with quality cuts:
    ruwe < 1.4, ipd_gof_harmonic_amplitude < 0.2, G < 20.5.
    Results are cached at cache_path (gzipped CSV).
    """
    need_download = redownload or not os.path.exists(cache_path)

    if not need_download:
        print(f'  Loading cached QSO candidates: {os.path.basename(cache_path)}')
        df = pd.read_csv(cache_path)
        missing = _QSO_REQUIRED_COLS - set(df.columns)
        if missing:
            print(f'  Cache missing columns: {sorted(missing)} — re-downloading...')
            need_download = True
        else:
            return df

    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
    Gaia.ROW_LIMIT = -1

    query = f"""
    SELECT source_id, ra, dec,
           pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr,
           parallax, parallax_error, parallax_pmra_corr, parallax_pmdec_corr,
           phot_g_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error,
           phot_bp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error,
           phot_rp_mean_mag, phot_rp_mean_flux, phot_rp_mean_flux_error,
           ruwe
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, {radius_deg:.6f})
    ) = 1
    AND in_qso_candidates = 'True'
    AND astrometric_params_solved >= 31
    AND pmra IS NOT NULL
    AND pmdec IS NOT NULL
    AND parallax IS NOT NULL
    AND ruwe < 1.4
    AND ipd_gof_harmonic_amplitude < 0.2
    AND phot_g_mean_mag < 20.5
    """

    print(f'  Querying Gaia DR3 QSO candidates: centre ({ra_deg:.4f}, {dec_deg:.4f}), '
          f'r = {radius_deg:.1f}°...')
    job = Gaia.launch_job_async(query, dump_to_file=False, verbose=False)
    df  = job.get_results().to_pandas()

    import math
    _c = 2.5 / math.log(10)
    for flux_col, ferr_col, err_col in [
        ('phot_g_mean_flux',  'phot_g_mean_flux_error',  'phot_g_mean_mag_error'),
        ('phot_bp_mean_flux', 'phot_bp_mean_flux_error', 'phot_bp_mean_mag_error'),
        ('phot_rp_mean_flux', 'phot_rp_mean_flux_error', 'phot_rp_mean_mag_error'),
    ]:
        flux = df[flux_col].to_numpy(dtype=float)
        ferr = df[ferr_col].to_numpy(dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            df[err_col] = np.where(flux > 0, _c * ferr / flux, np.nan)
    df = df.drop(columns=[c for c in df.columns if 'flux' in c], errors='ignore')

    df.to_csv(cache_path, index=False, compression='gzip')
    print(f'  Downloaded {len(df):,} QSO candidates → {os.path.basename(cache_path)}')
    return df


def crossmatch_milliquas(gaia_ra, gaia_dec, milliquas_path=None,
                          match_radius_arcsec=1.5):
    """
    Cross-match Gaia positions with the MILLIQUAS catalog using a 3-D
    unit-vector KD-tree (exact great-circle distances for small angles).

    Parameters
    ----------
    gaia_ra, gaia_dec   : (N,) degrees
    milliquas_path      : path to milliquas.fits; defaults to MILLIQUAS_PATH
    match_radius_arcsec : matching radius (default 1.5 arcsec)

    Returns
    -------
    matched : (N,) bool — True where a MILLIQUAS counterpart is found
    """
    from astropy.io import fits
    from scipy.spatial import cKDTree

    if milliquas_path is None:
        milliquas_path = MILLIQUAS_PATH

    print(f'  Loading MILLIQUAS from {os.path.basename(milliquas_path)}...')
    with fits.open(milliquas_path) as hdul:
        mq_ra  = hdul[1].data['RA'].astype(np.float64)
        mq_dec = hdul[1].data['DEC'].astype(np.float64)
    print(f'  {len(mq_ra):,} MILLIQUAS sources loaded')

    def _xyz(ra_deg, dec_deg):
        ra  = np.deg2rad(np.asarray(ra_deg,  dtype=np.float64))
        dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
        cd  = np.cos(dec)
        return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])

    gaia_xyz = _xyz(gaia_ra,  gaia_dec)
    mq_xyz   = _xyz(mq_ra,    mq_dec)

    tree    = cKDTree(mq_xyz)
    r_rad   = np.deg2rad(match_radius_arcsec / 3600.0)
    r_chord = 2.0 * np.sin(r_rad / 2.0)   # chord length ≈ θ for small angles

    dists, _ = tree.query(gaia_xyz, k=1, distance_upper_bound=r_chord * 1.01)
    return dists <= r_chord


def measure_pm_systematic(qso_df, min_qsos=20,
                           parallax_nsigma=3.0, pm_nsigma=3.0, n_iter=10):
    """
    Inverse-variance weighted mean PM of QSO candidates, with two-stage cleaning.

    Stage 1 — parallax cut: removes sources with |parallax| / parallax_error >
    parallax_nsigma.  True QSOs are at cosmological distances (parallax ≈ 0);
    stars with significantly non-zero parallax are rejected here.

    Stage 2 — iterative PM sigma-clipping: compute the inverse-variance
    weighted mean (mu_ra, mu_dec), remove sources where
    |pmra − mu_ra| / pmra_error > pm_nsigma or
    |pmdec − mu_dec| / pmdec_error > pm_nsigma, then repeat until convergence.
    Clipping relative to the running mean avoids biasing the result toward zero.

    Returns
    -------
    dict with keys:
        delta_pmra, delta_pmdec : float or None (None if < min_qsos survived)
        sigma_pmra, sigma_pmdec : float or None
        n_kept                  : int  — QSOs surviving all cleaning
        n_raw                   : int  — total rows in qso_df
        mask_valid              : (N,) bool — finite PM + error
        mask_plx_kept           : (N,) bool — survived parallax cut
        mask_kept               : (N,) bool — survived all cleaning
        parallax_nsigma, pm_nsigma : floats (echoed for plot labels)
    """
    N     = len(qso_df)
    pmra  = qso_df['pmra'].to_numpy(dtype=float)
    pmdec = qso_df['pmdec'].to_numpy(dtype=float)
    e_ra  = qso_df['pmra_error'].to_numpy(dtype=float)
    e_dec = qso_df['pmdec_error'].to_numpy(dtype=float)

    ok = (np.isfinite(pmra) & np.isfinite(pmdec)
          & np.isfinite(e_ra) & np.isfinite(e_dec)
          & (e_ra > 0) & (e_dec > 0))

    # Stage 1: parallax consistency cut
    ok_plx = ok.copy()
    if 'parallax' in qso_df.columns and 'parallax_error' in qso_df.columns:
        plx    = qso_df['parallax'].to_numpy(dtype=float)
        e_plx  = qso_df['parallax_error'].to_numpy(dtype=float)
        plx_ok = np.isfinite(plx) & np.isfinite(e_plx) & (e_plx > 0)
        ok_plx = ok & (~plx_ok | (np.abs(plx) < parallax_nsigma * e_plx))

    def _no_result(mask_kept):
        return dict(
            delta_pmra=None, delta_pmdec=None,
            sigma_pmra=None, sigma_pmdec=None,
            n_kept=int(mask_kept.sum()), n_raw=N,
            mask_valid=ok, mask_plx_kept=ok_plx, mask_kept=mask_kept,
            parallax_nsigma=parallax_nsigma, pm_nsigma=pm_nsigma,
        )

    if ok_plx.sum() < min_qsos:
        return _no_result(ok_plx)

    # Stage 2: iterative PM sigma-clipping around the running mean
    mask = ok_plx.copy()
    for _ in range(n_iter):
        w_ra  = 1.0 / e_ra[mask]**2
        w_dec = 1.0 / e_dec[mask]**2
        mu_ra  = np.sum(w_ra  * pmra[mask])  / np.sum(w_ra)
        mu_dec = np.sum(w_dec * pmdec[mask]) / np.sum(w_dec)

        chi_ra  = np.abs(pmra  - mu_ra)  / e_ra
        chi_dec = np.abs(pmdec - mu_dec) / e_dec
        new_mask = ok_plx & (chi_ra < pm_nsigma) & (chi_dec < pm_nsigma)
        if new_mask.sum() == mask.sum():
            break
        if new_mask.sum() < min_qsos:
            break
        mask = new_mask

    if mask.sum() < min_qsos:
        return _no_result(mask)

    w_ra  = 1.0 / e_ra[mask]**2
    w_dec = 1.0 / e_dec[mask]**2
    delta_pmra  = float(np.sum(w_ra  * pmra[mask])  / np.sum(w_ra))
    delta_pmdec = float(np.sum(w_dec * pmdec[mask]) / np.sum(w_dec))
    sigma_pmra  = float(np.sqrt(1.0 / np.sum(w_ra)))
    sigma_pmdec = float(np.sqrt(1.0 / np.sum(w_dec)))

    return dict(
        delta_pmra=delta_pmra, delta_pmdec=delta_pmdec,
        sigma_pmra=sigma_pmra, sigma_pmdec=sigma_pmdec,
        n_kept=int(mask.sum()), n_raw=N,
        mask_valid=ok, mask_plx_kept=ok_plx, mask_kept=mask,
        parallax_nsigma=parallax_nsigma, pm_nsigma=pm_nsigma,
    )


def build_qso_obs_arrays(qso_clean_df):
    """
    Build kinematic + photometric arrays from a cleaned QSO DataFrame.

    Rows with non-finite PM or parallax are silently dropped.

    Returns
    -------
    y_obs_qso   : (M, 3)    [pmra, pmdec, parallax] in mas(/yr)
    S_obs_qso   : (M, 3, 3) raw per-source measurement covariances
    gmags_qso   : (M,)      G magnitudes (NaN where unavailable)
    bpmags_qso  : (M,)      BP magnitudes
    rpmags_qso  : (M,)      RP magnitudes
    gmag_errs   : (M,)      G magnitude errors (NaN where unavailable)
    bpmag_errs  : (M,)      BP magnitude errors
    rpmag_errs  : (M,)      RP magnitude errors
    """
    df = qso_clean_df.copy()
    ok = (np.isfinite(df['pmra'].to_numpy(dtype=float)) &
          np.isfinite(df['pmdec'].to_numpy(dtype=float)) &
          np.isfinite(df['parallax'].to_numpy(dtype=float)))
    df = df[ok].reset_index(drop=True)
    M = len(df)

    pmra  = df['pmra'].to_numpy(dtype=float)
    pmdec = df['pmdec'].to_numpy(dtype=float)
    plx   = df['parallax'].to_numpy(dtype=float)
    e_ra  = df['pmra_error'].to_numpy(dtype=float)
    e_dec = df['pmdec_error'].to_numpy(dtype=float)
    e_plx = df['parallax_error'].to_numpy(dtype=float)

    y_obs_qso = np.column_stack([pmra, pmdec, plx])

    S = np.zeros((M, 3, 3))
    S[:, 0, 0] = e_ra**2
    S[:, 1, 1] = e_dec**2
    S[:, 2, 2] = e_plx**2

    for (i, j), col in [((0, 1), 'pmra_pmdec_corr'),
                         ((0, 2), 'parallax_pmra_corr'),
                         ((1, 2), 'parallax_pmdec_corr')]:
        if col in df.columns:
            c = df[col].to_numpy(dtype=float)
            c = np.where(np.isfinite(c), c, 0.0)
            ei = e_ra if i == 0 else e_dec
            ej = e_plx if j == 2 else e_dec
            S[:, i, j] = ei * ej * c
            S[:, j, i] = S[:, i, j]

    def _col(name, fallback=np.nan):
        return (df[name].to_numpy(dtype=float) if name in df.columns
                else np.full(M, fallback))

    gmags_qso  = _col('phot_g_mean_mag')
    bpmags_qso = _col('phot_bp_mean_mag')
    rpmags_qso = _col('phot_rp_mean_mag')
    gmag_errs  = _col('phot_g_mean_mag_error')
    bpmag_errs = _col('phot_bp_mean_mag_error')
    rpmag_errs = _col('phot_rp_mean_mag_error')

    return y_obs_qso, S, gmags_qso, bpmags_qso, rpmags_qso, gmag_errs, bpmag_errs, rpmag_errs


def compute_qso_surface_density(n_milliquas_clean, n_gaia_all, qso_radius_deg):
    """
    Geometric mean of MILLIQUAS (lower bound) and Gaia candidate (upper bound)
    QSO surface densities in QSOs per deg².

    MILLIQUAS-confirmed sources give a lower bound (high purity, incomplete).
    Gaia in_qso_candidates give an upper bound (complete but ~50% contaminated).
    """
    area = np.pi * float(qso_radius_deg)**2
    rho_lower = max(n_milliquas_clean, 0) / area
    rho_upper = max(n_gaia_all, 0) / area
    return float(np.sqrt(rho_lower * rho_upper))


def load_gaia_data(field, lvd_df=None, search_radius_deg=None, redownload=False):
    """
    Load Gaia DR3 data for *field*, using a local cache when available.

    Search radius defaults to GAIA_BG_RADIUS_FACTOR × rhalf (from the LVD
    catalog), clamped to [GAIA_MIN_RADIUS_DEG, GAIA_MAX_RADIUS_DEG].  This
    is large enough to include a generous background population without
    relying on any kinematic cuts to define it.

    Falls back to a legacy GaiaHub CSV if a download is not possible.
    Applies RUWE < 1.4 quality cut before returning.
    """
    # Cache lives in the field's output folder for a self-contained run.
    gaia_dir = os.path.join('gal_fitting_results', field)
    os.makedirs(gaia_dir, exist_ok=True)

    if search_radius_deg is None:
        search_radius_deg = _auto_search_radius_deg(field, lvd_df)
    radius_arcmin = search_radius_deg * 60.0
    cache_name    = f'gaia_dr3_{radius_arcmin:.0f}arcmin.csv.gz'
    cache_path    = os.path.join(gaia_dir, cache_name)

    need_download = redownload or not os.path.exists(cache_path)

    if not need_download:
        print(f'  Loading cached Gaia data: {cache_name}')
        df = pd.read_csv(cache_path)
        missing = _GAIA_REQUIRED_COLS - set(df.columns)
        if missing:
            print(f'  Cache missing columns: {sorted(missing)} — re-downloading...')
            need_download = True

    if need_download:
        try:
            ra_c, dec_c = _get_field_center(field, lvd_df)
            tbl = _query_gaia_dr3(ra_c, dec_c, search_radius_deg)
            df  = _standardise_gaia_df(tbl)
            df.to_csv(cache_path, index=False, compression='gzip')
            print(f'  Downloaded {len(df):,} stars → {cache_path}')
        except Exception as exc:
            warnings.warn(f'Gaia download failed ({exc}). Trying legacy CSV.')
            # Fall back to a pre-existing GaiaHub CSV in the old location
            legacy_dir = os.path.join(GAIA_DATA_PATH, field, 'Gaia')
            csvs = (
                sorted(f for f in os.listdir(legacy_dir) if f.endswith('.csv'))
                if os.path.isdir(legacy_dir) else []
            )
            if not csvs:
                raise RuntimeError(
                    f'No Gaia data available for field "{field}". '
                    f'Cache path checked: {cache_path}'
                ) from exc
            df = pd.read_csv(os.path.join(legacy_dir, csvs[0]))
            print(f'  Loaded {len(df):,} stars from legacy CSV {csvs[0]}')

    if 'ruwe' in df.columns:
        n_before = len(df)
        df = df[df['ruwe'] < 1.4].reset_index(drop=True)
        print(f'  RUWE < 1.4: {len(df):,} / {n_before:,} stars kept')
    return df


def get_radec_center(field, lvd_df):
    """Return (ra, dec) centre for *field* from the LVD catalog."""
    if field == 'Sagittarius_dSph':
        return SAGITTARIUS_RADEC_CENTER.copy()
    lvd_key = LVD_TRANSLATION.get(field)
    if lvd_key is None:
        raise ValueError(f'Field "{field}" not found in LVD_TRANSLATION.')
    row = lvd_df.loc[lvd_df['key'] == lvd_key]
    if row.empty:
        raise ValueError(f'LVD key "{lvd_key}" not found in catalog.')
    return row[['ra', 'dec']].to_numpy()[0]


def _read_lvd_row(field, lvd_df):
    lvd_key = LVD_TRANSLATION.get(field)
    if lvd_key is None:
        raise ValueError(f'Field "{field}" not found in LVD_TRANSLATION.')
    row = lvd_df.loc[lvd_df['key'] == lvd_key]
    if row.empty:
        raise ValueError(f'LVD key "{lvd_key}" not found in catalog.')
    return row


def load_field_priors(field, lvd_df):
    """
    Extract per-field priors from the LVD catalog.

    Returns a dict with keys:
        radec_center, mean_pm, mean_pm_err, mean_parallax, mean_parallax_err,
        vlos_implied_pm_sig, vlos_implied_pm_sig_err,
        ellipticity_mean, ellipticity_err,
        rhalf_mean, rhalf_err, pa_mean, pa_err,
        bad_rhalf, pm_and_para_mean, pm_and_para_mean_scatters,
        gmag_limit
    """
    row = _read_lvd_row(field, lvd_df)

    radec_center = get_radec_center(field, lvd_df)

    # Proper motion
    pmra_info  = row[['pmra',  'pmra_em',  'pmra_ep']].to_numpy()[0]
    pmdec_info = row[['pmdec', 'pmdec_em', 'pmdec_ep']].to_numpy()[0]
    if field == 'Leo_I':
        pmra_info, pmdec_info = pmdec_info, pmra_info

    mean_pm = np.array([pmra_info[0], pmdec_info[0]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean_pm_err = np.array([np.nanmean(pmra_info[1:]), np.nanmean(pmdec_info[1:])])
    if not np.all(np.isfinite(mean_pm)):
        mean_pm     = np.array([0.0, 0.0])
        mean_pm_err = np.array([0.5, 0.5])

    # Distance / parallax
    dist_info     = row[['distance', 'distance_em', 'distance_ep']].to_numpy()[0]
    mean_dist     = dist_info[0]
    mean_dist_err = np.nanmean(dist_info[1:])

    mean_parallax     = 1.0 / mean_dist if np.isfinite(mean_dist) else 0.0
    mean_parallax_err = mean_dist_err / mean_dist**2 if np.isfinite(mean_dist) else 0.1
    if not np.isfinite(mean_parallax_err) or mean_parallax_err == 0:
        mean_parallax_err = 0.1 * abs(mean_parallax)

    # Line-of-sight velocity dispersion → implied PM dispersion
    vlos_info         = row[['vlos_sigma', 'vlos_sigma_em', 'vlos_sigma_ep',
                              'vlos_sigma_ul']].to_numpy()[0]
    mean_vlos_sig     = vlos_info[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean_vlos_sig_err = np.nanmean(vlos_info[1:3])

    if np.isfinite(mean_vlos_sig) and np.isfinite(mean_dist) and mean_dist > 0:
        vlos_implied_pm_sig = mean_vlos_sig / (PM_TO_VEL_FACT * mean_dist)
        vlos_implied_pm_sig_err = np.sqrt(
            (mean_vlos_sig_err / (PM_TO_VEL_FACT * mean_dist))**2
            + (mean_dist_err * mean_vlos_sig / (PM_TO_VEL_FACT * mean_dist**2))**2
        )
    else:
        vlos_implied_pm_sig     = 10.0 / (PM_TO_VEL_FACT * mean_dist) if np.isfinite(mean_dist) else 0.1
        vlos_implied_pm_sig_err = vlos_implied_pm_sig

    # Morphology
    ellipticity_info = row[['ellipticity', 'ellipticity_em', 'ellipticity_ep']].to_numpy()[0]
    ellipticity_mean = ellipticity_info[0] if np.isfinite(ellipticity_info[0]) else 0.0
    ellipticity_err  = np.nanmean(ellipticity_info[1:])
    if not np.isfinite(ellipticity_err):
        ellipticity_err = 0.1

    pa_info  = row[['position_angle', 'position_angle_em', 'position_angle_ep']].to_numpy()[0]
    pa_mean  = pa_info[0] if np.isfinite(pa_info[0]) else 0.0
    pa_err   = np.nanmean(pa_info[1:])
    if not np.isfinite(pa_err):
        pa_err = 45.0

    # Half-light radius: try rcore → rhalf → rad_sersic
    rhalf_mean = np.nan
    rhalf_err  = np.nan
    for key in ['rcore', 'rhalf', 'rad_sersic']:
        info = row[[key, f'{key}_em', f'{key}_ep']].to_numpy()[0]
        if np.isfinite(info[0]):
            rhalf_mean = info[0]
            rhalf_err  = np.nanmean(info[1:])
            break

    bad_rhalf = not np.isfinite(rhalf_mean)
    if not np.isfinite(rhalf_err):
        rhalf_err = 0.5 * rhalf_mean if not bad_rhalf else 5.0

    # gmag bright cutoff
    gmag_limit = GMAG_LIMITS.get(field, None)
    if gmag_limit is None:
        gmag_limit = -10000.0

    pm_and_para_mean      = np.array([mean_pm[0], mean_pm[1], mean_parallax])
    pm_and_para_mean_scatters = np.array([PRIOR_MEAN_UNCERT, PRIOR_MEAN_UNCERT,
                                          mean_parallax_err])

    return dict(
        radec_center              = radec_center,
        mean_pm                   = mean_pm,
        mean_pm_err               = mean_pm_err,
        mean_parallax             = mean_parallax,
        mean_parallax_err         = mean_parallax_err,
        vlos_implied_pm_sig       = vlos_implied_pm_sig,
        vlos_implied_pm_sig_err   = vlos_implied_pm_sig_err,
        ellipticity_mean          = ellipticity_mean,
        ellipticity_err           = ellipticity_err,
        rhalf_mean                = rhalf_mean,
        rhalf_err                 = rhalf_err,
        pa_mean                   = pa_mean,
        pa_err                    = pa_err,
        bad_rhalf                 = bad_rhalf,
        gmag_limit                = gmag_limit,
        pm_and_para_mean          = pm_and_para_mean,
        pm_and_para_mean_scatters = pm_and_para_mean_scatters,
    )


def compute_kinematics(gaia_df, priors):
    """
    Build all photometric / astrometric arrays needed for modelling.

    Returns a dict with radec_offsets, pms, pm_and_paras, covariances,
    distances, selection masks, and survey geometry.
    """
    radec_center = priors['radec_center']
    mean_pm      = priors['mean_pm']
    mean_parallax= priors['mean_parallax']

    # --- Positions ---
    radecs         = gaia_df[['ra', 'dec']].to_numpy()
    radec_offsets  = radecs - radec_center
    #account for wrapping around
    radec_offsets[:, 0] = (radec_offsets[:, 0] + 180.0)%360.0 - 180.0
    radec_offsets[:, 0] *= np.cos(np.deg2rad(radec_center[1]))
    radec_dists    = np.sqrt(np.sum(radec_offsets**2, axis=1))

    survey_area       = np.pi * np.nanmax(radec_dists)**2
    survey_mean_width = np.mean(np.nanmax(radec_offsets, axis=0)
                                - np.nanmin(radec_offsets, axis=0))

    # --- Photometry ---
    gmags   = gaia_df[['gmag',  'gmag_error']].to_numpy()
    bpmags  = gaia_df[['bpmag', 'bpmag_error']].to_numpy()
    rpmags  = gaia_df[['rpmag', 'rpmag_error']].to_numpy()
    colors  = gaia_df[['bp_rp', 'bp_rp_error']].to_numpy()

    # --- Proper motions ---
    pms = gaia_df[['pmra', 'pmdec']].to_numpy()

    pm_covs = np.zeros((len(pms), 2, 2))
    pm_covs[:, 0, 0] = gaia_df['pmra_error'].to_numpy()**2
    pm_covs[:, 1, 1] = gaia_df['pmdec_error'].to_numpy()**2
    corr = gaia_df['pmra_pmdec_corr'].to_numpy()
    pm_covs[:, 0, 1] = gaia_df['pmra_error'] * gaia_df['pmdec_error'] * corr
    pm_covs[:, 1, 0] = pm_covs[:, 0, 1]

    pm_inv_covs = np.linalg.inv(pm_covs)
    pm_dists = np.sqrt(np.einsum('ni,ni->n',
                                  pms - mean_pm,
                                  np.einsum('nij,nj->ni', pm_inv_covs, pms - mean_pm)))

    # --- Parallax ---
    parallaxes      = gaia_df['parallax'].to_numpy()
    parallax_errors = gaia_df['parallax_error'].to_numpy()

    # --- Joint PM + parallax (3-D) ---
    pm_and_paras = np.zeros((len(pms), 3))
    pm_and_paras[:, :2] = pms
    pm_and_paras[:,  2] = parallaxes

    pm_and_para_covs = np.zeros((len(pms), 3, 3))
    pm_and_para_covs[:, :2, :2] = pm_covs
    pm_and_para_covs[:,  2,  2] = parallax_errors**2

    corr_pmra_plx = gaia_df['parallax_pmra_corr'].to_numpy()
    corr_pmdec_plx = gaia_df['parallax_pmdec_corr'].to_numpy()
    pm_and_para_covs[:, 0, 2] = gaia_df['pmra_error'] * parallax_errors * corr_pmra_plx
    pm_and_para_covs[:, 2, 0] = pm_and_para_covs[:, 0, 2]
    pm_and_para_covs[:, 1, 2] = gaia_df['pmdec_error'] * parallax_errors * corr_pmdec_plx
    pm_and_para_covs[:, 2, 1] = pm_and_para_covs[:, 1, 2]

    # Inflated inverse covariance for outlier rejection
    mean_pm_err               = priors['mean_pm_err']
    mean_parallax_err         = priors['mean_parallax_err']
    pm_and_para_sys_errs      = np.array([PM_SYS_ERR, PM_SYS_ERR, PARALLAX_SYS_ERR])
    pm_and_para_mean_errs     = np.array([mean_pm_err[0], mean_pm_err[0], mean_parallax_err])

    inflated_covs = (GAIA_MEAN_INFLATE**2 * pm_and_para_covs
                     + np.eye(3) * (pm_and_para_mean_errs * 2)**2
                     + np.eye(3) * pm_and_para_sys_errs**2)
    larger_pm_and_para_inv_covs = np.linalg.inv(inflated_covs)

    pm_and_para_mean = priors['pm_and_para_mean']
    pm_and_para_dists = np.sqrt(np.einsum(
        'ni,ni->n',
        pm_and_paras - pm_and_para_mean,
        np.einsum('nij,nj->ni', larger_pm_and_para_inv_covs,
                  pm_and_paras - pm_and_para_mean)
    ))

    has_pms = np.isfinite(pms[:, 0]) & np.isfinite(parallaxes)
    good_mags = (np.isfinite(gmags[:, 0])
                 & np.isfinite(bpmags[:, 0])
                 & np.isfinite(rpmags[:, 0]))

    return dict(
        radec_offsets               = radec_offsets,
        radec_dists                 = radec_dists,
        survey_area                 = survey_area,
        survey_mean_width           = survey_mean_width,
        gmags                       = gmags,
        bpmags                      = bpmags,
        rpmags                      = rpmags,
        colors                      = colors,
        pms                         = pms,
        pm_covs                     = pm_covs,
        pm_dists                    = pm_dists,
        parallaxes                  = parallaxes,
        parallax_errors             = parallax_errors,
        pm_and_paras                = pm_and_paras,
        pm_and_para_covs            = pm_and_para_covs,
        pm_and_para_dists           = pm_and_para_dists,
        larger_pm_and_para_inv_covs = larger_pm_and_para_inv_covs,
        has_pms                     = has_pms,
        good_mags                   = good_mags,
    )


def apply_field_spatial_cutoff(field, gaia_df, radec_offsets, radec_dists):
    """
    For certain large/contaminated fields, hard-clip the catalogue to a smaller
    region immediately after loading.  Returns updated (gaia_df, radec_offsets,
    radec_dists).
    """
    cutoff = FIELD_SPATIAL_CUTOFFS.get(field)
    if cutoff is None:
        return gaia_df, radec_offsets, radec_dists
    mask = np.all(np.abs(radec_offsets) < cutoff, axis=1)
    return (gaia_df[mask].reset_index(drop=True),
            radec_offsets[mask],
            radec_dists[mask])


def galactic_pm_rotation_matrices(ra, dec):
    """
    Per-star 2×2 rotation matrices that transform (pmra*, pmdec) to
    (pm_l_cosb, pm_b) via astropy's ICRS→Galactic PM transform.

    Parameters
    ----------
    ra, dec : array-like, degrees  (length N)

    Returns
    -------
    R : (N, 2, 2)  — R[i] @ [pmra*, pmdec] = [pm_l_cosb, pm_b]
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra_a  = np.asarray(ra,  dtype=float)
    dec_a = np.asarray(dec, dtype=float)
    N     = len(ra_a)
    kw    = dict(ra=ra_a*u.deg, dec=dec_a*u.deg,
                 distance=np.ones(N)*u.kpc, frame='icrs')

    # Transform two orthogonal unit PM vectors to Galactic frame
    g1 = SkyCoord(pm_ra_cosdec=np.ones(N)*u.mas/u.yr,
                  pm_dec=np.zeros(N)*u.mas/u.yr, **kw).galactic
    g2 = SkyCoord(pm_ra_cosdec=np.zeros(N)*u.mas/u.yr,
                  pm_dec=np.ones(N)*u.mas/u.yr, **kw).galactic

    # Columns of R: R[:, :, 0] = image of e1, R[:, :, 1] = image of e2
    R = np.zeros((N, 2, 2))
    R[:, 0, 0] = g1.pm_l_cosb.to(u.mas/u.yr).value
    R[:, 1, 0] = g1.pm_b.to(u.mas/u.yr).value
    R[:, 0, 1] = g2.pm_l_cosb.to(u.mas/u.yr).value
    R[:, 1, 1] = g2.pm_b.to(u.mas/u.yr).value
    return R


def load_bp3m_data(bp3m_dir):
    """
    Load BP3M stellar astrometry from stellar_astrometry.csv.

    Reads the marginal PM+parallax means and full 3×3 covariances for all
    stars with at least one HST image used (n_hst_used > 0) and a valid
    BP3M proper motion.

    The covariance returned is C_obs = C_vT + v_cov_marginalised — i.e. the
    marginal covariance already summed over the image-transformation
    uncertainty — read directly from the sigma and correlation columns.

    Parameters
    ----------
    bp3m_dir : str or Path
        Directory containing BP3M outputs with stellar_astrometry.csv.

    Returns
    -------
    bp3m_df : pd.DataFrame
        Per-star table (reset index) with columns including:
        Gaia_id, ra, dec, pmra_bp3m, pmdec_bp3m, parallax_bp3m,
        n_hst_used, gmag, bpmag, rpmag, plus raw sigma/corr columns.
    C_pm : (N, 3, 3) ndarray
        Marginal [pmra, pmdec, parallax] covariance matrices.
    """
    csv_path = os.path.join(str(bp3m_dir), 'stellar_astrometry.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'BP3M stellar_astrometry.csv not found: {csv_path}')

    df = pd.read_csv(csv_path)

    pmra_arr = df['pmra_bp3m'].to_numpy(dtype=float)
    has_bp3m = (df['n_hst_used'].to_numpy(dtype=int) > 0) & np.isfinite(pmra_arr)
    df = df[has_bp3m].reset_index(drop=True)
    N = len(df)
    if N == 0:
        raise ValueError(f'No BP3M stars with valid PMs found in {csv_path}')

    s_ra  = df['sigma_pmra_bp3m'].to_numpy(dtype=float)
    s_dec = df['sigma_pmdec_bp3m'].to_numpy(dtype=float)
    s_plx = df['sigma_parallax_bp3m'].to_numpy(dtype=float)
    c_ra_dec = np.where(np.isfinite(df['corr_pmra_pmdec'].to_numpy(dtype=float)),
                        df['corr_pmra_pmdec'].to_numpy(dtype=float), 0.0)
    c_ra_plx = np.where(np.isfinite(df['corr_pmra_plx'].to_numpy(dtype=float)),
                        df['corr_pmra_plx'].to_numpy(dtype=float), 0.0)
    c_dec_plx = np.where(np.isfinite(df['corr_pmdec_plx'].to_numpy(dtype=float)),
                         df['corr_pmdec_plx'].to_numpy(dtype=float), 0.0)

    C_pm = np.zeros((N, 3, 3))
    C_pm[:, 0, 0] = s_ra**2
    C_pm[:, 1, 1] = s_dec**2
    C_pm[:, 2, 2] = s_plx**2
    C_pm[:, 0, 1] = C_pm[:, 1, 0] = s_ra  * s_dec * c_ra_dec
    C_pm[:, 0, 2] = C_pm[:, 2, 0] = s_ra  * s_plx * c_ra_plx
    C_pm[:, 1, 2] = C_pm[:, 2, 1] = s_dec * s_plx * c_dec_plx

    g_a  = df['gmag'].to_numpy(dtype=float)
    g_b  = np.sum(np.isfinite(g_a) & (g_a > 20.7))
    _label = os.path.basename(str(bp3m_dir).rstrip('/\\')) or str(bp3m_dir)
    print(f'  BP3M: {N:,} stars with HST measurements '
          f'({g_b:,} faint, G > 20.7) from {_label}')
    return df, C_pm


def load_bp3m_latent_data(bp3m_dir, sigma_eff_threshold=0.015, lambda_min=1e-8):
    """
    Load BP3M data for the z-latent hierarchical model (v2).

    Reads K_matrices.npz, C_r.npy, C_vT.npy, star_indices.npz, and
    use_for_fit.npz produced by run_bp3m.py and constructs per-star
    sensitivity matrices S_i such that

        v_hat_i(r) ≈ v_hat_cond_i + S_i @ z,   z ~ Normal(0, I)

    where z encodes the image-transformation uncertainty as shared latent
    variables.  S_i @ S_i^T reconstructs the marginalised alignment
    covariance v_cov_marginalised[i] to the precision set by sigma_eff_threshold
    and lambda_min.

    Parameters
    ----------
    bp3m_dir : str or Path
        Directory containing BP3M outputs (K_matrices.npz etc.).
    sigma_eff_threshold : float
        Keep latent modes whose RMS PM sensitivity across stars exceeds
        this value (mas/yr).  Default 0.015 mas/yr retains the ~50 modes
        that account for ~99.5 % of alignment PM variance.
    lambda_min : float
        Minimum clamped C_r eigenvalue; modes below this are discarded
        before the sigma_eff filter.

    Returns
    -------
    S_pm : (N, 3, M) float64 ndarray
        Per-star sensitivity matrices in [pmra, pmdec, parallax] space.
        Rows of v_cov_marginalised[:, 2:5, 2:5] are reproduced by S@S^T.
    C_vT_pm : (N, 3, 3) float64 ndarray
        Conditional [pmra, pmdec, parallax] covariance (at MAP r_hat).
    cond_means : (N, 3) float64 ndarray
        Conditional means [pmra_bp3m_cond, pmdec_bp3m_cond, parallax_bp3m_cond].
    gaia_ids : (N,) int64 ndarray
        Gaia source IDs for the N stars (matching order of S_pm rows).

    Raises
    ------
    FileNotFoundError
        If K_matrices.npz or stellar_astrometry.csv are absent.
    ValueError
        If stellar_astrometry.csv lacks the *_cond columns written by
        the current run_bp3m.py (old-format output).
    """
    from pathlib import Path
    bp3m_dir = Path(bp3m_dir)

    K_path = bp3m_dir / 'K_matrices.npz'
    if not K_path.exists():
        raise FileNotFoundError(
            f'K_matrices.npz not found in {bp3m_dir}. '
            'This file is produced by the current run_bp3m.py. '
            'Use --bp3m-dir without --bp3m-latent for older BP3M runs.')

    csv_path = bp3m_dir / 'stellar_astrometry.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'stellar_astrometry.csv not found: {csv_path}')

    df = pd.read_csv(csv_path)

    if 'pmra_bp3m_cond' not in df.columns:
        raise ValueError(
            f'stellar_astrometry.csv in {bp3m_dir} is missing pmra_bp3m_cond. '
            'Re-run BP3M with the current run_bp3m.py to generate cond columns.')

    pmra_arr = df['pmra_bp3m'].to_numpy(dtype=float)
    has_bp3m = (df['n_hst_used'].to_numpy(dtype=int) > 0) & np.isfinite(pmra_arr)
    filtered_rows = np.where(has_bp3m)[0]  # row indices into full df / C_vT
    df_filt = df.iloc[filtered_rows].reset_index(drop=True)
    N = len(df_filt)

    gaia_ids = df_filt['Gaia_id'].to_numpy(dtype=np.int64)

    cond_means = np.column_stack([
        df_filt['pmra_bp3m_cond'].to_numpy(dtype=float),
        df_filt['pmdec_bp3m_cond'].to_numpy(dtype=float),
        df_filt['parallax_bp3m_cond'].to_numpy(dtype=float),
    ])

    # Conditional covariance directly from C_vT.npy (rows 2:5 = pmra,pmdec,plx)
    C_vT_full = np.load(bp3m_dir / 'C_vT.npy')   # (N_all, 5, 5)
    C_vT_pm = C_vT_full[filtered_rows, 2:5, 2:5]  # (N, 3, 3)

    # Image ordering from image_transformations.csv (matches C_r block layout)
    img_df = pd.read_csv(bp3m_dir / 'image_transformations.csv')
    image_names = img_df['image_name'].tolist()
    N_images = len(image_names)

    C_r = np.load(bp3m_dir / 'C_r.npy')
    n_r = C_r.shape[0]
    N_R = n_r // N_images   # transformation parameters per image

    # Eigendecompose C_r with clamped eigenvalues (matching sample_posteriors
    # fallback in solver.py which clamps negative eigenvalues to zero).
    vals, vecs = np.linalg.eigh(C_r)
    vals_clamped = np.maximum(vals, 0.0)
    mode_mask = vals_clamped > lambda_min
    V_kept   = vecs[:, mode_mask]                # (n_r, M_pre)
    sqrt_lam = np.sqrt(vals_clamped[mode_mask])  # (M_pre,)
    M_pre    = int(mode_mask.sum())

    # Load K matrices (keyed by image name)
    K_npz    = np.load(K_path, allow_pickle=False)
    sidx_npz = np.load(bp3m_dir / 'star_indices.npz', allow_pickle=False)
    use_npz  = np.load(bp3m_dir / 'use_for_fit.npz',  allow_pickle=False)

    # Pre-filter K matrices to used stars only; build global-index → image lookup.
    # star_to_images[g] = [(j_idx, local_used_row), ...]
    K_used_cache = {}    # j_idx → (n_used, 5, N_R)
    star_to_images = {}  # global_row → [(j_idx, local_used_row)]
    for j_idx, img in enumerate(image_names):
        if img not in K_npz:
            continue
        sidx_j = sidx_npz[img].astype(int)
        use_j  = use_npz[img].astype(bool)
        K_used_cache[j_idx] = K_npz[img][use_j]  # (n_used, 5, N_R)
        for loc, g in enumerate(sidx_j[use_j]):
            star_to_images.setdefault(int(g), []).append((j_idx, loc))

    # Build S_i = A_i @ V_kept * sqrt(λ) for each filtered star.
    # A_i[:, j*N_R:(j+1)*N_R] = C_vT_i @ K_i_j  (5, N_R)
    S_all = np.zeros((N, 5, M_pre))
    for i, g in enumerate(filtered_rows):
        imgs = star_to_images.get(int(g))
        if imgs is None:
            continue
        CvT_i = C_vT_full[g]     # (5, 5)
        A_i   = np.zeros((5, n_r))
        for j_idx, loc in imgs:
            K_ij = K_used_cache[j_idx][loc]          # (5, N_R)
            cs   = j_idx * N_R
            A_i[:, cs:cs + N_R] = CvT_i @ K_ij
        S_all[i] = A_i @ V_kept * sqrt_lam[None, :]  # (5, M_pre)

    # Filter modes by RMS PM sensitivity (modes k where σ_eff_pm_k > threshold).
    S_pm_pre     = S_all[:, 2:5, :]   # (N, 3, M_pre): pmra/pmdec/plx rows
    sigma_eff_pm = np.sqrt(np.mean(
        S_pm_pre[:, 0, :]**2 + S_pm_pre[:, 1, :]**2, axis=0))  # (M_pre,)
    pm_mask = sigma_eff_pm > sigma_eff_threshold
    S_pm = S_pm_pre[:, :, pm_mask]    # (N, 3, M_final)
    M_final = int(pm_mask.sum())

    _label = bp3m_dir.name or str(bp3m_dir)
    print(f'  BP3M latent ({_label}): {N} stars | '
          f'{M_pre} modes (λ>λ_min) → {M_final} kept '
          f'(σ_eff_pm > {sigma_eff_threshold:.3f} mas/yr)')

    return S_pm, C_vT_pm, cond_means, gaia_ids


def fit_background_gmm(pm_and_paras, good_backgrounds, n_components=5):
    """
    Fit a K-component sklearn GMM to background stars in (μ_α*, μ_δ, ϖ) space.

    Returns
    -------
    means   : (K, 3)
    covs    : (K, 3, 3)
    weights : (K,)
    """
    from sklearn.mixture import GaussianMixture

    X = pm_and_paras[good_backgrounds]
    ok = np.all(np.isfinite(X), axis=1)
    X = X[ok]

    K = min(n_components, max(1, len(X) // 10))
    gm = GaussianMixture(n_components=K, covariance_type='full',
                         random_state=42, n_init=3, max_iter=200)
    gm.fit(X)
    return gm.means_, gm.covariances_, gm.weights_


def compute_background_stats(pm_and_paras, r_ell, has_pms,
                              spatial_threshold=None):
    """
    Estimate the MW background from stars that are spatially far from the galaxy.

    Background is defined purely by elliptical radius: r_ell > spatial_threshold
    (in units of rhalf).  No PM-space cut is applied so that sources with PMs
    near zero (extragalactic background objects, halo stars) are not excluded.

    Parameters
    ----------
    pm_and_paras     : (N, 3)  [pmra, pmdec, parallax]
    r_ell            : (N,)    elliptical radius in units of rhalf (NaN = bad_rhalf)
    has_pms          : (N,) bool
    spatial_threshold: r_ell cut (default: GAIA_BG_SPATIAL_THRESHOLD from config)

    Returns
    -------
    background_mu, background_cov, background_errs, background_rhos,
    good_backgrounds (bool mask)
    """
    if spatial_threshold is None:
        spatial_threshold = GAIA_BG_SPATIAL_THRESHOLD

    valid_pm = has_pms & np.all(np.isfinite(pm_and_paras), axis=1)
    bad_rhalf = not np.any(np.isfinite(r_ell))

    if bad_rhalf:
        good_backgrounds = valid_pm
    else:
        # Try the requested threshold; lower it progressively if too few stars.
        used_threshold = spatial_threshold
        good_backgrounds = (r_ell >= used_threshold) & valid_pm
        for fallback in np.arange(GAIA_BG_SPATIAL_THRESHOLD,1-1e-10,-0.5).astype(float):
            if good_backgrounds.sum() >= GAIA_BG_MIN_STARS:
                break
            if fallback < used_threshold:
                used_threshold = fallback
                good_backgrounds = (r_ell >= used_threshold) & valid_pm
                warnings.warn(
                    f'Background sample below {GAIA_BG_MIN_STARS} at r_ell≥'
                    f'{spatial_threshold}; fell back to r_ell≥{used_threshold}.',
                    RuntimeWarning, stacklevel=2,
                )
        if good_backgrounds.sum() < GAIA_BG_MIN_STARS:
            warnings.warn(
                f'Only {good_backgrounds.sum()} background stars found. '
                'Results may be unreliable.', RuntimeWarning, stacklevel=2,
            )

    background_mu   = np.nanmedian(pm_and_paras[good_backgrounds], axis=0)
    background_cov  = np.cov(pm_and_paras[good_backgrounds], rowvar=False)
    background_errs = np.sqrt(np.diag(background_cov))
    background_rhos = np.array([
        background_cov[0, 1] / (background_errs[0] * background_errs[1]),
        background_cov[0, 2] / (background_errs[0] * background_errs[2]),
        background_cov[1, 2] / (background_errs[1] * background_errs[2]),
    ])

    return background_mu, background_cov, background_errs, background_rhos, good_backgrounds
