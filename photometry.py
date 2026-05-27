"""
Photometric membership prior based on CMD colour profiles.

Two methods are available:
  compute_photometric_prior()     – magnitude-binned profile approach (original)
  compute_photometric_prior_kde() – error-weighted 2-D KDE in (G, colour) space
"""

import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp


def _safe_comb_err(err_vals, min_err):
    """Combined inverse-variance error, floored at min_err."""
    s = np.sum(err_vals**-2) if len(err_vals) > 0 else 0.0
    if s <= 0:
        return min_err * 2
    return float(np.sqrt(1.0 / s + min_err**2))


def med_mags(gmags, colors, keep, nsigma=5, gmag_limit=-10000, magstep=0.1, min_err=0.05):
    """
    Compute magnitude-binned median colour profiles with iterative sigma-clipping.

    Parameters
    ----------
    gmags  : (N, 2)  – [mag, mag_err]
    colors : (N, 2)  – [color, color_err]
    keep   : (N,) bool mask
    nsigma : clipping threshold (use np.inf to skip clipping)
    gmag_limit : bright-end cutoff; bins brighter than this are skipped
    magstep : bin width in magnitudes
    min_err : minimum colour uncertainty floor

    Returns
    -------
    mag_bin_centers : (M,)
    red_branch  : (M, 2)  – [color, err]   (redder half of stars)
    all_branch  : (M, 2)  – [color, err]   (all stars)
    blue_branch : (M, 2)  – [color, err]   (bluer half of stars)
    """
    bright = max(gmag_limit, np.nanmin(gmags[keep, 0]))
    faint  = np.nanmax(gmags[keep, 0])
    mag_bins = np.sort(np.arange(faint + 1e-5, bright - magstep, -magstep))
    mag_bins[0] -= 0.001
    mag_bin_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])
    M = len(mag_bin_centers)

    c_all  = np.full((M, 3), np.nan)
    c_red  = np.full((M, 3), np.nan)
    c_blue = np.full((M, 3), np.nan)
    e_all  = np.full(M, np.nan)
    e_red  = np.full(M, np.nan)
    e_blue = np.full(M, np.nan)

    # First pass: raw percentiles
    for j in range(M):
        in_bin = keep & (gmags[:, 0] > mag_bins[j]) & (gmags[:, 0] <= mag_bins[j+1])
        if in_bin.sum() < 1:
            continue
        c_all[j]  = np.nanpercentile(colors[in_bin, 0], [16, 50, 84])
        in_red    = in_bin & (colors[:, 0] >= c_all[j, 1])
        in_blue   = in_bin & (colors[:, 0] <  c_all[j, 1])
        c_red[j]  = np.nanpercentile(colors[in_red,  0], [16, 50, 84])
        c_blue[j] = np.nanpercentile(colors[in_blue, 0], [16, 50, 84])

        e_all[j]  = _safe_comb_err(colors[in_bin,  1], min_err)
        e_red[j]  = _safe_comb_err(colors[in_red,  1], min_err)
        e_blue[j] = _safe_comb_err(colors[in_blue, 1], min_err)

    e_all  = np.where(np.isfinite(e_all),  e_all,  min_err * 2)
    e_red  = np.where(np.isfinite(e_red),  e_red,  min_err * 2)
    e_blue = np.where(np.isfinite(e_blue), e_blue, min_err * 2)

    # Second pass: sigma-clip
    for j in range(M):
        in_bin_orig = keep & (gmags[:, 0] > mag_bins[j]) & (gmags[:, 0] <= mag_bins[j+1])
        if in_bin_orig.sum() < 1:
            continue

        in_bin = in_bin_orig & (
            np.abs(colors[:, 0] - c_all[j, 1]) / (colors[:, 1] + e_all[j]) <= nsigma
        )
        if in_bin.sum() < 1:
            in_bin = in_bin_orig
        c_all[j] = np.nanpercentile(colors[in_bin, 0], [16, 50, 84])

        in_red  = in_bin_orig & (colors[:, 0] >= c_all[j, 1])
        in_red  = in_red & (
            np.abs(colors[:, 0] - c_red[j, 1]) / (colors[:, 1] + e_red[j]) <= nsigma
        )
        in_blue = in_bin_orig & (colors[:, 0] < c_all[j, 1])
        in_blue = in_blue & (
            np.abs(colors[:, 0] - c_blue[j, 1]) / (colors[:, 1] + e_blue[j]) <= nsigma
        )

        if in_red.sum()  > 0: c_red[j]  = np.nanpercentile(colors[in_red,  0], [16, 50, 84])
        if in_blue.sum() > 0: c_blue[j] = np.nanpercentile(colors[in_blue, 0], [16, 50, 84])

        if in_bin.sum()  > 0: e_all[j]  = _safe_comb_err(colors[in_bin,  1], min_err)
        if in_red.sum()  > 0: e_red[j]  = _safe_comb_err(colors[in_red,  1], min_err)
        if in_blue.sum() > 0: e_blue[j] = _safe_comb_err(colors[in_blue, 1], min_err)

    def _spread(c, e):
        spread = 0.5 * (c[:, 2] - c[:, 0])
        return np.sqrt(spread**2 + e**2)

    return (mag_bin_centers,
            np.column_stack([c_red[:, 1],  _spread(c_red,  e_red)]),
            np.column_stack([c_all[:, 1],  _spread(c_all,  e_all)]),
            np.column_stack([c_blue[:, 1], _spread(c_blue, e_blue)]))


def compute_photometric_prior(gmags, rpmags, bpmags, keep_member,
                               keep_background, gmag_limit):
    """
    Compute a per-star log-probability of galaxy membership from CMD colours.

    Uses RP (G-RP proxy) and BP colour profiles built from *keep_member* stars.
    Stars in *keep_background* define the MW background colour profile.

    Returns prior_log_probs (N,) and the colour profile arrays used.
    """
    # --- Member colour profiles ---
    x, r_rp, g_rp, b_rp = med_mags(gmags, rpmags, keep_member,
                                     nsigma=3, gmag_limit=gmag_limit)
    x, r_bp, g_bp, b_bp = med_mags(gmags, bpmags, keep_member,
                                     nsigma=3, gmag_limit=gmag_limit)

    def _interp(x_arr, profile_col, err_col, left_val):
        ok = np.isfinite(profile_col)
        if ok.sum() < 2:
            return np.full(len(gmags), left_val), np.full(len(gmags), 1.0)
        vals = np.interp(gmags[:, 0], x_arr[ok], profile_col[ok], left=left_val)
        errs = np.interp(gmags[:, 0], x_arr[ok], err_col[ok])
        return vals, errs

    gr_red,  gr_red_err  = _interp(x, r_rp[:, 0], r_rp[:, 1], -np.inf)
    gr_blue, gr_blue_err = _interp(x, b_rp[:, 0], b_rp[:, 1], 2.0)
    gb_red,  gb_red_err  = _interp(x, r_bp[:, 0], r_bp[:, 1], -np.inf)
    gb_blue, gb_blue_err = _interp(x, b_bp[:, 0], b_bp[:, 1], 2.0)

    # Initial background definition: stars far from centre & not on member CMD
    red_sigma  = np.sqrt(rpmags[:, 1]**2 + gr_red_err**2)
    blue_sigma = np.sqrt(rpmags[:, 1]**2 + gr_blue_err**2)
    red_dist   = np.minimum(np.abs((gr_red  - rpmags[:, 0]) / red_sigma),
                             np.abs((gr_blue - rpmags[:, 0]) / blue_sigma))
    bp_red_sigma  = np.sqrt(bpmags[:, 1]**2 + gb_red_err**2)
    bp_blue_sigma = np.sqrt(bpmags[:, 1]**2 + gb_blue_err**2)
    blue_dist  = np.minimum(np.abs((gb_red  - bpmags[:, 0]) / bp_red_sigma),
                             np.abs((gb_blue - bpmags[:, 0]) / bp_blue_sigma))

    red_logcdf  = stats.norm.logcdf(-red_dist)
    blue_logcdf = stats.norm.logcdf(-blue_dist)
    initial_prior = (red_logcdf + np.log(2)) + (blue_logcdf + np.log(2))

    # --- Background colour profiles ---
    # keep_background selects off-galaxy MW stars
    def _safe_interp(x_arr, profile_col, err_col):
        ok = np.isfinite(profile_col)
        if ok.sum() < 2:
            return np.full(len(gmags), np.nan), np.full(len(gmags), 1.0)
        vals = np.interp(gmags[:, 0], x_arr[ok], profile_col[ok])
        errs = np.interp(gmags[:, 0], x_arr[ok], err_col[ok])
        return vals, errs

    if keep_background.sum() > 5:
        x_bg, r_bg_rp, g_bg_rp, b_bg_rp = med_mags(gmags, rpmags, keep_background,
                                                      nsigma=np.inf)
        x_bg2, r_bg_bp, g_bg_bp, b_bg_bp = med_mags(gmags, bpmags, keep_background,
                                                      nsigma=np.inf)

        bg_gr_green, bg_gr_green_err = _safe_interp(x_bg,  g_bg_rp[:, 0], g_bg_rp[:, 1])
        bg_gb_green, bg_gb_green_err = _safe_interp(x_bg2, g_bg_bp[:, 0], g_bg_bp[:, 1])
    else:
        # No background stars available; use uninformative background
        bg_gr_green     = np.full(len(gmags), np.nan)
        bg_gr_green_err = np.full(len(gmags), 3.0)
        bg_gb_green     = np.full(len(gmags), np.nan)
        bg_gb_green_err = np.full(len(gmags), 3.0)

    # --- Member log-PDF (take better of red/blue branches) ---
    member_rp = np.maximum(
        stats.norm.logpdf(gr_red  - rpmags[:, 0],
                          scale=np.sqrt(rpmags[:, 1]**2 + gr_red_err**2)),
        stats.norm.logpdf(gr_blue - rpmags[:, 0],
                          scale=np.sqrt(rpmags[:, 1]**2 + gr_blue_err**2)),
    )
    member_bp = np.maximum(
        stats.norm.logpdf(gb_red  - bpmags[:, 0],
                          scale=np.sqrt(bpmags[:, 1]**2 + gb_red_err**2)),
        stats.norm.logpdf(gb_blue - bpmags[:, 0],
                          scale=np.sqrt(bpmags[:, 1]**2 + gb_blue_err**2)),
    )
    member_logpdfs = member_rp + member_bp

    # --- Background log-PDF ---
    bg_rp = stats.norm.logpdf(
        bg_gr_green - rpmags[:, 0],
        scale=np.sqrt(rpmags[:, 1]**2 + bg_gr_green_err**2),
    )
    bg_bp = stats.norm.logpdf(
        bg_gb_green - bpmags[:, 0],
        scale=np.sqrt(bpmags[:, 1]**2 + bg_gb_green_err**2),
    )
    background_logpdfs = bg_rp + bg_bp

    # --- Posterior membership probability in log-space ---
    prior_log_probs = member_logpdfs - np.logaddexp(member_logpdfs, background_logpdfs)
    prior_log_probs = np.clip(prior_log_probs, np.log(0.01), np.log(0.99))

    # Colour profiles for plotting (using all_branch = green)
    color_profiles = dict(
        x        = x,
        gr_red   = r_rp, gr_green = g_rp, gr_blue = b_rp,
        gb_red   = r_bp, gb_green = g_bp, gb_blue = b_bp,
    )

    return prior_log_probs, color_profiles


def build_log_prior_weights(prior_log_probs, keep, keep_inds, bad_rhalf,
                             test_keep, N_clusters):
    """
    Build the (N_obs, N_clusters) log-prior weight matrix fed into the GMM.

    In the normal case the photometric prior_log_probs drives component 0 (dwarf);
    in bad_rhalf we fall back to equal weights.
    """
    from utils import logdiffexp
    N = len(keep_inds)
    log_prior_ws = np.zeros((N, N_clusters))

    if bad_rhalf:
        log_prior_ws[:, 0] = np.log(0.5)
        log_prior_ws[test_keep[keep_inds], 0] = np.log(0.9)
    else:
        log_prior_ws[:, 0] = prior_log_probs[keep_inds]

    log_bg_priors = logdiffexp(np.zeros(N), log_prior_ws[:, 0]) - np.log(N_clusters-1)
    for j in range(1,N_clusters):
        log_prior_ws[:,j] = log_bg_priors
        
    return log_prior_ws


# ---------------------------------------------------------------------------
# 2-D error-weighted KDE prior (improved method)
# ---------------------------------------------------------------------------

def _kde_logpdf_3d(query_g, query_bp, query_rp,
                    query_g_err, query_bp_err, query_rp_err,
                    train_g, train_bp, train_rp,
                    train_g_err, train_bp_err, train_rp_err,
                    min_bw_g=0.15, min_bw_bp=0.05, min_bw_rp=0.05,
                    batch_size=500):
    """
    Variable-bandwidth 3-D KDE in (G, BP, RP) magnitude space.

    Each dimension is treated independently with bandwidth
    sqrt(σ_train² + σ_query² + min_bw²).  Using the three magnitudes directly
    avoids introducing artificial correlations that arise when working in
    colour-difference space (e.g. BP−G, RP−G would both share the G error).
    Returns log-density (N_query,), normalised by N_train.

    Speed: training stars are sorted by G; each query batch sums only the
    training stars within 3 × max_bw in G magnitude.
    """
    sort_idx = np.argsort(train_g)
    s_g   = train_g[sort_idx];   s_ge  = train_g_err[sort_idx]
    s_bp  = train_bp[sort_idx];  s_bpe = train_bp_err[sort_idx]
    s_rp  = train_rp[sort_idx];  s_rpe = train_rp_err[sort_idx]
    max_train_bw_g = np.sqrt(np.max(s_ge)**2 + min_bw_g**2)

    N_q      = len(query_g)
    log_N    = np.log(len(train_g))
    log_dens = np.empty(N_q)

    for start in range(0, N_q, batch_size):
        sl    = slice(start, start + batch_size)
        qg_b  = query_g[sl];   qge_b  = query_g_err[sl]
        qbp_b = query_bp[sl];  qbpe_b = query_bp_err[sl]
        qrp_b = query_rp[sl];  qrpe_b = query_rp_err[sl]

        max_bw = np.sqrt(max_train_bw_g**2 + np.max(qge_b)**2)
        lo = np.searchsorted(s_g, qg_b.min() - 3 * max_bw)
        hi = np.searchsorted(s_g, qg_b.max() + 3 * max_bw, side='right')

        if hi - lo < 1:
            log_dens[sl] = -np.inf
            continue

        t_g   = s_g[lo:hi];   t_ge  = s_ge[lo:hi]
        t_bp  = s_bp[lo:hi];  t_bpe = s_bpe[lo:hi]
        t_rp  = s_rp[lo:hi];  t_rpe = s_rpe[lo:hi]

        dg   = t_g[:, None]  - qg_b[None, :]    # (n_local, batch)
        dbp  = t_bp[:, None] - qbp_b[None, :]
        drp  = t_rp[:, None] - qrp_b[None, :]
        bw_g  = np.sqrt(t_ge[:, None]**2  + qge_b[None, :]**2  + min_bw_g**2)
        bw_bp = np.sqrt(t_bpe[:, None]**2 + qbpe_b[None, :]**2 + min_bw_bp**2)
        bw_rp = np.sqrt(t_rpe[:, None]**2 + qrpe_b[None, :]**2 + min_bw_rp**2)
        log_k = (stats.norm.logpdf(dg,  scale=bw_g) +
                 stats.norm.logpdf(dbp, scale=bw_bp) +
                 stats.norm.logpdf(drp, scale=bw_rp))
        log_dens[sl] = logsumexp(log_k, axis=0) - log_N

    return log_dens


def compute_photometric_prior_kde(gmags, rpmags, bpmags, keep_member,
                                   keep_background, gmag_limit,
                                   min_bw_g=0.15, min_bw_c=0.05,
                                   max_train=3000):
    """
    Compute per-star log-probability of galaxy membership using a 3-D
    error-weighted KDE in (G, BP, RP) magnitude space.

    Working directly in magnitude space avoids the spurious correlations that
    arise when using colour differences (e.g. BP−G and RP−G both share the G
    error).  Each training star contributes a Gaussian kernel independently in
    G, BP, and RP, with per-star bandwidths sqrt(σ_train² + σ_query² + min_bw²).

    Parameters
    ----------
    gmags, rpmags, bpmags : (N, 2) arrays of [value, error]
    keep_member     : bool mask – defines the member (galaxy) training set
    keep_background : bool mask – defines the MW background training set
    gmag_limit      : bright-end G cutoff (bins brighter than this are excluded)
    min_bw_g        : minimum bandwidth in G magnitude
    min_bw_c        : minimum bandwidth in BP and RP individually
    max_train       : random sub-sample cap for training sets (for speed)

    Returns
    -------
    prior_log_probs : (N,)  log P(member)
    """
    rng = np.random.default_rng(0)

    valid = np.isfinite(gmags[:, 0]) & np.isfinite(rpmags[:, 0]) & np.isfinite(bpmags[:, 0])
    if gmag_limit > -1000:
        bright_cut = gmags[:, 0] >= gmag_limit
    else:
        bright_cut = np.ones(len(gmags), dtype=bool)

    m_mask = keep_member     & valid & bright_cut
    b_mask = keep_background & valid & bright_cut
    q_mask = valid

    if m_mask.sum() < 3:
        return np.full(len(gmags), np.log(0.5))

    def _subsample(mask, n):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        return idx

    m_idx = _subsample(m_mask, max_train)
    b_idx = _subsample(b_mask, max_train) if b_mask.sum() >= 3 else None

    q_g   = gmags[q_mask, 0];  q_ge  = np.clip(gmags[q_mask, 1],  0.001, np.inf)
    q_bp  = bpmags[q_mask, 0]; q_bpe = np.clip(bpmags[q_mask, 1], 0.001, np.inf)
    q_rp  = rpmags[q_mask, 0]; q_rpe = np.clip(rpmags[q_mask, 1], 0.001, np.inf)

    kw = dict(min_bw_g=min_bw_g, min_bw_bp=min_bw_c, min_bw_rp=min_bw_c)

    def _errs(arr, idx):
        return np.clip(arr[idx, 1], 0.001, np.inf)

    member_logpdfs = _kde_logpdf_3d(
        q_g, q_bp, q_rp, q_ge, q_bpe, q_rpe,
        gmags[m_idx, 0], bpmags[m_idx, 0], rpmags[m_idx, 0],
        _errs(gmags, m_idx), _errs(bpmags, m_idx), _errs(rpmags, m_idx), **kw)

    if b_idx is not None:
        background_logpdfs = _kde_logpdf_3d(
            q_g, q_bp, q_rp, q_ge, q_bpe, q_rpe,
            gmags[b_idx, 0], bpmags[b_idx, 0], rpmags[b_idx, 0],
            _errs(gmags, b_idx), _errs(bpmags, b_idx), _errs(rpmags, b_idx), **kw)
    else:
        background_logpdfs = np.zeros(q_mask.sum())

    prior_lp_q = member_logpdfs - np.logaddexp(member_logpdfs, background_logpdfs)
    prior_lp_q = np.clip(prior_lp_q, np.log(0.01), np.log(0.99))

    prior_log_probs = np.full(len(gmags), np.log(0.5))
    prior_log_probs[q_mask] = prior_lp_q
    return prior_log_probs


def compute_qso_photometric_prior(gmags, rpmags, bpmags,
                                   qso_gmags, qso_rpmags, qso_bpmags,
                                   qso_gmag_errs=None, qso_bpmag_errs=None,
                                   qso_rpmag_errs=None,
                                   min_bw_g=0.15, min_bw_c=0.05):
    """
    Log P(photometry | QSO) for all N sources using a 3-D KDE in (G, BP, RP)
    space built from the cleaned QSO training sample.  Returns a flat 0.0
    (uninformative) if fewer than 10 finite QSO training points are available.

    Parameters
    ----------
    gmags, rpmags, bpmags : (N, 2) arrays [value, error] for all sources
    qso_gmags, qso_rpmags, qso_bpmags : (M,) arrays of QSO training magnitudes
    qso_gmag_errs, qso_bpmag_errs, qso_rpmag_errs : (M,) arrays of QSO training
        magnitude errors.  If None or all-NaN, falls back to 0.02 mag (dominated
        by min_bw floor, but real errors are preferred for faint sources).

    Returns
    -------
    log_qso_photo : (N,)
    """
    finite_qso = (np.isfinite(qso_gmags) & np.isfinite(qso_rpmags)
                  & np.isfinite(qso_bpmags))
    if finite_qso.sum() < 10:
        return np.zeros(len(gmags))

    t_g   = qso_gmags[finite_qso]
    t_rp  = qso_rpmags[finite_qso]
    t_bp  = qso_bpmags[finite_qso]

    def _train_errs(errs_arr, fallback=0.02):
        if errs_arr is None:
            return np.full(finite_qso.sum(), fallback)
        e = errs_arr[finite_qso]
        bad = ~np.isfinite(e) | (e <= 0)
        e = np.where(bad, fallback, e)
        return e

    t_ge  = _train_errs(qso_gmag_errs)
    t_bpe = _train_errs(qso_bpmag_errs)
    t_rpe = _train_errs(qso_rpmag_errs)

    valid = (np.isfinite(gmags[:, 0]) & np.isfinite(rpmags[:, 0])
             & np.isfinite(bpmags[:, 0]))
    log_qso_photo = np.zeros(len(gmags))

    kw = dict(min_bw_g=min_bw_g, min_bw_bp=min_bw_c, min_bw_rp=min_bw_c)
    q_g   = gmags[valid, 0];  q_ge  = np.clip(gmags[valid, 1],  0.001, np.inf)
    q_rp  = rpmags[valid, 0]; q_rpe = np.clip(rpmags[valid, 1], 0.001, np.inf)
    q_bp  = bpmags[valid, 0]; q_bpe = np.clip(bpmags[valid, 1], 0.001, np.inf)

    log_qso_photo[valid] = _kde_logpdf_3d(
        q_g, q_bp, q_rp, q_ge, q_bpe, q_rpe,
        t_g, t_bp, t_rp, t_ge, t_bpe, t_rpe, **kw)
    return log_qso_photo
