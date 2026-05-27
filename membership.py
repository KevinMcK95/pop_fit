"""
Per-star posterior membership probability computation.

Uses batched numpy operations over draws to avoid Python-loop overhead.
With draw_batch=50 and N=5000 stars, peak memory per iteration is ~150 MB.
"""

import numpy as np
from tqdm import tqdm

# Tril indices for a 3×3 lower-triangular matrix (packed Cholesky order)
_TRIL_R = np.array([0, 1, 1, 2, 2, 2])
_TRIL_C = np.array([0, 0, 1, 0, 1, 2])

def _unpack_batch(Ls):
    """Ls : (b, 6) → L : (b, 3, 3) lower-triangular."""
    b = Ls.shape[0]
    L = np.zeros((b, 3, 3))
    L[:, _TRIL_R, _TRIL_C] = Ls
    return L


def _chol_to_cov(Ls):
    """Ls : (b, 6) → cov : (b, 3, 3) via L @ Lᵀ."""
    L = _unpack_batch(Ls)
    return np.einsum('bij,bkj->bik', L, L)


def _bg_logpdf_flat(mu, C_inv, logdet, y_obs_flat):
    """
    MvNormal log-PDF for a single fixed (mu, C_inv, logdet) against N stars.

    Used for precomputing background log-PDFs outside the batch loop.

    mu          : (3,)
    C_inv       : (3, 3)   precomputed inverse of the fixed covariance
    logdet      : scalar   log-determinant of the fixed covariance
    y_obs_flat  : (N, 3)

    Returns : (N,)
    """
    delta = y_obs_flat - mu[None, :]                              # (N, 3)
    maha  = np.einsum('ni,ij,nj->n', delta, C_inv, delta)        # (N,)
    return -0.5 * (3 * np.log(2 * np.pi) + logdet + maha)


def _batch_logpdf(mu_b, cov_b, S_total, y_eff):
    """
    MvNormal log-PDF for a batch of draws and all stars.

    mu_b    : (b, 3)
    cov_b   : (b, 3, 3)
    S_total : (b, N, 3, 3)
    y_eff   : (b, N, 3)

    Returns : (b, N)
    """
    C      = cov_b[:, None, :, :] + S_total
    delta  = y_eff - mu_b[:, None, :]
    C_inv  = np.linalg.inv(C)
    _, logdet = np.linalg.slogdet(C)
    maha   = np.einsum('bni,bnij,bnj->bn', delta, C_inv, delta)
    return -0.5 * (3 * np.log(2 * np.pi) + logdet + maha)


def compute_membership_probs(trace, y_obs, pos_obs, S_obs, log_prior_ws,
                              survey_area, bg_means, bg_covs, bg_weights,
                              gmags_obs=None,
                              log_prior_qso=None,
                              spatial_profile='plummer',
                              n_samples=1000, seed=42, draw_batch=50,
                              y_obs_qso_train=None, S_obs_qso_train=None,
                              pos_obs_qso_train=None, log_prior_qso_train=None,
                              gmags_qso_train=None,
                              is_hst_main=None,
                              y_obs_hst_b=None, S_obs_hst_b=None,
                              log_spatial_hst_b=None, hst_area=None,
                              log_prior_ws_hst_b=None,
                              S_latent_hst_a=None,
                              S_latent_hst_b=None):
    """
    Average per-star membership probabilities over *n_samples* posterior draws.

    The MW background is represented by K fixed Gaussian components (pre-fit
    sklearn GMM): bg_means (K,3), bg_covs (K,3,3), bg_weights (K,).

    Parameters
    ----------
    trace           : ArviZ InferenceData from run_gmm_model
    y_obs           : (N, 3)
    pos_obs         : (N, 2)
    S_obs           : (N, 3, 3)
    log_prior_ws    : (N, 3)   – cols 1&2 are equal background weights
    survey_area     : float
    bg_means        : (K, 3)   pre-fit background component means
    bg_covs         : (K, 3, 3) pre-fit background component covariances
    bg_weights      : (K,)     pre-fit background mixing weights
    gmags_obs       : (N,) G magnitudes; ignored if None
    log_prior_qso   : (N,) per-source log QSO prior (photo + catalog); 3-class if not None
    spatial_profile : 'plummer' or 'sersic'
    n_samples       : draws to average over
    seed            : RNG seed
    draw_batch      : draws per vectorised batch
    y_obs_qso_train     : (M, 3) wide-field QSO training kinematic observations; optional
    S_obs_qso_train     : (M, 3, 3) corresponding measurement covariances
    pos_obs_qso_train   : (M, 2) sky positions (deg offsets from centre)
    log_prior_qso_train : (M,) per-source log catalog prior (e.g. log 0.99 for MILLIQUAS)
    gmags_qso_train     : (M,) G magnitudes for k(G) inflation
    is_hst_main         : (N,) bool or None — mask for Group A HST stars in main dataset
    y_obs_hst_b         : (N_B, 3) or None — Group B BP3M observations
    S_obs_hst_b         : (N_B, 3, 3) or None — Group B BP3M covariances
    log_spatial_hst_b   : (N_B,) or None — fixed log spatial density for Group B
    hst_area            : float or None — HST footprint area for Group B background
    log_prior_ws_hst_b  : (N_B, N_clusters) or None — Group B photometric prior weights
    S_latent_hst_a      : (N, 3, M) or None — z-latent sensitivity for Group A (v2 mode,
                          padded with zeros for Gaia-only stars).  When provided, z_latent
                          is read from the trace and used to compute per-draw y_eff for HST
                          stars: y_eff += einsum("ijk,bk->bij", S_latent_hst_a, z_batch).
    S_latent_hst_b      : (N_B, 3, M) or None — z-latent sensitivity for Group B (v2).

    Returns
    -------
    Without HST Group B:
      2-class : (final_probs, p_dwarf_samples, p_background_samples)
      3-class : (final_probs, p_dwarf_samples, p_background_samples, p_qso_samples[, qso_train_final_probs])
    With HST Group B (appended as last element in all cases):
      + p_dwarf_hst_b_final : (N_B,) median P(dwarf) for Group B stars
    """
    rng      = np.random.default_rng(seed)
    # trace.posterior may be an xr.Dataset (old ArviZ) or xr.DataTree node
    # (new ArviZ).  Extract the plain Dataset so stack/indexing behaves
    # identically across versions.
    _post = trace.posterior
    if hasattr(_post, 'ds'):          # DataTree node → get root Dataset
        _post = _post.ds
    elif not hasattr(_post, 'dims'):  # fallback: try to_dataset()
        _post = _post.to_dataset()
    stacked = _post.stack(sample=("chain", "draw"))
    n_total = stacked.sizes["sample"]
    idxs    = np.sort(rng.choice(n_total, size=min(n_samples, n_total), replace=False))
    n_draws = len(idxs)
    N       = len(y_obs)

    def _scalar(key):
        return np.array(stacked[key].values[idxs])

    def _vec(key):
        return np.array(stacked[key].values[:, idxs]).T

    all_dra    = _scalar('delta_ra_center')
    all_ddec   = _scalar('delta_dec_center')
    all_apl    = _scalar('a_plummer')
    all_pa     = _scalar('pa_deg')
    all_ell    = _scalar('ellipticity')
    all_k0     = _scalar('k')

    has_k1    = 'k_1'              in stacked
    has_fdw   = 'f_dwarf'          in stacked
    has_sys   = 'delta_pm_sys'     in stacked
    has_fstar = 'f_star'           in stacked
    has_khst  = 'k_hst'            in stacked
    has_sysh  = 'delta_pm_sys_hst' in stacked

    all_k1    = _scalar('k_1')       if has_k1    else np.zeros(n_draws)
    all_fdw   = _scalar('f_dwarf')   if has_fdw   else np.full(n_draws, 0.5)
    all_fstar = _scalar('f_star')    if has_fstar else np.ones(n_draws)
    all_dpm   = _vec('delta_pm_sys') if has_sys   else None   # (n_draws, 2)
    all_khst  = _scalar('k_hst')             if has_khst  else np.ones(n_draws)
    all_Lf_h  = _vec('chol_floor_hst')       if has_khst  else None
    all_dpm_h = _vec('delta_pm_sys_hst')     if has_sysh  else None  # (n_draws, 2)

    has_zlat  = 'z_latent' in stacked
    all_z     = _vec('z_latent') if has_zlat else None  # (n_draws, M)

    all_Lf  = _vec('chol_floor')
    all_Ld  = _vec('chol_intrinsic_dwarf')
    all_mud = _vec('mu_dwarf')

    # HST Group A and Group B flags
    _has_hst_main  = (is_hst_main is not None and np.any(is_hst_main))
    _has_latent_a  = (has_zlat and S_latent_hst_a is not None)
    _has_latent_b  = (has_zlat and S_latent_hst_b is not None)
    _has_hst_b    = (y_obs_hst_b is not None and len(y_obs_hst_b) > 0
                     and S_obs_hst_b is not None
                     and log_spatial_hst_b is not None
                     and log_prior_ws_hst_b is not None
                     and hst_area is not None)
    N_B = len(y_obs_hst_b) if _has_hst_b else 0

    # Pre-compute background prior weight: logaddexp of cols 1 and 2
    log_pw_bg = np.copy(log_prior_ws[:, 1])  # (N,)
    if len(log_prior_ws[0]) > 2:
        for j in range(2, len(log_prior_ws[0])):
            log_pw_bg = np.logaddexp(log_pw_bg, log_prior_ws[:, j])

    log_bg_w       = np.log(np.maximum(bg_weights, 1e-300))                # (K,)
    K              = len(bg_weights)

    # Precompute background log-PDFs — the background uses raw y_obs (never
    # shifted by delta_pm_sys), so the K-component mixture is constant across
    # all posterior draws.  Computing it once avoids K × n_batches redundant
    # 3×3 matrix inversions inside the main batch loop.
    _bg_Cinv = [np.linalg.inv(bg_covs[k]) for k in range(K)]
    _bg_ldet = [np.linalg.slogdet(bg_covs[k])[1] for k in range(K)]

    _bg_parts = np.stack([
        log_bg_w[k] + _bg_logpdf_flat(bg_means[k], _bg_Cinv[k], _bg_ldet[k], y_obs)
        for k in range(K)
    ], axis=0)                                                              # (K, N)
    bg_logL_fixed = np.logaddexp.reduce(_bg_parts, axis=0)                 # (N,)

    use_kg = gmags_obs is not None
    G_ref  = float(np.nanmedian(gmags_obs)) if use_kg else 0.0
    # Replace NaN G magnitudes with G_ref so missing photometry doesn't
    # propagate NaN through exp(k_1 * NaN) — matches the guard in models.py.
    G_arr  = (np.where(np.isfinite(gmags_obs), gmags_obs, G_ref)
              if use_kg else np.zeros(N))

    log_spatial_mw = np.log(1.0 / survey_area)

    has_3class = log_prior_qso is not None

    # Wide-field QSO training sample setup
    has_qso_train = (has_3class and y_obs_qso_train is not None
                     and S_obs_qso_train is not None
                     and pos_obs_qso_train is not None
                     and log_prior_qso_train is not None)
    M = len(y_obs_qso_train) if has_qso_train else 0

    if has_qso_train:
        G_ref_q   = G_ref   # reuse same reference magnitude
        G_arr_q   = (np.where(np.isfinite(gmags_qso_train), gmags_qso_train, G_ref_q)
                     if gmags_qso_train is not None else np.full(M, G_ref_q))
        # Background photometric weight for training sources: flat (no CMD prior)
        log_pw_bg_q = np.zeros(M)   # log(1) = 0

        # Precompute background log-PDFs for training QSOs (raw y_obs_qso_train,
        # constant across draws — same reasoning as the main-field precomputation).
        _bg_q_parts = np.stack([
            log_bg_w[k] + _bg_logpdf_flat(
                bg_means[k], _bg_Cinv[k], _bg_ldet[k], y_obs_qso_train)
            for k in range(K)
        ], axis=0)                                                          # (K, M)
        bg_logL_qso_fixed = np.logaddexp.reduce(_bg_q_parts, axis=0)      # (M,)

    # Group B precomputation: background log-PDFs (constant across draws)
    if _has_hst_b:
        _bg_b_parts = np.stack([
            log_bg_w[k] + _bg_logpdf_flat(
                bg_means[k], _bg_Cinv[k], _bg_ldet[k], y_obs_hst_b)
            for k in range(K)
        ], axis=0)                                                       # (K, N_B)
        bg_logL_hst_b_fixed = np.logaddexp.reduce(_bg_b_parts, axis=0) # (N_B,)
        log_spatial_hst_b_mw = np.log(1.0 / float(hst_area))

        log_pw_bg_b = np.copy(log_prior_ws_hst_b[:, 1])
        if log_prior_ws_hst_b.shape[1] > 2:
            for j in range(2, log_prior_ws_hst_b.shape[1]):
                log_pw_bg_b = np.logaddexp(log_pw_bg_b,
                                            log_prior_ws_hst_b[:, j])

    p_dwarf_samples      = np.zeros((n_draws, N))
    p_background_samples = np.zeros((n_draws, N))
    p_qso_samples        = np.zeros((n_draws, N)) if has_3class else None
    p_qso_train_samples  = np.zeros((n_draws, M)) if has_qso_train else None
    p_dwarf_b_samples    = np.zeros((n_draws, N_B)) if _has_hst_b else None

    for start in tqdm(range(0, n_draws, draw_batch), desc='Membership probs'):
        sl = slice(start, start + draw_batch)
        b  = min(draw_batch, n_draws - start)

        # ── Spatial log-density: (b, N) ──────────────────────────────────
        dra_b  = all_dra[sl][:, None]
        ddec_b = all_ddec[sl][:, None]
        apl_b  = all_apl[sl][:, None]
        pa_b   = all_pa[sl][:, None]
        ell_b  = all_ell[sl][:, None]
        q_b    = 1.0 - ell_b
        theta  = (pa_b + 90.0) * np.pi / 180.0

        x_sh  = pos_obs[None, :, 0] - dra_b  / 60.0
        y_sh  = pos_obs[None, :, 1] - ddec_b / 60.0
        Xm    = -x_sh * np.cos(theta) + y_sh * np.sin(theta)
        Ym    =  x_sh * np.sin(theta) + y_sh * np.cos(theta)
        r_bn  = np.sqrt(Xm**2 + (Ym / q_b)**2)

        a_deg_b = apl_b / 60.0
        if spatial_profile == 'sersic':
            log_sp_dw = (- np.log(2 * np.pi)
                         - 2 * np.log(a_deg_b) - np.log(q_b)
                         - r_bn / a_deg_b)
        else:
            log_sp_dw = (np.log(1.0 / (np.pi * a_deg_b**2 * q_b))
                         - 2.0 * np.log(1.0 + r_bn**2 / a_deg_b**2))

        # ── k(G) per-star inflation: (b, N) ──────────────────────────────
        # Clip k_1·ΔG to [-6, 6] to match the guard in models.py.
        k_i = (all_k0[sl][:, None]
               * np.exp(np.clip(all_k1[sl][:, None] * (G_arr[None, :] - G_ref),
                                -6.0, 6.0)))
        k_i = np.maximum(k_i, 0.1)

        # ── Floor covariance and total measurement cov: (b, N, 3, 3) ─────
        S_floor_b = _chol_to_cov(all_Lf[sl])
        # Gaia error model for all stars
        S_total_gaia = ((k_i[:, :, None, None]**2) * S_obs[None, :, :, :]
                        + S_floor_b[:, None, :, :])

        if _has_hst_main:
            # HST Group A stars use k_hst and chol_floor_hst
            k_hst_b   = all_khst[sl][:, None]
            S_floor_hb = _chol_to_cov(all_Lf_h[sl])
            S_total_hst = ((k_hst_b[:, :, None, None]**2) * S_obs[None, :, :, :]
                           + S_floor_hb[:, None, :, :])
            # Blend: is_hst_main (1, N, 1, 1)
            _m33 = is_hst_main[None, :, None, None].astype(np.float64)
            S_total = (1.0 - _m33) * S_total_gaia + _m33 * S_total_hst
        else:
            S_total = S_total_gaia

        # ── PM shifts and effective observations: (b, N, 3) ─────────────
        # Gaia shift (delta_pm_sys if QSO correction active)
        shift_gaia = np.zeros((b, 3))
        if has_sys:
            shift_gaia[:, :2] = all_dpm[sl]

        # HST shift (delta_pm_sys_hst; zeros if not present)
        shift_hst = np.zeros((b, 3))
        if has_sysh and all_dpm_h is not None:
            shift_hst[:, :2] = all_dpm_h[sl]

        if _has_hst_main:
            y_eff_gaia = y_obs[None, :, :] - shift_gaia[:, None, :]
            y_eff_hst  = y_obs[None, :, :] - shift_hst[:, None, :]
            _m1 = is_hst_main[None, :, None].astype(np.float64)
            y_eff = (1.0 - _m1) * y_eff_gaia + _m1 * y_eff_hst
        elif has_sys:
            y_eff = y_obs[None, :, :] - shift_gaia[:, None, :]
        else:
            y_eff = np.broadcast_to(y_obs[None, :, :], (b, N, 3))
        # Z-latent correction for Group A HST stars: S_latent_hst_a @ z
        # S_latent_hst_a is (N, 3, M) padded; z_batch is (b, M) → (b, N, 3).
        if _has_latent_a:
            z_batch    = all_z[sl]                                   # (b, M)
            y_z_corr_a = np.einsum('ijk,bk->bij', S_latent_hst_a, z_batch)
            y_eff      = np.array(y_eff) + y_z_corr_a               # copy if needed

        # ── Field-level fractions: (b, 1) ────────────────────────────────
        log_fdw_b   = np.log(np.clip(all_fdw[sl],           1e-9, 1-1e-9))[:, None]
        log_fdbg_b  = np.log(np.clip(1.0 - all_fdw[sl],     1e-9, 1-1e-9))[:, None]
        log_fstar_b = np.log(np.clip(all_fstar[sl],          1e-9, 1-1e-9))[:, None]

        # ── Dwarf log-PDF: (b, N) ────────────────────────────────────────
        logL_d  = _batch_logpdf(all_mud[sl], _chol_to_cov(all_Ld[sl]),
                                 S_total, y_eff)
        log_p_d = (log_fstar_b + log_fdw_b + log_prior_ws[None, :, 0]
                   + logL_d + log_sp_dw)                                  # (b, N)

        # ── Background logsumexp over K fixed components: (b, N) ─────────
        # bg_logL_fixed is precomputed before the batch loop (raw y_obs, no
        # delta_pm_sys shift) — see comment above.  Broadcast (N,) → (b, N).
        log_p_bg    = (log_fstar_b + log_fdbg_b + log_pw_bg[None, :]
                       + bg_logL_fixed[None, :] + log_spatial_mw)         # (b, N)

        log_p_floor = np.full_like(log_p_d, np.log(1e-300))

        if has_3class:
            # QSO: zero-mean MvNormal in shifted frame, measurement errors only
            log_fqso_b = np.log(np.clip(1.0 - all_fstar[sl], 1e-9, 1-1e-9))[:, None]
            zero_cov   = np.zeros((b, 3, 3))   # reused by training block below
            zero_mu    = np.zeros((b, 3))
            logL_qso_c = _batch_logpdf(zero_mu, zero_cov, S_total, y_eff)
            log_p_qso  = (log_fqso_b + log_prior_qso[None, :]
                          + logL_qso_c + log_spatial_mw)                  # (b, N)
            log_denom  = np.logaddexp(
                np.logaddexp(np.logaddexp(log_p_d, log_p_bg), log_p_qso),
                log_p_floor,
            )
            p_qso_samples[sl, :] = np.exp(log_p_qso - log_denom)
        else:
            log_denom = np.logaddexp(np.logaddexp(log_p_d, log_p_bg), log_p_floor)

        p_dwarf_samples[sl, :]      = np.exp(log_p_d  - log_denom)
        p_background_samples[sl, :] = np.exp(log_p_bg - log_denom)

        # ── Wide-field QSO training sample (3-class only) ─────────────────
        if has_qso_train:
            # k(G) inflation for training sources (clip matches models.py)
            k_q = (all_k0[sl][:, None]
                   * np.exp(np.clip(all_k1[sl][:, None] * (G_arr_q[None, :] - G_ref_q),
                                    -6.0, 6.0)))
            k_q = np.maximum(k_q, 0.1)

            S_floor_q = _chol_to_cov(all_Lf[sl])
            S_total_q = ((k_q[:, :, None, None]**2)
                         * S_obs_qso_train[None, :, :, :]
                         + S_floor_q[:, None, :, :])

            # QSO training sources use shifted frame (delta_pm_sys applied)
            if has_sys:
                shift_q = np.zeros((b, 3))
                shift_q[:, :2] = all_dpm[sl]
                y_eff_q = y_obs_qso_train[None, :, :] - shift_q[:, None, :]
            else:
                y_eff_q = np.broadcast_to(y_obs_qso_train[None, :, :], (b, M, 3))

            # Spatial log-density at training QSO positions: (b, M)
            x_sh_q = pos_obs_qso_train[None, :, 0] - dra_b / 60.0
            y_sh_q = pos_obs_qso_train[None, :, 1] - ddec_b / 60.0
            Xm_q   = -x_sh_q * np.cos(theta) + y_sh_q * np.sin(theta)
            Ym_q   =  x_sh_q * np.sin(theta) + y_sh_q * np.cos(theta)
            r_bq   = np.sqrt(Xm_q**2 + (Ym_q / q_b)**2)
            a_deg_b_q = apl_b / 60.0
            if spatial_profile == 'sersic':
                log_sp_dw_q = (- np.log(2 * np.pi)
                               - 2 * np.log(a_deg_b_q) - np.log(q_b)
                               - r_bq / a_deg_b_q)
            else:
                log_sp_dw_q = (np.log(1.0 / (np.pi * a_deg_b_q**2 * q_b))
                               - 2.0 * np.log(1.0 + r_bq**2 / a_deg_b_q**2))

            # Dwarf PDF for training sources
            logL_d_q  = _batch_logpdf(all_mud[sl], _chol_to_cov(all_Ld[sl]),
                                       S_total_q, y_eff_q)
            log_p_d_q = (log_fstar_b + log_fdw_b  # no photometric CMD prior
                         + logL_d_q + log_sp_dw_q)

            # Background PDF for training sources: precomputed from raw y_obs_qso_train
            log_p_bg_q  = (log_fstar_b + log_fdbg_b + log_pw_bg_q[None, :]
                           + bg_logL_qso_fixed[None, :] + log_spatial_mw)

            # QSO PDF for training sources
            logL_qso_q  = _batch_logpdf(zero_mu, zero_cov, S_total_q, y_eff_q)
            log_p_qso_q = (log_fqso_b + log_prior_qso_train[None, :]
                           + logL_qso_q + log_spatial_mw)

            log_p_floor_q = np.full_like(log_p_d_q, np.log(1e-300))
            log_denom_q   = np.logaddexp(
                np.logaddexp(np.logaddexp(log_p_d_q, log_p_bg_q), log_p_qso_q),
                log_p_floor_q,
            )
            p_qso_train_samples[sl, :] = np.exp(log_p_qso_q - log_denom_q)

        # ── HST Group B membership (frozen spatial prior) ─────────────────
        if _has_hst_b:
            k_hst_b    = all_khst[sl][:, None]
            S_floor_hb = _chol_to_cov(all_Lf_h[sl])
            S_total_b  = ((k_hst_b[:, :, None, None]**2)
                          * S_obs_hst_b[None, :, :, :]
                          + S_floor_hb[:, None, :, :])

            y_eff_b = (y_obs_hst_b[None, :, :] - shift_hst[:, None, :])
            # Z-latent correction for Group B
            if _has_latent_b:
                if not _has_latent_a:
                    z_batch = all_z[sl]          # (b, M) — compute if not already done
                y_eff_b = y_eff_b + np.einsum('ijk,bk->bij', S_latent_hst_b, z_batch)

            # 2-class model for Group B (HST stars are not QSOs)
            logL_b_d   = _batch_logpdf(all_mud[sl], _chol_to_cov(all_Ld[sl]),
                                        S_total_b, y_eff_b)
            log_p_b_d  = (log_fdw_b + log_prior_ws_hst_b[None, :, 0]
                          + logL_b_d + log_spatial_hst_b[None, :])   # (b, N_B)
            log_p_b_bg = (log_fdbg_b + log_pw_bg_b[None, :]
                          + bg_logL_hst_b_fixed[None, :]
                          + log_spatial_hst_b_mw)                     # (b, N_B)
            log_p_b_fl = np.full_like(log_p_b_d, np.log(1e-300))
            log_denom_b = np.logaddexp(
                np.logaddexp(log_p_b_d, log_p_b_bg), log_p_b_fl)
            p_dwarf_b_samples[sl, :] = np.exp(log_p_b_d - log_denom_b)

    final_membership_probs = np.median(p_dwarf_samples, axis=0)
    hst_b_final = (np.median(p_dwarf_b_samples, axis=0)
                   if _has_hst_b else None)

    if has_3class:
        if has_qso_train:
            result = (final_membership_probs, p_dwarf_samples,
                      p_background_samples, p_qso_samples,
                      np.median(p_qso_train_samples, axis=0))
        else:
            result = (final_membership_probs, p_dwarf_samples,
                      p_background_samples, p_qso_samples)
    else:
        result = (final_membership_probs, p_dwarf_samples,
                  p_background_samples)

    if _has_hst_b:
        return result + (hst_b_final,)
    return result
