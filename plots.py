"""
All plotting functions.  Every function saves to *result_path* and does not
call plt.show() so the script runs non-interactively.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import arviz as az

matplotlib.rc('font', family='serif', size=16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ellipse(ax, cx, cy, width, height, angle, color, lw, label=''):
    ell = Ellipse(xy=(cx, cy), width=width, height=height,
                  angle=angle, edgecolor=color, lw=lw, facecolor='none',
                  label=label)
    ax.add_patch(ell)


# ---------------------------------------------------------------------------
# Initial data plots (before spatial model)
# ---------------------------------------------------------------------------

def plot_initial_selection(radec_offsets, pms, colors, gmags,
                            keep, mean_pm, result_path, pm_labels=None):
    """Quick diagnostic: sky position and VPD for the initial keep mask."""
    if pm_labels is None:
        pm_labels = (r'$\mu_{\alpha*}$', r'$\mu_\delta$')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(radec_offsets[keep, 0], radec_offsets[keep, 1], s=1)
    ax.scatter(radec_offsets[:, 0],    radec_offsets[:, 1],
               s=1, c='grey', alpha=0.1, zorder=-1e10)
    ax.invert_xaxis(); ax.grid(True)
    ax.set_xlabel(r'$\Delta \alpha*$ (deg)')
    ax.set_ylabel(r'$\Delta \delta$ (deg)')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'diagnostic_position_initial.png'), dpi=100)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pms[keep, 0], pms[keep, 1], s=1)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.scatter(pms[:, 0], pms[:, 1], s=1, c='grey', alpha=0.1, zorder=-1e10)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.axvline(mean_pm[0], c='r', lw=1, ls='--')
    ax.axhline(mean_pm[1], c='r', lw=1, ls='--')
    ax.grid(True)
    ax.set_xlabel(f'{pm_labels[0]} (mas/yr)')
    ax.set_ylabel(f'{pm_labels[1]} (mas/yr)')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'diagnostic_VPD_initial.png'), dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Density-profile plots (spatial model output)
# ---------------------------------------------------------------------------

def plot_density_profile(radec_offsets, r_ell,
                          prior_params, posterior_params,
                          result_path, tag='update',
                          prior_center=(0.0, 0.0)):
    """
    Sky-position scatter coloured by elliptical radius, overlaid with
    prior (red) and posterior/updated (blue) ellipses.

    tag          : 'update'    → density_profile_update.png   (after spatial model)
                   'posterior' → density_profile_posterior.png (after GMM)
    prior_center : (ra_deg, dec_deg) position of the original LVD catalog centre
                   in the current radec_offsets frame.  Defaults to (0, 0).
                   Non-zero only when the frame has been shifted by iterated
                   centre updates, so the Prior ellipse always marks the
                   original LVD position rather than the updated centre.
    """
    # Prior ellipse geometry
    pa0  = prior_params['pa_mean']
    ell0 = prior_params['ellipticity_mean']
    r0   = prior_params['rhalf_mean'] / 60.0
    w0   = 2 * r0
    h0   = 2 * r0 * (1 - ell0)
    prior_cx, prior_cy = prior_center

    # Posterior ellipse geometry
    new_pa  = posterior_params['new_pa_deg']
    new_ell = posterior_params['new_ellipticity']
    new_a   = posterior_params['new_a_plummer'] / 60.0
    new_dra = posterior_params['new_delta_ra_center'] / 60.0
    new_ddec= posterior_params['new_delta_dec_center'] / 60.0
    w1  = 2 * new_a
    h1  = 2 * new_a * (1 - new_ell)

    label2 = 'Posterior' if tag == 'posterior' else 'Updated'

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(radec_offsets[:, 0], radec_offsets[:, 1],
                    s=1, alpha=1, c=r_ell)
    plt.colorbar(sc, ax=ax, label='Plummer Radii')
    ax.grid(True); ax.invert_xaxis(); ax.set_aspect('equal')

    for mult in [1, 2, 3]:
        lbl1 = 'Prior'  if mult == 1 else ''
        lbl2 = label2   if mult == 1 else ''
        _ellipse(ax, prior_cx, prior_cy, mult*w0, mult*h0, 90-pa0, 'red', 1, lbl1)
        _ellipse(ax, new_dra, new_ddec, mult*w1, mult*h1,
                 90-new_pa, 'C0', 2, lbl2)

    ax.legend(loc='best')
    ax.set_xlabel(r'$\Delta \alpha*$ (deg)', fontsize=20)
    ax.set_ylabel(r'$\Delta \delta$ (deg)',   fontsize=20)
    plt.tight_layout()
    fname = f'density_profile_{tag}.png'
    plt.savefig(os.path.join(result_path, fname), dpi=100)
    plt.close()
    print(f'  Saved {fname}')


# ---------------------------------------------------------------------------
# Pre-GMM "initial" diagnostic plots (Cell 92 in notebook)
# ---------------------------------------------------------------------------

def plot_pre_gmm(pos_obs, y_obs, colors, gmags, keep_inds,
                  log_prior_ws, mean_pm, result_path, pm_labels=None,
                  qso_pos=None, qso_pms=None, qso_prior_probs=None,
                  qso_cmd_colors=None, qso_cmd_gmags=None,
                  hst_b_data=None):
    """
    Combined 3×2 (or 3×3 with QSOs) summary figure saved as
    initial_population_summary.png.

    Rows: sky position, VPD, CMD.
    Cols: prior-member candidates (P>0.5), background candidates (P<0.5),
          and optionally QSO column coloured by prior P(QSO).
    """
    if pm_labels is None:
        pm_labels = (r'$\mu_{\alpha*}$', r'$\mu_\delta$')
    prob    = np.exp(log_prior_ws[:, 0])
    has_qso = (qso_pos is not None and len(qso_pos) > 0)
    ncols   = 3 if has_qso else 2

    n_qso_field = len(qso_pos) if has_qso else 0
    n_qso_cmd   = (int(np.isfinite(qso_cmd_gmags).sum())
                   if qso_cmd_gmags is not None else n_qso_field)
    col_labels = ['Prior member (P > 0.5)', 'Background (P < 0.5)']
    if has_qso:
        col_labels.append(
            f'QSO candidates  ({n_qso_field:,} in field  /  {n_qso_cmd:,} wide catalog)')

    high_prob = prob > 0.5
    low_prob  = prob < 0.5

    fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols, 18))

    _b_kw = dict(s=12, marker='^', edgecolors='k', linewidths=0.4,
                 vmin=0, vmax=1, cmap='RdYlBu', zorder=4)
    _b_prob = hst_b_data['probs'] if hst_b_data is not None else None

    for col, (mask, col_label) in enumerate(zip([high_prob, low_prob], col_labels)):
        # ── Sky position ──────────────────────────────────────────────────
        ax = axes[0, col]
        ax.scatter(pos_obs[~mask, 0], pos_obs[~mask, 1],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        sc = ax.scatter(pos_obs[mask, 0], pos_obs[mask, 1],
                        s=2, c=prob[mask], vmin=0, vmax=1, cmap='RdYlBu', zorder=2)
        plt.colorbar(sc, ax=ax, label='Prior P(member)')
        if hst_b_data is not None:
            ax.scatter(hst_b_data['pos'][:, 0], hst_b_data['pos'][:, 1],
                       c=_b_prob, **_b_kw,
                       label='BP3M Group B' if col == 0 else '')
            if col == 0:
                ax.legend(fontsize=8, loc='upper right')
        ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        ax.set_xlabel(r'$\Delta\alpha*$ (deg)'); ax.set_ylabel(r'$\Delta\delta$ (deg)')
        ax.set_title(col_label, fontsize=13)

        # ── VPD ───────────────────────────────────────────────────────────
        ax = axes[1, col]
        ax.scatter(y_obs[~mask, 0], y_obs[~mask, 1],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        sc = ax.scatter(y_obs[mask, 0], y_obs[mask, 1],
                        s=2, c=prob[mask], vmin=0, vmax=1, cmap='RdYlBu', zorder=2)
        plt.colorbar(sc, ax=ax, label='Prior P(member)')
        if hst_b_data is not None:
            ax.scatter(hst_b_data['pms'][:, 0], hst_b_data['pms'][:, 1],
                       c=_b_prob, **_b_kw)
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.axvline(mean_pm[0], c='r', lw=1, ls='--')
        ax.axhline(mean_pm[1], c='r', lw=1, ls='--')
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        ax.set_xlabel(f'{pm_labels[0]} (mas/yr)'); ax.set_ylabel(f'{pm_labels[1]} (mas/yr)')

        # ── CMD ───────────────────────────────────────────────────────────
        ax = axes[2, col]
        not_mask_inds = keep_inds[~mask]
        mask_inds     = keep_inds[mask]
        ax.scatter(colors[not_mask_inds, 0], gmags[not_mask_inds, 0],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        sc = ax.scatter(colors[mask_inds, 0], gmags[mask_inds, 0],
                        s=2, c=prob[mask], vmin=0, vmax=1, cmap='RdYlBu', zorder=2)
        plt.colorbar(sc, ax=ax, label='Prior P(member)')
        if hst_b_data is not None:
            _bc = hst_b_data['colors']
            _bg = hst_b_data['gmags']
            _bv = np.isfinite(_bc) & np.isfinite(_bg)
            ax.scatter(_bc[_bv], _bg[_bv],
                       c=_b_prob[_bv], **_b_kw)
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
        ax.set_xlabel('BP $-$ RP (mag)'); ax.set_ylabel('G (mag)')

    if has_qso:
        col    = 2
        _qp    = (qso_prior_probs if qso_prior_probs is not None
                  else np.ones(n_qso_field))
        qso_cm = 'Oranges'

        # ── Position ──────────────────────────────────────────────────────
        ax = axes[0, col]
        ax.scatter(pos_obs[:, 0], pos_obs[:, 1],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        sc = ax.scatter(qso_pos[:, 0], qso_pos[:, 1],
                        s=15, c=_qp, cmap=qso_cm, vmin=0, vmax=1,
                        marker='D', alpha=0.8, zorder=5)
        plt.colorbar(sc, ax=ax, label='Prior P(QSO)')
        ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        ax.set_xlabel(r'$\Delta\alpha*$ (deg)'); ax.set_ylabel(r'$\Delta\delta$ (deg)')
        ax.set_title(col_labels[2], fontsize=13)

        # ── VPD ───────────────────────────────────────────────────────────
        ax = axes[1, col]
        ax.scatter(y_obs[:, 0], y_obs[:, 1],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        _pms = qso_pms if qso_pms is not None else qso_pos
        sc = ax.scatter(_pms[:, 0], _pms[:, 1],
                        s=15, c=_qp, cmap=qso_cm, vmin=0, vmax=1,
                        marker='D', alpha=0.8, zorder=5)
        plt.colorbar(sc, ax=ax, label='Prior P(QSO)')
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.axhline(0, c='k', lw=0.5, ls='--')
        ax.axvline(0, c='k', lw=0.5, ls='--')
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        ax.set_xlabel(f'{pm_labels[0]} (mas/yr)'); ax.set_ylabel(f'{pm_labels[1]} (mas/yr)')

        # ── CMD (wide-field clean QSO catalog) ────────────────────────────
        ax = axes[2, col]
        ax.scatter(colors[keep_inds, 0], gmags[keep_inds, 0],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        if qso_cmd_gmags is not None and qso_cmd_colors is not None:
            valid_q = np.isfinite(qso_cmd_colors) & np.isfinite(qso_cmd_gmags)
            if valid_q.sum() > 0:
                sc = ax.scatter(qso_cmd_colors[valid_q], qso_cmd_gmags[valid_q],
                               s=10, c=np.full(valid_q.sum(), 0.99),
                               cmap=qso_cm, vmin=0, vmax=1,
                               marker='D', alpha=0.7, zorder=5,
                               label=f'Wide QSO catalog  N={valid_q.sum():,}')
                plt.colorbar(sc, ax=ax, label='Prior P(QSO)')
                ax.legend(fontsize=9)
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
        ax.set_xlabel('BP $-$ RP (mag)'); ax.set_ylabel('G (mag)')

    plt.suptitle('Initial population summary (pre-GMM)', fontsize=15, y=1.01)
    plt.tight_layout()
    fname = 'initial_population_summary.png'
    plt.savefig(os.path.join(result_path, fname), dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ---------------------------------------------------------------------------
# Background GMM diagnostic
# ---------------------------------------------------------------------------

def plot_background_gmm(pm_and_paras, good_backgrounds,
                         bg_means, bg_covs, bg_weights,
                         radec_offsets, r_ell,
                         result_path, tag='initial',
                         gmags=None, colors=None, pm_labels=None):
    """
    Two diagnostic figures for the pre-fit background GMM.

    Figure 1 – spatial_selection (sky + VPD + CMD):
      Panel 1: sky positions coloured by background selection
      Panel 2: VPD coloured by background selection
      Panel 3: G vs BP-RP CMD coloured by background selection (if photometry provided)

    Figure 2 – gmm_diagnostics (3 PM/parallax panels with GMM ellipses):
      Panel 1 – VPD (μ_α* vs μ_δ)
      Panel 2 – μ_α* vs ϖ
      Panel 3 – μ_δ vs ϖ

    Figure 3 – gmm_marginals (1-D histograms with GMM density overlaid)
    """
    from matplotlib.patches import Ellipse as MEllipse
    from matplotlib.cm import get_cmap
    import matplotlib.colors as mcolors

    if pm_labels is None:
        pm_labels = (r'$\mu_{\alpha*}$', r'$\mu_\delta$')

    has_cmd = (gmags is not None) and (colors is not None)
    ncols   = 3 if has_cmd else 2
    figw    = 7 * ncols

    # ── Figure 1: spatial selection ───────────────────────────────────────
    fig, axes1 = plt.subplots(1, ncols, figsize=(figw, 6))
    ax_sky, ax_vpd = axes1[0], axes1[1]
    ax_cmd = axes1[2] if has_cmd else None

    # Sky positions
    ax_sky.scatter(radec_offsets[~good_backgrounds, 0],
                   radec_offsets[~good_backgrounds, 1],
                   s=1, c='lightgrey', rasterized=True, label='galaxy region')
    ax_sky.scatter(radec_offsets[good_backgrounds, 0],
                   radec_offsets[good_backgrounds, 1],
                   s=4, c='steelblue', rasterized=True, label='background (training)')
    ax_sky.invert_xaxis()
    ax_sky.set_xlabel(r'$\Delta\alpha*$ (deg)', fontsize=13)
    ax_sky.set_ylabel(r'$\Delta\delta$ (deg)',  fontsize=13)
    ax_sky.set_title('Spatial selection', fontsize=13)
    ax_sky.legend(fontsize=10, markerscale=3)
    ax_sky.grid(True, alpha=0.3)

    # VPD coloured by selection
    has_pm = np.isfinite(pm_and_paras[:, 0])
    ax_vpd.scatter(pm_and_paras[has_pm & ~good_backgrounds, 0],
                   pm_and_paras[has_pm & ~good_backgrounds, 1],
                   s=1, c='lightgrey', rasterized=True)
    ax_vpd.scatter(pm_and_paras[good_backgrounds, 0],
                   pm_and_paras[good_backgrounds, 1],
                   s=4, c='steelblue', rasterized=True)
    # Clip to background bulk to keep axes legible
    X_bg = pm_and_paras[good_backgrounds]
    X_bg = X_bg[np.all(np.isfinite(X_bg), axis=1)]
    for dim, ax_lim in [(0, ax_vpd.set_xlim), (1, ax_vpd.set_ylim)]:
        lo, hi = np.nanpercentile(X_bg[:, dim], [1, 99])
        pad = 0.2 * (hi - lo)
        ax_lim(lo - pad, hi + pad)
    ax_vpd.set_xlabel(f'{pm_labels[0]} (mas/yr)', fontsize=13)
    ax_vpd.set_ylabel(f'{pm_labels[1]} (mas/yr)', fontsize=13)
    ax_vpd.set_title('VPD — background highlighted', fontsize=13)
    ax_vpd.grid(True, alpha=0.3)

    # CMD coloured by selection
    if has_cmd:
        g   = gmags[:, 0]   if gmags.ndim > 1 else gmags
        col = colors[:, 0]  if colors.ndim > 1 else colors
        valid_cmd = np.isfinite(g) & np.isfinite(col)
        ax_cmd.scatter(col[valid_cmd & ~good_backgrounds],
                       g[valid_cmd & ~good_backgrounds],
                       s=1, c='lightgrey', rasterized=True)
        ax_cmd.scatter(col[valid_cmd & good_backgrounds],
                       g[valid_cmd & good_backgrounds],
                       s=4, c='steelblue', rasterized=True)
        # Axis limits from full valid range (invert G so bright stars are at top)
        ax_cmd.invert_yaxis()
        ax_cmd.set_xlabel('BP $-$ RP (mag)', fontsize=13)
        ax_cmd.set_ylabel('G (mag)',          fontsize=13)
        ax_cmd.set_title('CMD — background highlighted', fontsize=13)
        ax_cmd.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'background_spatial_selection_{tag}.png'), dpi=100)
    plt.close()
    print(f'  Saved background_spatial_selection_{tag}.png')

    # ── Figures 2 & 3: GMM quality diagnostics ────────────────────────────
    X  = X_bg  # already finite-filtered background stars
    K  = len(bg_weights)
    cmap_k = get_cmap('plasma', K)
    norm_k = mcolors.Normalize(vmin=0, vmax=K - 1)

    dim_labels = [f'{pm_labels[0]} (mas/yr)', f'{pm_labels[1]} (mas/yr)',
                  r'$\varpi$ (mas)']
    pairs = [(0, 1), (0, 2), (1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (ix, iy) in zip(axes, pairs):
        ax.scatter(X[:, ix], X[:, iy], s=1, alpha=0.2, c='steelblue', rasterized=True)
        for k in range(K):
            mu_k  = bg_means[k]
            cov_k = bg_covs[k]
            col   = cmap_k(norm_k(k))
            sub   = cov_k[np.ix_([ix, iy], [ix, iy])]
            vals, vecs = np.linalg.eigh(sub)
            vals  = np.maximum(vals[vals.argsort()[::-1]], 0)
            vecs  = vecs[:, vals.argsort()[::-1]]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            for n_sig, lw, ls in [(1, 1.5, '-'), (2, 0.8, '--')]:
                ax.add_patch(MEllipse(
                    xy=(mu_k[ix], mu_k[iy]),
                    width=2 * n_sig * np.sqrt(vals[0]),
                    height=2 * n_sig * np.sqrt(vals[1]),
                    angle=angle, edgecolor=col, lw=lw, linestyle=ls,
                    facecolor='none', zorder=5,
                    label=f'k={k} (w={bg_weights[k]:.2f})' if n_sig == 1 else '',
                ))
        lo_x, hi_x = np.nanpercentile(X[:, ix], [1, 99])
        lo_y, hi_y = np.nanpercentile(X[:, iy], [1, 99])
        pad_x = 0.15 * (hi_x - lo_x); pad_y = 0.15 * (hi_y - lo_y)
        ax.set_xlim(lo_x - pad_x, hi_x + pad_x)
        ax.set_ylim(lo_y - pad_y, hi_y + pad_y)
        ax.set_xlabel(dim_labels[ix], fontsize=14)
        ax.set_ylabel(dim_labels[iy], fontsize=14)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, markerscale=3, loc='upper right')
    fig.suptitle(f'Background GMM  (K={K}, N={len(X):,})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'background_gmm_diagnostics_{tag}.png'), dpi=100)
    plt.close()
    print(f'  Saved background_gmm_diagnostics_{tag}.png')

    # ── Marginal 1-D histograms ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for dim, ax in enumerate(axes):
        d1d = X[:, dim]; d1d = d1d[np.isfinite(d1d)]
        lo, hi = np.nanpercentile(d1d, [0.5, 99.5])
        ax.hist(d1d, bins=80, density=True, alpha=0.5,
                range=(lo, hi), color='steelblue', label='data')
        t = np.linspace(lo, hi, 500)
        gmm_pdf = sum(
            bg_weights[k] / (np.sqrt(bg_covs[k, dim, dim]) * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((t - bg_means[k, dim]) / np.sqrt(bg_covs[k, dim, dim]))**2)
            for k in range(K)
        )
        ax.plot(t, gmm_pdf, 'r-', lw=2, label='GMM')
        ax.set_xlabel(dim_labels[dim], fontsize=14)
        ax.set_ylabel('density', fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'background_gmm_marginals_{tag}.png'), dpi=100)
    plt.close()
    print(f'  Saved background_gmm_marginals_{tag}.png')


# ---------------------------------------------------------------------------
# ArviZ diagnostic plots per component
# ---------------------------------------------------------------------------

def _az_plot_posterior(trace, var_names):
    """Call plot_posterior, handling its move to arviz_plots in newer ArviZ."""
    try:
        az.plot_posterior(trace, var_names=var_names)
        return
    except AttributeError:
        pass
    try:
        import arviz_plots as azp
        azp.plot_posterior(trace, var_names=var_names)
    except Exception:
        raise


def _az_plot_dist_comparison(trace, var_names):
    """Call plot_dist_comparison, handling its move to arviz_plots in newer ArviZ."""
    try:
        az.plot_dist_comparison(trace, var_names=var_names)
        return
    except AttributeError:
        pass
    try:
        import arviz_plots as azp
        azp.plot_dist_comparison(trace, var_names=var_names)
    except Exception:
        raise


def plot_arviz_diagnostics(trace, var_names, component_name, field, result_path):
    """
    Corner (pair), trace, posterior, and prior/posterior comparison plots
    for a named set of variables.  Variables absent from the trace are silently
    skipped so old traces and new traces both work.  Each plot is wrapped in
    try/except so an API change in one ArviZ version cannot crash the pipeline.
    """
    _post = trace.posterior
    if hasattr(_post, 'ds'):
        _post = _post.ds
    available = set(_post.data_vars)
    var_names = [v for v in var_names if v in available]
    if not var_names:
        return
    prefix = os.path.join(result_path, f'{field}')

    def _save(path):
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close()

    try:
        # Raise the subplot cap so large corner plots aren't silently dropped.
        old_max = matplotlib.rcParams.get('plot.max_subplots', 40)
        matplotlib.rcParams['plot.max_subplots'] = max(old_max,
                                                        len(var_names) ** 2 + 1)
        az.plot_pair(trace, var_names=var_names)
        matplotlib.rcParams['plot.max_subplots'] = old_max
        _save(f'{prefix}_corner_{component_name}.png')
    except Exception as e:
        plt.close('all')
        print(f'  WARNING: plot_pair failed for {component_name}: {e}')

    try:
        az.plot_trace(trace, var_names=var_names)
        _save(f'{prefix}_trace_{component_name}.png')
    except Exception as e:
        plt.close('all')
        print(f'  WARNING: plot_trace failed for {component_name}: {e}')

    try:
        _az_plot_posterior(trace, var_names)
        _save(f'{prefix}_posterior_{component_name}.png')
    except Exception as e:
        plt.close('all')
        print(f'  WARNING: plot_posterior failed for {component_name}: {e}')

    try:
        _az_plot_dist_comparison(trace, var_names)
        _save(f'{prefix}_prior_comp_posterior_{component_name}.png')
    except Exception as e:
        plt.close('all')
        print(f'  WARNING: plot_dist_comparison failed for {component_name}: {e}')

    print(f'  Saved ArviZ diagnostics for {component_name}')


def plot_spatial_diagnostics(trace, field, result_path, tag=''):
    """ArviZ trace/posterior/prior-comparison plots for a spatial model run."""
    var_names = ['delta_ra_center', 'delta_dec_center',
                 'a_plummer', 'ellipticity', 'pa_deg', 'f_dwarf']
    comp_name = f'spatial{"_" + tag if tag else ""}'
    plot_arviz_diagnostics(trace, var_names, comp_name, field, result_path)


def plot_all_diagnostics(trace, field, result_path):
    """Run ArviZ diagnostics for all named groups."""
    components = [
        ('density_profile', [
            'delta_ra_center', 'delta_dec_center', 'a_plummer',
            'ellipticity', 'pa_deg',
        ]),
        ('target', [
            'mu_dwarf', 'sigma_intrinsic_dwarf', 'rho_pm_intrinsic_dwarf',
            'rho_pmra_parallax_intrinsic_dwarf', 'rho_pmdec_parallax_intrinsic_dwarf',
        ]),
        ('systematics', [
            'f_dwarf', 'k', 'k_1', 'delta_pm_sys',
            'sigma_floor', 'rho_pm_floor',
            'rho_pmra_parallax_floor', 'rho_pmdec_parallax_floor',
        ]),
        # HST-specific group: only produces output when --bp3m-dir was used;
        # plot_arviz_diagnostics silently skips variables absent from the trace.
        ('hst_systematics', [
            'k_hst', 'delta_pm_sys_hst',
            'sigma_floor_hst', 'rho_pm_floor_hst',
            'rho_pmra_parallax_floor_hst', 'rho_pmdec_parallax_floor_hst',
        ]),
    ]
    for comp_name, var_names in components:
        plot_arviz_diagnostics(trace, var_names, comp_name, field, result_path)


# ---------------------------------------------------------------------------
# QSO systematic cleaning diagnostics
# ---------------------------------------------------------------------------

def plot_qso_cleaning(qso_df, result, result_path, n_gaia_raw=None):
    """
    4-panel diagnostic for the QSO-based PM systematic cleaning.

    Panel 1 — PM scatter of all raw candidates, coloured by fate:
        orange = rejected by parallax cut
        red    = rejected by iterative PM sigma-clip
        blue   = kept (clean QSO sample)
        black star = final measured systematic with error bars

    Panel 2 — zoomed PM scatter of the kept sample only, with the
        measured systematic (red star) and reference lines at (0,0).

    Panel 3 — histogram of parallax significance |ϖ|/σ_ϖ; vertical
        line marks the parallax_nsigma cut threshold.

    Panel 4 — histogram of PM significance max(|Δμ_α*|/σ, |Δμ_δ|/σ)
        computed relative to the final clean mean, for sources that
        survived the parallax cut; vertical line marks pm_nsigma.
    """
    pmra  = qso_df['pmra'].to_numpy(dtype=float)
    pmdec = qso_df['pmdec'].to_numpy(dtype=float)
    e_ra  = qso_df['pmra_error'].to_numpy(dtype=float)
    e_dec = qso_df['pmdec_error'].to_numpy(dtype=float)

    mask_valid    = result['mask_valid']
    mask_plx_kept = result['mask_plx_kept']
    mask_kept     = result['mask_kept']
    plx_rejected  = mask_valid & ~mask_plx_kept
    pm_clipped    = mask_plx_kept & ~mask_kept

    delta_pmra  = result['delta_pmra']
    delta_pmdec = result['delta_pmdec']
    sigma_pmra  = result['sigma_pmra']
    sigma_pmdec = result['sigma_pmdec']
    plx_nsig    = result['parallax_nsigma']
    pm_nsig     = result['pm_nsigma']
    n_kept      = result['n_kept']
    n_raw       = result['n_raw']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    if n_gaia_raw is not None and n_gaia_raw != n_raw:
        title = (f'QSO PM systematic cleaning — '
                 f'{n_gaia_raw:,} Gaia raw  →  {n_raw:,} MILLIQUAS-matched  '
                 f'→  {n_kept:,} kept after cleaning')
    else:
        title = f'QSO PM systematic cleaning — {n_kept:,} / {n_raw:,} candidates kept'
    fig.suptitle(title, fontsize=12)

    # ── Panel 1: PM scatter, all sources coloured by fate ─────────────────
    ax = axes[0, 0]
    clip = 15.0
    kw = dict(s=3, alpha=0.4, rasterized=True)
    ax.scatter(pmra[plx_rejected], pmdec[plx_rejected],
               c='orange', label=f'Parallax cut ({plx_rejected.sum():,})', **kw)
    ax.scatter(pmra[pm_clipped],   pmdec[pm_clipped],
               c='red',    label=f'PM clipped ({pm_clipped.sum():,})',    **kw)
    ax.scatter(pmra[mask_kept],    pmdec[mask_kept],
               c='steelblue', label=f'Kept ({n_kept:,})',                **kw)
    if delta_pmra is not None:
        ax.errorbar(delta_pmra, delta_pmdec,
                    xerr=sigma_pmra, yerr=sigma_pmdec,
                    fmt='k*', ms=12, zorder=6, label='Systematic')
    ax.axhline(0, c='k', lw=0.6, ls='--')
    ax.axvline(0, c='k', lw=0.6, ls='--')
    ax.set_xlim(-clip, clip)
    ax.set_ylim(-clip, clip)
    ax.set_xlabel(r'$\mu_{\alpha*}$ (mas yr$^{-1}$)')
    ax.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)')
    ax.set_title('All candidates')
    ax.legend(fontsize=9, markerscale=2)

    # ── Panel 2: Zoomed PM scatter of kept sample only ────────────────────
    ax = axes[0, 1]
    ax.scatter(pmra[mask_kept], pmdec[mask_kept],
               c='steelblue', s=4, alpha=0.4, rasterized=True)
    if delta_pmra is not None:
        ax.errorbar(delta_pmra, delta_pmdec,
                    xerr=sigma_pmra, yerr=sigma_pmdec,
                    fmt='r*', ms=14, zorder=5,
                    label=(fr'$\Delta\mu_{{\alpha*}} = {delta_pmra:+.4f} \pm {sigma_pmra:.4f}$'
                           '\n'
                           fr'$\Delta\mu_{{\delta}} = {delta_pmdec:+.4f} \pm {sigma_pmdec:.4f}$'))
        ax.legend(fontsize=9)
        spread = max(np.nanstd(pmra[mask_kept]),
                     np.nanstd(pmdec[mask_kept]), 0.3) * 4
        ax.set_xlim(delta_pmra - spread, delta_pmra + spread)
        ax.set_ylim(delta_pmdec - spread, delta_pmdec + spread)
    ax.axhline(0, c='k', lw=0.6, ls='--')
    ax.axvline(0, c='k', lw=0.6, ls='--')
    ax.set_xlabel(r'$\mu_{\alpha*}$ (mas yr$^{-1}$)')
    ax.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)')
    ax.set_title(f'Kept sample ({n_kept:,} QSOs)')

    # ── Panel 3: Parallax significance histogram ──────────────────────────
    ax = axes[1, 0]
    has_plx = ('parallax' in qso_df.columns and 'parallax_error' in qso_df.columns)
    if has_plx:
        plx   = qso_df['parallax'].to_numpy(dtype=float)
        e_plx = qso_df['parallax_error'].to_numpy(dtype=float)
        plx_ok = mask_valid & np.isfinite(plx) & np.isfinite(e_plx) & (e_plx > 0)
        plx_sig = np.full(len(qso_df), np.nan)
        plx_sig[plx_ok] = np.abs(plx[plx_ok]) / e_plx[plx_ok]

        sig_kept = plx_sig[mask_plx_kept & plx_ok]
        sig_rej  = plx_sig[plx_rejected  & plx_ok]
        bmax = max(float(np.nanpercentile(plx_sig[plx_ok], 99)) if plx_ok.any() else 10,
                   plx_nsig * 1.5)
        bins = np.linspace(0, bmax, 60)
        ax.hist(sig_kept, bins=bins, color='steelblue', alpha=0.7,
                label=f'Kept ({len(sig_kept):,})')
        ax.hist(sig_rej,  bins=bins, color='orange',    alpha=0.7,
                label=f'Rejected ({len(sig_rej):,})')
        ax.axvline(plx_nsig, c='k', lw=1.5, ls='--',
                   label=f'{plx_nsig:.0f}σ threshold')
        ax.set_xlabel(r'$|\varpi|\,/\,\sigma_\varpi$')
        ax.set_ylabel('Count')
        ax.set_title('Parallax significance')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No parallax data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    # ── Panel 4: PM significance histogram ───────────────────────────────
    ax = axes[1, 1]
    if delta_pmra is not None and mask_plx_kept.sum() > 0:
        chi_ra  = np.abs(pmra[mask_plx_kept]  - delta_pmra)  / e_ra[mask_plx_kept]
        chi_dec = np.abs(pmdec[mask_plx_kept] - delta_pmdec) / e_dec[mask_plx_kept]
        chi_max = np.maximum(chi_ra, chi_dec)

        sub_kept    = mask_kept[mask_plx_kept]
        sub_clipped = ~mask_kept[mask_plx_kept]

        bmax = max(float(np.nanpercentile(chi_max, 99)), pm_nsig * 1.5)
        bins = np.linspace(0, bmax, 60)
        ax.hist(chi_max[sub_kept],    bins=bins, color='steelblue', alpha=0.7,
                label=f'Kept ({sub_kept.sum():,})')
        ax.hist(chi_max[sub_clipped], bins=bins, color='red',       alpha=0.7,
                label=f'Clipped ({sub_clipped.sum():,})')
        ax.axvline(pm_nsig, c='k', lw=1.5, ls='--',
                   label=f'{pm_nsig:.0f}σ threshold')
        ax.set_xlabel(r'$\max\!\left(|\mu_i - \hat\mu_i|\,/\,\sigma_{\mu_i}\right)$')
        ax.set_ylabel('Count')
        ax.set_title('PM significance (post-parallax cut)')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Insufficient QSOs', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    out = os.path.join(result_path, 'qso_cleaning.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved QSO cleaning diagnostics → {out}')


# ---------------------------------------------------------------------------
# Final membership plots
# ---------------------------------------------------------------------------

def plot_final_membership(radec_offsets, pms, colors, gmags,
                           keep_inds, good_to_keep,
                           final_membership_probs, mu_d,
                           result_path, pm_labels=None,
                           final_qso_probs=None,
                           hst_b_data=None):
    """
    Combined 3×(2 or 3) summary figure saved as final_population_summary.png.

    Rows: sky position, VPD, CMD.
    Cols: members (P_target > 0.5), background (P_bg > 0.5),
          and optionally QSO column (P_qso >= 0.5) when final_qso_probs given.
    """
    if pm_labels is None:
        pm_labels = (r'$\mu_{\alpha*}$', r'$\mu_\delta$')

    has_qso_col    = final_qso_probs is not None
    final_bg_probs = (1.0 - final_membership_probs - final_qso_probs
                      if has_qso_col else 1.0 - final_membership_probs)
    final_members  = final_membership_probs > 0.5
    final_bg       = final_bg_probs > 0.5

    ncols = 3 if has_qso_col else 2

    col_data = [
        (final_members, r'Target members: $P > 0.5$',    final_membership_probs),
        (final_bg,      r'Background: $P_{\rm bg} > 0.5$', final_bg_probs),
    ]
    if has_qso_col:
        final_qso_high = final_qso_probs >= 0.5
        col_data.append(
            (final_qso_high, r'QSOs: $P_{\rm QSO} \geq 0.5$', final_qso_probs)
        )

    fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols, 18))

    # Group B mask arrays (members vs background split)
    _has_b = (hst_b_data is not None and 'probs' in hst_b_data
              and len(hst_b_data['probs']) > 0)
    if _has_b:
        _b_prob  = hst_b_data['probs']
        _b_memb  = _b_prob > 0.5
        _b_bg    = ~_b_memb
        _b_masks = [_b_memb, _b_bg]  # index matches col 0/1

    for col, (mask_gmm, col_label, prob_col) in enumerate(col_data):
        cidxs = keep_inds[mask_gmm]
        prob  = prob_col[mask_gmm]
        cmap  = 'Blues' if col == 0 else ('Reds' if col == 1 else 'Oranges')
        _b_kw = dict(s=15, marker='^', edgecolors='k', linewidths=0.4,
                     cmap=cmap, vmin=0, vmax=1, zorder=4)

        # ── Sky ──────────────────────────────────────────────────────────
        ax = axes[0, col]
        ax.scatter(radec_offsets[good_to_keep, 0], radec_offsets[good_to_keep, 1],
                   s=1, c='lightgrey', alpha=0.3, rasterized=True, zorder=1)
        sc = ax.scatter(radec_offsets[cidxs, 0], radec_offsets[cidxs, 1],
                        s=2, c=prob, cmap=cmap, vmin=0, vmax=1, zorder=2)
        plt.colorbar(sc, ax=ax)
        if _has_b and col < 2:
            _bm = _b_masks[col]
            if _bm.any():
                ax.scatter(hst_b_data['pos'][_bm, 0], hst_b_data['pos'][_bm, 1],
                           c=_b_prob[_bm], **_b_kw,
                           label='BP3M Group B' if col == 0 else '')
                if col == 0:
                    ax.legend(fontsize=8, loc='upper right')
        ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        ax.set_xlabel(r'$\Delta\alpha*$ (deg)'); ax.set_ylabel(r'$\Delta\delta$ (deg)')
        ax.set_title(col_label, fontsize=13)

        # ── VPD ──────────────────────────────────────────────────────────
        ax = axes[1, col]
        ax.scatter(pms[good_to_keep, 0], pms[good_to_keep, 1],
                   s=1, c='lightgrey', alpha=0.3, rasterized=True, zorder=1)
        sc = ax.scatter(pms[cidxs, 0], pms[cidxs, 1],
                        s=2, c=prob, cmap=cmap, vmin=0, vmax=1, zorder=2)
        plt.colorbar(sc, ax=ax)
        if _has_b and col < 2:
            _bm = _b_masks[col]
            if _bm.any():
                ax.scatter(hst_b_data['pms'][_bm, 0], hst_b_data['pms'][_bm, 1],
                           c=_b_prob[_bm], **_b_kw)
        ax.axvline(mu_d[0], c='r', lw=1, ls='--')
        ax.axhline(mu_d[1], c='r', lw=1, ls='--')
        # auto-zoom to include both Gaia and Group B in the selected column
        _vpd_pts = []
        if len(cidxs) > 0:
            _vpd_pts.append(pms[cidxs])
        if _has_b and col < 2 and _b_masks[col].any():
            _vpd_pts.append(hst_b_data['pms'][_b_masks[col]])
        if _vpd_pts:
            _all_vpd = np.concatenate(_vpd_pts, axis=0)
            lo0, hi0 = np.nanpercentile(_all_vpd[:, 0], [2, 98])
            lo1, hi1 = np.nanpercentile(_all_vpd[:, 1], [2, 98])
            pad = max(hi0 - lo0, hi1 - lo1, 1.0) * 0.5
            ax.set_xlim(lo0 - pad, hi0 + pad); ax.set_ylim(lo1 - pad, hi1 + pad)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f'{pm_labels[0]} (mas/yr)'); ax.set_ylabel(f'{pm_labels[1]} (mas/yr)')

        # ── CMD ──────────────────────────────────────────────────────────
        ax = axes[2, col]
        ax.scatter(colors[good_to_keep, 0], gmags[good_to_keep, 0],
                   s=1, c='lightgrey', alpha=0.3, rasterized=True, zorder=1)
        sc = ax.scatter(colors[cidxs, 0], gmags[cidxs, 0],
                        s=2, c=prob, cmap=cmap, vmin=0, vmax=1, zorder=2)
        plt.colorbar(sc, ax=ax)
        if _has_b and col < 2:
            _bm = _b_masks[col]
            if _bm.any():
                _bc = hst_b_data['colors'][_bm]
                _bg = hst_b_data['gmags'][_bm]
                _bv = np.isfinite(_bc) & np.isfinite(_bg)
                if _bv.any():
                    ax.scatter(_bc[_bv], _bg[_bv],
                               c=_b_prob[_bm][_bv], **_b_kw)
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
        ax.set_xlabel('BP $-$ RP (mag)'); ax.set_ylabel('G (mag)')

    plt.suptitle('Final membership summary', fontsize=15, y=1.01)
    plt.tight_layout()
    fname = 'final_population_summary.png'
    plt.savefig(os.path.join(result_path, fname), dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_qso_membership_diagnostics(radec_offsets, pms, colors, gmags,
                                     keep_inds, final_qso_probs,
                                     is_milliquas, is_gaia_only,
                                     result_path, pm_labels=None,
                                     qso_train_radec=None, y_obs_qso=None,
                                     qso_train_gmags=None, qso_train_bpmags=None,
                                     qso_train_rpmags=None,
                                     qso_train_probs=None):
    """
    3×3 (or 3×4 with wide-field training column) QSO diagnostic figure.

    Rows: sky position, VPD, CMD.
    Cols: MILLIQUAS-confirmed, Gaia-only candidates, model QSOs (P>=0.5),
          and optionally the wide-field clean QSO training sample.

    The first three columns colour highlighted points by posterior P(QSO).
    The optional 4th column shows the full wide-field training sample coloured
    by posterior P(QSO) (computed with the same GMM draws); CMD and VPD are
    zoomed to the training sample's photometric / kinematic range.
    """
    if pm_labels is None:
        pm_labels = (r'$\mu_{\alpha*}$', r'$\mu_\delta$')

    model_qso = final_qso_probs >= 0.5

    has_train = (qso_train_radec is not None and y_obs_qso is not None
                 and qso_train_gmags is not None)
    ncols = 4 if has_train else 3

    col_specs = [
        (is_milliquas[keep_inds], 'MILLIQUAS-confirmed'),
        (is_gaia_only[keep_inds], 'Gaia-only candidates'),
        (model_qso,               r'Model QSOs ($P_{\rm QSO} \geq 0.5$)'),
    ]

    fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols, 18))
    qso_cm = 'Oranges'

    for col, (mask, col_label) in enumerate(col_specs):
        cidxs     = keep_inds[mask]
        n_col     = mask.sum()
        prob_col  = final_qso_probs[mask]

        # ── Sky ──────────────────────────────────────────────────────────
        ax = axes[0, col]
        ax.scatter(radec_offsets[keep_inds, 0], radec_offsets[keep_inds, 1],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        if n_col > 0:
            sc = ax.scatter(radec_offsets[cidxs, 0], radec_offsets[cidxs, 1],
                            s=15, c=prob_col, cmap=qso_cm, vmin=0, vmax=1,
                            marker='D', alpha=0.8, zorder=3)
            plt.colorbar(sc, ax=ax, label='P(QSO)')
            ax.text(0.02, 0.98, f'N={n_col:,}', transform=ax.transAxes,
                    fontsize=10, va='top', ha='left')
        ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        ax.set_xlabel(r'$\Delta\alpha*$ (deg)'); ax.set_ylabel(r'$\Delta\delta$ (deg)')
        ax.set_title(col_label, fontsize=12)

        # ── VPD ──────────────────────────────────────────────────────────
        ax = axes[1, col]
        ax.scatter(pms[keep_inds, 0], pms[keep_inds, 1],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        if n_col > 0:
            sc = ax.scatter(pms[cidxs, 0], pms[cidxs, 1],
                            s=15, c=prob_col, cmap=qso_cm, vmin=0, vmax=1,
                            marker='D', alpha=0.8, zorder=3)
            plt.colorbar(sc, ax=ax, label='P(QSO)')
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.axhline(0, c='k', lw=0.5, ls='--'); ax.axvline(0, c='k', lw=0.5, ls='--')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f'{pm_labels[0]} (mas/yr)'); ax.set_ylabel(f'{pm_labels[1]} (mas/yr)')

        # ── CMD ──────────────────────────────────────────────────────────
        ax = axes[2, col]
        ax.scatter(colors[keep_inds, 0], gmags[keep_inds, 0],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        if n_col > 0:
            sc = ax.scatter(colors[cidxs, 0], gmags[cidxs, 0],
                            s=15, c=prob_col, cmap=qso_cm, vmin=0, vmax=1,
                            marker='D', alpha=0.8, zorder=3)
            plt.colorbar(sc, ax=ax, label='P(QSO)')
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
        ax.set_xlabel('BP $-$ RP (mag)'); ax.set_ylabel('G (mag)')

    # ── Column 3 (optional): wide-field clean QSO training sample ────────
    if has_train:
        col = 3
        n_train = len(qso_train_radec)
        has_train_probs = qso_train_probs is not None

        # Photometry for CMD
        fin_q = (np.isfinite(qso_train_gmags) & np.isfinite(qso_train_bpmags)
                 & np.isfinite(qso_train_rpmags))
        q_bprp = qso_train_bpmags[fin_q] - qso_train_rpmags[fin_q]
        q_gmag = qso_train_gmags[fin_q]
        q_prob = qso_train_probs[fin_q] if has_train_probs else None
        # CMD zoom limits with a small margin
        if fin_q.sum() > 0:
            cmd_xlo = np.nanpercentile(q_bprp, 1)  - 0.2
            cmd_xhi = np.nanpercentile(q_bprp, 99) + 0.2
            cmd_ylo = np.nanpercentile(q_gmag,  1)  - 0.5
            cmd_yhi = np.nanpercentile(q_gmag, 99)  + 0.5
        else:
            cmd_xlo, cmd_xhi, cmd_ylo, cmd_yhi = -1, 4, 12, 21

        # VPD zoom limits
        fin_v = np.isfinite(y_obs_qso[:, 0]) & np.isfinite(y_obs_qso[:, 1])
        if fin_v.sum() > 0:
            vpd_xlo = np.nanpercentile(y_obs_qso[fin_v, 0], 2) - 0.5
            vpd_xhi = np.nanpercentile(y_obs_qso[fin_v, 0], 98) + 0.5
            vpd_ylo = np.nanpercentile(y_obs_qso[fin_v, 1], 2) - 0.5
            vpd_yhi = np.nanpercentile(y_obs_qso[fin_v, 1], 98) + 0.5
            vpd_xlo = min(vpd_xlo, -1.0); vpd_xhi = max(vpd_xhi, 1.0)
            vpd_ylo = min(vpd_ylo, -1.0); vpd_yhi = max(vpd_yhi, 1.0)
        else:
            vpd_xlo, vpd_xhi, vpd_ylo, vpd_yhi = -3, 3, -3, 3

        prob_label = 'Posterior P(QSO)' if has_train_probs else 'Training QSOs'
        col_label_4 = f'Wide-field QSO training sample  (N={n_train:,})'

        def _scatter_train(ax, x, y, prob_arr):
            if prob_arr is not None:
                sc = ax.scatter(x, y, s=10, c=prob_arr, cmap=qso_cm,
                                vmin=0, vmax=1, marker='D', alpha=0.8, zorder=3,
                                rasterized=True)
                plt.colorbar(sc, ax=ax, label=prob_label)
            else:
                ax.scatter(x, y, s=10, c='darkorange', alpha=0.6,
                           marker='D', zorder=3, rasterized=True)

        # ── Sky ──────────────────────────────────────────────────────────
        ax = axes[0, col]
        ax.scatter(radec_offsets[keep_inds, 0], radec_offsets[keep_inds, 1],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        _tp = qso_train_probs if has_train_probs else None
        _scatter_train(ax, qso_train_radec[:, 0], qso_train_radec[:, 1], _tp)
        ax.text(0.02, 0.98, f'N={n_train:,}', transform=ax.transAxes,
                fontsize=10, va='top', ha='left')
        ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        ax.set_xlabel(r'$\Delta\alpha*$ (deg)'); ax.set_ylabel(r'$\Delta\delta$ (deg)')
        ax.set_title(col_label_4, fontsize=12)

        # ── VPD (zoomed) ─────────────────────────────────────────────────
        ax = axes[1, col]
        ax.scatter(pms[keep_inds, 0], pms[keep_inds, 1],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        _vp = qso_train_probs if has_train_probs else None
        _scatter_train(ax, y_obs_qso[:, 0], y_obs_qso[:, 1], _vp)
        ax.set_xlim(vpd_xlo, vpd_xhi); ax.set_ylim(vpd_ylo, vpd_yhi)
        ax.axhline(0, c='k', lw=0.5, ls='--'); ax.axvline(0, c='k', lw=0.5, ls='--')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f'{pm_labels[0]} (mas/yr)'); ax.set_ylabel(f'{pm_labels[1]} (mas/yr)')

        # ── CMD (zoomed) ─────────────────────────────────────────────────
        ax = axes[2, col]
        ax.scatter(colors[keep_inds, 0], gmags[keep_inds, 0],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        if fin_q.sum() > 0:
            _scatter_train(ax, q_bprp, q_gmag, q_prob)
        ax.set_xlim(cmd_xlo, cmd_xhi)
        ax.set_ylim(cmd_yhi, cmd_ylo)   # inverted: brighter at top
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('BP $-$ RP (mag)'); ax.set_ylabel('G (mag)')

    # Annotate model-vs-catalog agreement
    model_and_mq = model_qso & is_milliquas[keep_inds]
    model_not_mq = model_qso & ~is_milliquas[keep_inds] & ~is_gaia_only[keep_inds]
    mq_not_model = is_milliquas[keep_inds] & ~model_qso
    fig.suptitle(
        f'QSO membership diagnostics\n'
        f'Model∩MILLIQUAS: {model_and_mq.sum():,}  |  '
        f'New model QSOs (neither catalog): {model_not_mq.sum():,}  |  '
        f'MILLIQUAS missed by model: {mq_not_model.sum():,}',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    fname = 'qso_membership_diagnostics.png'
    plt.savefig(os.path.join(result_path, fname), dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ---------------------------------------------------------------------------
# CMD prior diagnostic plots
# ---------------------------------------------------------------------------

def plot_cmd_prior_diagnostics(gmags, rpmags, bpmags,
                                keep_member, keep_background,
                                prior_log_probs, color_profiles,
                                result_path, tag=''):
    """
    Six-panel CMD diagnostic for the photometric membership prior.

    Top row  – G vs G-RP colour
    Bottom row – G vs G-BP colour

    Columns: (1) member training stars with binned profiles, (2) background
    training stars, (3) all stars coloured by prior membership probability.
    """
    suffix = f'_{tag}' if tag else ''
    prob   = np.exp(np.clip(prior_log_probs, -20, 0))

    def _profile_lines(ax, x, branch, label, color, lw=1.5):
        ok = np.isfinite(branch[:, 0])
        if ok.sum() < 2:
            return
        ax.plot(branch[ok, 0], x[ok], color=color, lw=lw, label=label)
        ax.fill_betweenx(x[ok],
                         branch[ok, 0] - branch[ok, 1],
                         branch[ok, 0] + branch[ok, 1],
                         color=color, alpha=0.20)

    cp  = color_profiles
    x   = cp.get('x', np.array([]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for row, (col_arr, col_label, prof_red, prof_all, prof_blue) in enumerate([
        (rpmags, 'G $-$ RP (mag)', cp.get('gr_red'), cp.get('gr_green'), cp.get('gr_blue')),
        (bpmags, 'G $-$ BP (mag)', cp.get('gb_red'), cp.get('gb_green'), cp.get('gb_blue')),
    ]):
        valid_col = np.isfinite(col_arr[:, 0]) & np.isfinite(gmags[:, 0])

        # ── Column 0 : member training set ──
        ax = axes[row, 0]
        m_sel = keep_member & valid_col
        if m_sel.sum() > 0:
            ax.hexbin(col_arr[m_sel, 0], gmags[m_sel, 0],
                      gridsize=40, cmap='Blues', mincnt=1)
        if x.size > 0 and prof_red is not None:
            _profile_lines(ax, x, prof_red,  'Red branch',  'C3')
            _profile_lines(ax, x, prof_all,  'Median',      'C2')
            _profile_lines(ax, x, prof_blue, 'Blue branch', 'C0')
            ax.legend(fontsize=10, loc='best')
        ax.invert_yaxis()
        ax.set_xlabel(col_label, fontsize=14)
        ax.set_ylabel('G (mag)', fontsize=14)
        ax.set_title('Member training stars', fontsize=13)
        ax.grid(True, alpha=0.3)

        # ── Column 1 : background training set ──
        ax = axes[row, 1]
        b_sel = keep_background & valid_col
        if b_sel.sum() > 0:
            ax.hexbin(col_arr[b_sel, 0], gmags[b_sel, 0],
                      gridsize=40, cmap='Oranges', mincnt=1)
        if x.size > 0 and prof_red is not None:
            _profile_lines(ax, x, prof_all, 'Member median', 'C2')
        ax.invert_yaxis()
        ax.set_xlabel(col_label, fontsize=14)
        ax.set_ylabel('G (mag)', fontsize=14)
        ax.set_title('Background training stars', fontsize=13)
        ax.grid(True, alpha=0.3)

        # ── Column 2 : prior probability map ──
        ax = axes[row, 2]
        q_sel = valid_col
        sc = ax.scatter(col_arr[q_sel, 0], gmags[q_sel, 0],
                        s=1, c=prob[q_sel], vmin=0, vmax=1,
                        cmap='RdYlGn', rasterized=True)
        plt.colorbar(sc, ax=ax, label='Prior P(member)')
        ax.invert_yaxis()
        ax.set_xlabel(col_label, fontsize=14)
        ax.set_ylabel('G (mag)', fontsize=14)
        ax.set_title('Prior membership probability', fontsize=13)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'CMD photometric prior{" (" + tag + ")" if tag else ""}', fontsize=15)
    plt.tight_layout()
    fname = f'cmd_prior_diagnostics{suffix}.png'
    plt.savefig(os.path.join(result_path, fname), dpi=100)
    plt.close()
    print(f'  Saved {fname}')


def plot_qso_cmd_prior_diagnostics(gmags, rpmags, bpmags,
                                    log_qso_photo_all,
                                    log_prior_qso_src,
                                    qso_gmags, qso_rpmags, qso_bpmags,
                                    result_path):
    """
    Six-panel CMD diagnostic for the QSO photometric prior.

    Top row – G vs BP-RP; bottom row – G vs RP (each dimension of the 3-D KDE).
    Columns: (1) QSO training sample CMD, (2) sources by photometric prior only,
             (3) sources by combined (photo + catalog) prior P(QSO).
    """
    photo_prob   = np.exp(np.clip(log_qso_photo_all,                     -50, 0))
    combined_raw = log_qso_photo_all + log_prior_qso_src
    combined_raw -= np.nanmax(combined_raw)
    combined_prob = np.exp(np.clip(combined_raw, -50, 0))
    combined_prob = combined_prob / np.nanmax(combined_prob)

    color_bprp  = bpmags[:, 0] - rpmags[:, 0]
    valid_bprp  = np.isfinite(color_bprp) & np.isfinite(gmags[:, 0])
    valid_rp    = np.isfinite(rpmags[:, 0]) & np.isfinite(gmags[:, 0])

    finite_qso  = np.isfinite(qso_gmags) & np.isfinite(qso_bpmags) & np.isfinite(qso_rpmags)
    q_bprp      = qso_bpmags[finite_qso] - qso_rpmags[finite_qso]
    q_g         = qso_gmags[finite_qso]
    q_rp        = qso_rpmags[finite_qso]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    rows = [
        (color_bprp, 'BP $-$ RP (mag)', valid_bprp, q_bprp),
        (rpmags[:, 0], 'RP (mag)',       valid_rp,   q_rp),
    ]

    col_titles = [
        f'QSO training sample  (N={finite_qso.sum():,})',
        'Photometric prior  P(G, BP, RP | QSO)',
        'Combined prior  P(QSO | photo + catalog)',
    ]

    for row_idx, (x_arr, x_label, valid_mask, q_x) in enumerate(rows):
        # ── Column 0: QSO training CMD ────────────────────────────────────
        ax = axes[row_idx, 0]
        ax.scatter(x_arr[valid_mask], gmags[valid_mask, 0],
                   s=1, c='lightgrey', rasterized=True, zorder=1)
        if q_x.size > 0:
            ax.scatter(q_x, q_g, s=8, c='darkorange', alpha=0.6, zorder=5,
                       rasterized=True)
        ax.invert_yaxis()
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel('G (mag)', fontsize=13)
        ax.set_title(col_titles[0], fontsize=12)
        ax.grid(True, alpha=0.3)

        # ── Column 1: photometric prior ────────────────────────────────────
        ax = axes[row_idx, 1]
        sc = ax.scatter(x_arr[valid_mask], gmags[valid_mask, 0],
                        s=1, c=photo_prob[valid_mask], vmin=0, vmax=1,
                        cmap='RdYlGn', rasterized=True)
        plt.colorbar(sc, ax=ax, label='P(G, BP, RP | QSO)  [normalised]')
        ax.invert_yaxis()
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel('G (mag)', fontsize=13)
        ax.set_title(col_titles[1], fontsize=12)
        ax.grid(True, alpha=0.3)

        # ── Column 2: combined prior ───────────────────────────────────────
        ax = axes[row_idx, 2]
        sc = ax.scatter(x_arr[valid_mask], gmags[valid_mask, 0],
                        s=1, c=combined_prob[valid_mask], vmin=0, vmax=1,
                        cmap='RdYlGn', rasterized=True)
        plt.colorbar(sc, ax=ax, label='P(QSO | photo + catalog)  [normalised]')
        ax.invert_yaxis()
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel('G (mag)', fontsize=13)
        ax.set_title(col_titles[2], fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('QSO photometric prior diagnostics', fontsize=15)
    plt.tight_layout()
    fname = 'qso_cmd_prior_diagnostics.png'
    plt.savefig(os.path.join(result_path, fname), dpi=100)
    plt.close()
    print(f'  Saved {fname}')
