# pop_fit – Gaia GMM membership fitting tool

Translates `orig_code/PM_pop_fit_with_density_test.ipynb` into a modular CLI
pipeline.  Run with the `pymc_new` conda environment.

**Required ArviZ version: `arviz=0.23.4`.**  The diagnostic plot functions `plot_posterior` and `plot_dist_comparison` were removed in arviz 1.x.  The code falls back to `arviz_plots` equivalents automatically on newer installs, but the 0.23.4 output looks significantly better.  Install with:
```bash
conda install -n pymc_new -c conda-forge arviz=0.23.4
```

## Quick start

```bash
# Full run (default MCMC settings)
conda run -n pymc_new python fit.py NGC_55

# Stop after a specific step (useful for debugging)
conda run -n pymc_new python fit.py NGC_55 --stop-after 6

# Use the 2-D KDE photometric prior instead of the binned-profile approach
conda run -n pymc_new python fit.py NGC_55 --use-kde-prior

# Minimal MCMC steps for a fast smoke-test of the whole pipeline
conda run -n pymc_new python fit.py NGC_55 \
    --draws 50 --tune 50 --chains 2 \
    --spatial-draws 100 --spatial-tune 50 \
    --n-member-samples 100

# Re-run plots and membership from a saved trace (skips step 7 GMM only;
# steps 5–6 still re-run to get the correct refined spatial model and prior)
conda run -n pymc_new python fit.py NGC_55 --from-trace
conda run -n pymc_new python fit.py Leo_I --bp3m-dir ./data/ --from-trace

# Run with BP3M HST astrometry (Group A substitution + Group B faint stars)
conda run -n pymc_new python fit.py Leo_I \
    --bp3m-dir ./bp3m_data/Leo_I/BP3M_results/

# Z-latent mode (v2): embed image-transformation uncertainty as shared MCMC variables
# Requires current run_bp3m.py outputs (K_matrices.npz + *_cond columns).
# Falls back to v1 automatically if K_matrices.npz is absent.
conda run -n pymc_new python fit.py Leo_I \
    --bp3m-dir ./bp3m_data/Leo_I/BP3M_results/ \
    --bp3m-latent
```

All outputs (plots + trace NetCDF) go to `./gal_fitting_results/<field>/`.

## Module layout

| File | Purpose |
|------|---------|
| `config.py` | Hardcoded paths, field metadata tables (`LVD_TRANSLATION`, `GMAG_LIMITS`, `PROP_DICT`, `FIELD_SPATIAL_CUTOFFS`) |
| `utils.py` | Pure math: `compute_elliptical_r`, `batch_log_pdf`, `logdiffexp`, Cholesky pack/unpack helpers |
| `data.py` | `load_lvd_catalog`, `load_gaia_data`, `load_field_priors`, `compute_kinematics`, `compute_background_stats`, `fit_background_gmm`, `build_qso_obs_arrays`, `compute_qso_surface_density`, `load_bp3m_data` |
| `photometry.py` | `med_mags` (sigma-clipped binned colour profiles), `compute_photometric_prior` (profile-based), `compute_photometric_prior_kde` (3-D error-weighted KDE in G, BP, RP space), `compute_qso_photometric_prior` (QSO photo prior, 3-D KDE), `build_log_prior_weights` |
| `models.py` | `build_spatial_model` / `run_spatial_model`, `build_gmm_model` / `run_gmm_model` (dwarf + K fixed background components; 3-class with QSOs when `log_prior_qso` is passed) |
| `membership.py` | `compute_membership_probs` – loops over posterior draws, returns per-star P(dwarf); 3-tuple (2-class) or 4-tuple (3-class) |
| `plots.py` | All saved PNGs; uses `matplotlib.use('Agg')` (non-interactive) |
| `fit.py` | CLI entry point; orchestrates the 9-step pipeline |

## Pipeline steps

| Step | Description | Key outputs |
|------|-------------|-------------|
| 1 | Load LVD catalog + Gaia CSV; load BP3M `stellar_astrometry.csv` if `--bp3m-dir` given | – |
| 2 | Initial star selection (`keep` mask, `r_ell`) | – |
| 3 | Initial photometric membership prior | `cmd_prior_diagnostics_initial.png` |
| 4 | MW background statistics + pre-fit sklearn GMM (K components) | – |
| 5 | Spatial PyMC model (centre, Plummer scale, ellipticity, PA) | `density_profile_update.png` |
| 6 | Recompute morphology + updated photometric prior | `cmd_prior_diagnostics_updated.png` |
| 7 | GMM (dwarf + K pre-fit background components; 3-class with QSOs if `--qso-correction`); BP3M Group A y_obs/S_obs substituted in-place; Group B faint stars added as separate population | `{field}_trace.nc` |
| 8 | ArviZ diagnostic plots | `mcmc_summaries/{field}_{corner,trace,posterior,prior_comp}_{component}.png` |
| 9 | Per-star membership probabilities + final plots | `initial_population_summary.png`, `final_population_summary.png`, `qso_membership_diagnostics.png` (3-class only), `posterior_parameter_summaries.csv`, `run_metadata.json`, `photometry_kde_training.npz` |

Use `--stop-after N` to halt after step N (1–9).

`--from-trace` skips only step 7 (GMM MCMC). Steps 1–6 always re-run so that membership probabilities in steps 8–9 use the same refined spatial model and photometric prior that were used in the original run, not the initial step-4 estimates.

## Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--draws` | 2000 | GMM posterior draws |
| `--tune` | 2000 | GMM tuning steps |
| `--chains` | 4 | MCMC chains |
| `--spatial-draws` | 1000 | Spatial model draws |
| `--spatial-tune` | 100 | Spatial model tuning steps |
| `--stop-after` | 9 | Stop after step N |
| `--from-trace` | off | Skip step 7 (GMM MCMC) and load the saved `{field}_trace.nc` instead. Steps 1–6 still re-run. Restores `pm_labels` and `spatial_profile` from `run_metadata.json`. Backward compatible: traces saved before `keep_inds` was added to `constant_data` are handled via cKDTree nearest-neighbour reconstruction. |
| `--binned-prior` | off | Use binned colour profiles instead of 3-D KDE in (G, BP, RP) space (KDE is default) |
| `--spatial-profile` | plummer | `plummer` or `sersic` (Sérsic n=1) |
| `--seed` | 42 | Random seed |
| `--n-member-samples` | 1000 | Posterior draws for step-9 membership |
| `--membership-threshold` | 0.95 | P(dwarf) cut for "member" label |
| `--n-stars-max` | 100000 | Stars cap passed to GMM |
| `--bg-components` | 5 | sklearn GMM components for MW background pre-fit |
| `--n-init-density-fit` | 1 | Iterations of initial spatial density-profile fit; N>1 re-centers priors and refits background GMM between iterations |
| `--search-radius` | auto | Gaia cone-search radius in degrees (default: 8×rhalf, clamped 0.5–2.0 deg) |
| `--redownload` | off | Force re-download of Gaia data ignoring cache |
| `--qso-correction` | off | Enable the 3-class GMM: dwarf stars, MW background stars, and QSOs/compact extragalactic objects. Adds `f_star ~ Beta` (P(source is a star)) as a free MCMC parameter alongside `f_dwarf`. Per-source soft priors: MILLIQUAS-confirmed → log(0.99), Gaia-only `in_qso_candidates` → log(0.50), other → log(0.01); combined with a 3-D photometric KDE prior in (G, BP, RP) space. QSO surface density is the geometric mean of MILLIQUAS lower bound and Gaia upper bound. Constrains `delta_pm_sys` via wide-field QSO likelihood. Requires `milliquas.fits`. |
| `--qso-radius` | 5.0 | Search radius in degrees for the QSO candidate query |
| `--galactic-coords` | off | Fit in Galactic PM frame (pm_l_cosb, pm_b, parallax) instead of ICRS (pmra*, pmdec, parallax). Rotation is per-star and exact; Mahalanobis selection distances are preserved. |
| `--bp3m-dir` | None | Path to BP3M output directory containing `stellar_astrometry.csv`. Enables HST astrometry integration (Group A substitution + Group B faint stars). |
| `--bp3m-latent` | off | Enable z-latent image-transformation model (v2). Requires `K_matrices.npz` and `*_cond` columns from current `run_bp3m.py`. Adds `z_latent ~ Normal(0,I,M)` shared MCMC variables; uses conditional covariances C_vT instead of marginal C_obs. Falls back to v1 silently if `K_matrices.npz` is absent. |
| `--bp3m-latent-sigma-eff` | 0.015 | Mode selection threshold for `--bp3m-latent` (mas/yr). Only latent modes with RMS PM sensitivity > threshold are kept. Lower values → more modes → closer to full marginalised covariance reconstruction. |
| `--bp3m-a-chi2-threshold` | 16.0 | Chi-squared threshold for Group A BP3M–Gaia PM consistency pre-filter. Stars where `χ²=(y_bp3m−y_gaia)ᵀ C_gaia⁻¹ (y_bp3m−y_gaia) > threshold` revert to Gaia-only observations. Uses only Gaia-valid dimensions (positive finite diagonal); stars with no Gaia PM are never filtered. Set to `inf` to disable. |

## Data paths

- **LVD catalog**: `/Users/kevinm/Downloads/dwarf_all.csv` (hardcoded in `config.py`)
- **Gaia cache**: `./gal_fitting_results/{field}/gaia_dr3_{N}arcmin.csv.gz` (auto-downloaded; falls back to legacy GaiaHub CSV in `GAIA_DATA_PATH/{field}/Gaia/` if download fails)
- **QSO cache**: `./gal_fitting_results/{field}/gaia_dr3_qso_{R}deg.csv.gz` (downloaded when `--qso-correction` is passed; respects `--redownload`)
- **BP3M output**: `{bp3m_dir}/stellar_astrometry.csv` (path given via `--bp3m-dir`; e.g. `./bp3m_data/Leo_I/BP3M_results/`)

## Adding a new field

1. Add a `LVD_TRANSLATION` entry mapping the CLI field name to the LVD catalog key.
2. Add a `GMAG_LIMITS` entry (`None` = no bright cut).
3. If the field centre is not resolvable from the LVD catalog's RA/Dec columns,
   add a `PROP_DICT` entry with `ra`, `dec`, `search_radius`.
4. If the field needs a hard spatial pre-cut (e.g. very extended), add to
   `FIELD_SPATIAL_CUTOFFS`.

## Spatial density profiles

| Profile | CLI flag | Formula | Scale parameter |
|---------|----------|---------|----------------|
| Plummer (default) | `--spatial-profile plummer` | Σ ∝ (1 + r²/a²)⁻² | a = Plummer scale radius |
| Sérsic n=1 | `--spatial-profile sersic` | Σ ∝ exp(−r/a) | a = exponential scale length h (half-light r_e = 1.678 × h) |

Both have the same number of free parameters.  Sérsic falls off more slowly at large radii.

**Note on Sérsic parameterization:** `a_plummer` is the scale length h (e-folding radius), not the half-light radius (r_e = 1.678 × h).  Both `build_spatial_model` and `build_gmm_model` correctly centre the `a_plummer` prior at `rhalf_mean / 1.678` when `--spatial-profile sersic` is used.  The formula in `membership.py` exactly mirrors `models.py`.

## Photometric prior: two approaches

### Profile-based (default)
`compute_photometric_prior()` bins stars by G magnitude and fits median
G-RP and G-BP colour profiles with sigma-clipping.  Red, all, and blue branch
profiles are compared to each query star's colour via a Gaussian likelihood.

### 3-D KDE in (G, BP, RP) space (`--use-kde-prior`)
`compute_photometric_prior_kde()` places a 3-D Gaussian kernel in (G, BP, RP)
space for each training star, with per-dimension bandwidth
`sqrt(σ_train² + σ_query² + min_bw²)`.  Using absolute magnitudes rather than
colours avoids double-counting G and spurious correlations between photometric
systems.  No magnitude binning; naturally error-aware.  Slower for large
training sets (batched at 500 query stars; training capped at 3000 stars by
default).  `compute_qso_photometric_prior` uses the same 3-D KDE with fixed
0.02 mag training errors (dominated by the min_bw floor).

## GMM model structure

### Free MCMC parameters

| Parameter | Prior | Purpose |
|-----------|-------|---------|
| `mu_dwarf` | Normal (LVD-anchored) | Dwarf galaxy mean (μ_α*, μ_δ, ϖ) |
| `chol_intrinsic_dwarf` | LKJCholeskyCov | Dwarf intrinsic dispersion |
| `k` | LogNormal(log 1.2, 0.5) | Mean error inflation factor |
| `k_1` | Normal(0, 0.2) | Log-slope of error inflation vs G: k(G) = k₀ · exp(k₁·(G−G_ref)) |
| `chol_floor` | LKJCholeskyCov | Systematic floor covariance |
| `f_star` | Beta(α, β) | P(source is a star vs QSO); present only with `--qso-correction`. α/β derived from `f_qso_prior` estimate with concentration 5. |
| `delta_pm_sys` | Normal(0, 0.1), shape=(2,) | Field-level PM zero-point (μ_α*, μ_δ); present only with `--qso-correction`. Constrained by the QSO likelihood; applied to dwarf observations only. Background likelihoods use raw `y_obs` to decouple from this parameter. |
| `k_hst` | LogNormal(log 1.0, 0.3) | Global covariance scaling for BP3M measurements; present only with `--bp3m-dir`. |
| `chol_floor_hst` | LKJCholeskyCov (3×3) | Full 3×3 systematic floor covariance for HST/BP3M observations; present only with `--bp3m-dir`. Analogous to `chol_floor` for Gaia. |
| `delta_pm_sys_hst` | Normal(0, 0.5), shape=(2,) | Field-level PM zero-point for BP3M observations (μ_α*, μ_δ); present only with `--bp3m-dir`. Separate from the Gaia `delta_pm_sys` (QSO) parameter. Wide prior (0.5 mas/yr) allows self-calibration of large HST systematic offsets. |
| `z_latent` | Normal(0, 1), shape=(M,) | Image-transformation latent modes; present only with `--bp3m-latent`. M ≈ 20–50 modes (controlled by `--bp3m-latent-sigma-eff`). Shared across all BP3M stars; each star's effective PM = cond_mean + S_i @ z. Enables self-calibration of alignment and eliminates inflated `sigma_floor_hst`. |
| `delta_ra_center`, `delta_dec_center` | Normal (spatial posterior) | Dwarf centre offset |
| `a_plummer` | TruncatedNormal | Plummer/Sérsic scale radius (prior centred on `rhalf_mean` for Plummer; `rhalf_mean/1.678` for Sérsic) |
| `ellipticity` | TruncatedNormal | Axis ratio (0–0.99) |
| `dpa_rad` | VonMises(μ=0, κ=1/(2σ_PA)²) | PA deviation in doubled-angle space; `pa_deg` = `pa_prior%180 + 0.5·dpa_rad·(180/π)` is a Deterministic. The double-angle trick handles 180° periodicity of PA. |

### Fixed background components (no MCMC cost)

K Gaussian components pre-fit by scikit-learn `GaussianMixture` to background stars
in step 4.  Their means, covariances, and mixing weights are numpy constants inside
the PyMC model — only the error-model parameters (`k`, `k_1`, `chol_floor`) couple
them to the posterior.  Stored in `constant_data` in the trace NetCDF.

This removes ~18 MCMC free parameters vs the old disk+halo model while allowing
a richer background description (K=5 by default, tunable with `--bg-components`).

PyMC sampler: `nuts_sampler="numpyro"`, `target_accept=0.95`.

**f_dwarf and f_star prior concentration:** Both spatial and GMM models use `concentration = 5.0` (α = f₀·5, β = (1−f₀)·5), i.e., equivalent to 5 pseudo-observations.  This is intentionally weak so the posterior can move freely; earlier higher concentrations (10/20) were found to bias results.

### 3-class model (--qso-correction)

When `--qso-correction` is active, the GMM uses a two-Beta parameterisation:
- `f_star = P(source is a star)` — free MCMC parameter
- `f_dwarf = P(dwarf | star)` — free MCMC parameter (same as 2-class)
- P(dwarf) = f_star · f_dwarf
- P(MW background) = f_star · (1 − f_dwarf)
- P(QSO) = 1 − f_star

The QSO kinematic component uses zero mean proper motion and zero intrinsic dispersion (measurement errors + floor covariance only), evaluated in the shifted frame (`y_obs_eff`). The per-source `log_prior_qso` is the sum of a photometric KDE term (`compute_qso_photometric_prior`) and a catalog classification prior (log 0.99 / 0.50 / 0.01 for MILLIQUAS / Gaia-only / neither).

The `log_norm_cmd` term in the likelihood sums all three components so that `f_star` and `f_dwarf` are identified by kinematic and spatial evidence alone, not driven by CMD selection.

## Output files

All outputs go to `./gal_fitting_results/{field}/`.

| File | Description |
|------|-------------|
| `{field}_trace.nc` | ArviZ NetCDF trace with posterior, prior, constant_data |
| `mcmc_summaries/` | ArviZ diagnostic plots (corner, trace, posterior, prior_comp); spatial diagnostic plots |
| `posterior_parameter_summaries.csv` | Prior and posterior median + 68%/95% CI for all GMM free parameters; `ra_center`/`dec_center` are absolute coordinates (not deltas) |
| `run_metadata.json` | `G_ref`, `pm_labels`, catalog centre RA/Dec, UTC timestamp |
| `photometry_kde_training.npz` | G/BP/RP training mags + errors for member and background KDE sets |
| `cmd_prior_diagnostics_{initial,updated}.png` | Photometric prior diagnostics |
| `density_profile_update.png` | Spatial model density profile |
| `initial_population_summary.png` | Summary using initial prior-based membership |
| `final_population_summary.png` | Summary using full GMM posterior membership |
| `qso_membership_diagnostics.png` | 3-class QSO membership diagnostics (3-class only) |
| `gaia_dr3_{N}arcmin.csv.gz` | Cached Gaia cone-search result |
| `gaia_dr3_qso_{R}deg.csv.gz` | Cached QSO candidate cone-search (3-class only) |
| `bp3m_group_b_membership.csv` | Per-star membership probabilities for Group B (G>20.7 BP3M-only stars); present only with `--bp3m-dir` |

## Performance notes

- **Membership loop (step 9)**: vectorised over batches of 50 draws at a time using numpy einsum — ~50× faster than the previous per-draw Python loop.
- **Background precomputation**: the K fixed background Gaussian components use raw `y_obs` (never shifted by `delta_pm_sys`), so their log-PDFs are constant across all posterior draws. `membership.py` precomputes these once before the batch loop, eliminating K × n_batches redundant matrix inversions (e.g. 5 × 20 = 100 → 5 total).
- **KDE prior**: training stars are pre-sorted by G; each query batch only sums kernels from training stars within 3 bandwidths in magnitude, reducing O(N_train) to O(N_local).
- **JAX compilation cache**: enabled automatically at `~/.cache/pop_fit_jax`; second runs of the same model compile near-instantly.

## Gaia systematics (Vasiliev & Baumgardt 2021)

- Parallax systematic: 0.011 mas
- PM systematic: 0.026 mas/yr
- Mean error inflation factor: 1.20×

## Coordinate handling

- **RA wrapping** (`data.py:compute_kinematics`): RA offsets are wrapped to [−180°, +180°] before multiplying by cos(dec). This prevents spurious large distances for fields near RA = 0/360° (e.g. NGC_55 at RA ≈ 3.7°).
- **PA wrapping** (`models.py`): PA is 180°-periodic (0° and 180° are the same orientation). Both `build_spatial_model` and `build_gmm_model` parameterise PA using the double-angle trick: `dpa_rad ~ VonMises(0, κ)` in the doubled-angle space, then `pa_deg = pa_prior%180 + 0.5·dpa_rad·(180/π)`. This avoids bimodality at the 0/180° boundary.

## bad_rhalf fields

If the LVD catalog has no valid half-light radius for a field, `bad_rhalf=True`
and the pipeline:
- Uses a spatial fraction of the survey width instead of `r_ell` for the keep mask
- Estimates `rhalf_mean` from confident initial members
- Uses equal-weight prior weights (`log_prior_ws`) instead of morphology-based ones

## Statistical model verification

The following normalizations have been verified to integrate to 1 over the survey footprint:

- **Plummer**: `Σ = (1/(π·a²·q))·(1 + r²/a²)⁻²` — verified analytically ✓
- **Sérsic n=1**: `Σ = (1/(2π·a²·q))·exp(−r/a)` — verified analytically ✓
- **Uniform background**: `1/survey_area` where `survey_area = π·max_dist²` ✓
- **CMD normalization**: `log_norm_cmd` marginalises over the photometric selection so that `f_dwarf`/`f_star` are identified by kinematics and spatial position only, not CMD cuts ✓
- **3-class CMD norm**: sums all three components (dwarf + K background + QSO) ✓
- **KDE ratio**: `log_member_pdf − log_addexp(log_member_pdf, log_background_pdf)` = log P(member | CMD) ✓
- **Cholesky unpacking** (`_TRIL_R/C`): matches `np.tril_indices(3)` ordering from PyMC `LKJCholeskyCov` ✓
- **`models.py` ↔ `membership.py` consistency**: elliptical radius formula, k(G) error model, background decoupling from `delta_pm_sys`, and Plummer/Sérsic surface density all verified identical ✓

**k(G) clip**: both `models.py` and `membership.py` clip the exponent `k₁·(G−G_ref)` to `[−6, 6]` before applying `exp`, preventing overflow during NUTS warmup and posterior evaluation.

## Known limitations / future work

- **Survey area for non-circular fields**: `FIELD_SPATIAL_CUTOFFS` imposes a hard pre-cut on the catalogue, making the effective survey non-circular, but the background normalisation still uses `π·max_dist²`. This slightly overestimates the background fraction for fields with aggressive spatial cuts.
- **`delta_pm_sys` prior width**: currently fixed at `Normal(0, 0.1)` regardless of QSO count. A tighter prior is justified when many QSOs are available; width could be scaled as `0.1 / sqrt(N_qso / N_ref)`.
- **JAX-accelerated membership**: the step-9 membership loop runs in Python/NumPy. Porting to JAX (scan over posterior draws) would reduce wall time from minutes to seconds for large fields.

---

## HST + BP3M incorporation (v1, implemented)

### Overview

`--bp3m-dir` activates HST astrometry from BP3M (Bayesian Positions, Parallaxes, Proper
Motions; `~/Documents/Claude/Projects/bp3m_improved/`). For faint stars (G ≈ 20.7–21.5)
where Gaia measures only positions, the long HST–Gaia time baseline (typically 10–15 yr)
yields ~4.6× better PM precision than Gaia (measured for Leo_I).

**Science targets**: NGC_300, NGC_55, NGC_3109.
**Validation fields**: Draco_dSph, Sculptor_dSph, Leo_I (compare to literature PMs).

### Two star populations

**Group A** (G < 20.7, in both Gaia catalog and BP3M output):
- Matched by `gaia_df['source_id']` ↔ `bp3m_df['Gaia_id']`
- `y_obs[i]` and `S_obs[i]` are substituted in-place at step 7 with BP3M values
- Error model: `k_hst`, `chol_floor_hst`, `delta_pm_sys_hst`
- Spatial model: live GMM parameters (same as Gaia stars)

**Group B** (G > 20.7, BP3M-only — no Gaia PM solution):
- Added as a separate population at step 7 after Group A substitution
- Photometric prior: same KDE/profile used for Gaia stars, evaluated at Group B photometry. The KDE naturally becomes uninformative at G>20.7 where training data thins out, but provides genuine signal for stars on the isochrone sequence. Computed by concatenating Group B magnitudes with the Gaia arrays so training masks align; Group B rows are query-only.
- Spatial prior: frozen step-5 posterior medians (`sp` dict); not updated by GMM
- Background normalisation uses `hst_area` (bounding box of BP3M star positions in deg²)
- Shares `f_dwarf`, `mu_dwarf`, and all HST error model parameters with Group A
- Membership results saved to `bp3m_group_b_membership.csv`

**Star selection rule**: each star's data enters the likelihood exactly once — via either
raw Gaia PM or BP3M output, never both.

**`--galactic-coords` compatibility**: When both `--galactic-coords` and `--bp3m-dir` are
used, the BP3M ICRS PMs and covariances are rotated to (pm_l_cosb, pm_b) immediately after
the Group A substitution and Group B array construction in step 7, using the same per-star
rotation matrices as the Gaia rotation applied earlier.

### `load_bp3m_data(bp3m_dir)`

Reads `{bp3m_dir}/stellar_astrometry.csv`, filters to `n_hst_used > 0` and finite
`pmra_bp3m`, and reconstructs the 3×3 marginal covariance matrix from sigma+correlation
columns. Returns `(bp3m_df, C_pm)` where `C_pm` is shape `(N, 3, 3)`.

CSV columns used (marginal values, not `_cond` suffix):
- PM means: `pmra_bp3m`, `pmdec_bp3m`, `parallax_bp3m`
- Sigmas: `sigma_pmra_bp3m`, `sigma_pmdec_bp3m`, `sigma_parallax_bp3m`
- Correlations: `corr_pmra_pmdec`, `corr_pmra_plx`, `corr_pmdec_plx`

The marginal covariance `C_obs = C_vT + v_cov_marginalised` is encoded in these columns.

### Inter-star correlations

BP3M marginalising over image transformation parameters creates cross-star correlations.
**Quantified for Leo_I** (925 stars, 72 images): median 9.3% (p84: 28.8%) of each star's
variance comes from cross-image terms. Treating stars as independent (v1 approach) is
adequate for membership assignment.

### Error model

`S_eff_i = k_hst² · C_obs_i + chol_floor_hst @ chol_floor_hst^T`

where `C_obs_i` is the 3×3 marginal covariance from the CSV. Applied via PyTensor blending:
`S_total = (1−mask) * S_gaia + mask * S_hst` (avoids conditional control flow in the graph).
Similarly `y_obs_eff` blends per-star between `y_obs − pm_shift_gaia` and `y_obs − pm_shift_hst`.

### Bayesian consistency

BP3M compresses the joint (Gaia + HST) likelihood into a Gaussian posterior. Using that
posterior as input to pop_fit is equivalent to computing the full joint likelihood over
raw data — no double-counting as long as each star enters via exactly one path.

### Z-latent variable approach (v2, implemented)

Activated with `--bp3m-latent`. Requires `K_matrices.npz`, `star_indices.npz`,
`use_for_fit.npz` from current `run_bp3m.py` (plus `*_cond` columns in
`stellar_astrometry.csv`).

**How it works:**
1. `load_bp3m_latent_data` eigendecomposes `C_r` (clamping negative eigenvalues,
   matching `sample_posteriors` fallback in solver.py).
2. For each BP3M star, assembles the sensitivity `A_i[:, j*N_R:(j+1)*N_R] = C_vT_i @ K_ij`
   across all images where the star was used.
3. Computes `S_i = A_i @ V_kept * sqrt(λ_kept)` → `(5, M_pre)` factoring `v_cov_marginalised`.
4. Filters modes by RMS PM sensitivity `σ_eff_pm > σ_threshold`; returns `S_pm = S_i[2:5, :]`.
5. In the model: `z_latent ~ Normal(0, I, M)` shared across all BP3M stars; per-star
   effective PM = `y_obs_cond + S_pm @ z_latent`; covariance uses `C_vT` (conditional),
   not `C_obs` (marginal). The `S_pm @ z_latent` term replaces the inflated `sigma_floor_hst`.
6. Membership: z_latent draws from trace are used to compute per-draw `y_eff` for HST stars.

**Key files:**
- `data.py:load_bp3m_latent_data` — builds S_pm, C_vT_pm, cond_means from BP3M outputs
- `models.py:build_gmm_model` — `S_latent_hst_a` / `S_latent_hst_b` params add `z_latent`
- `membership.py:compute_membership_probs` — `S_latent_hst_a` / `S_latent_hst_b` params

**Image ordering**: Must use `image_transformations.csv` (not `K_matrices.npz.files` key order).
**C_r clamping**: Use `np.maximum(vals, 0)` in eigendecomposition (72 negative eigenvalues
for Leo_I from underdetermined image modes). This matches `sample_posteriors` fallback.
