# pop_fit

A modular command-line pipeline for measuring proper motions and stellar membership probabilities in Local Group dwarf galaxies using Gaia DR3 astrometry. Stellar populations are separated using a Gaussian Mixture Model (GMM) fit in proper-motion + parallax space, combined with spatial and photometric priors. The model is implemented in [PyMC](https://www.pymc.io/) and sampled with [numpyro](https://num.pyro.ai/).

Optional extensions support incorporating HST astrometry from the [BP3M pipeline](https://github.com/KevinMcK95/bp3m_improved) and a QSO-based proper-motion zero-point correction.

---

## Features

- Automated Gaia DR3 cone-search download and caching
- Plummer and Sérsic spatial density profiles
- 3-D photometric prior in (G, BP, RP) space via error-weighted KDE
- Flexible MW background model: K Gaussian components pre-fit by scikit-learn
- HST/BP3M astrometry integration for faint stars (G ≈ 20.7–21.5)
  - v1: marginal covariances
  - v2 (z-latent): image-transformation modes as shared MCMC variables
- QSO proper-motion zero-point self-calibration (`--qso-correction`)
- Galactic coordinate frame fitting (`--galactic-coords`)
- ArviZ diagnostic plots and posterior summary CSV

---

## Installation

The pipeline runs in a `conda` environment. The recommended setup uses `conda` with `mamba` for speed:

```bash
conda create -n pymc_new -c conda-forge \
    "pymc>=5" "numpyro>=0.15" "arviz=0.23.4" nutpie \
    astropy astroquery scikit-learn "matplotlib>=3.9" \
    pandas scipy h5netcdf
conda activate pymc_new
```

Key version requirements:
- **`arviz=0.23.4`**: pinned because the diagnostic plot functions `plot_posterior` and `plot_dist_comparison` were removed in arviz 1.x. The code falls back to arviz_plots equivalents if a newer version is installed, but the 0.23.x output looks significantly better.
- **`nutpie`**: Rust-based NUTS backend. Strongly recommended on multi-core CPU servers — gives ~15× speedup over the default numpyro backend by using native threads instead of JAX virtual devices. Use with `--sampler nutpie`.
- **`h5netcdf`**: required to save the MCMC trace to NetCDF4 format (which supports multiple groups). Without it the trace cannot be written to disk.

### Shell environment (Linux/server)

Add the following to `~/.bashrc` (or equivalent) on any Linux server:

```bash
# Limit MKL/OpenBLAS thread count for NumPy operations outside JAX.
# Without this, MKL defaults to using all available cores per BLAS call,
# which causes severe overhead for the small matrix operations in NUTS.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

These variables affect NumPy/SciPy operations (background GMM fitting, membership
post-processing, etc.) but not JAX's internal threading. For JAX, use `--sampler nutpie`
instead — nutpie's Rayon thread pool is controlled by `--sampler-threads` (default
`min(8, cpu_count)`, so 32 cores total with 4 chains).

Do **not** set `XLA_FLAGS` or `POP_FIT_N_DEVICES` manually — the code sets these
automatically.

Then clone this repository:

```bash
git clone https://github.com/KevinMcK95/pop_fit.git
cd pop_fit
```

---

## Required data

### 1. Local Volume Database (LVD) catalog — **required**

The pipeline reads structural parameters (half-light radius, ellipticity, position angle, bulk proper motion) from the [Local Volume Database](https://github.com/apace7/local_volume_database).

Download the combined dwarf catalog:

```bash
# Option A: clone the LVD repo and use the combined CSV
git clone https://github.com/apace7/local_volume_database.git
# The file you need is: local_volume_database/data/dwarf_all.csv

# Option B: download the CSV directly from the LVD data releases page
```

Then update the path in `config.py`:

```python
LVD_CATALOG_PATH = '/path/to/dwarf_all.csv'
```

### 2. Gaia DR3 — **auto-downloaded**

Gaia data are fetched automatically via the [astroquery](https://astroquery.readthedocs.io/) TAP interface on first run and cached to `./gal_fitting_results/<field>/gaia_dr3_<N>arcmin.csv.gz`. No manual download is needed. Use `--redownload` to refresh the cache.

### 3. MILLIQUAS catalog — **required for `--qso-correction` only**

The QSO zero-point correction uses the [Milliquas](https://www.quasars.org/milliquas.htm) (Million Quasars) catalog to identify confirmed QSOs in the field.

1. Download `milliquas.fits` from https://www.quasars.org/milliquas.htm (the HDF5/FITS version)
2. Place it in the `pop_fit/` project directory (same folder as `fit.py`)

The expected path is configured in `config.py`:

```python
MILLIQUAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'milliquas.fits')
```

### 4. BP3M HST astrometry — **required for `--bp3m-dir` only**

HST proper-motion outputs from the [BP3M pipeline](https://github.com/KevinMcK95/bp3m_improved). Pass the directory containing `stellar_astrometry.csv` via `--bp3m-dir`. For z-latent mode (v2), the directory must also contain `K_matrices.npz`, `star_indices.npz`, `use_for_fit.npz`, `C_r.npy`, `C_vT.npy`, and `image_transformations.csv`.

---

## Quick start

```bash
# Basic run with default settings
conda run -n pymc_new python fit.py Fornax_dSph

# Fast smoke-test (minimal MCMC)
conda run -n pymc_new python fit.py Fornax_dSph \
    --draws 50 --tune 50 --chains 2 --n-member-samples 100

# Sérsic profile instead of Plummer
conda run -n pymc_new python fit.py Leo_I --spatial-profile sersic

# QSO proper-motion zero-point correction
conda run -n pymc_new python fit.py Sculptor_dSph --qso-correction

# Fit in Galactic proper-motion frame
conda run -n pymc_new python fit.py NGC_55 --galactic-coords

# With HST/BP3M astrometry (v1)
conda run -n pymc_new python fit.py Leo_I \
    --bp3m-dir ./bp3m_data/Leo_I/BP3M_results/

# With HST/BP3M astrometry (v2 z-latent)
conda run -n pymc_new python fit.py Leo_I \
    --bp3m-dir ./bp3m_data/Leo_I/BP3M_results/ --bp3m-latent

# Stop after a specific pipeline step (useful for debugging)
conda run -n pymc_new python fit.py NGC_55 --stop-after 5
```

All outputs (plots and trace NetCDF) are written to `./gal_fitting_results/<field>/`.

---

## Supported fields

The following fields are pre-configured in `config.py`:

| Field name | LVD key |
|------------|---------|
| `Fornax_dSph` | fornax_1 |
| `Sculptor_dSph` | sculptor_1 |
| `Draco_dSph` | draco_1 |
| `Leo_I` | leo_1 |
| `Leo_II` | leo_2 |
| `NGC_55` | ngc_0055 |
| `NGC_300` | ngc_0300 |
| `NGC_3109` | ngc_3109 |
| `NGC_6822` | ngc_6822 |
| `Sagittarius_dSph` | sagittarius_1 |
| `Sextans_A` | sextans_a |
| `WLM` | wlm |
| `IC1613` | ic_1613 |
| `Antlia_II` | antlia_2 |
| `Crater_II` | crater_2 |
| `Hydrus_I` | hydrus_1 |
| … and more | see `config.py` |

To add a new field, see the **Adding a new field** section in `CLAUDE.md`.

---

## Key output files

| File | Description |
|------|-------------|
| `{field}_trace.nc` | ArviZ NetCDF trace (posterior + prior samples) |
| `posterior_parameter_summaries.csv` | Median and 68%/95% CI for all free parameters |
| `final_population_summary.png` | VPD + CMD + spatial map coloured by P(member) |
| `run_metadata.json` | Field centre, coordinate frame, timestamp |
| `mcmc_summaries/` | Corner, trace, and posterior plots per parameter group |
| `bp3m_group_b_membership.csv` | Per-star membership for HST-only (G > 20.7) stars |

---

## Module overview

| File | Purpose |
|------|---------|
| `fit.py` | CLI entry point; orchestrates the 9-step pipeline |
| `config.py` | Paths, field metadata, Gaia systematic error budget |
| `data.py` | Gaia download, kinematic computation, BP3M data loading |
| `models.py` | PyMC model definitions (spatial + GMM) |
| `membership.py` | Per-star P(dwarf) from posterior samples |
| `photometry.py` | Photometric prior via KDE or binned colour profiles |
| `plots.py` | All diagnostic and summary figures |
| `utils.py` | Math utilities (elliptical radius, Cholesky helpers) |

For full documentation of the statistical model, pipeline steps, and all CLI flags, see [`CLAUDE.md`](CLAUDE.md).

---

## Citation

If you use this code, please cite the Gaia DR3 catalogue and Vasiliev & Baumgardt (2021) for the adopted systematic error floor:

> Vasiliev, E. & Baumgardt, H. (2021), MNRAS, 505, 5978 — Gaia EDR3 proper motions of Milky Way satellite galaxies
