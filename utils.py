import numpy as np


def compute_elliptical_r(radec_offsets, delta_ra_arcmin, delta_dec_arcmin,
                         pa_deg, ellipticity, scale_deg):
    """
    Elliptical radius in units of *scale_deg*.

    Parameters
    ----------
    radec_offsets : (N, 2) array, degrees (RA offset already cos-corrected)
    delta_ra_arcmin, delta_dec_arcmin : centre offsets in arcmin
    pa_deg : position angle in degrees (N→E)
    ellipticity : e = 1 - b/a
    scale_deg : normalisation radius in degrees (e.g. rhalf_mean/60)
    """
    q = 1.0 - ellipticity
    theta = (pa_deg + 90.0) * np.pi / 180.0

    x = radec_offsets[:, 0] - delta_ra_arcmin / 60.0
    y = radec_offsets[:, 1] - delta_dec_arcmin / 60.0

    X_major = -x * np.cos(theta) + y * np.sin(theta)
    Y_minor =  x * np.sin(theta) + y * np.cos(theta)

    return np.sqrt(X_major**2 + (Y_minor / q)**2) / scale_deg


def batch_log_pdf(x, mu, covs):
    """
    Vectorised log-PDF of a multivariate Gaussian for N stars with
    per-star covariance matrices.

    x    : (N, D)
    mu   : (D,)
    covs : (N, D, D)
    """
    delta    = x - mu
    inv_covs = np.linalg.inv(covs)
    det      = np.linalg.det(covs)
    maha     = np.einsum('ni,nij,nj->n', delta, inv_covs, delta)
    D        = x.shape[1]
    return -0.5 * (D * np.log(2 * np.pi) + np.log(det) + maha)


def logdiffexp(a, b):
    """Numerically stable log(exp(a) - exp(b)).  Requires a >= b element-wise."""
    result = np.full_like(a, np.nan, dtype=float)
    result[a == b] = -np.inf
    good = a > b
    result[good] = a[good] + np.log1p(-np.exp(b[good] - a[good]))
    return result


def get_packed_init(sigmas, correlations):
    """
    Convert (sigmas, correlations) → packed lower-triangular Cholesky vector
    for a 3×3 covariance matrix.

    sigmas       : [s1, s2, s3]
    correlations : [r12, r13, r23]
    """
    C = np.eye(3)
    C[1, 0] = C[0, 1] = correlations[0]
    C[2, 0] = C[0, 2] = correlations[1]
    C[2, 1] = C[1, 2] = correlations[2]
    S   = np.diag(sigmas)
    cov = S @ C @ S
    L   = np.linalg.cholesky(cov)
    return L[np.tril_indices(3)]


def get_packed_init_from_cov(cov_matrix):
    """Cholesky-decompose a covariance matrix and return the packed lower triangle."""
    L = np.linalg.cholesky(cov_matrix)
    return L[np.tril_indices(3)]


def unpack_cholesky(packed_vals, n=3):
    """Reconstruct an (n, n) lower-triangular matrix from a packed 1-D vector."""
    L = np.zeros((n, n))
    L[np.tril_indices(n)] = packed_vals
    return L
