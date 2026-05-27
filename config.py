import os
import numpy as np

# --- Hardcoded data paths ---
LVD_CATALOG_PATH = '/Users/kevinm/Downloads/dwarf_all.csv'
GAIA_DATA_PATH   = '/Users/kevinm/Documents/UCSC/HST_Gaia_PMs/GaiaHub_results/'
# MILLIQUAS FITS file — expected in the same directory as this config file.
# Download from https://www.quasars.org/milliquas.htm
MILLIQUAS_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'milliquas.fits')

# --- Gaia systematic error budget (Vasiliev & Baumgardt 2021) ---
PARALLAX_SYS_ERR  = 0.011   # mas
PM_SYS_ERR        = 0.026   # mas/yr
GAIA_MEAN_INFLATE = 1.20    # mean error inflation factor
PM_TO_VEL_FACT    = 4.74    # km/s per (mas/yr * kpc)

N_CLUSTERS = 2           # dwarf galaxy, MW disk, MW halo (spatial model)
DEFAULT_BG_COMPONENTS = 5  # sklearn GMM components for pre-fit MW background

# Gaia download / background-selection geometry
GAIA_BG_RADIUS_FACTOR     = 8.0   # search radius = FACTOR × rhalf
GAIA_MIN_RADIUS_DEG       = 0.5   # minimum search radius (deg)
GAIA_MAX_RADIUS_DEG       = 2.0   # cap to avoid excessively large downloads (deg)
GAIA_BG_SPATIAL_THRESHOLD = 7.0   # r_ell > this (units of rhalf) → background
GAIA_BG_MIN_STARS         = 100   # fallback: lower threshold if fewer stars found

# Photometric membership prior: sigma on LVD bulk-motion estimate
PRIOR_MEAN_UNCERT = 0.5  # mas/yr

# --- Field-name to LVD catalog key ---
LVD_TRANSLATION = {
    'Fornax_dSph':     'fornax_1',
    'Draco_dSph':      'draco_1',
    'Draco_II':        'draco_2',
    'Leo_I':           'leo_1',
    'Leo_II':          'leo_2',
    'Leo_IV':          'leo_4',
    'Leo_V':           'leo_5',
    'Leo_T':           'leo_t',
    'Leo_A':           'leo_a',
    'NGC_3109':        'ngc_3109',
    'NGC_300':         'ngc_0300',
    'NGC_55':          'ngc_0055',
    'NGC_6822':        'ngc_6822',
    'Sextans_A':       'sextans_a',
    'Sextans_B':       'sextans_b',
    'Bootes1':         'bootes_1',
    'Sag_DIG':         'sagittarius_dirr',
    'Sagittarius_dSph':'sagittarius_1',
    'Sculptor_dSph':   'sculptor_1',
    'segue_1':         'segue_1',
    'sextans_1':       'sextans_1',
    'Phoenix':         'phoenix_1',
    'Phoenix_II':      'phoenix_2',
    'IC1613':          'ic_1613',
    'WLM':             'wlm',
    'UGC4879':         'ugc_04879',
    'M33':             'm33',
    'Antlia_II':       'antlia_2',
    'Crater_II':       'crater_2',
    'Hydrus_I':        'hydrus_1',
}

# Bright-star G-mag cutoff per field (stars brighter than this are excluded).
# None means no bright cut (uses -10000 internally).
GMAG_LIMITS = {
    'Fornax_dSph':      None,
    'Draco_dSph':       None,
    'Leo_I':            None,
    'NGC_3109':         None,
    'NGC_300':          None,
    'NGC_55':           None,
    'Sextans_A':        None,
    'Sextans_B':        None,
    'Bootes1':          None,
    'Sag_DIG':          None,
    'Sagittarius_dSph': None,
    'Sculptor_dSph':    None,
    'segue_1':          None,
    'sextans_1':        None,
}

# Fields that need explicit RA/Dec (not in Simbad or need overriding).
# search_radius in degrees.
PROP_DICT = {
    'Sag_DIG':         {'ra': 292.496,  'dec': -17.678,  'search_radius': 0.2},
    'Bootes1':         {'ra': 210.025,  'dec':  14.500,  'search_radius': None},
    'Sagittarius_dSph':{'ra': 283.829,  'dec': -30.545,  'search_radius': 0.5},
    'segue_1':         {'ra': 151.763,  'dec':  16.043,  'search_radius': None},
    'sextans_1':       {'ra': 153.275,  'dec':  -1.596,  'search_radius': None},
    'NGC_55':          {'ra':   3.723,  'dec': -39.197,  'search_radius': None},
    'NGC_300':         {'ra':  13.723,  'dec': -37.684,  'search_radius': None},
    'NGC_3109':        {'ra': None,     'dec': None,     'search_radius': None},
}

# Hard spatial cutoffs applied AFTER computing radec_offsets for certain crowded/large fields.
FIELD_SPATIAL_CUTOFFS = {
    # 'NGC_3109':         0.25,  # deg
    # 'Sextans_B':        0.15,
    # 'Sagittarius_dSph': 0.15,
}

# Sagittarius_dSph uses a custom centre instead of the LVD value.
SAGITTARIUS_RADEC_CENTER = np.array(
    [(18 + 55/60 + 19.5/3600) * 15, -(30 + 32/60 + 43/3600)]
)
