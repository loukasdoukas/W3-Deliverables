"""The 18 photometric bands of NGC 7469 and their filter curves.

Band wavelengths are pivot wavelengths in the observed frame, in micron
(SVO Filter Profile Service; ALMA values are the central wavelengths).
Filter curve files have two columns: wavelength in Angstrom and transmission.
ALMA bands are treated as a single wavelength.
"""

import os

import numpy as np

REDSHIFT = 0.01627

FILTER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "filters")

# (name, instrument, observed pivot wavelength in micron, filter curve file or None)
BANDS = [
    ("F275W", "HST/WFC3-UVIS", 0.270292, "HST_WFC3_UVIS1.F275W.txt"),
    ("F336W", "HST/WFC3-UVIS", 0.335460, "HST_WFC3_UVIS1.F336W.txt"),
    ("F330W", "HST/ACS-HRC", 0.336299, "HST_ACS_HRC.F330W.dat"),
    ("F435W", "HST/ACS-WFC", 0.432985, "HST_ACS_WFC.F435W.dat"),
    ("F547M", "HST/WFC3-UVIS", 0.544725, "HST_WFC3_UVIS1.F547M.txt"),
    ("F550M", "HST/ACS-WFC", 0.558140, "HST_ACS_WFC.F550M.dat"),
    ("F814W", "HST/ACS-WFC", 0.804553, "HST_ACS_WFC.F814W.dat"),
    ("F110W", "HST/WFC3-IR", 1.153446, "HST_WFC3_IR.F110W.txt"),
    ("F150W", "JWST/NIRCam", 1.500744, "JWST_NIRCam2025.F150W.txt"),
    ("F160W", "HST/WFC3-IR", 1.537034, "HST_WFC3_IR.F160W.txt"),
    ("F200W", "JWST/NIRCam", 1.988648, "JWST_NIRCam2025.F200W.txt"),
    ("F335M", "JWST/NIRCam", 3.362067, "JWST_NIRCam2025.F335M.txt"),
    ("F444W", "JWST/NIRCam", 4.404315, "JWST_NIRCam2025.F444W.txt"),
    ("F560W", "JWST/MIRI", 5.635256, "MIRI_F560.txt"),
    ("F770W", "JWST/MIRI", 7.639334, "MIRI_F770.txt"),
    ("F1500W", "JWST/MIRI", 15.063506, "MIRI_F1500.txt"),
    ("Band10", "ALMA", 330.0, None),
    ("Band8", "ALMA", 700.0, None),
]

BAND_NAMES = [b[0] for b in BANDS]
OBSERVED_UM = np.array([b[2] for b in BANDS], dtype=np.float64)
REST_FRAME_UM = (OBSERVED_UM / (1.0 + REDSHIFT)).astype(np.float32)


def load_curve(name):
    """Return (wavelength in micron, transmission) of a band, or None for ALMA bands."""
    filename = BANDS[BAND_NAMES.index(name)][3]
    if filename is None:
        return None
    table = np.loadtxt(os.path.join(FILTER_DIR, filename))
    wavelength, transmission = table[:, 0] * 1.0e-4, table[:, 1]
    keep = np.isfinite(wavelength) & np.isfinite(transmission) & (wavelength > 0) & (transmission > 0)
    return wavelength[keep], transmission[keep]


def conditioning_groups(wave):
    """Map each conditioning grid point to the list of bands that fall on it.

    Two pairs of bands share a grid point (F336W/F330W and F150W/F160W),
    so the 18 bands give 16 conditioning points.
    """
    groups = {}
    for band, rest_um in enumerate(REST_FRAME_UM):
        groups.setdefault(int(np.argmin(np.abs(wave - rest_um))), []).append(band)
    return groups


def band_to_grid_values(log_band_flux, wave):
    """Convert (N, 18) log10 band fluxes to {grid index: (N,) conditioning value}.

    Bands that share a grid point are averaged in log10.
    """
    return {j: log_band_flux[:, bands].mean(axis=1) for j, bands in conditioning_groups(wave).items()}
