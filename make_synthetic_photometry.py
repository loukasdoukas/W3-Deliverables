"""Compute the band-averaged flux of every library galaxy through the 18 filter curves.

    python make_synthetic_photometry.py --data-dir data

Writes data/synthetic_photometry/<library>_synthphot.npz with
    flux_band  (N, 18)  band-averaged F_nu, in the units of the library
    band       (18,)    band names
    pivot_um   (18,)    pivot wavelengths (observed frame, micron)
"""

import argparse
import os
import time

import numpy as np

from csdi_sed.bands import BAND_NAMES, OBSERVED_UM, REDSHIFT
from csdi_sed.data import LOG_EPS, library_files, photometry_file
from csdi_sed.photometry import all_band_operators, synthetic_photometry

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data-dir", default="data", help="folder with the SED libraries")
parser.add_argument("--out-dir", default=None, help="output folder (default: <data-dir>/synthetic_photometry)")
args = parser.parse_args()
out_dir = args.out_dir or os.path.join(args.data_dir, "synthetic_photometry")
os.makedirs(out_dir, exist_ok=True)

for path in library_files(args.data_dir):
    start = time.time()
    with np.load(path) as d:
        wave = d["wave"].astype(np.float64)
        flux = d["flux"]
        redshift = d["redshift"].reshape(-1)
    if not np.allclose(redshift, REDSHIFT, atol=1e-6):
        raise ValueError(f"{os.path.basename(path)}: the redshift is not {REDSHIFT}")

    operators = all_band_operators(wave)
    band_flux = np.empty((len(flux), len(BAND_NAMES)))
    for s in range(0, len(flux), 20000):
        band_flux[s:s + 20000] = synthetic_photometry(np.log10(flux[s:s + 20000] + 1.0e-300), operators)

    out = photometry_file(out_dir, path)
    np.savez(out, flux_band=band_flux, band=np.array(BAND_NAMES), pivot_um=OBSERVED_UM)
    print(f"{os.path.basename(path)}: {len(flux)} galaxies -> {out} ({time.time() - start:.0f} s)")
