"""Convert measured fluxes of NGC 7469 into conditioning values for the model."""

import csv

import numpy as np

from .bands import BAND_NAMES, OBSERVED_UM, conditioning_groups


def galactic_extinction_ratio(lam_um, rv=3.1):
    """A_lambda / A_V of Cardelli, Clayton & Mathis (1989), zero beyond 3.3 micron."""
    x = 1.0 / np.asarray(lam_um, float)
    a, b = np.zeros_like(x), np.zeros_like(x)

    ir = (x >= 0.3) & (x < 1.1)
    a[ir], b[ir] = 0.574 * x[ir] ** 1.61, -0.527 * x[ir] ** 1.61

    op = (x >= 1.1) & (x < 3.3)
    y = x[op] - 1.82
    a[op] = (1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4
             + 0.01979 * y ** 5 - 0.77530 * y ** 6 + 0.32999 * y ** 7)
    b[op] = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4
             - 0.62251 * y ** 5 + 5.30260 * y ** 6 - 2.09002 * y ** 7)

    uv = (x >= 3.3) & (x <= 8)
    xx = x[uv]
    fa = np.where(xx >= 5.9, -0.04473 * (xx - 5.9) ** 2 - 0.009779 * (xx - 5.9) ** 3, 0)
    fb = np.where(xx >= 5.9, 0.2130 * (xx - 5.9) ** 2 + 0.1207 * (xx - 5.9) ** 3, 0)
    a[uv] = 1.752 - 0.316 * xx - 0.104 / ((xx - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xx + 1.206 / ((xx - 4.62) ** 2 + 0.263) + fb
    return a + b / rv


def read_fluxes(csv_path):
    """Read one flux in mJy per band. Columns: band, F_mJy. Empty cells are missing."""
    with open(csv_path, encoding="utf-8") as f:
        rows = {r["band"].strip(): r for r in csv.DictReader(f)}
    flux = np.full(len(BAND_NAMES), np.nan)
    for b, name in enumerate(BAND_NAMES):
        row = rows.get(name)
        if row and row["F_mJy"].strip():
            flux[b] = float(row["F_mJy"])
    return flux


def conditioning_values(flux_mjy, norm, ebv=None):
    """Prepare the model input from the 18 measured fluxes.

    Steps: optional Galactic extinction correction; bands that share a grid point are
    averaged in log10; one global offset (in dex) moves the fluxes to the scale of the
    training library. The offset is the median of (training mean - measured log10 flux)
    over the conditioning points.

    Returns a dict with the conditioning grid indices, the log10 fluxes (measured scale and
    library scale), their z-scores against the training library, and the offset.
    """
    flux = np.array(flux_mjy, dtype=float)
    if ebv is not None:
        flux = flux * 10 ** (0.4 * 3.1 * ebv * galactic_extinction_ratio(OBSERVED_UM))

    wave, mean, std = norm["wave"], norm["mean"], norm["std"]
    log_flux = np.log10(flux)
    values = {}
    for grid_index, bands in conditioning_groups(wave).items():
        present = [b for b in bands if np.isfinite(log_flux[b])]
        if not present:
            raise ValueError("No flux for " + "/".join(BAND_NAMES[b] for b in bands)
                             + ". The model needs all 16 conditioning points.")
        values[grid_index] = log_flux[present].mean()

    index = np.array(sorted(values))
    observed = np.array([values[j] for j in index])
    offset = float(np.median(mean[index] - observed))
    return {
        "index": index,
        "log10_flux": observed,
        "log10_flux_library_scale": observed + offset,
        "z_score": (observed + offset - mean[index]) / std[index],
        "offset_dex": offset,
    }
