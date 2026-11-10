"""Synthetic photometry: band-averaged flux of a library SED through a filter curve.

    <F_nu> = integral(F_nu R dlam/lam) / integral(R dlam/lam)        (photon counting)

The SED is moved to the observed frame (lam_obs = lam_rest * (1 + z)) and
interpolated log-log onto the wavelengths of the filter curve.
ALMA bands use the flux at their observed wavelength.
"""

import numpy as np

from .bands import BAND_NAMES, BANDS, OBSERVED_UM, REDSHIFT, load_curve


def _interpolation_weights(wave_rest, lam_obs):
    """Indices and weights of a log-log interpolation of the SED at observed wavelengths."""
    ascending = np.argsort(wave_rest)
    log_wave = np.log10(wave_rest[ascending] * (1.0 + REDSHIFT))
    x = np.log10(lam_obs)
    if x.min() < log_wave[0] or x.max() > log_wave[-1]:
        raise ValueError("The filter is outside the wavelength range of the SED")
    k = np.clip(np.searchsorted(log_wave, x), 1, len(log_wave) - 1)
    w = (x - log_wave[k - 1]) / (log_wave[k] - log_wave[k - 1])
    return ascending[k - 1], ascending[k], w


def band_operator(wave_rest, name):
    """Return (i0, i1, w, q) so that <F_nu> = sum(q * 10 ** ((1 - w) log F[i0] + w log F[i1]))."""
    curve = load_curve(name)
    if curve is None:
        lam = np.array([OBSERVED_UM[BAND_NAMES.index(name)]])
        i0, i1, w = _interpolation_weights(wave_rest, lam)
        return i0, i1, w, np.ones(1)
    lam, transmission = curve
    i0, i1, w = _interpolation_weights(wave_rest, lam)
    weight = transmission / lam
    dlam = np.diff(lam)
    q = np.zeros_like(lam)
    q[:-1] += 0.5 * dlam * weight[:-1]
    q[1:] += 0.5 * dlam * weight[1:]
    return i0, i1, w, q / q.sum()


def synthetic_photometry(log_flux, operators):
    """Convert (N, L) log10 F_nu to (N, 18) band-averaged F_nu."""
    out = np.empty((len(log_flux), len(operators)))
    for b, (i0, i1, w, q) in enumerate(operators):
        out[:, b] = (10 ** ((1 - w) * log_flux[:, i0] + w * log_flux[:, i1])) @ q
    return out


def all_band_operators(wave_rest):
    return [band_operator(wave_rest, band[0]) for band in BANDS]
