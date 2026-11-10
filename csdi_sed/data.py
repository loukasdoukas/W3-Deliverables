"""Training data: SED libraries, synthetic photometry, split and normalisation.

Each library is an .npz file with the keys `wave` (L,), `flux` (N, L) and `redshift` (N,).
The model works on log10 flux, standardised at every wavelength with the mean and
standard deviation of the training set.

At the 16 conditioning positions the library flux is replaced by the band-averaged flux
through the real filter curves (see photometry.py). All other positions keep the library SED.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .bands import REST_FRAME_UM, band_to_grid_values

LOG_EPS = 1.0e-12
LIBRARY_PATTERN = "ngc7469_random_sampling_galaxies_AGN1_HOST[12].npz"


def library_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, LIBRARY_PATTERN)))
    if not files:
        raise FileNotFoundError(f"No library files {LIBRARY_PATTERN} in {data_dir}")
    return files


def photometry_file(photometry_dir, library_path):
    return os.path.join(photometry_dir, os.path.basename(library_path).replace(".npz", "_synthphot.npz"))


class SEDLibrary:
    """All library galaxies in memory, with split, normalisation and conditioning values."""

    def __init__(self, data_dir, photometry_dir, seed=1, valid_ratio=0.1, test_ratio=0.1):
        fluxes, host_ids, wave = [], [], None
        for path in library_files(data_dir):
            with np.load(path, allow_pickle=True) as d:
                flux, w = d["flux"].astype(np.float32), d["wave"].astype(np.float32)
            if wave is not None and not np.allclose(w, wave, rtol=1e-6, atol=1e-8):
                raise ValueError("All libraries must share the same wavelength grid")
            wave = w
            fluxes.append(flux)
            host_ids.append(np.full(len(flux), 1 if "HOST2" in os.path.basename(path).upper() else 0))
        flux = np.concatenate(fluxes)
        if np.any(flux <= 0):
            raise ValueError("The library contains non-positive fluxes")
        log_flux = np.log10(flux + LOG_EPS).astype(np.float32)

        self.wave = wave
        self.host_id = np.concatenate(host_ids)
        log_wave = np.log10(wave)
        self.timepoints = ((log_wave - log_wave.min()) / (log_wave.max() - log_wave.min() + 1e-12)).astype(np.float32)
        self.band_indices = np.array(sorted({int(np.argmin(np.abs(wave - u))) for u in REST_FRAME_UM}), dtype=np.int64)

        order = np.arange(len(log_flux))
        np.random.RandomState(seed).shuffle(order)
        n_test = int(round(len(order) * test_ratio))
        n_valid = int(round(len(order) * valid_ratio))
        n_train = len(order) - n_valid - n_test
        self.split = {
            "train": order[:n_train],
            "valid": order[n_train:n_train + n_valid],
            "test": order[n_train + n_valid:],
        }

        train_log_flux = log_flux[self.split["train"]]
        self.mean = train_log_flux.mean(axis=0).astype(np.float32)
        self.std = train_log_flux.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0

        self.data = ((log_flux - self.mean) / self.std).astype(np.float32)
        self._set_conditioning_values(data_dir, photometry_dir, log_flux)

    def _set_conditioning_values(self, data_dir, photometry_dir, log_flux):
        band_flux = []
        for path in library_files(data_dir):
            file = photometry_file(photometry_dir, path)
            if not os.path.isfile(file):
                raise FileNotFoundError(f"{file} is missing. Run make_synthetic_photometry.py first.")
            with np.load(file) as d:
                band_flux.append(d["flux_band"])
        band_flux = np.concatenate(band_flux)
        if len(band_flux) != len(self.data):
            raise ValueError("Synthetic photometry and libraries have different numbers of galaxies")
        values = band_to_grid_values(np.log10(band_flux + LOG_EPS), self.wave)
        self.band_minus_grid_dex = {}
        for j, v in values.items():
            self.band_minus_grid_dex[j] = float(np.median(v - log_flux[:, j]))
            self.data[:, j] = ((v - self.mean[j]) / self.std[j]).astype(np.float32)

    def save_normalisation(self, path):
        np.savez(path, wave=self.wave, mean=self.mean, std=self.std, timepoints=self.timepoints,
                 band_indices=self.band_indices)


class SEDDataset(Dataset):
    def __init__(self, library, mode):
        self.library = library
        self.indices = library.split[mode]
        mask = np.zeros((library.data.shape[1], 1), dtype=np.float32)
        mask[library.band_indices, 0] = 1.0
        self.condition_mask = mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.indices[i]
        flux = self.library.data[row][:, None]
        return {
            "observed_data": flux,
            "observed_mask": np.ones_like(flux),
            "gt_mask": self.condition_mask,
            "timepoints": self.library.timepoints,
            "host_id": np.array(self.library.host_id[row], dtype=np.int64),
        }


def get_dataloaders(library, batch_size):
    """Return train, validation and test loaders."""
    loaders = []
    for mode in ("train", "valid", "test"):
        loaders.append(DataLoader(SEDDataset(library, mode), batch_size=batch_size,
                                  shuffle=(mode == "train"), num_workers=0))
    return loaders


def scalers(library, device):
    """Per-wavelength standard deviation and mean as tensors."""
    return torch.tensor(library.std, device=device).float(), torch.tensor(library.mean, device=device).float()
