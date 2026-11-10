"""Reconstruct the full SED of NGC 7469 from the measured fluxes.

    python reconstruct.py observation.csv --model trained_model --out ngc7469_reconstruction
    python reconstruct.py observation.csv --ebv 0.061

observation.csv has one flux in mJy per band (see observation_template.csv).
All 16 conditioning points are needed (F336W/F330W and F150W/F160W share a point,
so one band of each pair is enough).

Writes <out>.csv and <out>.png. Wavelengths are rest-frame, fluxes are in mJy.
Columns of the csv: wavelength_um, median, p16, p84, p2_5, p97_5, is_input.
At the conditioning points the input fluxes are reported unchanged.
The measurement errors are not used by the model.
"""

import argparse
import os

import matplotlib
import numpy as np
import torch

from csdi_sed.model import load_trained_model
from csdi_sed.observation import conditioning_values, read_fluxes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("observation", help="csv file with the columns band, F_mJy")
parser.add_argument("--model", default="trained_model", help="folder with config.json, model.pth, normalisation.npz")
parser.add_argument("--out", default="ngc7469_reconstruction", help="output file name without extension")
parser.add_argument("--ebv", type=float, default=None, help="Galactic E(B-V) to correct for (optional)")
parser.add_argument("--nsample", type=int, default=50, help="number of posterior samples")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

norm = dict(np.load(os.path.join(args.model, "normalisation.npz")))
obs = conditioning_values(read_fluxes(args.observation), norm, args.ebv)
index, offset = obs["index"], obs["offset_dex"]
wave, mean, std = norm["wave"], norm["mean"], norm["std"]

print(f"Offset to the library scale: {offset:+.3f} dex (subtracted again from the result)")
for j, z in zip(index, obs["z_score"]):
    flag = "   <-- unusual compared with the training library" if abs(z) > 3 else ""
    print(f"  {wave[j]:9.3f} um   z-score {z:+.2f}{flag}")

torch.manual_seed(args.seed)
model, _ = load_trained_model(args.model, args.device)

n_wave = len(wave)
data = np.zeros((1, 1, n_wave), dtype=np.float32)
data[0, 0, index] = (obs["log10_flux_library_scale"] - mean[index]) / std[index]
cond_mask = np.zeros((1, 1, n_wave), dtype=np.float32)
cond_mask[0, 0, index] = 1.0
data = torch.tensor(data, device=args.device)
cond_mask = torch.tensor(cond_mask, device=args.device)
timepoints = torch.tensor(norm["timepoints"][None], device=args.device)

samples = model.impute(data, cond_mask, model.side_info(timepoints, cond_mask), args.nsample)
log_flux = samples[0, :, 0, :].cpu().numpy() * std + mean - offset
log_flux[:, index] = obs["log10_flux"]

p2_5, p16, median, p84, p97_5 = (10 ** np.percentile(log_flux, q, axis=0) for q in (2.5, 16, 50, 84, 97.5))
is_input = np.zeros(n_wave, dtype=int)
is_input[index] = 1
np.savetxt(args.out + ".csv", np.column_stack([wave, median, p16, p84, p2_5, p97_5, is_input]),
           delimiter=",", comments="", fmt="%.8g", header="wavelength_um,median,p16,p84,p2_5,p97_5,is_input")

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(wave, p2_5, p97_5, color="tab:blue", alpha=0.15, label="95 % interval")
ax.fill_between(wave, p16, p84, color="tab:blue", alpha=0.35, label="68 % interval")
ax.plot(wave, median, color="tab:blue", label="Median")
ax.plot(wave[index], median[index], "o", color="tab:red", label="Input fluxes")
ax.axvspan(5, 35, color="gold", alpha=0.2, lw=0)
ax.set(xscale="log", yscale="log", xlabel="Rest-frame wavelength (micron)", ylabel="Flux density (mJy)")
ax.legend()
fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
print(f"Wrote {args.out}.csv and {args.out}.png")
