"""Train the CSDI model on the NGC 7469 libraries.

    python make_synthetic_photometry.py --data-dir data     # once
    python train.py --data-dir data

Writes runs/<date>/ with config.json, normalisation.npz, model.pth (best weights),
model_last.pth (final weights). Use --dry-run to check the setup with one training step.
Then run evaluate.py on the output folder.
"""

import argparse
import datetime
import json
import os

import torch
import yaml

from csdi_sed.data import SEDLibrary, get_dataloaders
from csdi_sed.engine import train
from csdi_sed.model import CSDI_GalaxySED

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--data-dir", default=None, help="folder with the libraries (default: from the config)")
parser.add_argument("--out-dir", default="runs", help="folder where the run folder is created")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=None, help="split seed (default: from the config)")
parser.add_argument("--dry-run", action="store_true", help="load the data and do one training step only")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)
data_cfg = config["dataset"]
data_dir = args.data_dir or data_cfg["data_dir"]
photometry_dir = os.path.join(data_dir, "synthetic_photometry") if args.data_dir else data_cfg["photometry_dir"]

library = SEDLibrary(data_dir, photometry_dir, seed=args.seed or data_cfg["split_seed"])
train_loader, valid_loader, _ = get_dataloaders(library, config["train"]["batch_size"])
print(f"Galaxies: {len(train_loader.dataset)} train, {len(valid_loader.dataset)} validation")

model = CSDI_GalaxySED(config, args.device).to(args.device)
n_parameters = sum(p.numel() for p in model.parameters())

if args.dry_run:
    loss = model(next(iter(train_loader)))
    loss.backward()
    print(f"Dry run OK. One step, loss {loss.item():.4f}, {n_parameters} parameters.")
    raise SystemExit

folder = os.path.join(args.out_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(folder, exist_ok=True)
with open(os.path.join(folder, "config.json"), "w") as f:
    json.dump(config, f, indent=4)
library.save_normalisation(os.path.join(folder, "normalisation.npz"))
print("Run folder:", folder)

train(model, config["train"], train_loader, valid_loader, folder)
print("Training finished. Best weights:", os.path.join(folder, "model.pth"))
