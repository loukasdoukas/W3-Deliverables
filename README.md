# CSDI model for SED reconstruction of NGC 7469 (WP3)

Code and trained model that go with the WP3 report.

The model reconstructs the full SED of a galaxy from a few photometric bands.
It is a conditional diffusion model (CSDI, Tashiro et al. 2021).
It is trained on synthetic SEDs from the WP2 tool, at the redshift of NGC 7469 (z = 0.01627),
and it uses the 18 HST, JWST and ALMA bands of this galaxy.

## Repository structure

| File or folder | What it is |
|---|---|
| `config.yaml` | All settings of the model and the training (the values used for the trained model). |
| `requirements.txt` | Python packages needed. |
| `csdi_sed/bands.py` | The 18 bands: names, wavelengths, filter files. Maps bands to the 16 conditioning points. |
| `csdi_sed/photometry.py` | Computes the band-averaged flux of an SED through a filter curve (synthetic photometry). |
| `csdi_sed/data.py` | Loads the libraries, takes the log, splits train/validation/test, standardises, builds the model input. |
| `csdi_sed/network.py` | The denoising network (8 residual blocks, linear attention along wavelength). |
| `csdi_sed/model.py` | The CSDI model: training loss and DDIM sampling of the posterior. |
| `csdi_sed/engine.py` | Training loop. |
| `csdi_sed/observation.py` | Turns measured fluxes into model input (optional Galactic extinction, flux offset). |
| `make_synthetic_photometry.py` | Step 1 of training: band fluxes of all library galaxies. |
| `train.py` | Step 2: trains the model. |
| `reconstruct.py` | Applies the trained model to measured fluxes. |
| `observation_template.csv` | Template for the measured fluxes (one flux in mJy per band). |
| `filters/` | Transmission curves of the 16 bands that have one (the 2 ALMA bands are single wavelengths). |
| `trained_model/` | The trained model (see below). |
| `data/` | Create this folder and put the SED libraries in it (see section 2). |
| `LICENSE_CSDI.txt` | License of the original CSDI code, on which this code is based. |

## Trained model (`trained_model/`)

| File | Content |
|---|---|
| `model.pth` | Weights (4,208,545 parameters). These are the weights of the final model. |
| `config.json` | Settings used for this model. |
| `normalisation.npz` | Wavelength grid (223 points), and the mean and standard deviation of log10 flux of the training set at each wavelength. Needed to use the model. |

The weights are the ones with the lowest validation loss. This was reached at the last
validation (epoch 100), so they are also the final weights.

## Setup

Python 3.11 and a GPU are recommended. The model was trained with PyTorch 2.10 (CUDA 12.6).

```bash
pip install -r requirements.txt
```

Run all commands from this folder.

## 1. Reconstruct a galaxy from measured fluxes

1. Copy `observation_template.csv` and fill in one flux in mJy per band.
   Both bands of the pairs F336W/F330W and F150W/F160W are used if given, but one of each pair is enough.
   All other bands are needed.
2. Run:

```bash
python reconstruct.py observation.csv --out ngc7469_reconstruction
```

Use `--ebv <E(B-V)>` to correct the input fluxes for Galactic extinction. Without it, no correction is applied.

Output: `ngc7469_reconstruction.csv` and `.png`. The csv has the rest-frame wavelength (micron),
the median flux (mJy), the 68 % and 95 % intervals, and a flag for the input points.
The script also prints a z-score of every input flux against the training library.
A value above 3 means the flux is unusual for this library.

Notes:
- The fluxes are moved by one global offset to the scale of the library. The offset is printed and removed again from the result.
- The measurement errors are not used by the model.

## 2. Train the model

### Data

We created two SED libraries, equivalent to the WP2 library and targeted at the galaxy under study (NGC 7469).
They were made with the WP2 random SED generation approach:

- two libraries of 200,000 galaxies, with the CYGNUS torus,
- a spheroidal host (`..._HOST1.npz`) and a disc host (`..._HOST2.npz`),
- redshift 0.01627, all parameters drawn uniformly over the full model grids,
- starburst and polar dust each present in a random 50 % of the galaxies.

We tested the CYGNUS torus. The other three AGN torus models of the WP2 tool (Fritz06, SKIRTOR, Siebenmorgen)
can be tested too: generate the libraries with the same format and set the file name pattern (`LIBRARY_PATTERN` in `csdi_sed/data.py`).

The code looks for these file names in the data folder (`data/` by default).
Each file needs the keys `wave` (223 rest-frame wavelengths in micron), `flux` (N x 223, F_nu) and `redshift` (N).

### Steps

```bash
python make_synthetic_photometry.py --data-dir data
python train.py --data-dir data
```

- Step 1 writes `data/synthetic_photometry/`.
- Step 2 writes `runs/<date>/` with the weights. Use `--dry-run` first to check the setup.
- The training is not seeded, so a new training gives a similar model, not identical weights.
- The split of the data into train, validation and test is fixed (seed 1).

## Method overview

1. Each library SED is taken in log10 flux and standardised at every wavelength.
2. At 16 wavelength points, the library flux is replaced by the flux that a telescope measures in the band (filter average).
   These 16 values are the input. The other 207 points are hidden.
3. The model is trained to predict the noise that was added to the hidden points (diffusion, 50 steps).
4. To reconstruct a galaxy, 50 samples are drawn with DDIM (eta = 0.6). The median is the reconstruction and the spread is the uncertainty.

## Known limits

- The model is trained only on synthetic SEDs from a library equivalent to the WP2 library, not on the WP2 library itself. It can only reconstruct what such a library can produce.
- The 68 % and 95 % intervals are well calibrated in the core, but too narrow in the tails.
- Only the 18 bands of NGC 7469 are supported. Another set of bands needs a new training.

## Credits

Author: Loukas Doukas, PhD candidate, European University Cyprus (EUC).
Developed for the ESA project on machine-learning reconstruction of galaxy spectra (WP3), 2026.

The training libraries follow the WP2 SED generation tool, which is based on SMART
(Varnava & Efstathiou 2024a,b).

References:
- Varnava C. & Efstathiou A. 2024a, MNRAS, 531, 2304.
- Varnava C. & Efstathiou A. 2024b, Astrophysics Source Code Library, ascl:2406.003.

The model is based on CSDI by Tashiro, Song, Song and Ermon (NeurIPS 2021), MIT license (`LICENSE_CSDI.txt`),
with linear attention (`linear-attention-transformer`) and DDIM sampling (Song et al. 2021).
