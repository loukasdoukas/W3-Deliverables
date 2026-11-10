"""CSDI model for galaxy SEDs: training loss and DDIM sampling.

Data layout used throughout: (B, 1, L) = galaxies, one feature, L = 223 wavelength points.
The "condition mask" is 1 at the 16 conditioning positions and 0 at the hidden positions.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

from .network import DenoisingNetwork


class CSDI_GalaxySED(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        model_cfg, diff_cfg = config["model"], config["diffusion"]

        self.time_embedding_dim = model_cfg["timeemb"]
        feature_embedding_dim = model_cfg["featureemb"]
        side_dim = self.time_embedding_dim + feature_embedding_dim + 1
        diff_cfg["side_dim"] = side_dim

        self.embed_layer = nn.Embedding(1, feature_embedding_dim)
        self.diffmodel = DenoisingNetwork(diff_cfg, input_dim=2)

        self.num_steps = diff_cfg["num_steps"]
        if diff_cfg["schedule"] == "quad":
            beta = np.linspace(diff_cfg["beta_start"] ** 0.5, diff_cfg["beta_end"] ** 0.5, self.num_steps) ** 2
        else:
            beta = np.linspace(diff_cfg["beta_start"], diff_cfg["beta_end"], self.num_steps)
        self.alpha = np.cumprod(1.0 - beta)
        self.alpha_torch = torch.tensor(self.alpha).float().to(device).unsqueeze(1).unsqueeze(1)

        self.sampling_steps = min(diff_cfg.get("eval_num_steps", self.num_steps), self.num_steps)
        self.eta = float(diff_cfg.get("eta", 1.0))
        self.ddim_sigma_fix = bool(diff_cfg.get("ddim_sigma_fix", False))

    def wavelength_embedding(self, timepoints):
        """Sinusoidal embedding of the normalised log-wavelength, (B, L) -> (B, L, d)."""
        d = self.time_embedding_dim
        pe = torch.zeros(timepoints.shape[0], timepoints.shape[1], d, device=self.device)
        position = timepoints.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d, 2, device=self.device) / d)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def side_info(self, timepoints, cond_mask):
        """Side information (B, 161, 1, L): wavelength embedding, feature embedding, condition mask."""
        B, K, L = cond_mask.shape
        wave_embed = self.wavelength_embedding(timepoints).unsqueeze(2)
        feature_embed = self.embed_layer(torch.arange(K, device=self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        info = torch.cat([wave_embed, feature_embed], dim=-1).permute(0, 3, 2, 1)
        return torch.cat([info, cond_mask.unsqueeze(1)], dim=1)

    def prepare(self, batch):
        """Move a batch to the device and return (data, mask, timepoints, cond_mask), data as (B, 1, L)."""
        data = batch["observed_data"].to(self.device).float().permute(0, 2, 1)
        mask = batch["observed_mask"].to(self.device).float().permute(0, 2, 1)
        timepoints = batch["timepoints"].to(self.device).float()
        cond_mask = batch["gt_mask"].to(self.device).float().permute(0, 2, 1)
        return data, mask, timepoints, cond_mask

    def network_input(self, noisy, data, cond_mask):
        """Two input channels: conditioning values, and the noisy target at hidden positions."""
        return torch.cat([(cond_mask * data).unsqueeze(1), ((1 - cond_mask) * noisy).unsqueeze(1)], dim=1)

    def calc_loss(self, data, mask, cond_mask, side_info, step=None):
        """Noise-prediction MSE at the hidden positions. A random step is used unless `step` is given."""
        B = data.shape[0]
        if step is None:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        else:
            t = torch.full((B,), step, dtype=torch.long, device=self.device)
        alpha = self.alpha_torch[t]
        noise = torch.randn_like(data)
        noisy = alpha ** 0.5 * data + (1.0 - alpha) ** 0.5 * noise
        predicted = self.diffmodel(self.network_input(noisy, data, cond_mask), side_info, t)
        target_mask = mask - cond_mask
        residual = (noise - predicted) * target_mask
        n_eval = target_mask.sum()
        return (residual ** 2).sum() / (n_eval if n_eval > 0 else 1)

    def forward(self, batch, is_train=True):
        """Training loss (random step) or validation loss (average over all steps)."""
        data, mask, timepoints, cond_mask = self.prepare(batch)
        side_info = self.side_info(timepoints, cond_mask)
        if is_train:
            return self.calc_loss(data, mask, cond_mask, side_info)
        total = 0
        for t in range(self.num_steps):
            total += self.calc_loss(data, mask, cond_mask, side_info, step=t).detach()
        return total / self.num_steps

    @torch.no_grad()
    def impute(self, data, cond_mask, side_info, n_samples):
        """Draw `n_samples` DDIM samples for each galaxy. Returns (B, n_samples, 1, L)."""
        B, K, L = data.shape
        data = data.repeat_interleave(n_samples, dim=0)
        cond_mask = cond_mask.repeat_interleave(n_samples, dim=0)
        side_info = side_info.repeat_interleave(n_samples, dim=0)

        sample = torch.randn_like(data)
        steps = np.linspace(0, self.num_steps - 1, self.sampling_steps, dtype=int)

        for i in range(len(steps) - 1, -1, -1):
            t = int(steps[i])
            t_prev = int(steps[i - 1]) if i > 0 else -1

            eps = self.diffmodel(
                self.network_input(sample, data, cond_mask), side_info, torch.tensor([t], device=self.device)
            )

            alpha_t = self.alpha[t]
            alpha_prev = self.alpha[t_prev] if t_prev >= 0 else 1.0
            predicted_x0 = (sample - (1 - alpha_t) ** 0.5 * eps) / alpha_t ** 0.5

            if t_prev >= 0:
                sigma = self.eta * ((1.0 - alpha_prev) / (1.0 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
            else:
                sigma = 0.0

            if self.ddim_sigma_fix:
                direction = max(1 - alpha_prev - sigma ** 2, 0.0) ** 0.5 * eps
            else:
                direction = (1 - alpha_prev) ** 0.5 * eps
            sample = alpha_prev ** 0.5 * predicted_x0 + direction
            if sigma > 0.0:
                sample = sample + sigma * torch.randn_like(sample)

        return sample.reshape(B, n_samples, K, L)

    @torch.no_grad()
    def evaluate(self, batch, n_samples):
        """Sample a test batch. Returns samples, truth, hidden-point mask, timepoints."""
        data, mask, timepoints, cond_mask = self.prepare(batch)
        side_info = self.side_info(timepoints, cond_mask)
        samples = self.impute(data, cond_mask, side_info, n_samples)
        return samples, data, mask - cond_mask, timepoints


def load_trained_model(folder, device):
    """Load config.json and model.pth from `folder`. Returns (model, config)."""
    with open(os.path.join(folder, "config.json")) as f:
        config = json.load(f)
    model = CSDI_GalaxySED(config, device).to(device)
    model.load_state_dict(torch.load(os.path.join(folder, "model.pth"), map_location=device))
    model.eval()
    return model, config
