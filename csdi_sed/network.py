"""Denoising network of the CSDI model.

Eight residual blocks with a linear-attention transformer along the wavelength axis,
gated activations, and skip connections (Tashiro et al. 2021; Shen et al. 2021).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer


def linear_transformer(heads, channels):
    return LinearAttentionTransformer(
        dim=channels,
        depth=1,
        heads=heads,
        max_seq_len=256,
        n_local_attn_heads=0,
        local_attn_window_size=0,
    )


def conv1d(in_channels, out_channels):
    layer = nn.Conv1d(in_channels, out_channels, 1)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """Sinusoidal embedding of the diffusion step followed by two linear layers."""

    def __init__(self, num_steps, embedding_dim=128):
        super().__init__()
        self.register_buffer("embedding", self._build_table(num_steps, embedding_dim / 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, diffusion_step):
        x = F.silu(self.projection1(self.embedding[diffusion_step]))
        return F.silu(self.projection2(x))

    @staticmethod
    def _build_table(num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = conv1d(side_dim, 2 * channels)
        self.mid_projection = conv1d(channels, 2 * channels)
        self.output_projection = conv1d(channels, 2 * channels)
        self.time_layer = linear_transformer(nheads, channels)
        # Feature attention of the general CSDI design. It is not used here
        # (one feature), but it is kept so that the saved weights load.
        self.feature_layer = linear_transformer(nheads, channels)

    def forward(self, x, side_info, diffusion_emb):
        B, channels, K, L = x.shape
        x = x.reshape(B, channels, K * L)
        y = x + self.diffusion_projection(diffusion_emb).unsqueeze(-1)

        y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.mid_projection(y)

        side_info = side_info.reshape(B, side_info.shape[1], K * L)
        y = y + self.cond_projection(side_info)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(B, channels, K, L)
        residual = residual.reshape(B, channels, K, L)
        skip = skip.reshape(B, channels, K, L)
        return (x + residual) / math.sqrt(2.0), skip


class DenoisingNetwork(nn.Module):
    def __init__(self, config, input_dim=2):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(config["num_steps"], config["diffusion_embedding_dim"])
        self.input_projection = conv1d(input_dim, self.channels)
        self.output_projection1 = conv1d(self.channels, self.channels)
        self.output_projection2 = conv1d(self.channels, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(config["side_dim"], self.channels, config["diffusion_embedding_dim"], config["nheads"])
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, side_info, diffusion_step):
        B, input_dim, K, L = x.shape
        x = F.relu(self.input_projection(x.reshape(B, input_dim, K * L)))
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skips = []
        for layer in self.residual_layers:
            x, skip = layer(x, side_info, diffusion_emb)
            skips.append(skip)

        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = F.relu(self.output_projection1(x))
        return self.output_projection2(x).reshape(B, K, L)
