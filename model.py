"""Two-level wavelet fusion, three dense feature blocks, one spatial GCN, PCU heads."""
import numpy as np
import pywt
import torch
from torch import nn
import torch.nn.functional as F
from common import config


class WaveletMultiScaleDecomposition(nn.Module):
    def __init__(self, wavelet_basis="db4", levels=2, mode="symmetric"):
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet_basis)
        self.levels, self.mode = levels, mode

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected RGB tensor (B,3,H,W)")
        size = (x.shape[-2] // 2, x.shape[-1] // 2)
        samples = []
        for sample in x.detach().float().cpu().numpy():
            channels = []
            for channel in sample:
                current = channel
                for _ in range(self.levels):
                    ll, (lh, hl, hh) = pywt.dwt2(current, self.wavelet, mode=self.mode)
                    bands = torch.from_numpy(np.stack([ll, lh, hl, hh])).unsqueeze(0)
                    channels.append(F.interpolate(bands, size=size, mode="bilinear", align_corners=False)[0])
                    current = ll
            samples.append(torch.cat(channels, dim=0))
        return torch.stack(samples).to(device=x.device, dtype=x.dtype)


class MultiScaleFusion(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class DenseBlock(nn.Module):
    """One densely connected feature block adds 32 channels to its input."""
    def __init__(self, in_ch, growth_rate=32, dropout=0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, growth_rate, 3, padding=1, bias=False), nn.Dropout2d(dropout))
        self.out_channels = in_ch + growth_rate

    def forward(self, x):
        return torch.cat([x, self.features(x)], dim=1)


def build_grid_edge_index(g):
    edges = []
    for i in range(g):
        for j in range(g):
            u = i * g + j
            for ni, nj in [(i + 1, j), (i, j + 1)]:
                if ni < g and nj < g:
                    v = ni * g + nj
                    edges.extend([(u, v), (v, u)])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, g=14, dropout=0.1):
        super().__init__()
        self.g = g
        self.register_buffer("edge_index", build_grid_edge_index(g), persistent=False)
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nei = nn.Linear(in_dim, out_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, nodes, dim = x.shape
        if nodes != self.g ** 2:
            raise ValueError("Node count does not match the spatial grid")
        flat = x.reshape(batch * nodes, dim)
        src, dst = self.edge_index
        offsets = torch.arange(batch, device=x.device).repeat_interleave(src.numel()) * nodes
        src = src.repeat(batch) + offsets
        dst = dst.repeat(batch) + offsets
        messages = self.lin_nei(flat[src])
        neighbors = torch.zeros(batch * nodes, messages.shape[-1], device=x.device, dtype=messages.dtype)
        neighbors.index_add_(0, dst, messages)
        degrees = torch.bincount(dst, minlength=batch * nodes).to(messages.dtype).clamp_min(1)
        updated = self.lin_self(flat) + neighbors / degrees[:, None]
        return self.drop(F.gelu(updated)).reshape(batch, nodes, -1)


class FractureDetectionWaveletDenseGNN(nn.Module):
    def __init__(self, cfg=None, use_gcn=True):
        super().__init__()
        cfg = cfg or config
        self.in_ms_ch = cfg.wavelet_ms_channels
        self.wavelet_ms = WaveletMultiScaleDecomposition(cfg.wavelet_basis, cfg.wavelet_levels)
        self.fusion = MultiScaleFusion(self.in_ms_ch, 64)
        self.dense = nn.Sequential(DenseBlock(64, dropout=cfg.dropout),
                                   DenseBlock(96, dropout=cfg.dropout), DenseBlock(128, dropout=cfg.dropout))
        self.to_gnn = MultiScaleFusion(160, cfg.gnn_hidden_dim)
        self.g = cfg.gnn_grid_size
        self.pool = nn.AdaptiveAvgPool2d((self.g, self.g))
        self.gnn = GraphConv(cfg.gnn_hidden_dim, cfg.gnn_hidden_dim, self.g, cfg.dropout) if use_gcn else nn.Identity()
        self.fc_p = nn.Linear(cfg.gnn_hidden_dim, 1)
        self.fc_c = nn.Linear(cfg.gnn_hidden_dim, 1)
        self.fc_u = nn.Linear(cfg.gnn_hidden_dim, 1)

    def forward(self, x, return_feature_map=False):
        if x.shape[1] == 3:
            x = self.wavelet_ms(x)
        elif x.shape[1] != self.in_ms_ch:
            raise ValueError("Expected RGB or aligned wavelet channels")
        features = self.pool(self.to_gnn(self.dense(self.fusion(x))))
        nodes = self.gnn(features.flatten(2).transpose(1, 2))
        embedding = nodes.mean(dim=1)
        result = (self.fc_p(embedding), self.fc_c(embedding), self.fc_u(embedding))
        if return_feature_map:
            return (*result, nodes.transpose(1, 2).reshape(x.shape[0], -1, self.g, self.g))
        return result

    def get_classifier_weights(self):
        return self.fc_p.weight.detach()


def get_model(config, pretrained=False, drop_rate=None):
    return FractureDetectionWaveletDenseGNN(config).to(config.device)
