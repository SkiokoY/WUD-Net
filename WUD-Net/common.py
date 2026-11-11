"""Configuration for the manuscript WUD-Net architecture and evaluation protocol."""
from dataclasses import dataclass, asdict
from pathlib import Path
import torch

DATASET_COUNTS = {"mura": (1312, 1115, 197), "fracatlas": (4083, 3470, 613), "hbfmid": (1536, 1382, 154)}

@dataclass
class Config:
    image_size: int = 384
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    wavelet_basis: str = "db4"
    wavelet_levels: int = 2
    wavelet_ms_channels: int = 24
    fusion_out_channels: int = 64
    dense_growth_rate: int = 32
    dense_layers: int = 3
    gnn_grid_size: int = 14
    gnn_hidden_dim: int = 128
    dropout: float = 0.1
    conf_loss_weight: float = 0.1
    uncert_loss_weight: float = 0.1
    memory_bank_size: int = 2048
    memory_select_ratio: float = 0.10
    stage1_end: int = 30
    stage2_end: int = 70
    stage2_mem_ratio: float = 0.30
    stage3_mem_ratio: float = 0.50
    alpha: float = 1.0
    beta: float = 1.0
    threshold: float = 0.5
    seed: int = 20261030
    device: str = "cuda:0"

    def validate(self):
        if self.alpha < 0 or self.beta < 0 or self.alpha + self.beta <= 0:
            raise ValueError("Difficulty weights must be nonnegative and not both zero")
        if self.wavelet_ms_channels != 12 * self.wavelet_levels:
            raise ValueError("Wavelet channels must equal 12 times the decomposition depth")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; explicitly select --device cpu for CPU execution")
        return self

    def to_dict(self):
        return asdict(self)

config = Config()
