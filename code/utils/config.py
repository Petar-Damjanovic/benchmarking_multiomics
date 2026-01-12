# root/code/utils/config.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class IntegraoConfig:
    # IntegrAO hyperparams (your values)
    neighbor_size: int = 20
    embedding_dims: int = 64
    fusing_iteration: int = 30
    normalization_factor: float = 1.0
    alignment_epochs: int = 1000
    beta: float = 1.0
    mu: float = 0.5

    # Experiment
    n_splits: int = 5
    random_state: int = 42
    cluster_number: int = 4
    n_keep_features: int = 500
    finetune_epochs: int = 800
