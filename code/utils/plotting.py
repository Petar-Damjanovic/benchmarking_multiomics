# root/code/utils/plotting.py
from __future__ import annotations
from pathlib import Path
from matplotlib.patches import Patch
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_cv_boxplot(metrics: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = ["F1_micro", "F1_weighted", "Bal Accuracy", "Accuracy"]
    plt.figure(figsize=(8, 5))
    plt.boxplot(metrics, labels=names)
    plt.title(title)
    plt.ylabel("Score")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def save_top_features_barplot(df_top: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    df_top expected columns: label, score, modality
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = df_top.copy().sort_values("score", ascending=True)

    # --- color map for modalities (rna/dna/rppa) ---
    palette = {
        "rna":  "#1f77b4",  # blue
        "dna":  "#2ca02c",  # green
        "rppa": "#ff7f0e",  # orange
    }

    mods = df["modality"].astype(str).str.lower().str.strip()
    colors = mods.map(palette).fillna("#7f7f7f")  # gray fallback

    # Debug (optional): confirms mapping is active
    print("Legend modalities:", list(pd.unique(mods)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["label"].astype(str), df["score"].astype(float), color=colors)

    ax.set_xlabel("Absolute correlation with IntegrAO output")
    ax.set_ylabel("Feature")
    ax.set_title(title)

    # Legend only for modalities present
    pretty_label = {
        "rna": "mRNA",
        "dna": "DNAm",
        "rppa": "RPPA",
    }

    present = [m for m in pd.unique(mods) if m in palette]

    handles = [
        Patch(color=palette[m], label=pretty_label.get(m, m))
        for m in present
    ]

    if handles:
        ax.legend(
            handles=handles,          # <-- your custom modality handles
            title="Omic view",
            bbox_to_anchor=(1.02, 1), # outside to the right
            loc="upper left",
            borderaxespad=0.0,
        )


    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
