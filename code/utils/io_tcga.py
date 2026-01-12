# root/code/utils/io_tcga.py
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, Any, List

def load_tcga_brca_pickles(data_dir: Path, omics: List[str]) -> Dict[str, Any]:
    """
    Expects files like:
      data_dir/ mRNA.pkl, DNAm.pkl, RPPA.pkl
    and each pickle contains keys like ["expr", "meta"].
    """
    out = {}
    for omic in omics:
        p = data_dir / f"{omic}.pkl"
        with p.open("rb") as f:
            out[omic] = pickle.load(f)
    return out
