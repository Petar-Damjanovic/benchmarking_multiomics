# root/code/utils/preprocessing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

@dataclass
class OmicsViews:
    dna: pd.DataFrame     # DNAm
    rna: pd.DataFrame     # mRNA
    rppa: pd.DataFrame    # RPPA
    meta_union: pd.DataFrame

def build_union_views(data: Dict) -> OmicsViews:
    dna = data["DNAm"]["expr"]
    rna = data["mRNA"]["expr"]
    rppa = data["RPPA"]["expr"]

    all_union = dna.index.union(rna.index).union(rppa.index)

    dna_u = dna.reindex(all_union)
    rna_u = rna.reindex(all_union)
    rppa_u = rppa.reindex(all_union)

    dna_meta_u = data["DNAm"]["meta"].reindex(all_union)
    rna_meta_u = data["mRNA"]["meta"].reindex(all_union)
    rppa_meta_u = data["RPPA"]["meta"].reindex(all_union)

    meta_union = dna_meta_u.combine_first(rna_meta_u).combine_first(rppa_meta_u)
    return OmicsViews(dna=dna_u, rna=rna_u, rppa=rppa_u, meta_union=meta_union)

def encode_pam50(meta_union: pd.DataFrame,
                 label_col: str = "paper_BRCA_Subtype_PAM50") -> Tuple[pd.Series, List[str]]:
    """
    Returns:
      labels_1based: pd.Series index=subjects, values in {1..K}
      classes: label names in encoder order
    """
    labels_raw = meta_union[label_col]
    if labels_raw.isna().any():
        # Professional handling: drop unlabeled subjects for supervised training/eval
        labels_raw = labels_raw.dropna()

    le = LabelEncoder()
    y_1based = pd.Series(le.fit_transform(labels_raw) + 1, index=labels_raw.index, name="cluster_id")
    return y_1based, list(le.classes_)

def variance_feature_select(expr_df: pd.DataFrame, n_keep: int) -> pd.Index:
    var = expr_df.var(axis=0, skipna=True)
    return var.sort_values(ascending=False).head(n_keep).index

def select_features_for_views(views: Dict[str, pd.DataFrame], n_keep: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Index]]:
    """
    views keys expected: "dna","rna","rppa"
    """
    fs_views = {}
    feat_lists = {}
    for name, df in views.items():
        if name == "rppa":
            fs_views[name] = df.copy()
            feat_lists[name] = df.columns
        else:
            top = variance_feature_select(df, n_keep)
            fs_views[name] = df.loc[:, top]
            feat_lists[name] = top
    return fs_views, feat_lists
