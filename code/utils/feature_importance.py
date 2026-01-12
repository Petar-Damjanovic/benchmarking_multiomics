# root/code/utils/feature_importance.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Union, Iterable
import numpy as np
import pandas as pd

from .metrics import preds_to_index

@dataclass
class FeatureImportanceResult:
    table: pd.DataFrame        # combined
    top_k: pd.DataFrame        # top K rows

def preds_to_dataframe(preds: np.ndarray, preds_index) -> pd.DataFrame:
    idx = preds_to_index(preds_index)
    arr = np.asarray(preds)
    df = pd.DataFrame(arr, index=idx)
    if df.shape[1] == 1:
        df.columns = ["pred_class_0based"]
    else:
        df.columns = [f"logit_{i}" for i in range(df.shape[1])]
    return df

def compute_scalar_output(pred_df: pd.DataFrame, y_true_0based: np.ndarray) -> pd.Series:
    if "pred_class_0based" in pred_df.columns:
        return pred_df["pred_class_0based"].astype(float).rename("model_output")

    # else: use logit of true class
    tmp = pred_df.copy()
    tmp["true_class"] = y_true_0based
    def pick(row):
        c = int(row["true_class"])
        return row[f"logit_{c}"]
    out = tmp.apply(pick, axis=1).astype(float)
    out.name = "model_output"
    return out

def corr_with_output(X: pd.DataFrame, modality: str, target: pd.Series) -> pd.DataFrame:
    corrs = X.apply(lambda col: col.corr(target), axis=0)
    df = pd.DataFrame({"feature": corrs.index, "corr": corrs.values})
    df["modality"] = modality
    df["score"] = df["corr"].abs()
    return df.dropna(subset=["score"])

def add_symbols_for_rna_features(df: pd.DataFrame,
                                feature_col="feature",
                                modality_col="modality",
                                rna_modality_value="rna") -> pd.DataFrame:
    """
    Optional: requires `mygene` and internet.
    Falls back safely if mygene fails.
    """
    out = df.copy()
    out["label"] = out[feature_col].astype(str)
    out["symbol"] = pd.NA
    out["ensembl_base"] = pd.NA

    is_rna = out[modality_col].astype(str).str.lower().eq(str(rna_modality_value).lower())
    if is_rna.sum() == 0:
        return out

    base_ids = out.loc[is_rna, feature_col].astype(str).str.split(".").str[0]
    out.loc[is_rna, "ensembl_base"] = base_ids.values

    try:
        import mygene
        mg = mygene.MyGeneInfo()
        unique_ids = pd.Index(base_ids.unique())

        hits = mg.querymany(
            unique_ids.tolist(),
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            as_dataframe=True,
        )

        mapping = {}
        for ensg, row in hits.iterrows():
            sym = row.get("symbol")
            mapping[str(ensg)] = sym if isinstance(sym, str) and len(sym) > 0 else str(ensg)

        symbols = base_ids.map(mapping)
        out.loc[is_rna, "symbol"] = symbols.values
        out.loc[is_rna, "label"] = symbols.values
    except Exception:
        # keep Ensembl IDs as label if lookup fails
        pass

    return out

def global_corr_importance(
    views: Dict[str, pd.DataFrame],          # keys: rna/dna/rppa
    labels_1based: pd.Series,
    preds: np.ndarray,
    preds_index,
    top_k: int = 10,
) -> FeatureImportanceResult:
    pred_df = preds_to_dataframe(preds, preds_index)

    common = pred_df.index.intersection(labels_1based.index)
    for m, df in views.items():
        common = common.intersection(df.index)

    pred_df = pred_df.loc[common]
    y_true_0 = (labels_1based.loc[common].values - 1).astype(int)

    scalar = compute_scalar_output(pred_df, y_true_0)

    parts = []
    for modality, X in views.items():
        Xc = X.loc[common]
        parts.append(corr_with_output(Xc, modality, scalar))
    all_feat = pd.concat(parts, ignore_index=True)

    top = all_feat.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    top = add_symbols_for_rna_features(top, rna_modality_value="rna")

    return FeatureImportanceResult(table=all_feat, top_k=top)
