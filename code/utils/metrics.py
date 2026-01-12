# root/code/utils/metrics.py
from __future__ import annotations
from typing import Dict, Iterable, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

def preds_to_index(preds_index: Union[Dict, Iterable]) -> pd.Index:
    if isinstance(preds_index, dict):
        return pd.Index(list(preds_index.keys()), name="sample")
    return pd.Index(list(preds_index), name="sample")

def extract_pred_labels_1based(preds: np.ndarray,
                               preds_index: Union[Dict, Iterable],
                               sample_ids: Iterable) -> pd.Series:
    """
    IntegrAO commonly returns class indices (0..K-1) or an (N x K) logit/prob matrix.
    We convert to 1..K labels aligned to `sample_ids`.
    """
    idx = preds_to_index(preds_index)
    arr = np.asarray(preds)

    df = pd.DataFrame(arr, index=idx)
    df = df.loc[list(sample_ids)]

    if df.shape[1] == 1:
        yhat0 = df.iloc[:, 0].astype(int).values
    else:
        yhat0 = df.values.argmax(axis=1)

    return pd.Series(yhat0 + 1, index=df.index, name="pred_label")

def compute_fold_metrics(y_true_1based: np.ndarray, y_pred_1based: np.ndarray) -> Tuple[float, float, float, float]:
    f1_micro = f1_score(y_true_1based, y_pred_1based, average="micro")
    f1_weighted = f1_score(y_true_1based, y_pred_1based, average="weighted")
    bal_acc = balanced_accuracy_score(y_true_1based, y_pred_1based)
    acc = accuracy_score(y_true_1based, y_pred_1based)
    return f1_micro, f1_weighted, bal_acc, acc
