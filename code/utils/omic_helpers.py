"""
omics_helpers.py

Unified helper module for the TCGA BRCA multi-omics experiments.

This file collects in a single place:

- basic utilities
  * z-scoring of DataFrames
  * concatenation of omic views with optional block scaling

- PCA / embedding helpers
  * per-view PCA and PCA on top-K most variable features
  * variance explained per view
  * pairwise silhouette scores and correlation matrices between embeddings
  * per-factor R² across views

- MOFA-related helpers
  * 2D / 3D plots of MOFA factors
  * factor–class correlation matrices
  * feature-importance rankings (global and per modality) based on MOFA weights

- labels
  * PAM50 extraction from per-view meta
  * combined PAM50_any labels across views

- per-feature statistics
  * ANOVA per feature (F, p, η²)
  * variance per feature

- feature selection
  * variance- and ANOVA-based SelectKBest score functions
  * small “view blocks” for sklearn Pipelines (ANOVA / most-variable)
  * helpers to select top features per view (same k or per-view k)

- plotting helpers
  * 2D / 3D scatter plots for PCA and MOFA embeddings

- classification baselines
  * logistic regression on precomputed factor embeddings (e.g. MOFA / PCA)
  * PCA + logistic regression on concatenated raw features (no leakage)
  * generic GridSearchCV wrapper with reporting
  * late-fusion (per-view) logistic regression baseline
"""


from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.linalg import lstsq

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import (
    silhouette_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LogisticRegression
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def zscore_df(
    df: pd.DataFrame,
    with_mean: bool = True,
    with_std: bool = True,
) -> pd.DataFrame:
    """
    Z-score a DataFrame column-wise using sklearn's StandardScaler.

    Parameters
    ----------
    df : DataFrame (N x P)
    with_mean : bool
    with_std : bool

    Returns
    -------
    DataFrame (N x P) z-scored per feature.
    """
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    vals = scaler.fit_transform(df.values)
    return pd.DataFrame(vals, index=df.index, columns=df.columns)


# ---------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------

def run_pca_view(
    data: dict,
    view_name: str,
    patients=None,
    n_components: int = 10,
):
    """
    Run PCA on one view (e.g. "mRNA", "DNAm", "RPPA").

    Parameters
    ----------
    data : dict
        multi-omics dict, data[view_name]["expr"] is a DataFrame.
    view_name : str
    patients : list-like or index, optional
        If given, restrict rows to these patient IDs.
    n_components : int

    Returns
    -------
    pca : sklearn.decomposition.PCA
    scores_df : DataFrame (N x K)
        Sample scores (embedding).
    load_df : DataFrame (P x K)
        Feature loadings.
    """
    X = data[view_name]["expr"]
    if patients is not None:
        X = X.loc[patients]

    X_z = zscore_df(X)

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X_z.values)   # N x K
    loadings = pca.components_.T             # P x K

    pc_names = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=X.index, columns=pc_names)
    load_df = pd.DataFrame(loadings, index=X.columns, columns=pc_names)
    return pca, scores_df, load_df


def get_concat_matrix(
    data: dict,
    patients,
    block_scale: bool = True,
) -> pd.DataFrame:
    """
    Concatenate mRNA, DNAm, RPPA matrices for the same patients.

    Each view is z-scored per feature; if block_scale=True, each block
    is divided by sqrt(#features) so that large views don't dominate.

    Parameters
    ----------
    data : dict
    patients : list-like
    block_scale : bool

    Returns
    -------
    X_concat : DataFrame (N x sum(P_view))
    """
    X_rna = data["mRNA"]["expr"].loc[patients]
    X_meth = data["DNAm"]["expr"].loc[patients]
    X_prot = data["RPPA"]["expr"].loc[patients]

    X_rna_z = zscore_df(X_rna)
    X_meth_z = zscore_df(X_meth)
    X_prot_z = zscore_df(X_prot)

    if block_scale:
        X_rna_z = X_rna_z / np.sqrt(X_rna_z.shape[1])
        X_meth_z = X_meth_z / np.sqrt(X_meth_z.shape[1])
        X_prot_z = X_prot_z / np.sqrt(X_prot_z.shape[1])

    X_concat = pd.concat([X_rna_z, X_meth_z, X_prot_z], axis=1)
    return X_concat


def run_pca_view_topKvar(
    data: dict,
    view_name: str,
    patients,
    K_feat: int,
    K_pca: int,
):
    """
    PCA on the top K_feat most variable features, with K_pca components.

    Parameters
    ----------
    data : dict
        data[view_name]["expr"] must be DataFrame (N x P).
    view_name : str
    patients : index-like
    K_feat : int
        Number of most-variable features to keep.
    K_pca : int
        Number of PCA components.

    Returns
    -------
    pca : PCA
    scores : ndarray (N x n_components)
    loadings : ndarray (K_feat x n_components)
    top_cols : Index of selected feature names
    """
    X_df = data[view_name]["expr"].loc[patients]    # (N x P)

    var = X_df.var(axis=0)
    top_cols = var.sort_values(ascending=False).index[:K_feat]
    X_top = X_df[top_cols].values                  # (N x K_feat)

    X_scaled = StandardScaler().fit_transform(X_top)

    n_components = min(K_pca, X_scaled.shape[0], X_scaled.shape[1])

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)           # N x n_components
    loadings = pca.components_.T                   # K_feat x n_components

    return pca, scores, loadings, top_cols


# ---------------------------------------------------------------------
# Numerical comparison helpers
# ---------------------------------------------------------------------

def variance_explained_view(
    X_df: pd.DataFrame,
    Z_df: pd.DataFrame,
) -> float:
    """
    Fraction of total variance in X (view) that can be reconstructed
    from embedding Z using linear least squares.

    Parameters
    ----------
    X_df : DataFrame (N x P)
    Z_df : DataFrame (N x K)

    Returns
    -------
    R2 : float in [0, 1]
    """
    common = X_df.index.intersection(Z_df.index)
    X = X_df.loc[common].values      # N x P
    Z = Z_df.loc[common].values      # N x K

    X_centered = X - X.mean(axis=0, keepdims=True)

    W, *_ = lstsq(Z, X_centered, rcond=None)  # K x P
    X_hat = Z @ W                             # N x P

    resid = X_centered - X_hat
    sse = np.sum(resid ** 2)
    sst = np.sum(X_centered ** 2)
    return 1.0 - sse / sst


def silhouette_in_embedding(
    Z_df: pd.DataFrame,
    labels: pd.Series,
    n_dims: int = 2,
) -> float:
    """
    Silhouette score of `labels` in first n_dims of embedding Z.
    """
    labels = labels.dropna()
    common = Z_df.index.intersection(labels.index)
    Z = Z_df.loc[common].iloc[:, :n_dims].values
    y = labels.loc[common].values
    return silhouette_score(Z, y, metric="euclidean")


def corr_matrix(
    Z1_df: pd.DataFrame,
    Z2_df: pd.DataFrame,
    n1: int | None = None,
    n2: int | None = None,
) -> pd.DataFrame:
    """
    Correlation between columns of two embeddings (e.g. PCs vs MOFA).

    Returns a DataFrame (K1_used x K2_used).
    """
    common = Z1_df.index.intersection(Z2_df.index)
    A = Z1_df.loc[common]
    B = Z2_df.loc[common]

    if n1 is not None:
        A = A.iloc[:, :n1]
    if n2 is not None:
        B = B.iloc[:, :n2]

    C = np.corrcoef(A.values.T, B.values.T)
    nA = A.shape[1]
    corr_AB = C[:nA, nA:]
    return pd.DataFrame(corr_AB, index=A.columns, columns=B.columns)


def pairwise_silhouette_views(
    embeddings: dict,
    labels: pd.Series,
    n_dims: int = 2,
) -> pd.DataFrame:
    """
    Pairwise silhouette for every unordered label pair, for each embedding.

    Parameters
    ----------
    embeddings : dict {name: scores_df}
    labels : Series (e.g. PAM50)
    n_dims : int

    Returns
    -------
    DataFrame:
        index = embedding names
        columns = 'class1 vs class2'
    """
    from itertools import combinations

    lbl = labels.dropna()
    classes = sorted(lbl.unique())
    pairs = list(combinations(classes, 2))

    col_names = [f"{a} vs {b}" for (a, b) in pairs]
    res = pd.DataFrame(index=embeddings.keys(), columns=col_names, dtype=float)

    for emb_name, Z_df in embeddings.items():
        common = Z_df.index.intersection(lbl.index)
        Z_all = Z_df.loc[common].iloc[:, :n_dims]
        y_all = lbl.loc[common]

        for (a, b), col in zip(pairs, col_names):
            mask = y_all.isin([a, b])
            Z_pair = Z_all[mask]
            y_pair = y_all[mask]

            counts = y_pair.value_counts()
            if len(counts) < 2 or (counts < 2).any():
                res.loc[emb_name, col] = np.nan
                continue

            try:
                score = silhouette_score(Z_pair.values, y_pair.values,
                                         metric="euclidean")
            except Exception:
                score = np.nan
            res.loc[emb_name, col] = score

    return res


def per_factor_r2_matrix(
    views: dict[str, pd.DataFrame],
    Z_df: pd.DataFrame,
    n_factors: int = 15,
) -> pd.DataFrame:
    """
    Compute R^2 per factor (column of Z_df) and per view.

    Parameters
    ----------
    views : dict
        {view_name: X_df} with X_df (N x P).
    Z_df : DataFrame
        Embedding (N x K), e.g. PCA scores.
    n_factors : int
        Max number of factors/PCs to use (starting from PC1).

    Returns
    -------
    DataFrame (n_factors_used x n_views)
        index  = Factor1, Factor2, ...
        columns = view names
        values = R^2 (0..1)
    """
    K = min(n_factors, Z_df.shape[1])
    factor_names = [f"Factor{i+1}" for i in range(K)]
    view_names = list(views.keys())
    R2 = pd.DataFrame(index=factor_names, columns=view_names, dtype=float)

    for vname, X_df in views.items():
        common = X_df.index.intersection(Z_df.index)
        X = X_df.loc[common].values             # N x P
        X_centered = X - X.mean(axis=0, keepdims=True)

        Z = Z_df.loc[common].iloc[:, :K].values # N x K

        for k in range(K):
            z_k = Z[:, [k]]                     # N x 1
            W_k, *_ = lstsq(z_k, X_centered, rcond=None)  # (1 x P)
            X_hat_k = z_k @ W_k                              # N x P

            resid_k = X_centered - X_hat_k
            sse = np.sum(resid_k ** 2)
            sst = np.sum(X_centered ** 2)
            R2.iloc[k, R2.columns.get_loc(vname)] = 1.0 - sse / sst

    return R2


# ---------------------------------------------------------------------
# PAM50 label extraction and combinations
# ---------------------------------------------------------------------

def get_pam50(data: dict, view_name: str) -> pd.Series:
    """
    Extract PAM50 labels from the meta table of a given view.

    Parameters
    ----------
    data : dict
        data[view_name]["meta"] must exist.
    view_name : {"mRNA", "DNAm", "RPPA", ...}

    Returns
    -------
    s : pd.Series
        Index = patient IDs, values = PAM50 labels (str or NaN).
    """
    meta = data[view_name].get("meta", None)
    if meta is None or "paper_BRCA_Subtype_PAM50" not in meta.columns:
        return pd.Series(index=[], dtype="object")

    s = meta["paper_BRCA_Subtype_PAM50"].astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    s.name = view_name
    return s


def get_pam50_any(
    data: dict,
    label_views: tuple[str, ...] = ("mRNA", "DNAm", "RPPA"),
) -> pd.Series:
    """
    Combine PAM50 labels across multiple views into one Series.

    The first non-missing label across the views is kept for each sample.
    """
    combined: pd.Series | None = None
    for v in label_views:
        if v not in data:
            continue
        s = get_pam50(data, v)
        if combined is None:
            combined = s.copy()
        else:
            combined = combined.combine_first(s)

    if combined is None:
        return pd.Series(index=[], dtype="object")

    combined.name = "PAM50_any"
    return combined


# ---------------------------------------------------------------------
# Per-feature ANOVA / variance tables (for one view)
# ---------------------------------------------------------------------

def per_feature_anova_np(
    X: np.ndarray,               # shape (N, P)
    y: np.ndarray,               # shape (N,), integer classes
    view_name: str,              # e.g. "mRNA", "DNAm", "RPPA"
    topN: int = 30,              # kept for backwards compatibility (not used)
    topK_scatter: int = 3000,    # kept for backwards compatibility (not used)
    save_prefix: str | None = None,
    feature_names: list[str] | None = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compute ANOVA F, p and eta^2 per feature for one view.

    Returns
    -------
    res : DataFrame
        Columns: ["feature", "F", "p", "eta2", "rank", "view"].
        Sorted in descending eta2.
    """
    N, P = X.shape
    K = len(np.unique(y))

    if feature_names is None:
        width = len(str(P))
        feature_names = [f"{view_name.lower()}_{i:0{width}d}" for i in range(P)]

    F, p = f_classif(X, y)  # length P
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)

    eta2 = ((K - 1) * F) / (((K - 1) * F) + (N - K) + 1e-12)

    res = pd.DataFrame(
        {
            "feature": feature_names,
            "F": F,
            "p": p,
            "eta2": eta2,
        }
    ).sort_values("eta2", ascending=False).reset_index(drop=True)

    res["rank"] = np.arange(1, P + 1)
    res["view"] = view_name

    if plot:
        plt.figure()
        plt.hist(res["eta2"].values, bins=50)
        plt.title(f"{view_name}: η² distribution (N={N}, K={K}, P={P})")
        plt.xlabel("eta²")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

    if save_prefix is not None:
        res.to_csv(f"{save_prefix}_{view_name}_anova.csv", index=False)

    return res


def per_feature_variance_np(
    X: np.ndarray,
    view_name: str,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compute per-feature variance for one view and (optionally) plot its
    distribution.

    Returns
    -------
    res : DataFrame with columns ["feature", "var", "rank", "view"].
    """
    N, P = X.shape

    var = np.var(X, axis=0)
    res = pd.DataFrame(
        {
            "feature": [
                f"{view_name.lower()}_{i:0{len(str(P))}d}" for i in range(P)
            ],
            "var": var,
        }
    ).sort_values("var", ascending=False).reset_index(drop=True)

    res["rank"] = np.arange(1, P + 1)
    res["view"] = view_name

    if plot:
        plt.figure()
        plt.hist(res["var"].values, bins=50)
        plt.title(f"{view_name}: variance distribution (N={N}, P={P})")
        plt.xlabel("variance")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

    return res


# ---------------------------------------------------------------------
# Feature-selection score + view blocks (for sklearn Pipelines)
# ---------------------------------------------------------------------

def variance_score(
    X: np.ndarray,
    y: np.ndarray | None = None,
):
    """
    Score function for SelectKBest: feature variance (higher = better).

    Parameters
    ----------
    X : array-like, shape (N, P)
    y : ignored (for compatibility with SelectKBest)

    Returns
    -------
    scores : np.ndarray, shape (P,)
        Variance of each feature.
    pvalues : np.ndarray, shape (P,)
        Dummy NaNs (not used by SelectKBest).
    """
    scores = np.var(X, axis=0)
    pvalues = np.full_like(scores, np.nan, dtype=float)
    return scores, pvalues


def _make_view_block(
    score_func,
    k: int | None = None,
    scale_before: bool = True,
    step_name: str = "kbest",
) -> Pipeline:
    """
    Generic helper: scaling + SelectKBest(score_func, k).
    """
    steps: list[tuple[str, object]] = []

    if scale_before:
        steps.append(("scale", StandardScaler()))

    if k is not None:
        steps.append((step_name, SelectKBest(score_func=score_func, k=k)))

    if not scale_before:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)


def view_block_anova(k: int | None = None) -> Pipeline:
    """
    View block using ANOVA F-score (f_classif) for feature ranking.
    """
    return _make_view_block(
        score_func=f_classif,
        k=k,
        scale_before=True,
        step_name="anova",
    )


def view_block_mostVar(k: int | None = None) -> Pipeline:
    """
    View block selecting the k most variable features.
    """
    return _make_view_block(
        score_func=variance_score,
        k=k,
        scale_before=False,
        step_name="var",
    )


# ---------------------------------------------------------------------
# Feature selection on full data dict (per view)
# ---------------------------------------------------------------------

def select_top_variable_features(
    data: dict,
    n_keep: int = 2000,
    views: list[str] | None = None,
    use: str = "var",
):
    """
    Select top `n_keep` most variable features per view.

    Parameters
    ----------
    data : dict
        data[view]["expr"] must be a DataFrame (samples x features).
    n_keep : int
        Maximum number of features to keep per view.
    views : list of str or None
    use : {"var", "mad"}

    Returns
    -------
    filtered_data : dict
    feature_indices : dict
        view -> Index of selected feature names.
    """
    if views is None:
        views = list(data.keys())

    filtered_data = {}
    feature_indices = {}

    for v in views:
        X = data[v]["expr"]  # samples x features

        if use == "var":
            var = X.var(axis=0, ddof=1)
        elif use == "mad":
            med = X.median(axis=0)
            var = (X - med).abs().median(axis=0)
        else:
            raise ValueError("use must be 'var' or 'mad'")

        var = var.dropna()

        k = min(n_keep, var.shape[0])
        top_feats = var.sort_values(ascending=False).head(k).index

        feature_indices[v] = top_feats

        new_view = dict(data[v])               # shallow copy
        new_view["expr"] = data[v]["expr"][top_feats]
        filtered_data[v] = new_view

        print(f"{v}: kept {k} / {X.shape[1]} features (most variable, {use})")

    return filtered_data, feature_indices


def select_top_variable_features_per_view(
    data: dict,
    n_keep_per_view: dict,
    views: list[str] | None = None,
    use: str = "var",
):
    """
    Like `select_top_variable_features`, but different #features per view.

    Parameters
    ----------
    data : dict
    n_keep_per_view : dict
        e.g. {"mRNA": 2000, "DNAm": 2000, "RPPA": 464}.
    views : list[str] or None
    use : {"var", "mad"}

    Returns
    -------
    filtered_data : dict
    feature_indices : dict
    """
    if views is None:
        views = list(n_keep_per_view.keys())

    filtered_data = {}
    feature_indices = {}

    for v in views:
        if v not in data:
            raise KeyError(f"View '{v}' not found in data.")
        if v not in n_keep_per_view:
            raise KeyError(f"n_keep_per_view has no entry for view '{v}'.")

        X = data[v]["expr"]

        if use == "var":
            var = X.var(axis=0, ddof=1)
        elif use == "mad":
            med = X.median(axis=0)
            var = (X - med).abs().median(axis=0)
        else:
            raise ValueError("use must be 'var' or 'mad'")

        var = var.dropna()
        k = min(int(n_keep_per_view[v]), var.shape[0])
        top_feats = var.sort_values(ascending=False).head(k).index

        feature_indices[v] = top_feats

        new_view = dict(data[v])
        new_view["expr"] = data[v]["expr"][top_feats]
        filtered_data[v] = new_view

        print(f"{v}: kept {k} / {X.shape[1]} features (most variable, {use})")

    return filtered_data, feature_indices


def select_top_anova_features(
    data: dict,
    n_keep: int = 2000,
    views: list[str] | None = None,
    label_view: str = "mRNA",
):
    """
    Select top `n_keep` features per view using ANOVA F-score wrt PAM50.

    Parameters
    ----------
    data : dict
    n_keep : int
    views : list[str] or None
    label_view : str

    Returns
    -------
    filtered_data : dict
    feature_indices : dict
    """
    if views is None:
        views = list(data.keys())

    y_all = get_pam50(data, label_view).dropna()
    if y_all.empty:
        raise ValueError(
            f"No PAM50 labels found for view '{label_view}' "
            "in ANOVA feature selection."
        )

    filtered_data: dict = {}
    feature_indices: dict = {}

    for v in views:
        X = data[v]["expr"]  # samples x features

        common = X.index.intersection(y_all.index)
        if len(common) == 0:
            raise ValueError(
                f"No overlapping samples between '{v}' and label view "
                f"'{label_view}'."
            )

        X_sub = X.loc[common]
        y_sub = y_all.loc[common]

        var = X_sub.var(axis=0, ddof=1)
        good_cols = var[var > 0].index
        X_sub = X_sub[good_cols]

        X_sub = X_sub.fillna(X_sub.mean(axis=0))

        f_vals, _ = f_classif(X_sub.values, y_sub.values)
        scores = pd.Series(f_vals, index=good_cols)

        scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        k = min(n_keep, scores.shape[0])
        top_feats = scores.sort_values(ascending=False).head(k).index

        feature_indices[v] = top_feats

        new_view = dict(data[v])
        new_view["expr"] = data[v]["expr"][top_feats]
        filtered_data[v] = new_view

        print(f"{v}: kept {k} / {X.shape[1]} features (ANOVA)")

    return filtered_data, feature_indices


def select_top_anova_features_per_view(
    data: dict,
    n_keep_per_view: dict,
    views: list[str] | None = None,
    label_view: str = "mRNA",
):
    """
    Select top features per view using ANOVA F-score w.r.t. PAM50 labels.

    Parameters
    ----------
    data : dict
    n_keep_per_view : dict
    views : list[str] or None
    label_view : str

    Returns
    -------
    filtered_data : dict
    feature_indices : dict
    """
    if views is None:
        views = list(n_keep_per_view.keys())

    y_all = get_pam50(data, label_view).dropna()
    if y_all.empty:
        raise ValueError(
            f"No PAM50 labels found for view '{label_view}' in ANOVA selection."
        )

    filtered_data = {}
    feature_indices = {}

    for v in views:
        if v not in data:
            raise KeyError(f"View '{v}' not found in data.")
        if v not in n_keep_per_view:
            raise KeyError(f"n_keep_per_view has no entry for view '{v}'.")

        X = data[v]["expr"]

        common = X.index.intersection(y_all.index)
        if len(common) == 0:
            raise ValueError(
                f"No overlapping samples between '{v}' and label view '{label_view}'."
            )

        X_sub = X.loc[common]
        y_sub = y_all.loc[common]

        var = X_sub.var(axis=0, ddof=1)
        good_cols = var[var > 0].index
        X_sub = X_sub[good_cols]

        X_sub = X_sub.fillna(X_sub.mean(axis=0))

        f_vals, _ = f_classif(X_sub.values, y_sub.values)
        scores = pd.Series(f_vals, index=good_cols)
        scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        k = min(int(n_keep_per_view[v]), scores.shape[0])
        top_feats = scores.sort_values(ascending=False).head(k).index

        feature_indices[v] = top_feats

        new_view = dict(data[v])
        new_view["expr"] = data[v]["expr"][top_feats]
        filtered_data[v] = new_view

        print(f"{v}: kept {k} / {X.shape[1]} features (ANOVA)")

    return filtered_data, feature_indices


# ---------------------------------------------------------------------
# Plotting helpers: PCA / MOFA
# ---------------------------------------------------------------------

def plot_pca_2d(
    scores_df: pd.DataFrame,
    labels: pd.Series | None = None,
    x_pc: int = 1,
    y_pc: int = 2,
    title: str | None = None,
    hue_name: str | None = None,
    figsize=(6, 5),
):
    """
    2D scatter of two PCs/factors with optional coloring by labels.
    """
    x_col = f"PC{x_pc}"
    y_col = f"PC{y_pc}"

    if labels is not None:
        if hue_name is None:
            hue_name = labels.name if labels.name is not None else "label"
        df = scores_df.join(labels.rename(hue_name)).dropna()
    else:
        df = scores_df.copy()
        hue_name = None

    plt.figure(figsize=figsize)
    if hue_name is not None:
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue_name,
            s=40,
            alpha=0.8,
        )
    else:
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            s=40,
            alpha=0.8,
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pca_3d(
    scores_df: pd.DataFrame,
    labels: pd.Series | None = None,
    pcs=(1, 2, 3),
    title: str | None = None,
    hue_name: str | None = None,
    figsize=(7, 6),
):
    """
    3D scatter of three PCs/factors with optional coloring by labels.
    """
    x_pc, y_pc, z_pc = pcs
    x_col = f"PC{x_pc}"
    y_col = f"PC{y_pc}"
    z_col = f"PC{z_pc}"

    if labels is not None:
        if hue_name is None:
            hue_name = labels.name if labels.name is not None else "label"
        df = scores_df.join(labels.rename(hue_name)).dropna()
    else:
        df = scores_df.copy()
        hue_name = None

    x = df[x_col]
    y = df[y_col]
    z = df[z_col]

    if hue_name is not None:
        lab = df[hue_name]
        classes = lab.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    else:
        lab = None
        classes = [None]
        colors = [plt.cm.tab10(0.0)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if hue_name is not None:
        for c, col in zip(classes, colors):
            mask = (lab == c)
            ax.scatter(x[mask], y[mask], z[mask],
                       label=c, s=40, alpha=0.8, color=col)
        ax.legend()
    else:
        ax.scatter(x, y, z, s=40, alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_mofa_2d(
    scores_df: pd.DataFrame,
    labels: pd.Series | None = None,
    x_f: int = 1,
    y_f: int = 2,
    title: str | None = None,
    savepath: str | None = None,
):
    """
    2D scatter of two MOFA factors with optional coloring by labels.
    """
    x_col = f"Factor{x_f}"
    y_col = f"Factor{y_f}"

    if labels is not None:
        hue_name = labels.name or "label"
        df = scores_df.join(labels.rename(hue_name)).dropna()
    else:
        df = scores_df.copy()
        hue_name = None

    fig, ax = plt.subplots(figsize=(6, 5))

    if hue_name is not None:
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue_name,
            s=40,
            alpha=0.8,
            ax=ax,
        )
    else:
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            s=40,
            alpha=0.8,
            ax=ax,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_mofa_3d(
    scores_df: pd.DataFrame,
    labels: pd.Series | None = None,
    factors: tuple[int, int, int] = (1, 2, 3),
    title: str | None = None,
    figsize=(7, 6),
):
    """
    3D scatter of three MOFA factors with optional coloring by labels.
    """
    f1, f2, f3 = factors
    x_col = f"Factor{f1}"
    y_col = f"Factor{f2}"
    z_col = f"Factor{f3}"

    if labels is not None:
        hue_name = labels.name or "label"
        df = scores_df.join(labels.rename(hue_name)).dropna()
    else:
        df = scores_df.copy()
        hue_name = None

    x = df[x_col]
    y = df[y_col]
    z = df[z_col]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if hue_name is not None:
        lab = df[hue_name]
        classes = lab.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        for c, col in zip(classes, colors):
            mask = lab == c
            ax.scatter(x[mask], y[mask], z[mask],
                       label=c, s=40, alpha=0.8, color=col)
        ax.legend()
    else:
        ax.scatter(x, y, z, s=40, alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Factor–class correlation
# ---------------------------------------------------------------------

def factor_class_correlation_matrix(
    Z_df: pd.DataFrame,
    labels: pd.Series,
    n_factors: int | None = None,
) -> pd.DataFrame:
    """
    Correlation between each factor/PC and each class indicator.

    For each class c we build a 0/1 indicator and compute the Pearson
    correlation with each factor column in Z_df.

    Returns
    -------
    DataFrame (n_factors_used x n_classes)
    """
    labels = labels.dropna()
    common = Z_df.index.intersection(labels.index)
    Z = Z_df.loc[common]
    y = labels.loc[common]

    if n_factors is not None:
        Z = Z.iloc[:, :n_factors]

    classes = sorted(y.unique())
    corr = pd.DataFrame(index=Z.columns, columns=classes, dtype=float)

    for c in classes:
        m = (y == c).astype(float).values
        for factor in Z.columns:
            z = Z[factor].values
            if np.all(z == z[0]) or np.all(m == m[0]):
                r = np.nan
            else:
                r = np.corrcoef(z, m)[0, 1]
            corr.loc[factor, c] = r

    return corr


# ---------------------------------------------------------------------
# Logistic regression on factors / PCA
# ---------------------------------------------------------------------

def run_logreg_on_factors(
    X_factors: pd.DataFrame,
    labels_any: pd.Series,
    title: str,
    verbose: bool = True,
):
    """
    Logistic regression with CV on factor embeddings (e.g. MOFA / PCA).

    Returns
    -------
    dict with keys:
        - title
        - best_params
        - mean_cv_bal_acc
        - std_cv_bal_acc
        - test_bal_acc
        - test_acc
        - test_roc_auc_ovr
    """
    y = labels_any.reindex(X_factors.index)
    mask = y.notna()
    X = X_factors.loc[mask]
    y = y.loc[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                # multinomial logistic regression
                __import__("sklearn.linear_model").linear_model.LogisticRegression(
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=5000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l1", "l2"],
    }

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    gs.fit(Xtr, ytr)

    best_params = gs.best_params_
    mean_cv = gs.best_score_
    std_cv = gs.cv_results_["std_test_score"][gs.best_index_]

    if verbose:
        print(f"{title}: best params {best_params}")
        print(f"CV balanced accuracy: {mean_cv:.3f} ± {std_cv:.3f}")

    y_pred = gs.predict(Xte)
    proba = gs.predict_proba(Xte)

    bal_acc_test = balanced_accuracy_score(yte, y_pred)
    acc_test = accuracy_score(yte, y_pred)
    roc_auc_ovr = roc_auc_score(
        yte, proba, multi_class="ovr", average="weighted"
    )

    if verbose:
        print(f"Test balanced accuracy: {bal_acc_test:.3f}")
        print(f"Test accuracy:        {acc_test:.3f}")
        print(f"Test ROC-AUC (OvR):   {roc_auc_ovr:.3f}")
        print("\nClassification report:")
        print(classification_report(yte, y_pred, target_names=le.classes_))

        cm = confusion_matrix(yte, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(ax=ax, cmap="viridis", colorbar=True)
        ax.grid(False)
        plt.tight_layout()

        safe_title = title.replace(" ", "_")
        fig.savefig(
            f"confmat_{safe_title}.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

    return {
        "title": title,
        "best_params": best_params,
        "mean_cv_bal_acc": mean_cv,
        "std_cv_bal_acc": std_cv,
        "test_bal_acc": bal_acc_test,
        "test_acc": acc_test,
        "test_roc_auc_ovr": roc_auc_ovr,
    }


def run_logreg_with_pca(
    X_concat: pd.DataFrame,
    labels_any: pd.Series,
    title: str,
    n_components: int = 15,
    verbose: bool = True,
):
    """
    Logistic regression with PCA (no data leakage) on concatenated features.

    Pipeline: StandardScaler -> PCA(n_components) -> multinomial LR.
    """
    y = labels_any.reindex(X_concat.index)
    mask = y.notna()
    X = X_concat.loc[mask]
    y = y.loc[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            (
                "clf",
                __import__("sklearn.linear_model").linear_model.LogisticRegression(
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=5000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l1", "l2"],
    }

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    gs.fit(Xtr, ytr)

    best_params = gs.best_params_
    mean_cv = gs.best_score_
    std_cv = gs.cv_results_["std_test_score"][gs.best_index_]

    if verbose:
        print(f"{title} (PCA {n_components}): best params {best_params}")
        print(f"CV balanced accuracy: {mean_cv:.3f} ± {std_cv:.3f}")

    y_pred = gs.predict(Xte)
    proba = gs.predict_proba(Xte)

    bal_acc_test = balanced_accuracy_score(yte, y_pred)
    acc_test = accuracy_score(yte, y_pred)
    roc_auc_ovr = roc_auc_score(
        yte, proba, multi_class="ovr", average="weighted"
    )

    if verbose:
        print(f"Test balanced accuracy: {bal_acc_test:.3f}")
        print(f"Test accuracy:        {acc_test:.3f}")
        print(f"Test ROC-AUC (OvR):   {roc_auc_ovr:.3f}")
        print("\nClassification report:")
        print(classification_report(yte, y_pred, target_names=le.classes_))

        cm = confusion_matrix(yte, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(ax=ax, cmap="viridis", colorbar=True)
        ax.grid(False)
        plt.title(f"Confusion matrix – {title} (PCA {n_components})")
        plt.tight_layout()

        safe_title = title.replace(" ", "_")
        fig.savefig(
            f"confmat_{safe_title}_PCA{n_components}.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

    return {
        "title": f"{title} (PCA {n_components})",
        "best_params": best_params,
        "mean_cv_bal_acc": mean_cv,
        "std_cv_bal_acc": std_cv,
        "test_bal_acc": bal_acc_test,
        "test_acc": acc_test,
        "test_roc_auc_ovr": roc_auc_ovr,
    }


def top_mofa_features_global(
    model,
    mdata,
    modalities: list[str] | None = None,
    top_n: int = 20,
    agg: str = "max",
) -> pd.DataFrame:
    """
    Top-N most informative MOFA features across *all* modalities.

    Parameters
    ----------
    model : mofax.mofa_model
        Loaded MOFA model.
    mdata : mu.MuData
        MuData object built from the same filtered views.
    modalities : list of str or None
        Modalities to include (e.g. ["rna", "dna", "rppa"]).
        If None, use mdata.mod keys.
    top_n : int
        Number of features to return in total.
    agg : {"max", "l2"}
        How to aggregate loadings across factors for each feature:
        - "max": max absolute weight over factors
        - "l2" : L2-norm of weights over factors

    Returns
    -------
    DataFrame with columns:
        - modality
        - feature
        - score   (importance score)
    """
    if modalities is None:
        modalities = list(mdata.mod.keys())

    rows = []

    for mod in modalities:
        # weights for THIS modality: (n_features x n_factors)
        W = model.get_weights(mod)

        # feature names for this modality
        feat_names = mdata[mod].var_names

        if agg == "max":
            scores = np.max(np.abs(W), axis=1)
        elif agg == "l2":
            scores = np.linalg.norm(W, axis=1)
        else:
            raise ValueError("agg must be 'max' or 'l2'")

        df_mod = pd.DataFrame(
            {
                "modality": mod,
                "feature": feat_names,
                "score": scores,
            }
        )
        rows.append(df_mod)

    all_feats = pd.concat(rows, ignore_index=True)
    all_feats = all_feats.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])

    top_global = (
        all_feats.sort_values("score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top_global
    
    
def top_mofa_features_factor(
    model,
    mdata,
    factor: int,
    modalities: list[str] | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Top-N most informative MOFA features for a *single* factor across modalities.

    Parameters
    ----------
    model : mofax.mofa_model
        Loaded MOFA model.
    mdata : mu.MuData
        MuData object built from the same filtered views.
    factor : int
        Index of the factor to inspect (0-based).
    modalities : list of str or None
        Modalities to include (e.g. ["rna", "dna", "rppa"]).
        If None, use mdata.mod keys.
    top_n : int
        Number of features to return in total.

    Returns
    -------
    DataFrame with columns:
        - modality
        - feature
        - factor      (factor index)
        - loading     (raw loading for that factor)
        - score       (|loading|, used for ranking)
    """
    if modalities is None:
        modalities = list(mdata.mod.keys())

    rows = []

    for mod in modalities:
        # weights for THIS modality: (n_features x n_factors)
        W = model.get_weights(mod)  # assumed ndarray

        n_features, n_factors = W.shape
        if not (0 <= factor < n_factors):
            raise ValueError(
                f"Factor index {factor} is out of bounds for modality '{mod}' "
                f"with {n_factors} factors."
            )

        # feature names for this modality
        feat_names = mdata[mod].var_names

        # loading for this factor
        loading = W[:, factor]
        scores = np.abs(loading)

        df_mod = pd.DataFrame(
            {
                "modality": mod,
                "feature": feat_names,
                "factor": factor,
                "loading": loading,
                "score": scores,
            }
        )
        rows.append(df_mod)

    all_feats = pd.concat(rows, ignore_index=True)
    all_feats = all_feats.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])

    top_factor = (
        all_feats.sort_values("score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top_factor
    
  
def top_mofa_features_per_view(
    model,
    mdata,
    modality: str,
    top_n: int = 30,
    agg: str = "max",
) -> pd.DataFrame:
    """
    Top-N most informative MOFA features *within one modality*.

    Parameters
    ----------
    model : mofax.mofa_model
    mdata : mu.MuData
    modality : str
        Modality key in mdata (e.g. "rna", "dna", "rppa").
    top_n : int
        Number of features to keep for this modality.
    agg : {"max", "l2"}
        How to aggregate loadings across factors per feature:
        - "max": max absolute loading over factors
        - "l2" : L2 norm over factors

    Returns
    -------
    DataFrame with columns: feature, score.
    """
    W = model.get_weights(modality)         # (n_features x n_factors)
    feat_names = mdata[modality].var_names

    if agg == "max":
        scores = np.max(np.abs(W), axis=1)
    elif agg == "l2":
        scores = np.linalg.norm(W, axis=1)
    else:
        raise ValueError("agg must be 'max' or 'l2'")

    df = pd.DataFrame(
        {"feature": feat_names, "score": scores}
    ).sort_values("score", ascending=False)

    return df.head(top_n).reset_index(drop=True)
    
    
def run_gridsearch_and_report(
    name,
    pipeline,
    param_grid,
    Xtr,
    y_tr,
    Xte,
    y_te,
    cv,
    class_names
):
    """
    Fit GridSearchCV with multiple scorings, print CV mean±std + test metrics.
    Returns the fitted GridSearchCV and predictions on test.
    """
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring={
            "bal_acc": "balanced_accuracy",
            "logloss": "neg_log_loss",
            "auc": "roc_auc_ovr_weighted",
        },
        refit="bal_acc",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    gs.fit(Xtr, y_tr)

    print(f"\n=== {name}: best params ===")
    print(gs.best_params_)

    best_idx = gs.best_index_
    for key, nice, negate in [
        ("bal_acc", "Balanced accuracy", False),
        ("auc", "ROC-AUC (OvR, weighted)", False),
        ("logloss", "Log-loss", True),
    ]:
        mean = gs.cv_results_[f"mean_test_{key}"][best_idx]
        std = gs.cv_results_[f"std_test_{key}"][best_idx]
        if negate:
            mean = -mean
        print(f"{nice} (CV): {mean:.3f} ± {std:.3f}")

    # test-set evaluation
    y_pred = gs.predict(Xte)
    proba = gs.predict_proba(Xte)

    print("\nTest balanced accuracy:", balanced_accuracy_score(y_te, y_pred).round(3))
    print(
        "Test ROC-AUC (OvR, weighted):",
        roc_auc_score(y_te, proba, multi_class="ovr", average="weighted").round(3),
    )
    print("\nClassification report (test):")
    print(classification_report(y_te, y_pred, target_names=class_names, digits=3))

   
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, xticks_rotation=45, cmap="viridis", colorbar=True)
    ax.grid(False)            
    plt.savefig("confmat_early.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    
    return gs, y_pred, proba


def late_fusion_logreg(
    Xtr_views,
    Xte_views,
    y_tr,
    y_te,
    k_per_view,
    cv_outer,
    param_grid,
    class_names,
    use_variance=False,
):
    """
    Late fusion with per-view logistic regression models tuned via GridSearchCV
    over the same param grid (e.g. param_grid_ef).

    - Outer CV (cv_outer) gives mean ± std of late-fusion balanced accuracy.
    - For each outer fold and each view:
        * run GridSearchCV on the fold's training data
        * use best_estimator_ to predict probabilities on the fold's val data.
    - Then, refit per-view models on the full training set (with GridSearchCV)
      and evaluate late fusion on the held-out test set.
    """

    def make_base_view_pipeline(k):
        # ANOVA vs variance for feature ranking
        if use_variance:
            selector = SelectKBest(score_func=variance_score, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)

        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("kbest", selector),
                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        multi_class="multinomial",
                        class_weight="balanced",
                        max_iter=5000,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        return pipe

    # --------------------------------------------------
    # 1) Outer CV: BA mean ± std for late fusion
    # --------------------------------------------------
    cv_scores = []

    for tr_idx, val_idx in cv_outer.split(Xtr_views["mRNA"], y_tr):
        Xm_tr, Xm_val = Xtr_views["mRNA"][tr_idx], Xtr_views["mRNA"][val_idx]
        Xd_tr, Xd_val = Xtr_views["DNAm"][tr_idx], Xtr_views["DNAm"][val_idx]
        Xp_tr, Xp_val = Xtr_views["RPPA"][tr_idx], Xtr_views["RPPA"][val_idx]
        y_tr_fold, y_val = y_tr[tr_idx], y_tr[val_idx]

        models_fold = {}

        # tune each view on the fold's training data
        for view_name, X_tr_view, k in [
            ("mRNA", Xm_tr, k_per_view["mRNA"]),
            ("DNAm", Xd_tr, k_per_view["DNAm"]),
            ("RPPA", Xp_tr, k_per_view["RPPA"]),
        ]:
            base_pipe = make_base_view_pipeline(k)
            gs_view = GridSearchCV(
                base_pipe,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=3,            # inner CV for that fold (can change)
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            gs_view.fit(X_tr_view, y_tr_fold)
            models_fold[view_name] = gs_view.best_estimator_

        # late fusion on validation fold
        logp = np.log(models_fold["mRNA"].predict_proba(Xm_val) + 1e-12)
        logp += np.log(models_fold["DNAm"].predict_proba(Xd_val) + 1e-12)
        logp += np.log(models_fold["RPPA"].predict_proba(Xp_val) + 1e-12)
        logp /= 3.0

        y_val_pred = logp.argmax(axis=1)
        cv_scores.append(balanced_accuracy_score(y_val, y_val_pred))

    cv_scores = np.array(cv_scores)
    print(
        f"Late fusion CV balanced accuracy: "
        f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    )

    # --------------------------------------------------
    # 2) Refit per-view models on full training set,
    #    with the same grid, and evaluate on test set
    # --------------------------------------------------
    models_full = {}

    for view_name, X_tr_view, k in [
        ("mRNA", Xtr_views["mRNA"], k_per_view["mRNA"]),
        ("DNAm", Xtr_views["DNAm"], k_per_view["DNAm"]),
        ("RPPA", Xtr_views["RPPA"], k_per_view["RPPA"]),
    ]:
        base_pipe = make_base_view_pipeline(k)
        gs_view = GridSearchCV(
            base_pipe,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=cv_outer,      # use same outer CV splits for inner tuning
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        gs_view.fit(X_tr_view, y_tr)
        models_full[view_name] = gs_view.best_estimator_

        print(f"\nBest params for {view_name}: {gs_view.best_params_}")

    # late fusion on test set
    logp_test = np.log(models_full["mRNA"].predict_proba(Xte_views["mRNA"]) + 1e-12)
    logp_test += np.log(models_full["DNAm"].predict_proba(Xte_views["DNAm"]) + 1e-12)
    logp_test += np.log(models_full["RPPA"].predict_proba(Xte_views["RPPA"]) + 1e-12)
    logp_test /= 3.0

    y_pred_lf = logp_test.argmax(axis=1)

    print(
        "\nLate fusion test balanced accuracy:",
        balanced_accuracy_score(y_te, y_pred_lf).round(3),
    )
   
   
    print("\nClassification report (test, late fusion):")
    print(classification_report(y_te, y_pred_lf, target_names=class_names, digits=3))

    cm = confusion_matrix(y_te, y_pred_lf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap="viridis", colorbar=True)
    plt.show()

    return y_pred_lf, cv_scores


def plot_var_eta2_grid(
    res_var: dict[str, pd.DataFrame],
    res_anova: dict[str, pd.DataFrame],
    views: list[str] = ("mRNA", "DNAm", "RPPA"),
    bins_var: int = 50,
    bins_eta: int = 50,
    savepath: str | None = None,
):
    """
    Make a 3x2 histogram grid:
        rows  = views (e.g. mRNA, DNAm, RPPA)
        col 0 = variance per feature
        col 1 = ANOVA eta^2 per feature

    Each subplot has its own y-axis scale (no shared axis), so RPPA
    histograms remain visible despite fewer features.
    """
    n_rows = len(views)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 9))

    if n_rows == 1:
        axes = np.array([axes])  # ensure 2D

    for i, view in enumerate(views):
        df_var = res_var[view]
        df_an  = res_anova[view]

        # left column: variance
        ax_var = axes[i, 0]
        ax_var.hist(df_var["var"].values, bins=bins_var)
        ax_var.set_xlabel("variance")
        ax_var.set_ylabel("count")
        ax_var.set_title(f"{view}: variance distribution")

        # right column: eta^2
        ax_eta = axes[i, 1]
        ax_eta.hist(df_an["eta2"].values, bins=bins_eta)
        ax_eta.set_xlabel(r"$\eta^2$")
        ax_eta.set_ylabel("count")
        ax_eta.set_title(f"{view}: " + r"$\eta^2$" + " distribution")

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axes
