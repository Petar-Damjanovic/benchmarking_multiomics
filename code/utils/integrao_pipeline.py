# root/code/utils/integrao_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from .config import IntegraoConfig
from .metrics import extract_pred_labels_1based, compute_fold_metrics

# Import your IntegrAO objects
from integrao.integrater import integrao_integrater, integrao_predictor


@dataclass
class CVResult:
    metrics: np.ndarray                 # shape (K, 4)
    confusion_matrices: List[np.ndarray]
    fold_model_paths: List[Path]

def _build_integrater(datasets: List[pd.DataFrame],
                      cfg: IntegraoConfig,
                      dataset_name: str,
                      modalities: List[str]) -> "integrao_integrater":
    return integrao_integrater(
        datasets,
        dataset_name,
        modalities_name_list=modalities,
        neighbor_size=cfg.neighbor_size,
        embedding_dims=cfg.embedding_dims,
        fusing_iteration=cfg.fusing_iteration,
        normalization_factor=cfg.normalization_factor,
        alighment_epochs=cfg.alignment_epochs,
        beta=cfg.beta,
        mu=cfg.mu,
    )

def _build_predictor(datasets: List[pd.DataFrame],
                     cfg: IntegraoConfig,
                     dataset_name: str,
                     modalities: List[str]) -> "integrao_predictor":
    return integrao_predictor(
        datasets,
        dataset_name,
        modalities_name_list=modalities,
        neighbor_size=cfg.neighbor_size,
        embedding_dims=cfg.embedding_dims,
        fusing_iteration=cfg.fusing_iteration,
        normalization_factor=cfg.normalization_factor,
        alighment_epochs=cfg.alignment_epochs,
        beta=cfg.beta,
        mu=cfg.mu,
        num_classes=cfg.cluster_number,
    )

def train_one_fold(train_views: Dict[str, pd.DataFrame],
                   y_train_1based: pd.Series,
                   cfg: IntegraoConfig,
                   dataset_name: str,
                   run_dir: Path,
                   model_out_path: Path,
                   modalities: List[str]) -> Path:
    """
    Unsupervised alignment + supervised fine-tune, then save model.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = [train_views[m].dropna() for m in modalities]
    integrater = _build_integrater(datasets, cfg, dataset_name, modalities)

    integrater.network_diffusion()
    _, _, model = integrater.unsupervised_alignment()

    # IntegrAO expects 0-based cluster.id
    label_df = pd.DataFrame({"cluster.id": (y_train_1based - 1).astype(int)})
    label_df.index.name = "subjects"

    _, _, model, _ = integrater.classification_finetuning(
        label_df,
        str(run_dir),
        finetune_epochs=cfg.finetune_epochs,
    )

    torch.save(model.state_dict(), model_out_path)
    return model_out_path

def infer_on_graph(model_path: Path,
                   inference_views: Dict[str, pd.DataFrame],
                   cfg: IntegraoConfig,
                   dataset_name: str,
                   modalities: List[str]) -> Tuple[np.ndarray, object, dict]:
    """
    Build diffusion on inference graph and run supervised inference.
    Returns (preds, predictor, preds_index)
    """
    datasets = [inference_views[m].dropna() for m in modalities]
    predictor = _build_predictor(datasets, cfg, dataset_name, modalities)
    predictor.network_diffusion()

    preds = predictor.inference_supervised(
        str(model_path),
        new_datasets=datasets,
        modalities_names=modalities,
    )
    preds_index = predictor.dict_sampleToIndexs
    return preds, predictor, preds_index


# -------------------------
# Scenarios
# -------------------------

def run_cv_scenario(
    scenario_name: str,
    labelled_ids: pd.Index,
    y_all_1based: pd.Series,
    views_for_training: Dict[str, pd.DataFrame],
    views_for_inference_builder,  # callable(fold_train_ids, fold_test_ids) -> Dict[str, pd.DataFrame]
    cfg: IntegraoConfig,
    dataset_name: str,
    run_dir: Path,
    models_dir: Path,
    modalities: List[str],
) -> CVResult:
    """
    Generic CV runner:
      - splits only over labelled_ids
      - trains on views_for_training restricted to X_train
      - builds inference views via views_for_inference_builder
    """
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    metrics_rows = []
    cms = []
    model_paths = []

    y_vec = y_all_1based.loc[labelled_ids].values

    for fold, (tr, te) in enumerate(skf.split(labelled_ids, y_vec), start=1):
        X_train = labelled_ids[tr]
        X_test = labelled_ids[te]
        y_train = y_all_1based.loc[X_train]
        y_test = y_all_1based.loc[X_test].values

        # Train views (train only)
        train_views = {m: views_for_training[m].loc[X_train] for m in modalities}

        model_path = models_dir / scenario_name / f"model_fold{fold}.pth"
        fold_run_dir = run_dir / scenario_name / f"fold{fold}"

        train_one_fold(
            train_views=train_views,
            y_train_1based=y_train,
            cfg=cfg,
            dataset_name=dataset_name,
            run_dir=fold_run_dir,
            model_out_path=model_path,
            modalities=modalities,
        )

        # Inference graph for this scenario
        inf_views = views_for_inference_builder(X_train, X_test)

        preds, _, preds_index = infer_on_graph(
            model_path=model_path,
            inference_views=inf_views,
            cfg=cfg,
            dataset_name=dataset_name,
            modalities=modalities,
        )

        y_pred_1based = extract_pred_labels_1based(preds, preds_index, X_test).values
        f1_micro, f1_weighted, bal_acc, acc = compute_fold_metrics(y_test, y_pred_1based)

        metrics_rows.append((f1_micro, f1_weighted, bal_acc, acc))
        cms.append(confusion_matrix(y_test, y_pred_1based, labels=list(range(1, cfg.cluster_number + 1))))
        model_paths.append(model_path)

        print(f"[{scenario_name}] Fold {fold}/{cfg.n_splits} | F1_micro={f1_micro:.3f} F1_w={f1_weighted:.3f} Acc={acc:.3f}")

    return CVResult(metrics=np.asarray(metrics_rows), confusion_matrices=cms, fold_model_paths=model_paths)
