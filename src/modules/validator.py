from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import src.feature_engineering as default_fe_module


def higher_is_better(metric: str) -> bool:
    metric = metric.lower()
    return metric in {"f1", "accuracy", "roc_auc", "auc", "r2"}


def _build_splitter(
    task_type: str,
    cv_type: str,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> Tuple[Any, str]:
    cv_type = cv_type.lower()
    task_type = task_type.lower()
    if cv_type == "auto":
        cv_type = "stratified" if task_type == "classification" else "kfold"

    if cv_type == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return splitter, "stratified"

    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return splitter, "kfold"


def _build_model(
    task_type: str,
    model_type: str,
    random_state: int,
    model_params: Dict[str, Any],
    n_classes: Optional[int],
) -> Tuple[Any, str]:
    model_type = model_type.lower()
    params = dict(model_params)

    if model_type == "xgboost":
        try:
            import xgboost as xgb

            if task_type == "classification":
                default = {
                    "n_estimators": 300,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "random_state": random_state,
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "eval_metric": "logloss",
                }
                if n_classes and n_classes > 2:
                    default["objective"] = "multi:softprob"
                    default["num_class"] = n_classes
                    default["eval_metric"] = "mlogloss"
                else:
                    default["objective"] = "binary:logistic"
                default.update(params)
                return xgb.XGBClassifier(**default), "xgboost"

            default = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": random_state,
                "n_jobs": -1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            }
            default.update(params)
            return xgb.XGBRegressor(**default), "xgboost"
        except Exception:
            pass

    if model_type == "lightgbm":
        try:
            import lightgbm as lgb

            if task_type == "classification":
                default = {
                    "n_estimators": 400,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "random_state": random_state,
                }
                default.update(params)
                return lgb.LGBMClassifier(**default), "lightgbm"

            default = {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": random_state,
            }
            default.update(params)
            return lgb.LGBMRegressor(**default), "lightgbm"
        except Exception:
            pass

    if task_type == "classification":
        default = {
            "n_estimators": 500,
            "random_state": random_state,
            "n_jobs": -1,
        }
        default.update(params)
        return RandomForestClassifier(**default), "random_forest"

    default = {
        "n_estimators": 500,
        "random_state": random_state,
        "n_jobs": -1,
    }
    default.update(params)
    return RandomForestRegressor(**default), "random_forest"


def _score_classification(
    metric: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    label_encoder: LabelEncoder,
) -> float:
    metric = metric.lower()
    y_true_arr = y_true.astype(str).to_numpy()
    y_pred_arr = np.asarray(y_pred).astype(str)
    if metric == "f1":
        if label_encoder.classes_.shape[0] == 2:
            positive_label = str(label_encoder.classes_[1])
            return float(f1_score(y_true_arr, y_pred_arr, average="binary", pos_label=positive_label))
        return float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    if metric == "accuracy":
        return float(accuracy_score(y_true_arr, y_pred_arr))
    if metric in {"roc_auc", "auc"}:
        if y_proba is None:
            return float("nan")
        y_true_enc = label_encoder.transform(y_true_arr)
        if y_proba.ndim == 1 or y_proba.shape[1] == 1:
            return float("nan")
        if y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true_enc, y_proba[:, 1]))
        return float(roc_auc_score(y_true_enc, y_proba, multi_class="ovr"))
    if metric == "logloss":
        if y_proba is None:
            return float("nan")
        y_true_enc = label_encoder.transform(y_true_arr)
        return float(log_loss(y_true_enc, y_proba))
    raise ValueError(f"Unsupported classification metric: {metric}")


def _score_regression(metric: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    metric = metric.lower()
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"Unsupported regression metric: {metric}")


def _get_fe_module(fe_module: Optional[Any]) -> Any:
    return fe_module if fe_module is not None else default_fe_module


def _build_features_with_module(
    fe_module: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    config: Dict[str, Any],
    enabled_blocks: Optional[Iterable[str]],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, Any]]:
    if hasattr(fe_module, "build_features"):
        return fe_module.build_features(
            train_df=train_df,
            test_df=test_df,
            label_col=label_col,
            config=config,
            enabled_blocks=enabled_blocks,
        )

    fe_state = fe_module.fit_fe(
        train_df=train_df,
        label_col=label_col,
        config=config,
        enabled_blocks=enabled_blocks,
    )
    x_train = fe_module.transform_fe(train_df, fe_state, config=config)
    y_train = train_df[label_col].copy()
    x_test = fe_module.transform_fe(test_df, fe_state, config=config)
    return x_train, y_train, x_test, fe_state


def run_cross_validation(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    enabled_blocks: Optional[Iterable[str]] = None,
    fe_module: Optional[Any] = None,
) -> Dict[str, Any]:
    fe_module = _get_fe_module(fe_module)
    task_cfg = config["task"]
    data_cfg = config["data"]
    val_cfg = config["validation"]

    task_type = str(task_cfg["type"]).lower()
    metric = str(task_cfg["metric"]).lower()
    label_col = data_cfg["label_col"]
    model_type = str(val_cfg.get("model_type", "random_forest"))
    model_params = val_cfg.get("model_params", {}) or {}
    n_splits = int(val_cfg["n_splits"])
    shuffle = bool(val_cfg["shuffle"])
    random_state = int(val_cfg["random_state"])

    splitter, cv_type_used = _build_splitter(
        task_type=task_type,
        cv_type=str(val_cfg.get("cv_type", "auto")),
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    y_full = train_df[label_col]
    label_encoder: Optional[LabelEncoder] = None
    y_for_split = None
    n_classes: Optional[int] = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_full.astype(str))
        y_for_split = y_encoded
        n_classes = int(len(label_encoder.classes_))

    fold_scores: List[float] = []
    fold_details: List[Dict[str, Any]] = []
    model_type_used = model_type
    last_state: Dict[str, Any] = {}

    for fold_idx, (train_idx, valid_idx) in enumerate(
        splitter.split(train_df, y_for_split if cv_type_used == "stratified" else None),
        start=1,
    ):
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        valid_fold = train_df.iloc[valid_idx].reset_index(drop=True)

        fe_state = fe_module.fit_fe(
            train_df=train_fold,
            label_col=label_col,
            config=config,
            enabled_blocks=enabled_blocks,
        )
        last_state = fe_state
        x_train = fe_module.transform_fe(train_fold, fe_state, config=config)
        x_valid = fe_module.transform_fe(valid_fold, fe_state, config=config)

        if task_type == "classification":
            assert label_encoder is not None
            y_train = label_encoder.transform(train_fold[label_col].astype(str))
            y_valid = valid_fold[label_col].astype(str)
            model, model_type_used = _build_model(
                task_type=task_type,
                model_type=model_type,
                random_state=random_state + fold_idx,
                model_params=model_params,
                n_classes=n_classes,
            )
            model.fit(x_train, y_train)

            pred_encoded = np.asarray(model.predict(x_valid)).astype(int)
            y_pred = label_encoder.inverse_transform(pred_encoded)
            y_proba = model.predict_proba(x_valid) if hasattr(model, "predict_proba") else None
            score = _score_classification(metric, y_valid, y_pred, y_proba, label_encoder)
        else:
            y_train = pd.to_numeric(train_fold[label_col], errors="coerce").fillna(0.0).to_numpy()
            y_valid = pd.to_numeric(valid_fold[label_col], errors="coerce").fillna(0.0)
            model, model_type_used = _build_model(
                task_type=task_type,
                model_type=model_type,
                random_state=random_state + fold_idx,
                model_params=model_params,
                n_classes=None,
            )
            model.fit(x_train, y_train)
            y_pred = np.asarray(model.predict(x_valid))
            score = _score_regression(metric, y_valid, y_pred)

        fold_scores.append(float(score))
        fold_details.append(
            {
                "fold": fold_idx,
                "score": float(score),
                "num_train_rows": int(len(train_fold)),
                "num_valid_rows": int(len(valid_fold)),
            }
        )

    mean_cv = float(np.nanmean(fold_scores))
    std_cv = float(np.nanstd(fold_scores, ddof=0))
    objective_mean = mean_cv if higher_is_better(metric) else -mean_cv

    return {
        "task_type": task_type,
        "metric": metric,
        "higher_is_better": higher_is_better(metric),
        "cv_type_used": cv_type_used,
        "n_splits": n_splits,
        "model_type_requested": model_type,
        "model_type_used": model_type_used,
        "enabled_blocks": list(enabled_blocks) if enabled_blocks is not None else None,
        "fold_scores": fold_scores,
        "fold_details": fold_details,
        "mean_cv": mean_cv,
        "std_cv": std_cv,
        "objective_mean": objective_mean,
        "feature_registry": (
            fe_module.feature_registry_from_state(last_state)
            if last_state and hasattr(fe_module, "feature_registry_from_state")
            else []
        ),
    }


def fit_full_and_predict(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    enabled_blocks: Optional[Iterable[str]] = None,
    fe_module: Optional[Any] = None,
) -> Dict[str, Any]:
    fe_module = _get_fe_module(fe_module)
    task_cfg = config["task"]
    data_cfg = config["data"]
    val_cfg = config["validation"]

    task_type = str(task_cfg["type"]).lower()
    label_col = data_cfg["label_col"]
    id_col = data_cfg["id_col"]
    random_state = int(val_cfg["random_state"])
    model_type = str(val_cfg.get("model_type", "random_forest"))
    model_params = val_cfg.get("model_params", {}) or {}

    x_train, y_train_raw, x_test, fe_state = _build_features_with_module(
        fe_module=fe_module,
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        config=config,
        enabled_blocks=enabled_blocks,
    )

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw.astype(str))
        model, model_type_used = _build_model(
            task_type=task_type,
            model_type=model_type,
            random_state=random_state,
            model_params=model_params,
            n_classes=int(len(label_encoder.classes_)),
        )
        model.fit(x_train, y_train)
        pred_encoded = np.asarray(model.predict(x_test)).astype(int)
        y_test_pred = label_encoder.inverse_transform(pred_encoded)
    else:
        y_train = pd.to_numeric(y_train_raw, errors="coerce").fillna(0.0).to_numpy()
        model, model_type_used = _build_model(
            task_type=task_type,
            model_type=model_type,
            random_state=random_state,
            model_params=model_params,
            n_classes=None,
        )
        model.fit(x_train, y_train)
        y_test_pred = np.asarray(model.predict(x_test))

    submission = sample_submission_df.copy()
    if id_col in test_df.columns:
        submission[id_col] = test_df[id_col].values
    submission[label_col] = y_test_pred

    return {
        "model": model,
        "fe_state": fe_state,
        "submission_df": submission,
        "model_type_used": model_type_used,
    }
