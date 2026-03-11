from __future__ import annotations

import re
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

SAFE_FEATURE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def higher_is_better(metric: str) -> bool:
    return metric.lower() in {"f1", "accuracy", "roc_auc", "auc", "r2"}


def run_cross_validation(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    preprocessor_module: Any,
    feature_module: Any,
    enabled_blocks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    _assert_module_methods(
        module=preprocessor_module,
        required_methods=["fit_preprocessor", "transform_preprocessor"],
        module_label="preprocessor_module",
    )
    _assert_module_methods(
        module=feature_module,
        required_methods=["fit_feature_engineering", "transform_feature_engineering"],
        module_label="feature_module",
    )

    label_col = _resolve_label_col(config=config, train_df=train_df)
    runtime_config = _build_module_runtime_config(config=config, label_col=label_col, enabled_blocks=enabled_blocks)
    modeling_cfg = _resolve_modeling_cfg(config)
    task_type = modeling_cfg["task_type"]
    metric = modeling_cfg["metric"]
    model_type = modeling_cfg["model_type"]
    model_params = modeling_cfg["model_params"]
    cv_type = modeling_cfg["cv_type"]
    n_splits = modeling_cfg["n_splits"]
    shuffle = modeling_cfg["shuffle"]
    random_state = modeling_cfg["random_state"]

    splitter, cv_type_used = _build_splitter(
        task_type=task_type,
        cv_type=cv_type,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    y_full = train_df[label_col]
    y_for_split = None
    n_classes: Optional[int] = None
    if task_type == "classification":
        split_encoder = LabelEncoder()
        y_for_split = split_encoder.fit_transform(y_full.astype(str))
        n_classes = int(len(split_encoder.classes_))

    fold_scores: List[float] = []
    fold_details: List[Dict[str, Any]] = []
    last_fe_state: Dict[str, Any] = {}
    model_type_used = model_type

    split_target = y_for_split if cv_type_used == "stratified" else None
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(train_df, split_target), start=1):
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        valid_fold = train_df.iloc[valid_idx].reset_index(drop=True)

        prep_state = preprocessor_module.fit_preprocessor(train_fold.copy(), label_col, runtime_config)
        target_transform_method = _resolve_target_transform_method(
            prep_state=prep_state,
            label_col=label_col,
        )
        train_pre = preprocessor_module.transform_preprocessor(train_fold.copy(), prep_state, runtime_config)
        valid_pre = preprocessor_module.transform_preprocessor(valid_fold.copy(), prep_state, runtime_config)
        train_pre, valid_pre = _normalize_preprocessed_pair(
            train_pre=train_pre,
            other_pre=valid_pre,
            label_col=label_col,
        )
        _assert_label_exists(train_pre, label_col, "train_preprocessed")
        _assert_label_exists(valid_pre, label_col, "valid_preprocessed")

        fe_state = feature_module.fit_feature_engineering(
            train_df=train_pre.copy(),
            label_col=label_col,
            config=runtime_config,
            enabled_blocks=enabled_blocks,
        )
        last_fe_state = fe_state
        x_train_raw = feature_module.transform_feature_engineering(train_pre.copy(), fe_state, runtime_config)
        x_valid_raw = feature_module.transform_feature_engineering(valid_pre.copy(), fe_state, runtime_config)

        x_train, x_valid = _align_and_encode_pair(x_train_raw, x_valid_raw)

        if task_type == "classification":
            y_train_str = train_pre[label_col].astype(str)
            y_valid_str = valid_pre[label_col].astype(str)
            fold_encoder = LabelEncoder()
            y_train = fold_encoder.fit_transform(y_train_str)
            y_valid_encoded = fold_encoder.transform(y_valid_str)

            model, model_type_used = _build_model(
                task_type=task_type,
                model_type=model_type,
                random_state=random_state + fold_idx,
                model_params=model_params,
                n_classes=n_classes,
            )
            model.fit(x_train, y_train)
            y_pred_enc = np.asarray(model.predict(x_valid)).astype(int)
            y_pred = fold_encoder.inverse_transform(y_pred_enc)
            y_proba = model.predict_proba(x_valid) if hasattr(model, "predict_proba") else None
            score = _score_classification(
                metric=metric,
                y_true=y_valid_str,
                y_pred=y_pred,
                y_proba=y_proba,
                label_encoder=fold_encoder,
                y_true_encoded=y_valid_encoded,
            )
        else:
            y_train = pd.to_numeric(train_pre[label_col], errors="coerce").fillna(0.0).to_numpy()
            y_valid_transformed = pd.to_numeric(valid_pre[label_col], errors="coerce").fillna(0.0)
            y_valid_original = pd.to_numeric(valid_fold[label_col], errors="coerce").fillna(0.0)
            model, model_type_used = _build_model(
                task_type=task_type,
                model_type=model_type,
                random_state=random_state + fold_idx,
                model_params=model_params,
                n_classes=None,
            )
            model.fit(x_train, y_train)
            y_pred = np.asarray(model.predict(x_valid))
            if target_transform_method is not None:
                y_pred_eval = _inverse_transform_target_array(y_pred, target_transform_method)
                y_valid_eval = y_valid_original
            else:
                y_pred_eval = y_pred
                y_valid_eval = y_valid_transformed
            score = _score_regression(metric=metric, y_true=y_valid_eval, y_pred=y_pred_eval)

        fold_scores.append(float(score))
        fold_details.append(
            {
                "fold": fold_idx,
                "score": float(score),
                "num_train_rows": int(len(train_fold)),
                "num_valid_rows": int(len(valid_fold)),
                "num_features": int(x_train.shape[1]),
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
        "feature_registry": _feature_registry_from_state(feature_module, last_fe_state),
        "feature_blocks": _feature_blocks_from_module(feature_module),
        "interface_contract": {
            "preprocessor_required": ["fit_preprocessor", "transform_preprocessor"],
            "feature_required": ["fit_feature_engineering", "transform_feature_engineering"],
        },
    }


def _resolve_modeling_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    modeling = config.get("modeling", {}) or {}
    validation = modeling.get("validation", {}) or {}
    model = modeling.get("model", {}) or {}
    return {
        "task_type": str(modeling.get("task_type", "regression")).lower(),
        "metric": str(modeling.get("metric", "rmse")).lower(),
        "cv_type": str(validation.get("cv_type", "auto")).lower(),
        "n_splits": int(validation.get("n_splits", 5)),
        "shuffle": bool(validation.get("shuffle", True)),
        "random_state": int(validation.get("random_state", 42)),
        "model_type": str(model.get("type", "random_forest")).lower(),
        "model_params": dict(model.get("params", {}) or {}),
    }


def _build_module_runtime_config(
    config: Dict[str, Any],
    label_col: str,
    enabled_blocks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    runtime = dict(config)
    runtime["label_col"] = label_col

    data_cfg = dict(runtime.get("data", {}) or {})
    data_cfg["label_col"] = label_col
    runtime["data"] = data_cfg

    if enabled_blocks is not None:
        runtime["enabled_blocks"] = [str(block) for block in enabled_blocks]
    return runtime


def _resolve_label_col(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
) -> str:
    data_cfg = config.get("data", {}) or {}
    configured = data_cfg.get("label_col")
    if configured is not None:
        return str(configured)

    return str(train_df.columns[-1])


def _build_splitter(
    task_type: str,
    cv_type: str,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> Tuple[Any, str]:
    if cv_type == "auto":
        cv_type = "stratified" if task_type == "classification" else "kfold"
    if cv_type == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), "stratified"
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), "kfold"


def _build_model(
    task_type: str,
    model_type: str,
    random_state: int,
    model_params: Dict[str, Any],
    n_classes: Optional[int],
) -> Tuple[Any, str]:
    params = dict(model_params)
    if model_type == "xgboost":
        try:
            import xgboost as xgb

            if task_type == "classification":
                defaults: Dict[str, Any] = {
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
                if n_classes is not None and n_classes > 2:
                    defaults["objective"] = "multi:softprob"
                    defaults["num_class"] = n_classes
                    defaults["eval_metric"] = "mlogloss"
                else:
                    defaults["objective"] = "binary:logistic"
                defaults.update(params)
                return xgb.XGBClassifier(**defaults), "xgboost"

            defaults = {
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
            defaults.update(params)
            return xgb.XGBRegressor(**defaults), "xgboost"
        except Exception:
            pass

    if model_type == "lightgbm":
        try:
            import lightgbm as lgb

            if task_type == "classification":
                defaults = {
                    "n_estimators": 250,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "min_child_samples": 10,
                    "subsample": 0.8,
                    "subsample_freq": 1,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "random_state": random_state,
                    "n_jobs": -1,
                    "verbosity": -1,
                }
                defaults.update(params)
                return lgb.LGBMClassifier(**defaults), "lightgbm"

            defaults = {
                "n_estimators": 250,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 10,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": -1,
            }
            defaults.update(params)
            return lgb.LGBMRegressor(**defaults), "lightgbm"
        except Exception:
            pass

    if task_type == "classification":
        defaults = {"n_estimators": 500, "random_state": random_state, "n_jobs": -1}
        defaults.update(params)
        return RandomForestClassifier(**defaults), "random_forest"
    defaults = {"n_estimators": 500, "random_state": random_state, "n_jobs": -1}
    defaults.update(params)
    return RandomForestRegressor(**defaults), "random_forest"


def _score_classification(
    metric: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    label_encoder: LabelEncoder,
    y_true_encoded: np.ndarray,
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
        if y_proba.ndim == 1 or y_proba.shape[1] == 1:
            return float("nan")
        if y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true_encoded, y_proba[:, 1]))
        return float(roc_auc_score(y_true_encoded, y_proba, multi_class="ovr"))
    if metric == "logloss":
        if y_proba is None:
            return float("nan")
        return float(log_loss(y_true_encoded, y_proba))
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


def _resolve_target_transform_method(
    prep_state: Any,
    label_col: str,
) -> Optional[str]:
    if not isinstance(prep_state, dict):
        return None

    info = prep_state.get("target_transform_info")
    if isinstance(info, dict):
        method = str(info.get("method", "")).strip().lower()
        column = str(info.get("column", "")).strip()
        if method in {"log1p"} and (not column or column == label_col):
            return method

    apply_log_transform = prep_state.get("apply_log_transform")
    label_col_name = str(prep_state.get("label_col_name", "")).strip()
    if bool(apply_log_transform) and (not label_col_name or label_col_name == label_col):
        return "log1p"

    return None


def _inverse_transform_target_array(values: np.ndarray, method: str) -> np.ndarray:
    method = str(method).strip().lower()
    arr = np.asarray(values, dtype=float)
    if method == "log1p":
        return np.expm1(arr)
    return arr


def _align_and_encode_pair(train_df: pd.DataFrame, other_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, other = _normalize_feature_pair(train_df=train_df, other_df=other_df)

    train_out = pd.DataFrame(index=train.index)
    other_out = pd.DataFrame(index=other.index)
    for col in train.columns:
        s_train = train[col]
        s_other = other[col]
        if pd.api.types.is_numeric_dtype(s_train):
            train_out[col] = pd.to_numeric(s_train, errors="coerce").fillna(0.0)
            other_out[col] = pd.to_numeric(s_other, errors="coerce").fillna(0.0)
        else:
            train_str = s_train.astype("string").fillna("__NA__")
            categories = pd.Index(train_str.unique())
            other_str = s_other.astype("string").fillna("__NA__")
            train_out[col] = pd.Categorical(train_str, categories=categories).codes.astype(float)
            other_out[col] = pd.Categorical(other_str, categories=categories).codes.astype(float)
    return train_out, other_out


def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return pd.DataFrame(df)


def _normalize_feature_pair(train_df: pd.DataFrame, other_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = _ensure_dataframe(train_df)
    other = _ensure_dataframe(other_df)

    train_raw_cols = [str(col) for col in train.columns]
    other_raw_cols = [str(col) for col in other.columns]
    train.columns = train_raw_cols
    other.columns = other_raw_cols

    # Align using raw train schema first, then enforce safe/unique names consistently.
    other = other.reindex(columns=train_raw_cols)
    normalized_cols = _build_safe_unique_feature_names(train_raw_cols)
    train.columns = normalized_cols
    other.columns = normalized_cols
    return train, other


def _normalize_preprocessed_pair(
    train_pre: pd.DataFrame,
    other_pre: pd.DataFrame,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _ensure_dataframe(train_pre)
    other_df = _ensure_dataframe(other_pre)

    train_label = train_df[label_col].copy() if label_col in train_df.columns else None
    other_label = other_df[label_col].copy() if label_col in other_df.columns else None

    train_feat = train_df.drop(columns=[label_col], errors="ignore")
    other_feat = other_df.drop(columns=[label_col], errors="ignore")
    train_feat, other_feat = _normalize_feature_pair(train_df=train_feat, other_df=other_feat)

    train_out = train_feat.copy()
    other_out = other_feat.copy()
    if train_label is not None:
        train_out[label_col] = train_label.values
    if other_label is not None:
        other_out[label_col] = other_label.values
    return train_out, other_out


def _build_safe_unique_feature_names(names: Iterable[Any]) -> List[str]:
    used: set[str] = set()
    out: List[str] = []
    for idx, raw_name in enumerate(names):
        base = _sanitize_feature_name(raw_name)
        if not base:
            base = f"feature_{idx}"
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        out.append(candidate)
    return out


def _sanitize_feature_name(name: Any) -> str:
    text = str(name)
    text = re.sub(r"[^A-Za-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    if not text:
        return ""
    if not SAFE_FEATURE_NAME_PATTERN.fullmatch(text):
        return ""
    return text


def _feature_registry_from_state(feature_module: Any, fe_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    if fe_state and hasattr(feature_module, "feature_registry_from_state"):
        try:
            registry = feature_module.feature_registry_from_state(fe_state)
            if isinstance(registry, list):
                return registry
        except Exception:
            pass
    return []


def _feature_blocks_from_module(feature_module: Any) -> Dict[str, Any]:
    blocks = getattr(feature_module, "FEATURE_BLOCKS", None)
    if isinstance(blocks, dict):
        return blocks
    return {}


def _assert_label_exists(df: pd.DataFrame, label_col: str, stage_name: str) -> None:
    if label_col not in df.columns:
        raise ValueError(f"{stage_name} must contain label column '{label_col}'")


def _assert_module_methods(module: Any, required_methods: List[str], module_label: str) -> None:
    missing = [name for name in required_methods if not hasattr(module, name)]
    if missing:
        raise ValueError(f"{module_label} module missing required function(s): {missing}")
