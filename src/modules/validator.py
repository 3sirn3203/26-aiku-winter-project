from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType
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


class _DefaultPreprocessor:
    @staticmethod
    def fit_preprocessor(train_df: pd.DataFrame, label_col: str, config: Dict[str, Any]) -> Dict[str, Any]:
        del label_col, config
        return {"columns": [str(col) for col in train_df.columns]}

    @staticmethod
    def transform_preprocessor(df: pd.DataFrame, prep_state: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        del prep_state, config
        return df.copy()


class _DefaultFeatureEngineering:
    FEATURE_BLOCKS = {"identity": {"description": "No feature engineering", "enabled_by_default": True}}

    @staticmethod
    def fit_feature_engineering(
        train_df: pd.DataFrame,
        label_col: str,
        config: Dict[str, Any],
        enabled_blocks: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        del config
        blocks = list(enabled_blocks) if enabled_blocks is not None else ["identity"]
        feature_cols = [str(col) for col in train_df.columns if str(col) != str(label_col)]
        return {"feature_cols": feature_cols, "enabled_blocks": blocks}

    @staticmethod
    def transform_feature_engineering(df: pd.DataFrame, fe_state: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        del config
        feature_cols = [col for col in fe_state.get("feature_cols", []) if col in df.columns]
        out = df.loc[:, feature_cols].copy()
        return out

    @staticmethod
    def feature_registry_from_state(fe_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"feature": col, "block": "identity"}
            for col in fe_state.get("feature_cols", [])
        ]


def load_preprocessor_module(module_path: Optional[str]) -> Any:
    if module_path is None:
        return _DefaultPreprocessor()
    module = _load_module_from_path(module_path, "generated_preprocessor")
    _assert_module_functions(module, ["fit_preprocessor", "transform_preprocessor"], "preprocessor")
    return module


def load_feature_engineering_module(module_path: Optional[str]) -> Any:
    if module_path is None:
        return _DefaultFeatureEngineering()
    module = _load_module_from_path(module_path, "generated_feature_engineering")
    _assert_module_functions(module, ["fit_feature_engineering", "transform_feature_engineering"], "feature_engineering")
    return module


def higher_is_better(metric: str) -> bool:
    return metric.lower() in {"f1", "accuracy", "roc_auc", "auc", "r2"}


def run_cross_validation(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    preprocessor_module: Any,
    feature_module: Any,
    enabled_blocks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
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
    label_encoder: Optional[LabelEncoder] = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_for_split = label_encoder.fit_transform(y_full.astype(str))
        n_classes = int(len(label_encoder.classes_))

    fold_scores: List[float] = []
    fold_details: List[Dict[str, Any]] = []
    last_fe_state: Dict[str, Any] = {}
    model_type_used = model_type

    split_target = y_for_split if cv_type_used == "stratified" else None
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(train_df, split_target), start=1):
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        valid_fold = train_df.iloc[valid_idx].reset_index(drop=True)

        prep_state = preprocessor_module.fit_preprocessor(train_fold.copy(), label_col, runtime_config)
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
            y_valid = pd.to_numeric(valid_pre[label_col], errors="coerce").fillna(0.0)
            model, model_type_used = _build_model(
                task_type=task_type,
                model_type=model_type,
                random_state=random_state + fold_idx,
                model_params=model_params,
                n_classes=None,
            )
            model.fit(x_train, y_train)
            y_pred = np.asarray(model.predict(x_valid))
            score = _score_regression(metric=metric, y_true=y_valid, y_pred=y_pred)

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


def fit_full_and_predict(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    preprocessor_module: Any,
    feature_module: Any,
    enabled_blocks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    label_col = _resolve_label_col(config=config, train_df=train_df, test_df=test_df, sample_submission_df=sample_submission_df)
    runtime_config = _build_module_runtime_config(config=config, label_col=label_col, enabled_blocks=enabled_blocks)
    id_col = _resolve_id_col(config=config, train_df=train_df, test_df=test_df, sample_submission_df=sample_submission_df)
    modeling_cfg = _resolve_modeling_cfg(config)
    task_type = modeling_cfg["task_type"]
    model_type = modeling_cfg["model_type"]
    model_params = modeling_cfg["model_params"]
    random_state = modeling_cfg["random_state"]

    prep_state = preprocessor_module.fit_preprocessor(train_df.copy(), label_col, runtime_config)
    train_pre = preprocessor_module.transform_preprocessor(train_df.copy(), prep_state, runtime_config)
    test_pre = preprocessor_module.transform_preprocessor(test_df.copy(), prep_state, runtime_config)
    train_pre, test_pre = _normalize_preprocessed_pair(
        train_pre=train_pre,
        other_pre=test_pre,
        label_col=label_col,
    )
    _assert_label_exists(train_pre, label_col, "train_preprocessed")

    fe_state = feature_module.fit_feature_engineering(
        train_df=train_pre.copy(),
        label_col=label_col,
        config=runtime_config,
        enabled_blocks=enabled_blocks,
    )
    x_train_raw = feature_module.transform_feature_engineering(train_pre.copy(), fe_state, runtime_config)
    x_test_raw = feature_module.transform_feature_engineering(test_pre.copy(), fe_state, runtime_config)
    x_train, x_test = _align_and_encode_pair(x_train_raw, x_test_raw)

    if task_type == "classification":
        y_train_str = train_pre[label_col].astype(str)
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_str)
        n_classes = int(len(label_encoder.classes_))
        model, model_type_used = _build_model(
            task_type=task_type,
            model_type=model_type,
            random_state=random_state,
            model_params=model_params,
            n_classes=n_classes,
        )
        model.fit(x_train, y_train)
        pred_encoded = np.asarray(model.predict(x_test)).astype(int)
        y_test_pred = label_encoder.inverse_transform(pred_encoded)
    else:
        y_train = pd.to_numeric(train_pre[label_col], errors="coerce").fillna(0.0).to_numpy()
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
    if id_col is not None and id_col in test_df.columns and id_col in submission.columns:
        submission[id_col] = test_df[id_col].values
    submission[label_col] = y_test_pred

    return {
        "submission_df": submission,
        "model_type_used": model_type_used,
        "fe_state": fe_state,
        "feature_registry": _feature_registry_from_state(feature_module, fe_state),
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
    test_df: Optional[pd.DataFrame] = None,
    sample_submission_df: Optional[pd.DataFrame] = None,
) -> str:
    data_cfg = config.get("data", {}) or {}
    configured = data_cfg.get("label_col")
    if configured is not None:
        return str(configured)

    if sample_submission_df is not None and test_df is not None:
        candidate = [col for col in sample_submission_df.columns if col not in test_df.columns]
        if len(candidate) == 1:
            return str(candidate[0])
        if len(sample_submission_df.columns) >= 2:
            return str(sample_submission_df.columns[-1])

    return str(train_df.columns[-1])


def _resolve_id_col(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    data_cfg = config.get("data", {}) or {}
    configured = data_cfg.get("id_col")
    if configured is not None:
        return str(configured)

    if sample_submission_df is not None:
        for col in sample_submission_df.columns:
            if col in test_df.columns:
                return str(col)

    for col in test_df.columns:
        if col in train_df.columns:
            return str(col)
    return None


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
                    "n_estimators": 400,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "random_state": random_state,
                }
                defaults.update(params)
                return lgb.LGBMClassifier(**defaults), "lightgbm"

            defaults = {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": random_state,
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


def _load_module_from_path(module_path: str, module_name: str) -> ModuleType:
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"Module path not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_module_functions(module: Any, required_functions: List[str], module_label: str) -> None:
    missing = [name for name in required_functions if not hasattr(module, name)]
    if missing:
        raise ValueError(f"{module_label} module missing required function(s): {missing}")
