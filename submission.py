# 실행 방법:
# 1) 기본 실행 (final_selection 우선 사용, 없으면 run의 best iteration 자동 선택)
#    python -m submission --config config/dacon.json --run_id <RUN_ID>
#
# 2) iteration 수동 지정
#    python -m submission --config config/dacon.json --run_id <RUN_ID> --iteration <ITERATION>
#
# 3) 출력 경로 지정
#    python -m submission --config config/dacon.json --run_id <RUN_ID> --output_path submissions/dacon/submission_<RUN_ID>_iter_<ITERATION>.csv

import importlib.util
import inspect
import json
import os
import re
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _pick_best_iteration_from_report_payload(payload: Dict[str, Any]) -> Optional[int]:
    best_summary = payload.get("best_summary")
    if isinstance(best_summary, dict):
        try:
            return int(best_summary.get("iteration"))
        except Exception:
            pass

    rows = payload.get("iterations")
    if not isinstance(rows, list):
        return None

    best_iteration: Optional[int] = None
    best_objective: Optional[float] = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("success", False)):
            continue
        try:
            iteration = int(row.get("iteration"))
            objective_mean = float(row.get("objective_mean"))
        except Exception:
            continue
        if best_objective is None or objective_mean > best_objective:
            best_iteration = iteration
            best_objective = objective_mean
    return best_iteration


def _pick_best_iteration_from_execute_results(run_dir: Path) -> Optional[int]:
    best_iteration: Optional[int] = None
    best_objective: Optional[float] = None
    for path in sorted(run_dir.glob("iteration_*/execute/execute_result.json")):
        match = re.match(r"iteration_(\d+)$", path.parent.parent.name)
        if not match:
            continue
        try:
            payload = read_json(str(path))
        except Exception:
            continue
        if not bool(payload.get("success", False)):
            continue
        cv_result = payload.get("cv_result", {})
        if not isinstance(cv_result, dict):
            continue
        try:
            iteration = int(match.group(1))
            objective_mean = float(cv_result.get("objective_mean"))
        except Exception:
            continue
        if best_objective is None or objective_mean > best_objective:
            best_iteration = iteration
            best_objective = objective_mean
    return best_iteration


def resolve_best_iteration_from_run(run_id: str, config: Dict[str, Any]) -> Tuple[Optional[int], str]:
    run_dir = Path("runs") / str(run_id).strip()
    if not run_dir.exists():
        return None, f"run_dir_not_found:{run_dir}"

    fe_cfg = dict(config.get("feature_engineering", {}) or {})
    report_cfg = dict(fe_cfg.get("report", {}) or {})
    output_dir = str(report_cfg.get("output_dir", "")).strip()
    report_json_filename = str(report_cfg.get("report_json_filename", "report.json")).strip() or "report.json"

    candidate_paths: List[Path] = []
    if output_dir:
        candidate_paths.append(run_dir / output_dir / report_json_filename)
    candidate_paths.append(run_dir / report_json_filename)
    if report_json_filename != "report.json":
        candidate_paths.append(run_dir / "report.json")

    seen: set[str] = set()
    unique_candidates: List[Path] = []
    for path in candidate_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(path)

    for report_path in unique_candidates:
        if not report_path.exists():
            continue
        try:
            payload = read_json(str(report_path))
        except Exception:
            continue
        best_iteration = _pick_best_iteration_from_report_payload(payload)
        if best_iteration is not None:
            return best_iteration, f"report:{report_path}"

    fallback_iteration = _pick_best_iteration_from_execute_results(run_dir)
    if fallback_iteration is not None:
        return fallback_iteration, f"execute_results_scan:{run_dir}"

    return None, f"best_iteration_not_found:{run_dir}"


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def infer_label_col(train_df: pd.DataFrame, sample_submission_df: pd.DataFrame, id_col: str) -> str:
    submission_cols = [col for col in sample_submission_df.columns if col != id_col]
    if len(submission_cols) == 1:
        return submission_cols[0]
    if len(train_df.columns) >= 1:
        return str(train_df.columns[-1])
    raise ValueError("Unable to infer label column. Please set submission.data.label_col in config.")


def split_holdout(train_df: pd.DataFrame, holdout_frac: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < holdout_frac < 1:
        raise ValueError("submission.validation.holdout_frac must be between 0 and 1.")
    if len(train_df) < 2:
        raise ValueError("Train data must have at least 2 rows for holdout split.")

    tuning_df = train_df.sample(frac=holdout_frac, random_state=random_state)
    train_only_df = train_df.drop(index=tuning_df.index)

    if tuning_df.empty or train_only_df.empty:
        tuning_size = int(round(len(train_df) * holdout_frac))
        tuning_size = max(1, min(len(train_df) - 1, tuning_size))
        shuffled = train_df.sample(frac=1.0, random_state=random_state)
        tuning_df = shuffled.iloc[:tuning_size]
        train_only_df = shuffled.iloc[tuning_size:]

    return train_only_df.reset_index(drop=True), tuning_df.reset_index(drop=True)


def build_fit_kwargs(model_cfg: Dict[str, Any], fit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    fit_kwargs: Dict[str, Any] = {}

    presets = model_cfg.get("presets", "best_quality")
    if presets is not None:
        fit_kwargs["presets"] = presets

    time_limit = model_cfg.get("time_limit")
    if time_limit is not None:
        fit_kwargs["time_limit"] = time_limit

    num_gpus = model_cfg.get("num_gpus")
    if num_gpus is not None:
        fit_kwargs["num_gpus"] = num_gpus

    fit_kwargs["num_bag_folds"] = 0
    fit_kwargs["num_stack_levels"] = 0
    fit_kwargs.update(fit_cfg)

    fit_signature = inspect.signature(TabularPredictor.fit)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in fit_signature.parameters.values()
    )
    if accepts_var_kwargs:
        return fit_kwargs

    valid_fit_params = set(fit_signature.parameters.keys())
    filtered_kwargs: Dict[str, Any] = {}
    ignored_keys: List[str] = []
    for key, value in fit_kwargs.items():
        if key in valid_fit_params:
            filtered_kwargs[key] = value
        else:
            ignored_keys.append(key)

    if ignored_keys:
        print(f"[WARN] Ignored unsupported fit keys: {ignored_keys}")

    return filtered_kwargs


def load_module_from_path(module_path: str, module_name: str) -> ModuleType:
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"Module path not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_module_functions(module: Any, required_functions: List[str], module_label: str) -> None:
    missing = [name for name in required_functions if not hasattr(module, name)]
    if missing:
        raise ValueError(f"{module_label} module missing required function(s): {missing}")


def ensure_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame()
    return pd.DataFrame(value)


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col) for col in out.columns]
    return out.loc[:, ~out.columns.duplicated()].copy()


def sanitize_feature_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def sanitize_feature_columns(
    train_df: pd.DataFrame,
    other_dfs: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    sanitized = [sanitize_feature_name(col) for col in train_df.columns]
    seen: Dict[str, int] = {}
    final_cols: List[str] = []
    for col in sanitized:
        if col not in seen:
            seen[col] = 0
            final_cols.append(col)
        else:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")

    out_train = train_df.copy()
    out_train.columns = final_cols

    out_others: List[pd.DataFrame] = []
    for other in other_dfs:
        out = other.copy()
        out.columns = final_cols
        out_others.append(out)
    return out_train, out_others


def fill_missing_for_autogluon(
    train_df: pd.DataFrame,
    other_dfs: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    train_out = train_df.copy()
    others_out = [df.copy() for df in other_dfs]

    for col in train_out.columns:
        if pd.api.types.is_numeric_dtype(train_out[col]):
            train_out[col] = pd.to_numeric(train_out[col], errors="coerce").fillna(0.0)
            for other in others_out:
                other[col] = pd.to_numeric(other[col], errors="coerce").fillna(0.0)
        else:
            train_out[col] = train_out[col].astype("string").fillna("__MISSING__")
            for other in others_out:
                other[col] = other[col].astype("string").fillna("__MISSING__")

    return train_out, others_out


def parse_enabled_blocks(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        blocks = [str(item).strip() for item in raw if str(item).strip()]
        return blocks if blocks else None
    text = str(raw).strip()
    if text == "":
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def build_runtime_config(config: Dict[str, Any], submission_data_cfg: Dict[str, Any], label_col: str, enabled_blocks: Optional[Iterable[str]]) -> Dict[str, Any]:
    runtime = dict(config)
    runtime["label_col"] = label_col
    data_cfg = dict(runtime.get("data", {}) or {})
    data_cfg.update(submission_data_cfg)
    data_cfg["label_col"] = label_col
    runtime["data"] = data_cfg
    if enabled_blocks is not None:
        runtime["enabled_blocks"] = [str(block) for block in enabled_blocks]
    return runtime


def resolve_target_transform_method(prep_state: Any, label_col: str) -> Optional[str]:
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


def inverse_transform_predictions(predictions: Any, method: Optional[str]) -> pd.Series:
    if isinstance(predictions, pd.Series):
        pred_series = predictions.copy()
    else:
        pred_series = pd.Series(predictions)

    if method is None:
        return pred_series

    normalized = str(method).strip().lower()
    if normalized != "log1p":
        return pred_series

    numeric = pd.to_numeric(pred_series, errors="coerce")
    inverted = np.expm1(numeric.to_numpy(dtype=float))
    return pd.Series(inverted, index=pred_series.index, name=pred_series.name)


def is_regression_submission_task(config: Dict[str, Any], submission_model_cfg: Dict[str, Any]) -> bool:
    modeling_cfg = dict(config.get("modeling", {}) or {})
    modeling_task_type = str(modeling_cfg.get("task_type", "")).strip().lower()
    if modeling_task_type:
        return modeling_task_type == "regression"

    problem_type = str(submission_model_cfg.get("problem_type", "")).strip().lower()
    if problem_type:
        return problem_type in {"regression", "quantile"}

    eval_metric = str(submission_model_cfg.get("eval_metric", "")).strip().lower()
    return eval_metric in {"rmse", "mae", "mse", "r2", "rmsle", "pinball"}


def resolve_module_paths(
    run_id: str,
    iteration: int,
    submission_cfg: Dict[str, Any],
    preprocessor_override: Optional[str],
    feature_override: Optional[str],
) -> Tuple[str, str]:
    impl_dir = Path("runs") / run_id / f"iteration_{iteration}" / "implement"

    preprocessor_path = preprocessor_override or submission_cfg.get("preprocessor_path")
    if not preprocessor_path:
        preprocessor_path = str(impl_dir / "preprocessor.py")

    feature_path = feature_override or submission_cfg.get("feature_engineering_path")
    if not feature_path:
        feature_path = str(impl_dir / "feature_engineering.py")
    if not os.path.exists(feature_path):
        fallback = submission_cfg.get("feature_generator_path")
        if fallback:
            feature_path = str(fallback)

    return str(preprocessor_path), str(feature_path)


def _resolve_relative_path(raw_path: str, impl_dir: Path) -> str:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return str(path)

    direct = Path(path)
    if direct.exists():
        return str(direct)

    impl_relative = impl_dir / path
    if impl_relative.exists():
        return str(impl_relative)

    return str(direct)


def resolve_pipeline_script_path(
    run_id: str,
    iteration: int,
    submission_cfg: Dict[str, Any],
    pipeline_override: Optional[str],
) -> Tuple[str, bool]:
    impl_dir = Path("runs") / run_id / f"iteration_{iteration}" / "implement"
    configured = pipeline_override or submission_cfg.get("pipeline_script_path")
    explicitly_configured = bool(configured)
    if configured:
        return _resolve_relative_path(str(configured), impl_dir), explicitly_configured

    summary_path = impl_dir / "implement_summary.json"
    if summary_path.exists():
        try:
            summary = read_json(str(summary_path))
            summary_pipeline_path = str(summary.get("pipeline_script_path", "")).strip()
            if summary_pipeline_path:
                return _resolve_relative_path(summary_pipeline_path, impl_dir), False
        except Exception:
            pass

    return str(impl_dir / "implement_pipeline.py"), False


def resolve_generated_module_bundle(
    run_id: str,
    iteration: int,
    submission_cfg: Dict[str, Any],
) -> Tuple[Optional[str], List[str], str]:
    impl_dir = Path("runs") / run_id / f"iteration_{iteration}" / "implement"

    # Optional explicit overrides from submission config (if user sets them).
    configured_pre = submission_cfg.get("preprocessor_module_path")
    configured_blocks = submission_cfg.get("feature_block_module_paths")
    if configured_pre:
        pre_path = _resolve_relative_path(str(configured_pre), impl_dir)
        block_paths: List[str] = []
        if isinstance(configured_blocks, list):
            for item in configured_blocks:
                text = str(item or "").strip()
                if text:
                    block_paths.append(_resolve_relative_path(text, impl_dir))
        return pre_path, block_paths, "submission_cfg"

    summary_path = impl_dir / "implement_summary.json"
    if not summary_path.exists():
        return None, [], f"summary_not_found:{summary_path}"

    try:
        summary = read_json(str(summary_path))
    except Exception as exc:  # noqa: BLE001
        return None, [], f"summary_parse_failed:{summary_path}:{exc}"

    pre_raw = str(summary.get("preprocessor_module_path", "")).strip()
    if not pre_raw:
        return None, [], f"module_bundle_not_declared:{summary_path}"

    pre_path = _resolve_relative_path(pre_raw, impl_dir)
    block_paths: List[str] = []
    raw_blocks = summary.get("feature_block_module_paths")
    if isinstance(raw_blocks, list):
        for item in raw_blocks:
            text = str(item or "").strip()
            if text:
                block_paths.append(_resolve_relative_path(text, impl_dir))

    return pre_path, block_paths, f"implement_summary:{summary_path}"


def resolve_final_selection_bundle(
    run_id: str,
    submission_cfg: Dict[str, Any],
) -> Tuple[Optional[str], List[str], str]:
    run_dir = Path("runs") / str(run_id).strip()
    configured_path = str(submission_cfg.get("final_selection_path", "") or "").strip()

    candidate_paths: List[Path] = []
    if configured_path:
        candidate_paths.append(Path(configured_path))
    candidate_paths.append(run_dir / "final" / "final_selection.json")

    seen: set[str] = set()
    for candidate in candidate_paths:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)

        if not candidate.exists():
            continue

        try:
            payload = read_json(str(candidate))
        except Exception as exc:  # noqa: BLE001
            return None, [], f"final_selection_parse_failed:{candidate}:{exc}"

        pre_raw = str(payload.get("selected_preprocessor_module_path", "") or "").strip()
        if not pre_raw:
            best_pre = payload.get("best_preprocessor", {})
            if isinstance(best_pre, dict):
                pre_raw = str(best_pre.get("path", "") or "").strip()
        if not pre_raw:
            return None, [], f"final_selection_missing_preprocessor:{candidate}"

        block_paths: List[str] = []
        raw_blocks = payload.get("selected_feature_block_module_paths")
        if isinstance(raw_blocks, list):
            for item in raw_blocks:
                text = str(item or "").strip()
                if text:
                    block_paths.append(_resolve_relative_path(text, candidate.parent))

        pre_path = _resolve_relative_path(pre_raw, candidate.parent)
        return pre_path, block_paths, f"final_selection:{candidate}"

    return None, [], f"final_selection_not_found:{run_dir}"


def _load_generated_preprocessor_module(preprocessor_module_path: str) -> Any:
    module = load_module_from_path(
        preprocessor_module_path,
        "submission_generated_preprocessor_module",
    )
    candidate = getattr(module, "GeneratedPreprocessor", None)
    obj = candidate() if inspect.isclass(candidate) else candidate
    if obj is None:
        obj = module
    assert_module_functions(obj, ["fit_preprocessor", "transform_preprocessor"], "generated_preprocessor_module")
    return obj


def _is_feature_block_class(candidate: Any, module_name: str) -> bool:
    return (
        inspect.isclass(candidate)
        and getattr(candidate, "__module__", "") == module_name
        and hasattr(candidate, "fit")
        and hasattr(candidate, "transform")
    )


def _extract_block_index_from_path(path: str) -> Optional[int]:
    stem = Path(path).stem
    match = re.search(r"feature_block_(\d+)$", stem)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _load_generated_feature_block_module(feature_block_module_path: str, index: int) -> Any:
    module = load_module_from_path(
        feature_block_module_path,
        f"submission_generated_feature_block_module_{index}",
    )
    module_name = str(getattr(module, "__name__", "") or "")
    local_index = _extract_block_index_from_path(feature_block_module_path)

    expected_names: List[str] = []
    if isinstance(local_index, int):
        expected_names.append(f"GeneratedFeatureBlock{local_index}")
    expected_names.append(f"GeneratedFeatureBlock{index}")
    expected_names = list(dict.fromkeys(expected_names))

    for class_name in expected_names:
        candidate = getattr(module, class_name, None)
        if _is_feature_block_class(candidate, module_name):
            obj = candidate()
            assert_module_functions(obj, ["fit", "transform"], class_name)
            return obj

    fallback_candidates = []
    for name, value in module.__dict__.items():
        if _is_feature_block_class(value, module_name):
            fallback_candidates.append((str(name), value))

    if len(fallback_candidates) == 1:
        class_name, candidate = fallback_candidates[0]
        obj = candidate()
        assert_module_functions(obj, ["fit", "transform"], class_name)
        return obj

    if len(fallback_candidates) == 0:
        raise ValueError(
            "Feature block class not found: "
            f"index={index}, path={feature_block_module_path}, expected_names={expected_names}"
        )

    candidate_names = [name for name, _ in fallback_candidates]
    raise ValueError(
        "Multiple feature block classes found in module: "
        f"path={feature_block_module_path}, expected_names={expected_names}, "
        f"candidates={candidate_names}"
    )


class SubmissionComposedFeatureEngineering:
    def __init__(self, blocks: List[Any]) -> None:
        self._blocks = list(blocks)
        self.FEATURE_BLOCKS = {
            "generated": {
                "description": "One generated feature per hypothesis block",
                "enabled_by_default": True,
                "count": len(self._blocks),
            }
        }

    def fit_feature_engineering(
        self,
        train_df: pd.DataFrame,
        label_col: str,
        config: Dict[str, Any],
        enabled_blocks: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        del enabled_blocks
        train_in = ensure_dataframe(train_df)
        block_states: List[Dict[str, Any]] = []
        for idx, block in enumerate(self._blocks, start=1):
            default_name = f"generated_feature_{idx}"
            try:
                state = block.fit(train_df=train_in.copy(), label_col=label_col, config=config)
            except Exception as exc:  # noqa: BLE001
                state = {"fit_error": str(exc)}
            if not isinstance(state, dict):
                state = {"fit_return_type": type(state).__name__}
            raw_name = state.get("feature_name")
            if raw_name is None and hasattr(block, "FEATURE_NAME"):
                raw_name = getattr(block, "FEATURE_NAME")
            safe_name = sanitize_feature_name(raw_name) if raw_name is not None else ""
            safe_name = safe_name.strip("_")
            if not safe_name:
                safe_name = default_name
            state["feature_name"] = safe_name
            state["_block_index"] = idx
            state["_block_class"] = block.__class__.__name__
            block_states.append(state)
        return {"block_states": block_states, "feature_cols": []}

    def transform_feature_engineering(self, df: pd.DataFrame, fe_state: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        df_in = ensure_dataframe(df)
        label_col = ""
        if isinstance(config, dict):
            data_cfg = config.get("data", {})
            if isinstance(data_cfg, dict):
                label_col = str(data_cfg.get("label_col", "") or "")
        base_df = df_in.drop(columns=[label_col], errors="ignore") if label_col else df_in.copy()
        out_df = base_df.copy()

        states = fe_state.get("block_states", []) if isinstance(fe_state, dict) else []
        for idx, block in enumerate(self._blocks, start=1):
            state = {}
            if idx - 1 < len(states) and isinstance(states[idx - 1], dict):
                state = states[idx - 1]
            default_name = f"generated_feature_{idx}"
            feature_name = str(state.get("feature_name", default_name))
            feature_name = sanitize_feature_name(feature_name).strip("_") or default_name
            try:
                raw_feature = block.transform(
                    df=df_in.copy(),
                    block_state=state,
                    label_col=label_col,
                    config=config,
                )
            except Exception:
                raw_feature = pd.Series([0.0] * len(df_in), index=df_in.index, name=feature_name)

            feature_series = self._coerce_single_feature(
                raw=raw_feature,
                index=df_in.index,
                feature_name=feature_name,
            )
            out_df[feature_name] = pd.to_numeric(feature_series, errors="coerce").fillna(0.0).to_numpy()

        out_df.columns = _dedupe_names_runtime(out_df.columns.tolist())
        if isinstance(fe_state, dict):
            fe_state["feature_cols"] = list(out_df.columns)
        return out_df

    def feature_registry_from_state(self, fe_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        cols = [str(col) for col in fe_state.get("feature_cols", [])] if isinstance(fe_state, dict) else []
        generated_set = set()
        if isinstance(fe_state, dict):
            for item in fe_state.get("block_states", []):
                if isinstance(item, dict):
                    name = str(item.get("feature_name", "")).strip()
                    if name:
                        generated_set.add(name)
        out: List[Dict[str, Any]] = []
        for col in cols:
            block = "generated" if col in generated_set else "base"
            out.append({"feature": col, "block": block})
        return out

    @staticmethod
    def _coerce_single_feature(raw: Any, index: pd.Index, feature_name: str) -> pd.Series:
        if isinstance(raw, pd.DataFrame):
            if raw.shape[1] == 0:
                series = pd.Series([0.0] * len(index), index=index)
            else:
                series = raw.iloc[:, 0]
        elif isinstance(raw, pd.Series):
            series = raw
        else:
            try:
                series = pd.Series(raw)
            except Exception:
                series = pd.Series([0.0] * len(index))

        series = series.reset_index(drop=True)
        if len(series) < len(index):
            series = series.reindex(range(len(index)))
        if len(series) > len(index):
            series = series.iloc[: len(index)]
        series.index = index
        series.name = feature_name
        return series


def _dedupe_names_runtime(names: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: Dict[str, int] = {}
    for idx, raw in enumerate(names):
        base = sanitize_feature_name(str(raw)).strip("_")
        if not base:
            base = f"feature_{idx}"
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def load_modules_from_module_bundle(
    preprocessor_module_path: str,
    feature_block_module_paths: List[str],
) -> Tuple[Any, Any]:
    preprocessor_obj = _load_generated_preprocessor_module(preprocessor_module_path)
    blocks = [
        _load_generated_feature_block_module(path, index)
        for index, path in enumerate(feature_block_module_paths, start=1)
    ]
    feature_obj = SubmissionComposedFeatureEngineering(blocks)
    assert_module_functions(feature_obj, ["fit_feature_engineering", "transform_feature_engineering"], "composed_feature_engineering")
    return preprocessor_obj, feature_obj


def load_modules_from_pipeline_script(pipeline_script_path: str) -> Tuple[Any, Any]:
    module = load_module_from_path(pipeline_script_path, "submission_generated_pipeline_module")

    # Support assembled execute script layout:
    # PREPROCESSOR_MODULE_PATH + FEATURE_BLOCK_MODULE_PATHS.
    pre_path = str(getattr(module, "PREPROCESSOR_MODULE_PATH", "") or "").strip()
    raw_blocks = getattr(module, "FEATURE_BLOCK_MODULE_PATHS", None)
    if pre_path:
        block_paths: List[str] = []
        if isinstance(raw_blocks, list):
            block_paths = [str(item).strip() for item in raw_blocks if str(item).strip()]
        return load_modules_from_module_bundle(
            preprocessor_module_path=pre_path,
            feature_block_module_paths=block_paths,
        )

    preprocessor_obj: Any = None
    if hasattr(module, "GeneratedPreprocessor"):
        candidate = getattr(module, "GeneratedPreprocessor")
        preprocessor_obj = candidate() if inspect.isclass(candidate) else candidate

    feature_obj: Any = None
    if hasattr(module, "GeneratedFeatureEngineering"):
        candidate = getattr(module, "GeneratedFeatureEngineering")
        feature_obj = candidate() if inspect.isclass(candidate) else candidate

    if preprocessor_obj is not None and feature_obj is not None:
        assert_module_functions(preprocessor_obj, ["fit_preprocessor", "transform_preprocessor"], "generated_preprocessor")
        assert_module_functions(feature_obj, ["fit_feature_engineering", "transform_feature_engineering"], "generated_feature_engineering")
        return preprocessor_obj, feature_obj

    assert_module_functions(module, ["fit_preprocessor", "transform_preprocessor"], "pipeline_script(preprocessor)")
    assert_module_functions(module, ["fit_feature_engineering", "transform_feature_engineering"], "pipeline_script(feature_engineering)")
    return module, module


def apply_generated_feature_pipeline(
    config: Dict[str, Any],
    submission_data_cfg: Dict[str, Any],
    submission_cfg: Dict[str, Any],
    label_col: str,
    train_only_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor_module: Any,
    feature_module: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[str]]:
    enabled_blocks = parse_enabled_blocks(submission_cfg.get("enabled_blocks"))
    runtime_config = build_runtime_config(
        config=config,
        submission_data_cfg=submission_data_cfg,
        label_col=label_col,
        enabled_blocks=enabled_blocks,
    )

    prep_state = preprocessor_module.fit_preprocessor(train_only_df.copy(), label_col, runtime_config)
    target_transform_method = resolve_target_transform_method(prep_state=prep_state, label_col=label_col)
    train_pre = ensure_dataframe(preprocessor_module.transform_preprocessor(train_only_df.copy(), prep_state, runtime_config))
    tuning_pre = ensure_dataframe(preprocessor_module.transform_preprocessor(tuning_df.copy(), prep_state, runtime_config))
    test_pre = ensure_dataframe(preprocessor_module.transform_preprocessor(test_df.copy(), prep_state, runtime_config))

    if label_col not in train_pre.columns:
        raise ValueError(f"train preprocessor output must contain label column '{label_col}'")
    if label_col not in tuning_pre.columns:
        raise ValueError(f"tuning preprocessor output must contain label column '{label_col}'")

    fe_state = feature_module.fit_feature_engineering(
        train_df=train_pre.copy(),
        label_col=label_col,
        config=runtime_config,
        enabled_blocks=enabled_blocks,
    )

    x_train = ensure_dataframe(feature_module.transform_feature_engineering(train_pre.copy(), fe_state, runtime_config))
    x_tuning = ensure_dataframe(feature_module.transform_feature_engineering(tuning_pre.copy(), fe_state, runtime_config))
    x_test = ensure_dataframe(feature_module.transform_feature_engineering(test_pre.copy(), fe_state, runtime_config))

    x_train = deduplicate_columns(x_train)
    x_tuning = deduplicate_columns(x_tuning)
    x_test = deduplicate_columns(x_test)

    if label_col in x_train.columns:
        x_train = x_train.drop(columns=[label_col])
    if label_col in x_tuning.columns:
        x_tuning = x_tuning.drop(columns=[label_col])
    if label_col in x_test.columns:
        x_test = x_test.drop(columns=[label_col])

    x_tuning = x_tuning.reindex(columns=x_train.columns)
    x_test = x_test.reindex(columns=x_train.columns)

    if bool(submission_cfg.get("sanitize_feature_names", True)):
        x_train, [x_tuning, x_test] = sanitize_feature_columns(x_train, [x_tuning, x_test])

    x_train, [x_tuning, x_test] = fill_missing_for_autogluon(x_train, [x_tuning, x_test])

    train_ag = x_train.copy()
    train_ag[label_col] = train_pre[label_col].values
    tuning_ag = x_tuning.copy()
    tuning_ag[label_col] = tuning_pre[label_col].values
    return train_ag, tuning_ag, x_test, target_transform_method


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/dacon.json")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--pipeline_script_path", type=str, default=None)
    parser.add_argument("--preprocessor_path", type=str, default=None)
    parser.add_argument("--feature_path", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    config = read_json(args.config)
    submission_cfg = dict(config.get("submission", {}) or {})

    run_id = args.run_id or submission_cfg.get("run_id")
    if not run_id:
        raise ValueError("run_id is required. Set --run_id or submission.run_id in config.")

    iteration_source = ""
    if args.iteration is not None:
        iteration = int(args.iteration)
        iteration_source = "cli_arg"
    else:
        auto_best = bool(submission_cfg.get("use_best_iteration_from_run", True))
        selected_iteration: Optional[int] = None
        best_source = ""
        if auto_best:
            selected_iteration, best_source = resolve_best_iteration_from_run(run_id=run_id, config=config)
        if selected_iteration is not None:
            iteration = selected_iteration
            iteration_source = best_source
        else:
            iteration = int(submission_cfg.get("iteration", 1))
            iteration_source = "submission.iteration_fallback"

    submission_data_cfg = dict(submission_cfg.get("data", {}) or {})
    submission_model_cfg = dict(submission_cfg.get("model", {}) or {})
    submission_validation_cfg = dict(submission_cfg.get("validation", {}) or {})
    submission_fit_cfg = dict(submission_cfg.get("fit", {}) or {})

    train_path = submission_data_cfg.get("train_path", "data/dacon/train.csv")
    test_path = submission_data_cfg.get("test_path", "data/dacon/test.csv")
    sample_submission_path = submission_data_cfg.get(
        "sample_submission_path",
        submission_data_cfg.get("submission_path", "data/dacon/sample_submission.csv"),
    )
    id_col = str(submission_data_cfg.get("id_col", "ID"))

    configured_output_path = submission_data_cfg.get("output_path") or submission_cfg.get("output_path")
    default_output_dir = "submissions/dacon"
    if configured_output_path:
        candidate_dir = os.path.dirname(str(configured_output_path))
        if candidate_dir:
            default_output_dir = candidate_dir
    default_output_name = f"submission_{run_id}_iter_{iteration}.csv"
    output_path = str(args.output_path or os.path.join(default_output_dir, default_output_name))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    allow_overwrite_output = bool(submission_cfg.get("allow_overwrite_output", True))
    if os.path.exists(output_path) and not allow_overwrite_output:
        raise FileExistsError(f"Submission file already exists: {output_path}")

    if args.pipeline_script_path and (args.preprocessor_path or args.feature_path):
        raise ValueError(
            "Use either --pipeline_script_path or (--preprocessor_path/--feature_path), not both."
        )

    module_source = "pipeline_script"
    pipeline_script_path: Optional[str] = None
    preprocessor_path: Optional[str] = None
    feature_path: Optional[str] = None
    feature_block_paths: List[str] = []

    legacy_override = bool(args.preprocessor_path or args.feature_path)
    if legacy_override:
        module_source = "legacy_modules"
        preprocessor_path, feature_path = resolve_module_paths(
            run_id=run_id,
            iteration=iteration,
            submission_cfg=submission_cfg,
            preprocessor_override=args.preprocessor_path,
            feature_override=args.feature_path,
        )
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor module not found: {preprocessor_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature engineering module not found: {feature_path}")

        preprocessor_module = load_module_from_path(preprocessor_path, "submission_preprocessor_module")
        feature_module = load_module_from_path(feature_path, "submission_feature_engineering_module")
        assert_module_functions(preprocessor_module, ["fit_preprocessor", "transform_preprocessor"], "preprocessor")
        assert_module_functions(feature_module, ["fit_feature_engineering", "transform_feature_engineering"], "feature_engineering")
    else:
        can_use_final_selection = (
            args.iteration is None
            and not bool(args.pipeline_script_path)
            and bool(submission_cfg.get("use_final_selection_from_run", True))
        )
        final_pre: Optional[str] = None
        final_blocks: List[str] = []
        if can_use_final_selection:
            final_pre, final_blocks, final_source = resolve_final_selection_bundle(
                run_id=run_id,
                submission_cfg=submission_cfg,
            )
            if final_pre is not None:
                if not os.path.exists(final_pre):
                    raise FileNotFoundError(f"Final-selection preprocessor module not found: {final_pre}")
                missing_blocks = [path for path in final_blocks if not os.path.exists(path)]
                if missing_blocks:
                    raise FileNotFoundError(f"Final-selection feature block module(s) not found: {missing_blocks}")
                preprocessor_path = final_pre
                feature_block_paths = final_blocks
                preprocessor_module, feature_module = load_modules_from_module_bundle(
                    preprocessor_module_path=preprocessor_path,
                    feature_block_module_paths=feature_block_paths,
                )
                module_source = f"module_bundle:{final_source}"

        can_use_bundle = not bool(args.pipeline_script_path)
        bundle_pre: Optional[str] = None
        bundle_blocks: List[str] = []
        if can_use_bundle and final_pre is None:
            bundle_pre, bundle_blocks, bundle_source = resolve_generated_module_bundle(
                run_id=run_id,
                iteration=iteration,
                submission_cfg=submission_cfg,
            )
            if bundle_pre is not None:
                if not os.path.exists(bundle_pre):
                    raise FileNotFoundError(f"Generated preprocessor module not found: {bundle_pre}")
                missing_blocks = [path for path in bundle_blocks if not os.path.exists(path)]
                if missing_blocks:
                    raise FileNotFoundError(f"Generated feature block module(s) not found: {missing_blocks}")
                preprocessor_path = bundle_pre
                feature_block_paths = bundle_blocks
                preprocessor_module, feature_module = load_modules_from_module_bundle(
                    preprocessor_module_path=preprocessor_path,
                    feature_block_module_paths=feature_block_paths,
                )
                module_source = f"module_bundle:{bundle_source}"

        if bundle_pre is None and final_pre is None:
            pipeline_script_path, explicitly_configured = resolve_pipeline_script_path(
                run_id=run_id,
                iteration=iteration,
                submission_cfg=submission_cfg,
                pipeline_override=args.pipeline_script_path,
            )
            if not os.path.exists(pipeline_script_path):
                if explicitly_configured:
                    raise FileNotFoundError(f"Pipeline script not found: {pipeline_script_path}")

                module_source = "legacy_modules"
                preprocessor_path, feature_path = resolve_module_paths(
                    run_id=run_id,
                    iteration=iteration,
                    submission_cfg=submission_cfg,
                    preprocessor_override=args.preprocessor_path,
                    feature_override=args.feature_path,
                )
                if not os.path.exists(preprocessor_path):
                    raise FileNotFoundError(
                        f"Pipeline script not found ({pipeline_script_path}) and preprocessor module not found: {preprocessor_path}"
                    )
                if not os.path.exists(feature_path):
                    raise FileNotFoundError(
                        f"Pipeline script not found ({pipeline_script_path}) and feature engineering module not found: {feature_path}"
                    )

                preprocessor_module = load_module_from_path(preprocessor_path, "submission_preprocessor_module")
                feature_module = load_module_from_path(feature_path, "submission_feature_engineering_module")
                assert_module_functions(preprocessor_module, ["fit_preprocessor", "transform_preprocessor"], "preprocessor")
                assert_module_functions(feature_module, ["fit_feature_engineering", "transform_feature_engineering"], "feature_engineering")
            else:
                preprocessor_module, feature_module = load_modules_from_pipeline_script(pipeline_script_path)

                # If assembled-script style was resolved from constants, report module bundle source.
                if hasattr(feature_module, "__class__") and feature_module.__class__.__name__ == "SubmissionComposedFeatureEngineering":
                    module_source = "module_bundle:assembled_pipeline_constants"
                else:
                    module_source = "pipeline_script"

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)
    sample_submission_df = load_csv(sample_submission_path)

    label_col = submission_data_cfg.get("label_col")
    if label_col is None:
        label_col = infer_label_col(train_df, sample_submission_df, id_col)
        print(f"[INFO] label_col inferred as '{label_col}'")
    label_col = str(label_col)
    if label_col not in train_df.columns:
        raise ValueError(f"label_col '{label_col}' not found in train columns")

    holdout_frac = float(submission_validation_cfg.get("holdout_frac", 0.2))
    random_state = int(submission_validation_cfg.get("random_state", 42))
    train_only_df, tuning_df = split_holdout(train_df=train_df, holdout_frac=holdout_frac, random_state=random_state)

    train_ag, tuning_ag, test_ag, target_transform_method = apply_generated_feature_pipeline(
        config=config,
        submission_data_cfg=submission_data_cfg,
        submission_cfg=submission_cfg,
        label_col=label_col,
        train_only_df=train_only_df,
        tuning_df=tuning_df,
        test_df=test_df,
        preprocessor_module=preprocessor_module,
        feature_module=feature_module,
    )

    fit_kwargs = build_fit_kwargs(model_cfg=submission_model_cfg, fit_cfg=submission_fit_cfg)
    temp_model_path = tempfile.mkdtemp(prefix=f"ag_submission_{run_id}_iter_{iteration}_")
    try:
        predictor = TabularPredictor(
            label=label_col,
            eval_metric=submission_model_cfg.get("eval_metric"),
            problem_type=submission_model_cfg.get("problem_type"),
            path=temp_model_path,
        )
        predictor.fit(train_data=train_ag, tuning_data=tuning_ag, **fit_kwargs)
        predictions = predictor.predict(test_ag)
    finally:
        shutil.rmtree(temp_model_path, ignore_errors=True)

    if is_regression_submission_task(config=config, submission_model_cfg=submission_model_cfg):
        predictions = inverse_transform_predictions(predictions, target_transform_method)
        if target_transform_method:
            print(f"[INFO] applied inverse target transform for submission predictions: {target_transform_method}")

    submission_df = sample_submission_df.copy()
    if id_col in test_df.columns and id_col in submission_df.columns:
        submission_df[id_col] = test_df[id_col].values
    submission_df[label_col] = predictions.values
    submission_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] run_id: {run_id}, iteration: {iteration}")
    print(f"[INFO] iteration_source: {iteration_source}")
    print(f"[INFO] module_source: {module_source}")
    if module_source.startswith("pipeline_script"):
        print(f"[INFO] pipeline_script: {pipeline_script_path}")
    elif module_source.startswith("module_bundle"):
        print(f"[INFO] preprocessor_module: {preprocessor_path}")
        print(f"[INFO] feature_block_modules: {len(feature_block_paths)}")
    else:
        print(f"[INFO] preprocessor: {preprocessor_path}")
        print(f"[INFO] feature_engineering: {feature_path}")
    print("[INFO] model artifacts: disabled (temporary path auto-removed)")
    print(f"[INFO] submission saved to: {output_path}")


if __name__ == "__main__":
    main()
