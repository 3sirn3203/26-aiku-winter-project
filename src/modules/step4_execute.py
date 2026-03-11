from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def execute(
    execute_cfg: Dict[str, Any],
    implement_result: Dict[str, Any],
    iteration_dir: str,
) -> Dict[str, Any]:
    execute_dir = os.path.join(iteration_dir, "execute")
    os.makedirs(execute_dir, exist_ok=True)

    pipeline_script_path = str(implement_result.get("pipeline_script_path", "")).strip()
    preprocessor_module_path = str(implement_result.get("preprocessor_module_path", "")).strip()
    feature_block_module_paths = _normalize_path_list(implement_result.get("feature_block_module_paths"))

    assembly_detail: Dict[str, Any] = {}
    if preprocessor_module_path:
        try:
            pipeline_script_path, assembly_detail = _assemble_pipeline_script(
                execute_dir=execute_dir,
                requested_pipeline_script_path=pipeline_script_path,
                preprocessor_module_path=preprocessor_module_path,
                feature_block_module_paths=feature_block_module_paths,
            )
            print(
                "    [Step4:assemble] generated assembled pipeline "
                f"script={pipeline_script_path} blocks={len(feature_block_module_paths)}"
            )
        except Exception as exc:  # noqa: BLE001
            return _hard_failure(
                execute_dir=execute_dir,
                reason="pipeline_assembly_failed",
                detail={
                    "error": str(exc),
                    "preprocessor_module_path": preprocessor_module_path,
                    "feature_block_module_paths": feature_block_module_paths,
                },
            )

    if not pipeline_script_path:
        return _hard_failure(
            execute_dir=execute_dir,
            reason="missing_pipeline_script_path",
            detail={
                "pipeline_script_path": pipeline_script_path,
                "preprocessor_module_path": preprocessor_module_path,
                "feature_block_module_paths": feature_block_module_paths,
            },
        )

    return _execute_pipeline_script(
        execute_cfg=execute_cfg,
        execute_dir=execute_dir,
        pipeline_script_path=pipeline_script_path,
        assembly_detail=assembly_detail,
    )


def _normalize_path_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _assemble_pipeline_script(
    execute_dir: str,
    requested_pipeline_script_path: str,
    preprocessor_module_path: str,
    feature_block_module_paths: List[str],
) -> Tuple[str, Dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    preprocessor_path = _resolve_existing_path(preprocessor_module_path, project_root)
    block_paths = [_resolve_existing_path(path, project_root) for path in feature_block_module_paths]

    if requested_pipeline_script_path:
        output_path = Path(requested_pipeline_script_path)
        if not output_path.is_absolute():
            output_path = (project_root / output_path).resolve()
    else:
        output_path = (Path(execute_dir) / "assembled_pipeline.py").resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = _render_assembled_pipeline_script(
        preprocessor_module_path=str(preprocessor_path),
        feature_block_module_paths=[str(path) for path in block_paths],
    )
    output_path.write_text(script, encoding="utf-8")

    return (
        str(output_path),
        {
            "mode": "module_assembly",
            "preprocessor_module_path": str(preprocessor_path),
            "feature_block_module_paths": [str(path) for path in block_paths],
            "assembled_pipeline_script_path": str(output_path),
        },
    )


def _resolve_existing_path(raw_path: str, project_root: Path) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return path.resolve()

    direct = (project_root / path).resolve()
    if direct.exists():
        return direct

    alt = Path(path)
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"Path not found: {raw_path}")


def _render_assembled_pipeline_script(
    preprocessor_module_path: str,
    feature_block_module_paths: List[str],
) -> str:
    template = '''from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

try:
    from category_encoders import TargetEncoder
except Exception:  # noqa: BLE001
    TargetEncoder = None

from src.modules.validator import run_cross_validation

PREPROCESSOR_MODULE_PATH = __PREPROCESSOR_PATH__
FEATURE_BLOCK_MODULE_PATHS = __FEATURE_BLOCK_PATHS__
SAFE_FEATURE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
COMMON_SYMBOLS = {
    "np": np,
    "ColumnTransformer": ColumnTransformer,
    "SimpleImputer": SimpleImputer,
    "OneHotEncoder": OneHotEncoder,
    "TargetEncoder": TargetEncoder,
}


def _safe_name(name: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "feature"
    if not SAFE_FEATURE_NAME_PATTERN.fullmatch(text):
        return "feature"
    return text


def _dedupe_names(names: Iterable[Any]) -> List[str]:
    out: List[str] = []
    used: set[str] = set()
    for idx, raw in enumerate(names):
        base = _safe_name(raw)
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


def _ensure_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _load_module_from_path(path: str, module_name: str) -> Any:
    module_path = Path(path)
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    for symbol_name, symbol_value in COMMON_SYMBOLS.items():
        if symbol_name not in module.__dict__ and symbol_value is not None:
            module.__dict__[symbol_name] = symbol_value
    spec.loader.exec_module(module)
    return module


def _assert_methods(obj: Any, required: List[str], label: str) -> None:
    missing = [name for name in required if not hasattr(obj, name)]
    if missing:
        raise ValueError(f"{label} missing required methods: {missing}")


def _load_preprocessor(path: str) -> Any:
    module = _load_module_from_path(path, "generated_preprocessor_module")
    candidate = getattr(module, "GeneratedPreprocessor", None)
    obj = candidate() if isinstance(candidate, type) else candidate
    if obj is None:
        obj = module
    _assert_methods(obj, ["fit_preprocessor", "transform_preprocessor"], "GeneratedPreprocessor")
    return obj


def _is_feature_block_class(candidate: Any, module_name: str) -> bool:
    return (
        isinstance(candidate, type)
        and getattr(candidate, "__module__", "") == module_name
        and hasattr(candidate, "fit")
        and hasattr(candidate, "transform")
    )


def _extract_block_index_from_path(path: str) -> Optional[int]:
    stem = Path(path).stem
    match = re.search(r"feature_block_(\\d+)$", stem)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _load_feature_block(path: str, index: int) -> Any:
    module = _load_module_from_path(path, f"generated_feature_block_{index}")
    module_name = str(getattr(module, "__name__", "") or "")
    local_index = _extract_block_index_from_path(path)

    expected_names: List[str] = []
    if isinstance(local_index, int):
        expected_names.append(f"GeneratedFeatureBlock{local_index}")
    expected_names.append(f"GeneratedFeatureBlock{index}")
    expected_names = list(dict.fromkeys(expected_names))

    for class_name in expected_names:
        candidate = getattr(module, class_name, None)
        if _is_feature_block_class(candidate, module_name):
            block = candidate()
            _assert_methods(block, ["fit", "transform"], class_name)
            return block

    fallback_candidates = []
    for name, value in module.__dict__.items():
        if _is_feature_block_class(value, module_name):
            fallback_candidates.append((str(name), value))

    if len(fallback_candidates) == 1:
        class_name, candidate = fallback_candidates[0]
        block = candidate()
        _assert_methods(block, ["fit", "transform"], class_name)
        return block

    if len(fallback_candidates) == 0:
        raise ValueError(
            "Feature block class not found: "
            f"index={index}, path={path}, expected_names={expected_names}"
        )

    candidate_names = [name for name, _ in fallback_candidates]
    raise ValueError(
        "Multiple feature block classes found in module: "
        f"path={path}, expected_names={expected_names}, candidates={candidate_names}"
    )


class GeneratedFeatureEngineering:
    FEATURE_BLOCKS = {
        "generated": {
            "description": "One generated feature per hypothesis block",
            "enabled_by_default": True,
            "count": len(FEATURE_BLOCK_MODULE_PATHS),
        }
    }

    def __init__(self) -> None:
        self._blocks = [
            _load_feature_block(path=path, index=idx)
            for idx, path in enumerate(FEATURE_BLOCK_MODULE_PATHS, start=1)
        ]

    def fit_feature_engineering(
        self,
        train_df: pd.DataFrame,
        label_col: str,
        config: Dict[str, Any],
        enabled_blocks: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        del enabled_blocks
        train_in = _ensure_dataframe(train_df)
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
            safe_name = _safe_name(raw_name) if raw_name is not None else ""
            if not safe_name:
                safe_name = default_name
            state["feature_name"] = safe_name
            state["_block_index"] = idx
            state["_block_class"] = block.__class__.__name__
            block_states.append(state)
        return {"block_states": block_states, "feature_cols": []}

    def transform_feature_engineering(self, df: pd.DataFrame, fe_state: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        df_in = _ensure_dataframe(df)
        label = ""
        if isinstance(config, dict):
            data_cfg = config.get("data", {})
            if isinstance(data_cfg, dict):
                label = str(data_cfg.get("label_col", "") or "")
        base_df = df_in.drop(columns=[label], errors="ignore") if label else df_in.copy()
        out_df = base_df.copy()

        states = fe_state.get("block_states", []) if isinstance(fe_state, dict) else []
        for idx, block in enumerate(self._blocks, start=1):
            state = {}
            if idx - 1 < len(states) and isinstance(states[idx - 1], dict):
                state = states[idx - 1]
            default_name = f"generated_feature_{idx}"
            feature_name = _safe_name(state.get("feature_name", default_name))
            if not feature_name:
                feature_name = default_name
            try:
                raw_feature = block.transform(
                    df=df_in.copy(),
                    block_state=state,
                    label_col=label,
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

        final_cols = _dedupe_names(out_df.columns.tolist())
        out_df.columns = final_cols
        if isinstance(fe_state, dict):
            fe_state["feature_cols"] = final_cols
        return out_df

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


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _parse_enabled_blocks(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Assembled E2E pipeline script")
    parser.add_argument("--config", type=str, default="config/dacon.json")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--enabled-blocks", type=str, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    train_path = str(args.train_path or data_cfg.get("train_path", "data/dacon/train.csv"))
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")

    preprocessor_module = _load_preprocessor(PREPROCESSOR_MODULE_PATH)
    feature_module = GeneratedFeatureEngineering()
    enabled_blocks = _parse_enabled_blocks(args.enabled_blocks)

    cv_result = run_cross_validation(
        config=config,
        train_df=train_df,
        preprocessor_module=preprocessor_module,
        feature_module=feature_module,
        enabled_blocks=enabled_blocks,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(cv_result, file, ensure_ascii=False, indent=2)

    print(f"Metric: {cv_result.get('metric')}")
    print(f"Mean CV: {float(cv_result.get('mean_cv', 0.0)):.6f}")
    print(f"Std CV : {float(cv_result.get('std_cv', 0.0)):.6f}")
    print(f"Saved CV result to: {args.output_json}")


if __name__ == "__main__":
    main()
'''
    return (
        template.replace("__PREPROCESSOR_PATH__", json.dumps(preprocessor_module_path))
        .replace("__FEATURE_BLOCK_PATHS__", json.dumps(feature_block_module_paths, ensure_ascii=False))
    )


def _execute_pipeline_script(
    execute_cfg: Dict[str, Any],
    execute_dir: str,
    pipeline_script_path: str,
    assembly_detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    syntax_check = bool(execute_cfg.get("syntax_check_generated_modules", True))
    if syntax_check:
        ok, reason = _py_compile_check([pipeline_script_path])
        if not ok:
            return _hard_failure(
                execute_dir=execute_dir,
                reason="generated_pipeline_syntax_error",
                detail={"error": reason, "pipeline_script_path": pipeline_script_path, "assembly": assembly_detail or {}},
            )

    output_json = str(execute_cfg.get("output_json", os.path.join(execute_dir, "cv_result.json")))
    timeout_sec = int(execute_cfg.get("timeout_sec", 1800))
    python_bin = str(execute_cfg.get("python_bin", sys.executable))
    config_path = str(execute_cfg.get("config_path", "config/dacon.json"))
    success_stdout_log_max_chars = int(execute_cfg.get("success_stdout_log_max_chars", 3000))
    success_stderr_log_max_chars = int(execute_cfg.get("success_stderr_log_max_chars", 1500))
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = str(env.get("PYTHONPATH", "")).strip()
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(project_root)
    )

    cmd: List[str] = [
        python_bin,
        pipeline_script_path,
        "--config",
        config_path,
        "--output-json",
        output_json,
    ]
    _append_optional_arg(cmd, "--train-path", execute_cfg.get("train_path"))

    enabled_blocks = execute_cfg.get("enabled_blocks")
    if isinstance(enabled_blocks, list):
        enabled_blocks = ",".join([str(x).strip() for x in enabled_blocks if str(x).strip()])
    _append_optional_arg(cmd, "--enabled-blocks", enabled_blocks)

    try:
        run_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            cwd=str(project_root),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return _hard_failure(
            execute_dir=execute_dir,
            reason="validation_timeout",
            detail={
                "timeout_sec": timeout_sec,
                "command": cmd,
                "stdout_tail": str(exc.stdout or "")[-2000:],
                "stderr_tail": str(exc.stderr or "")[-2000:],
                "pipeline_script_path": pipeline_script_path,
                "assembly": assembly_detail or {},
            },
        )

    stdout_path = os.path.join(execute_dir, "execute_stdout.log")
    stderr_path = os.path.join(execute_dir, "execute_stderr.log")
    Path(stdout_path).write_text(run_result.stdout or "", encoding="utf-8")
    Path(stderr_path).write_text(run_result.stderr or "", encoding="utf-8")

    if run_result.returncode != 0:
        return _hard_failure(
            execute_dir=execute_dir,
            reason="validation_process_failed",
            detail={
                "returncode": run_result.returncode,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "stdout_tail": str(run_result.stdout or "")[-2000:],
                "stderr_tail": str(run_result.stderr or "")[-2000:],
                "command": cmd,
                "pipeline_script_path": pipeline_script_path,
                "assembly": assembly_detail or {},
            },
        )

    if not os.path.exists(output_json):
        return _hard_failure(
            execute_dir=execute_dir,
            reason="missing_cv_result_json",
            detail={"output_json": output_json, "command": cmd, "pipeline_script_path": pipeline_script_path},
        )

    try:
        with open(output_json, "r", encoding="utf-8") as file:
            cv_result = json.load(file)
    except Exception as exc:  # noqa: BLE001
        return _hard_failure(
            execute_dir=execute_dir,
            reason="invalid_cv_result_json",
            detail={"output_json": output_json, "error": str(exc), "pipeline_script_path": pipeline_script_path},
        )

    required_keys = ["mean_cv", "std_cv", "metric", "objective_mean"]
    missing = [key for key in required_keys if key not in cv_result]
    if missing:
        return _hard_failure(
            execute_dir=execute_dir,
            reason="missing_required_cv_keys",
            detail={"missing_keys": missing, "output_json": output_json, "pipeline_script_path": pipeline_script_path},
        )

    result = {
        "success": True,
        "hard_failure": False,
        "reason": "",
        "command": cmd,
        "pipeline_script_path": pipeline_script_path,
        "assembly": assembly_detail or {},
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "cv_result_path": output_json,
        "cv_result": cv_result,
    }
    with open(os.path.join(execute_dir, "execute_result.json"), "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(
        "    [Step4:execute] SUCCESS "
        f"metric={cv_result.get('metric')} "
        f"mean_cv={cv_result.get('mean_cv')} "
        f"std_cv={cv_result.get('std_cv')} "
        f"objective_mean={cv_result.get('objective_mean')}"
    )
    print(
        "    [Step4:execute] artifacts "
        f"stdout={stdout_path} stderr={stderr_path} cv_json={output_json}"
    )
    stdout_text = str(run_result.stdout or "").strip()
    if stdout_text:
        print("    [Step4:execute] stdout (tail):")
        print(_tail_for_log(stdout_text, success_stdout_log_max_chars))
    stderr_text = str(run_result.stderr or "").strip()
    if stderr_text:
        print("    [Step4:execute] stderr (tail):")
        print(_tail_for_log(stderr_text, success_stderr_log_max_chars))

    return result


def _append_optional_arg(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text == "":
        return
    cmd.extend([flag, text])


def _py_compile_check(module_paths: List[str]) -> Tuple[bool, str]:
    for path in module_paths:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            return False, f"path={path}, stderr={stderr}, stdout={stdout}"
    return True, ""


def _hard_failure(execute_dir: str, reason: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = {
        "success": False,
        "hard_failure": True,
        "reason": reason,
        "detail": detail or {},
    }
    with open(os.path.join(execute_dir, "execute_result.json"), "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def _tail_for_log(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    normalized = str(text or "")
    if len(normalized) <= max_chars:
        return normalized
    return "(truncated, showing tail)\n" + normalized[-max_chars:]
