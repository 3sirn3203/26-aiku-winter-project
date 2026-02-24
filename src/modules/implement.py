from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

import pandas as pd
from google import genai
from jinja2 import Template

SAFE_FEATURE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")

PREP_SYSTEM_INSTRUCTION = """You generate only executable Python code for a tabular preprocessor module.
Return code only.
Do not return markdown fences.
Do not return explanations.
The code must implement:
- fit_preprocessor(train_df, label_col, config) -> dict
- transform_preprocessor(df, prep_state, config) -> pandas.DataFrame"""

FE_SYSTEM_INSTRUCTION = """You generate only executable Python code for a tabular feature engineering module.
Return code only.
Do not return markdown fences.
Do not return explanations.
The code must implement:
- fit_feature_engineering(train_df, label_col, config, enabled_blocks=None) -> dict
- transform_feature_engineering(df, fe_state, config) -> pandas.DataFrame
Optional:
- feature_registry_from_state(fe_state) -> list
- FEATURE_BLOCKS dictionary"""


def implement(
    client: genai.Client,
    implement_cfg: Dict[str, Any],
    profile_result: Dict[str, Any],
    hypotheses: Dict[str, Any],
    output_dir: str,
    train_path: Optional[str] = None,
    label_col: Optional[str] = None,
    pipeline_config: Optional[Dict[str, Any]] = None,
    external_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    implement_dir = os.path.join(output_dir, "implement")
    os.makedirs(implement_dir, exist_ok=True)

    model = str(implement_cfg.get("model", "gemini-2.5-flash"))
    temperature = float(implement_cfg.get("temperature", 0.2))
    top_p = float(implement_cfg.get("top_p", 0.9))
    max_output_tokens = int(implement_cfg.get("max_output_tokens", implement_cfg.get("max_tokens", 8192)))
    max_codegen_attempts = int(implement_cfg.get("max_codegen_attempts", 2))
    syntax_check = bool(implement_cfg.get("syntax_check", True))
    enable_contract_check = bool(implement_cfg.get("enable_contract_check", True))
    contract_sample_rows = int(implement_cfg.get("contract_sample_rows", 512))
    max_contract_attempts = int(implement_cfg.get("max_contract_attempts", 2))

    preprocessing_hypotheses = _normalize_text_list(hypotheses.get("preprocessing"))
    feature_hypotheses = _normalize_text_list(hypotheses.get("feature_engineering"))
    profile_context = _build_profile_context(profile_result=profile_result, implement_cfg=implement_cfg)

    final_prep_meta: Dict[str, Any] = {}
    final_fe_meta: Dict[str, Any] = {}
    final_contract_meta: Dict[str, Any] = {}
    contract_feedback: Optional[Dict[str, Any]] = None

    for contract_attempt in range(1, max_contract_attempts + 1):
        prep_code, prep_meta = _generate_script(
            client=client,
            stage_name="preprocessor",
            template_path=Path("src/prompt/3_implement_prep.j2"),
            render_ctx={
                "profile_result_json": json.dumps(profile_context, ensure_ascii=False),
                "preprocessing_hypotheses_json": json.dumps(preprocessing_hypotheses, ensure_ascii=False),
            },
            required_functions=["fit_preprocessor", "transform_preprocessor"],
            output_script_path=Path(implement_dir) / "preprocessor.py",
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            max_codegen_attempts=max_codegen_attempts,
            syntax_check=syntax_check,
            system_instruction=str(implement_cfg.get("preprocessor_system_instruction", PREP_SYSTEM_INSTRUCTION)),
            external_feedback=external_feedback,
            contract_feedback=contract_feedback,
        )

        fe_code, fe_meta = _generate_script(
            client=client,
            stage_name="feature_engineering",
            template_path=Path("src/prompt/3_implement_fe.j2"),
            render_ctx={
                "profile_result_json": json.dumps(profile_context, ensure_ascii=False),
                "feature_hypotheses_json": json.dumps(feature_hypotheses, ensure_ascii=False),
            },
            required_functions=["fit_feature_engineering", "transform_feature_engineering"],
            output_script_path=Path(implement_dir) / "feature_engineering.py",
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            max_codegen_attempts=max_codegen_attempts,
            syntax_check=syntax_check,
            system_instruction=str(implement_cfg.get("feature_engineering_system_instruction", FE_SYSTEM_INSTRUCTION)),
            external_feedback=external_feedback,
            contract_feedback=contract_feedback,
        )

        _ = prep_code
        _ = fe_code
        final_prep_meta = prep_meta
        final_fe_meta = fe_meta

        if not enable_contract_check:
            final_contract_meta = {
                "checked": False,
                "success": True,
                "attempt": contract_attempt,
                "message": "contract_check_disabled",
            }
            break

        try:
            contract_stats = _contract_smoke_check(
                preprocessor_path=Path(implement_dir) / "preprocessor.py",
                feature_engineering_path=Path(implement_dir) / "feature_engineering.py",
                train_path=train_path,
                label_col=label_col,
                pipeline_config=pipeline_config or {},
                sample_rows=contract_sample_rows,
            )
            final_contract_meta = {
                "checked": True,
                "success": True,
                "attempt": contract_attempt,
                "stats": contract_stats,
            }
            contract_feedback = None
            break
        except Exception as exc:  # noqa: BLE001
            contract_feedback = {
                "attempt": contract_attempt,
                "error": str(exc),
            }
            final_contract_meta = {
                "checked": True,
                "success": False,
                "attempt": contract_attempt,
                "error": str(exc),
            }
            with open(os.path.join(implement_dir, f"contract_attempt_{contract_attempt}.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "attempt": contract_attempt,
                        "error": str(exc),
                        "external_feedback": external_feedback,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            if contract_attempt >= max_contract_attempts:
                raise RuntimeError(f"Contract smoke check failed after {max_contract_attempts} attempt(s): {exc}") from exc

    summary = {
        "preprocessor_path": str(Path(implement_dir) / "preprocessor.py"),
        "feature_engineering_path": str(Path(implement_dir) / "feature_engineering.py"),
        "contract": {
            "preprocessor": {"required_functions": ["fit_preprocessor", "transform_preprocessor"]},
            "feature_engineering": {
                "required_functions": ["fit_feature_engineering", "transform_feature_engineering"],
                "optional": ["feature_registry_from_state", "FEATURE_BLOCKS"],
            },
        },
        "meta": {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "max_codegen_attempts": max_codegen_attempts,
            "syntax_check": syntax_check,
            "enable_contract_check": enable_contract_check,
            "contract_sample_rows": contract_sample_rows,
            "max_contract_attempts": max_contract_attempts,
            "profile_context_keys": list(profile_context.keys()),
            "external_feedback": external_feedback,
            "preprocessor": final_prep_meta,
            "feature_engineering": final_fe_meta,
            "contract_check": final_contract_meta,
        },
    }

    with open(os.path.join(implement_dir, "implement_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def _generate_script(
    client: genai.Client,
    stage_name: str,
    template_path: Path,
    render_ctx: Dict[str, Any],
    required_functions: List[str],
    output_script_path: Path,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_codegen_attempts: int,
    syntax_check: bool,
    system_instruction: str,
    external_feedback: Optional[Dict[str, Any]],
    contract_feedback: Optional[Dict[str, Any]],
) -> tuple[str, Dict[str, Any]]:
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    template = Template(template_path.read_text(encoding="utf-8"))
    stage_dir = output_script_path.parent
    last_raw_text = ""
    last_response_meta: Dict[str, Any] = {}
    last_error = ""
    final_code: Optional[str] = None
    feedback: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_codegen_attempts + 1):
        prompt = template.render(
            **render_ctx,
            generation_feedback_json=json.dumps(feedback, ensure_ascii=False) if feedback is not None else "null",
            external_feedback_json=json.dumps(external_feedback, ensure_ascii=False) if external_feedback is not None else "null",
            contract_feedback_json=json.dumps(contract_feedback, ensure_ascii=False) if contract_feedback is not None else "null",
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "text/plain",
                "system_instruction": system_instruction,
            },
        )
        response_meta = _extract_response_meta(response)
        raw_text = str(getattr(response, "text", "") or "").strip()
        last_raw_text = raw_text
        last_response_meta = response_meta
        code = _extract_python_code(raw_text)
        output_script_path.write_text(code, encoding="utf-8")

        try:
            _assert_required_functions(code=code, required_functions=required_functions)
            if syntax_check:
                _syntax_check(script_path=output_script_path)
            final_code = code
            last_error = ""
            feedback = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            feedback = {"attempt": attempt, "error": last_error}
            with open(stage_dir / f"{stage_name}_attempt_{attempt}.json", "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "attempt": attempt,
                        "error": last_error,
                        "response_meta": response_meta,
                        "raw_text": raw_text,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

    if final_code is None:
        raise RuntimeError(f"Failed to generate valid {stage_name} module: {last_error}")

    with open(stage_dir / f"{stage_name}_raw_response.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "raw_text": last_raw_text,
                "response_meta": last_response_meta,
                "last_error": last_error,
                "required_functions": required_functions,
                "syntax_check": syntax_check,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return final_code, {"required_functions": required_functions, "last_error": last_error}


def _contract_smoke_check(
    preprocessor_path: Path,
    feature_engineering_path: Path,
    train_path: Optional[str],
    label_col: Optional[str],
    pipeline_config: Dict[str, Any],
    sample_rows: int,
) -> Dict[str, Any]:
    if not train_path:
        raise ValueError("train_path is required for contract smoke check.")
    train_path_obj = Path(train_path)
    if not train_path_obj.exists():
        raise FileNotFoundError(f"train_path not found: {train_path_obj}")

    train_df = pd.read_csv(train_path_obj, encoding="utf-8-sig", nrows=max(32, sample_rows))
    if train_df.empty:
        raise ValueError("train.csv is empty; cannot run contract smoke check.")

    resolved_label = _resolve_label_col(label_col=label_col, train_df=train_df, pipeline_config=pipeline_config)
    if resolved_label not in train_df.columns:
        raise ValueError(f"Resolved label column '{resolved_label}' not found in train dataframe.")

    runtime_config = dict(pipeline_config)
    runtime_config.setdefault("label_col", resolved_label)
    runtime_config.setdefault("data", {})
    if isinstance(runtime_config["data"], dict):
        runtime_config["data"].setdefault("label_col", resolved_label)

    preprocessor_module = _load_module_from_path(preprocessor_path, "contract_preprocessor_module")
    feature_module = _load_module_from_path(feature_engineering_path, "contract_feature_engineering_module")
    _assert_module_functions(preprocessor_module, ["fit_preprocessor", "transform_preprocessor"])
    _assert_module_functions(feature_module, ["fit_feature_engineering", "transform_feature_engineering"])

    probe_df = train_df.copy().head(min(len(train_df), 256)).reset_index(drop=True)
    probe_without_label = probe_df.drop(columns=[resolved_label], errors="ignore")

    prep_state = preprocessor_module.fit_preprocessor(probe_df.copy(), resolved_label, runtime_config)
    train_pre = preprocessor_module.transform_preprocessor(probe_df.copy(), prep_state, runtime_config)
    test_pre = preprocessor_module.transform_preprocessor(probe_without_label.copy(), prep_state, runtime_config)
    train_pre, test_pre, pre_norm_stats = _normalize_preprocessed_pair_for_contract(
        train_pre=train_pre,
        test_pre=test_pre,
        label_col=resolved_label,
    )

    if not isinstance(train_pre, pd.DataFrame):
        raise ValueError("transform_preprocessor(train_with_label) must return pandas.DataFrame.")
    if not isinstance(test_pre, pd.DataFrame):
        raise ValueError("transform_preprocessor(test_without_label) must return pandas.DataFrame.")
    if resolved_label not in train_pre.columns:
        raise ValueError("transform_preprocessor must preserve label_col when input contains label_col.")

    fe_state = feature_module.fit_feature_engineering(
        train_df=train_pre.copy(),
        label_col=resolved_label,
        config=runtime_config,
        enabled_blocks=None,
    )
    x_train = feature_module.transform_feature_engineering(train_pre.copy(), fe_state, runtime_config)
    x_test = feature_module.transform_feature_engineering(test_pre.copy(), fe_state, runtime_config)

    if not isinstance(x_train, pd.DataFrame):
        raise ValueError("transform_feature_engineering(train_pre) must return pandas.DataFrame.")
    if not isinstance(x_test, pd.DataFrame):
        raise ValueError("transform_feature_engineering(test_pre) must return pandas.DataFrame.")
    if resolved_label in x_train.columns or resolved_label in x_test.columns:
        raise ValueError("transform_feature_engineering output must not contain label_col.")

    x_train, x_test, normalization_stats = _normalize_feature_pair_for_contract(x_train=x_train, x_test=x_test)
    _assert_valid_feature_frame(x_train, "x_train")
    _assert_valid_feature_frame(x_test, "x_test")

    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("Feature schema mismatch between train and test transformed outputs.")

    return {
        "rows_checked": int(len(probe_df)),
        "num_features": int(x_train.shape[1]),
        "preprocessor_normalization_stats": pre_norm_stats,
        "normalization_stats": normalization_stats,
    }


def _assert_valid_feature_frame(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} is empty.")
    col_names = [str(c) for c in df.columns]
    if len(col_names) != len(set(col_names)):
        dup_counts: Dict[str, int] = {}
        for col in col_names:
            dup_counts[col] = dup_counts.get(col, 0) + 1
        duplicates = [col for col, count in dup_counts.items() if count > 1]
        sample = duplicates[:10]
        raise ValueError(
            f"{name} contains duplicated columns. duplicate_count={len(duplicates)}, sample={sample}"
        )
    if any(col.strip() == "" for col in col_names):
        raise ValueError(f"{name} contains empty column names.")
    unsafe = [col for col in col_names if SAFE_FEATURE_NAME_PATTERN.fullmatch(col) is None]
    if unsafe:
        sample = unsafe[:10]
        raise ValueError(
            f"{name} contains feature names with unsupported characters. unsafe_count={len(unsafe)}, sample={sample}"
        )


def _resolve_label_col(label_col: Optional[str], train_df: pd.DataFrame, pipeline_config: Dict[str, Any]) -> str:
    if label_col and label_col in train_df.columns:
        return str(label_col)
    data_cfg = pipeline_config.get("data", {}) if isinstance(pipeline_config, dict) else {}
    cfg_label = data_cfg.get("label_col") if isinstance(data_cfg, dict) else None
    if cfg_label and str(cfg_label) in train_df.columns:
        return str(cfg_label)
    return str(train_df.columns[-1])


def _load_module_from_path(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_module_functions(module: Any, required_functions: List[str]) -> None:
    missing = [name for name in required_functions if not hasattr(module, name)]
    if missing:
        raise ValueError(f"Missing required function(s): {missing}")


def _extract_python_code(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    fence_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text


def _normalize_feature_pair_for_contract(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    train = x_train.copy()
    test = x_test.copy()
    train_raw_cols = [str(col) for col in train.columns]
    test.columns = [str(col) for col in test.columns]
    test = test.reindex(columns=train_raw_cols)

    normalized_cols = _build_safe_unique_feature_names(train_raw_cols)
    train.columns = normalized_cols
    test.columns = normalized_cols

    duplicate_count_before = len(train_raw_cols) - len(set(train_raw_cols))
    renamed_count = sum(1 for raw, new in zip(train_raw_cols, normalized_cols) if str(raw) != str(new))
    return (
        train,
        test,
        {
            "duplicate_count_before": int(duplicate_count_before),
            "renamed_count": int(renamed_count),
        },
    )


def _normalize_preprocessed_pair_for_contract(
    train_pre: pd.DataFrame,
    test_pre: pd.DataFrame,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    train_df = train_pre.copy()
    test_df = test_pre.copy()
    train_label = train_df[label_col].copy() if label_col in train_df.columns else None
    test_label = test_df[label_col].copy() if label_col in test_df.columns else None

    train_feat = train_df.drop(columns=[label_col], errors="ignore")
    test_feat = test_df.drop(columns=[label_col], errors="ignore")
    train_feat_norm, test_feat_norm, feat_stats = _normalize_feature_pair_for_contract(
        x_train=train_feat,
        x_test=test_feat,
    )

    train_out = train_feat_norm.copy()
    test_out = test_feat_norm.copy()
    if train_label is not None:
        train_out[label_col] = train_label.values
    if test_label is not None:
        test_out[label_col] = test_label.values
    return (
        train_out,
        test_out,
        {
            "feature_normalization": feat_stats,
            "label_preserved_train": bool(train_label is not None),
            "label_preserved_test": bool(test_label is not None),
        },
    )


def _build_safe_unique_feature_names(names: List[str]) -> List[str]:
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


def _sanitize_feature_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    if not text:
        return ""
    if SAFE_FEATURE_NAME_PATTERN.fullmatch(text) is None:
        return ""
    return text


def _assert_required_functions(code: str, required_functions: List[str]) -> None:
    tree = ast.parse(code)
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = [name for name in required_functions if name not in function_names]
    if missing:
        raise ValueError(f"Missing required function(s): {missing}")


def _syntax_check(script_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise ValueError(f"Python syntax check failed. stderr={stderr} stdout={stdout}")


def _normalize_text_list(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _build_profile_context(profile_result: Dict[str, Any], implement_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile_result, dict):
        return {}

    max_items = int(implement_cfg.get("profile_context_max_items", 8))
    max_text_chars = int(implement_cfg.get("profile_context_max_text_chars", 800))
    max_columns = int(implement_cfg.get("profile_context_max_columns", 80))

    preferred_keys = [
        "summary",
        "insights",
        "risks",
        "recommended_next_actions",
        "shape",
        "columns",
        "dtypes",
        "missing_top",
        "cardinality_top",
    ]

    context: Dict[str, Any] = {}
    for key in preferred_keys:
        if key not in profile_result:
            continue
        value = profile_result.get(key)
        if key == "columns" and isinstance(value, list):
            context[key] = [str(col) for col in value[:max_columns]]
        else:
            context[key] = _trim_profile_value(value=value, max_items=max_items, max_text_chars=max_text_chars)

    if not context:
        for key, value in list(profile_result.items())[:max_items]:
            context[str(key)] = _trim_profile_value(value=value, max_items=max_items, max_text_chars=max_text_chars)

    context["_keys_in_original_profile"] = list(profile_result.keys())
    return context


def _trim_profile_value(value: Any, max_items: int, max_text_chars: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if len(text) <= max_text_chars:
            return text
        return text[:max_text_chars] + "...(truncated)"
    if isinstance(value, list):
        return [_trim_profile_value(item, max_items=max_items, max_text_chars=max_text_chars) for item in value[:max_items]]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                break
            out[str(k)] = _trim_profile_value(v, max_items=max_items, max_text_chars=max_text_chars)
        return out
    return str(value)[:max_text_chars]


def _extract_response_meta(response: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        finish_reason = getattr(first, "finish_reason", None)
        if finish_reason is not None:
            meta["finish_reason"] = str(finish_reason)
        all_finish_reasons: List[str] = []
        for cand in candidates:
            cand_reason = getattr(cand, "finish_reason", None)
            if cand_reason is not None:
                all_finish_reasons.append(str(cand_reason))
        if all_finish_reasons:
            meta["candidate_finish_reasons"] = all_finish_reasons

    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is not None:
        meta["usage_metadata"] = _to_jsonable(usage_metadata)

    model_version = getattr(response, "model_version", None)
    if model_version:
        meta["model_version"] = str(model_version)
    return meta


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            return _to_jsonable(dumped)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            return _to_jsonable(dumped)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            pass
    return str(value)
