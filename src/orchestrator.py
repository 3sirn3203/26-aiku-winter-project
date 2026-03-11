from __future__ import annotations
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from google import genai

from src.utils import utc_run_id
from src.modules.step1_profile import profiling
from src.modules.step2_hypothesis import generate_hypotheses
from src.modules.step3_implement import implement
from src.modules.step4_execute import execute
from src.modules.step5_diagnose import diagnose
from src.modules.final_report import make_report
from src.modules.validator import run_cross_validation


def _load_llm() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def run_pipeline(config: Dict) -> Dict:
    run_id = utc_run_id()
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    run_config_path = os.path.join(run_dir, "config.json")
    with open(run_config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    task_context = _build_task_context(config=config)
    with open(os.path.join(run_dir, "task_context.json"), "w", encoding="utf-8") as file:
        json.dump(task_context, file, ensure_ascii=False, indent=2)

    data_cfg = config.get("data", {})
    fe_cfg = config.get("feature_engineering", {})

    budget_cfg = fe_cfg.get("budget", {})
    profile_cfg = fe_cfg.get("profile", {})
    hypothesis_cfg = fe_cfg.get("hypothesis", {})
    implement_cfg = fe_cfg.get("implement", {})
    execute_cfg = fe_cfg.get("execute", {})
    diagnose_cfg = fe_cfg.get("diagnose", {})
    report_cfg = fe_cfg.get("report", {})

    train_path = data_cfg.get("train_path", "data/dacon/train.csv")
    test_path = data_cfg.get("test_path", "data/dacon/test.csv")
    label_col = data_cfg.get("label_col")
    _ = test_path

    client = _load_llm()

    baseline_result = _run_baseline_cv(config=config, train_path=train_path, run_dir=run_dir)
    baseline_cv = baseline_result.get("cv_result", {}) if isinstance(baseline_result, dict) else {}
    baseline_objective = _safe_float(baseline_cv.get("objective_mean"), default=float("-inf"))
    previous_success_objective = baseline_objective

    iter_num = int(budget_cfg.get("iterations", 3))
    diagnose_results: List[Dict] = []
    execute_results: List[Dict] = []
    prev_diagnose_result = None
    iteration_summaries: List[Dict] = []
    successful_iterations: List[Dict] = []

    best_preprocessor_path = ""
    best_preprocessor_iteration: Optional[int] = None
    best_preprocessor_objective = float("-inf")
    kept_feature_blocks: List[Dict[str, Any]] = []

    for iteration in range(1, iter_num + 1):
        print(f"=== Iteration {iteration}/{iter_num} ===")

        iter_dir = os.path.join(run_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        implement_result: Dict[str, Any] = {}
        execute_result: Dict[str, Any] = {}
        execute_attempts: List[Dict[str, Any]] = []
        fallback_count = 0
        execute_appended = False

        keep_reference_objective = previous_success_objective
        selected_prior_block_paths = _merge_unique_paths(
            [str(item.get("path", "")).strip() for item in kept_feature_blocks],
            [],
        )
        current_feature_block_paths: List[str] = []
        execute_feature_block_paths: List[str] = list(selected_prior_block_paths)
        kept_block_added_count = 0
        kept_blocks_this_iteration = False
        keep_decision_reason = "iteration_not_completed"

        try:
            print("  Step 1: Profiling")
            profile_result = profiling(
                client=client,
                profile_cfg=profile_cfg,
                train_path=train_path,
                output_dir=iter_dir,
                iteration=iteration,
                prev_diagnose_result=prev_diagnose_result,
                task_context=task_context,
            )

            print("  Step 2: Hypothesis Generation")
            hypotheses = generate_hypotheses(
                client=client,
                hypothesis_cfg=hypothesis_cfg,
                profile_result=profile_result,
                output_dir=iter_dir,
                prev_diagnose_result=prev_diagnose_result,
                task_context=task_context,
            )

            print("  Step 3: Implement")
            implement_result = implement(
                client=client,
                implement_cfg=implement_cfg,
                profile_result=profile_result,
                hypotheses=hypotheses,
                output_dir=iter_dir,
                train_path=train_path,
                label_col=label_col,
                pipeline_config=config,
                external_feedback=None,
                task_context=task_context,
            )

            print("  Step 4: Execute")
            execute_cfg_for_iter = dict(execute_cfg)
            execute_cfg_for_iter.setdefault("config_path", run_config_path)
            max_fallbacks = int(execute_cfg_for_iter.get("max_implement_fallbacks", 1))
            fallback_count = 0

            for attempt in range(max_fallbacks + 1):
                current_feature_block_paths = _normalize_path_list(
                    implement_result.get("feature_block_module_paths")
                )
                execute_feature_block_paths = _merge_unique_paths(
                    selected_prior_block_paths,
                    current_feature_block_paths,
                )
                execute_implement_result = dict(implement_result)
                execute_implement_result["feature_block_module_paths"] = execute_feature_block_paths
                execute_implement_result["selected_prior_feature_block_paths"] = selected_prior_block_paths
                execute_result = execute(
                    execute_cfg=execute_cfg_for_iter,
                    implement_result=execute_implement_result,
                    iteration_dir=iter_dir,
                )
                execute_attempts.append(execute_result)

                if not execute_result.get("hard_failure", True):
                    break

                if attempt < max_fallbacks:
                    fallback_count += 1
                    print(f"    Hard failure in execute. Re-running implement fallback ({fallback_count}/{max_fallbacks})")
                    implement_feedback = _build_implement_feedback(execute_result)
                    implement_result = implement(
                        client=client,
                        implement_cfg=implement_cfg,
                        profile_result=profile_result,
                        hypotheses=hypotheses,
                        output_dir=iter_dir,
                        train_path=train_path,
                        label_col=label_col,
                        pipeline_config=config,
                        external_feedback=implement_feedback,
                        task_context=task_context,
                    )
                else:
                    break

            print("  Step 5: Diagnose")
            execute_results.append({"iteration": iteration, **execute_result})
            execute_appended = True
            best_before_iteration = max(successful_iterations, key=lambda x: x["objective_mean"]) if successful_iterations else None
            diagnose_result = diagnose(
                client=client,
                diagnose_cfg=diagnose_cfg,
                execute_result=execute_result,
                output_dir=iter_dir,
                iteration=iteration,
                best_before_iteration=best_before_iteration,
                task_context=task_context,
            )
            diagnose_results.append(diagnose_result)
            prev_diagnose_result = diagnose_result

            execute_cv = execute_result.get("cv_result", {}) if isinstance(execute_result, dict) else {}
            objective_value = _safe_float(execute_cv.get("objective_mean"), default=float("-inf"))

            if execute_result.get("success", False):
                if objective_value > keep_reference_objective:
                    for block_path in current_feature_block_paths:
                        if any(item.get("path") == block_path for item in kept_feature_blocks):
                            continue
                        kept_feature_blocks.append(
                            {
                                "path": block_path,
                                "source_iteration": iteration,
                                "trigger_objective_mean": objective_value,
                            }
                        )
                        kept_block_added_count += 1
                    kept_blocks_this_iteration = True
                    keep_decision_reason = (
                        f"objective_improved:{keep_reference_objective:.6f}->{objective_value:.6f}"
                    )
                else:
                    keep_decision_reason = (
                        f"objective_not_improved:{objective_value:.6f}<={keep_reference_objective:.6f}"
                    )

                previous_success_objective = objective_value

                preprocessor_path = str(implement_result.get("preprocessor_module_path", "")).strip()
                if preprocessor_path and objective_value > best_preprocessor_objective:
                    best_preprocessor_objective = objective_value
                    best_preprocessor_path = preprocessor_path
                    best_preprocessor_iteration = iteration

            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "profile_path": os.path.join(iter_dir, "profile", "profile.json"),
                    "hypothesis_path": os.path.join(iter_dir, "hypothesis", "hypothesis.json"),
                    "implement_summary_path": os.path.join(iter_dir, "implement", "implement_summary.json"),
                    "execute_result_path": os.path.join(iter_dir, "execute", "execute_result.json"),
                    "execute_attempts": len(execute_attempts),
                    "implement_fallback_count": fallback_count,
                    "success": bool(execute_result.get("success", False)),
                    "hard_failure": bool(execute_result.get("hard_failure", True)),
                    "reason": str(execute_result.get("reason", "")),
                    "mean_cv": execute_cv.get("mean_cv"),
                    "std_cv": execute_cv.get("std_cv"),
                    "objective_mean": execute_cv.get("objective_mean"),
                    "metric": execute_cv.get("metric"),
                    "diagnose_path": os.path.join(iter_dir, "diagnose", "diagnose.json"),
                    "diagnose_status": diagnose_result.get("status"),
                    "selection_keep_reference_objective": keep_reference_objective,
                    "selection_kept_blocks_this_iteration": kept_blocks_this_iteration,
                    "selection_kept_block_added_count": kept_block_added_count,
                    "selection_keep_decision_reason": keep_decision_reason,
                    "selection_total_kept_block_count": len(kept_feature_blocks),
                    "execute_selected_prior_block_count": len(selected_prior_block_paths),
                    "execute_current_iteration_block_count": len(current_feature_block_paths),
                    "execute_total_block_count": len(execute_feature_block_paths),
                }
            )
            if execute_result.get("success", False):
                successful_iterations.append(
                    {
                        "iteration": iteration,
                        "mean_cv": float(execute_cv["mean_cv"]),
                        "std_cv": float(execute_cv["std_cv"]),
                        "objective_mean": float(execute_cv["objective_mean"]),
                        "metric": str(execute_cv["metric"]),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            error_message = str(exc)
            print(
                f"  [WARN] Iteration {iteration} failed and will continue to next iteration: "
                f"{error_type}: {error_message}"
            )
            error_payload = {
                "iteration": iteration,
                "error_type": error_type,
                "error_message": error_message,
            }
            with open(os.path.join(iter_dir, "iteration_error.json"), "w", encoding="utf-8") as file:
                json.dump(error_payload, file, ensure_ascii=False, indent=2)

            if not execute_result:
                execute_result = {
                    "success": False,
                    "hard_failure": True,
                    "reason": "iteration_exception",
                    "detail": {
                        "error_type": error_type,
                        "error_message": error_message,
                    },
                }

            if not execute_appended:
                execute_results.append({"iteration": iteration, **execute_result})

            execute_cv = execute_result.get("cv_result", {}) if isinstance(execute_result, dict) else {}
            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "profile_path": os.path.join(iter_dir, "profile", "profile.json"),
                    "hypothesis_path": os.path.join(iter_dir, "hypothesis", "hypothesis.json"),
                    "implement_summary_path": os.path.join(iter_dir, "implement", "implement_summary.json"),
                    "execute_result_path": os.path.join(iter_dir, "execute", "execute_result.json"),
                    "execute_attempts": len(execute_attempts),
                    "implement_fallback_count": fallback_count,
                    "success": bool(execute_result.get("success", False)),
                    "hard_failure": bool(execute_result.get("hard_failure", True)),
                    "reason": str(execute_result.get("reason", "iteration_exception")),
                    "mean_cv": execute_cv.get("mean_cv"),
                    "std_cv": execute_cv.get("std_cv"),
                    "objective_mean": execute_cv.get("objective_mean"),
                    "metric": execute_cv.get("metric"),
                    "diagnose_path": os.path.join(iter_dir, "diagnose", "diagnose.json"),
                    "diagnose_status": "skipped_iteration_exception",
                    "iteration_error": error_payload,
                    "selection_keep_reference_objective": keep_reference_objective,
                    "selection_kept_blocks_this_iteration": False,
                    "selection_kept_block_added_count": 0,
                    "selection_keep_decision_reason": "iteration_exception",
                    "selection_total_kept_block_count": len(kept_feature_blocks),
                    "execute_selected_prior_block_count": len(selected_prior_block_paths),
                    "execute_current_iteration_block_count": len(current_feature_block_paths),
                    "execute_total_block_count": len(execute_feature_block_paths),
                }
            )

            if execute_result.get("success", False):
                try:
                    successful_iterations.append(
                        {
                            "iteration": iteration,
                            "mean_cv": float(execute_cv["mean_cv"]),
                            "std_cv": float(execute_cv["std_cv"]),
                            "objective_mean": float(execute_cv["objective_mean"]),
                            "metric": str(execute_cv["metric"]),
                        }
                    )
                except Exception:
                    pass
            continue

    best = max(successful_iterations, key=lambda x: x["objective_mean"]) if successful_iterations else None

    final_selection_result = _build_final_selection_artifacts(
        run_dir=run_dir,
        run_config_path=run_config_path,
        execute_cfg=execute_cfg,
        baseline_result=baseline_result,
        best_preprocessor_path=best_preprocessor_path,
        best_preprocessor_iteration=best_preprocessor_iteration,
        best_preprocessor_objective=best_preprocessor_objective,
        kept_feature_blocks=kept_feature_blocks,
    )
    final_selection_path = str(final_selection_result.get("final_selection_path", "") or "")
    if final_selection_path:
        print(f"[INFO] Final selection saved to: {final_selection_path}")

    report_result = make_report(
        report_cfg=report_cfg,
        diagnose_results=diagnose_results,
        iteration_summaries=iteration_summaries,
        execute_results=execute_results,
        run_dir=run_dir,
        best_summary=best,
    )

    if not successful_iterations:
        raise RuntimeError(
            "No successful iteration found. Check run artifacts for error logs. "
            f"Report: {report_result.get('report_path')}"
        )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "iterations": iteration_summaries,
        "best_summary": best,
        "report_path": report_result.get("report_path"),
        "report_json_path": report_result.get("report_json_path"),
        "final_selection_path": final_selection_result.get("final_selection_path"),
        "final_execute_result_path": final_selection_result.get("final_execute_result_path"),
    }


def _build_task_context(config: Dict[str, Any]) -> Dict[str, Any]:
    fe_cfg = config.get("feature_engineering", {}) if isinstance(config, dict) else {}
    root_task_cfg = config.get("task_description", {}) if isinstance(config, dict) else {}
    fe_task_cfg = fe_cfg.get("task_description", {}) if isinstance(fe_cfg, dict) else {}

    merged_cfg: Dict[str, Any] = {}
    if isinstance(root_task_cfg, dict):
        merged_cfg.update(root_task_cfg)
    if isinstance(fe_task_cfg, dict):
        merged_cfg.update(fe_task_cfg)

    enabled = bool(merged_cfg.get("enabled", True))
    if not enabled:
        return {}

    max_chars = int(merged_cfg.get("max_chars", 3000))
    raw_text = str(merged_cfg.get("text", "") or "").strip()
    source_path = str(merged_cfg.get("path", "") or "").strip()

    if not raw_text and source_path:
        path = Path(source_path)
        if path.exists() and path.is_file():
            raw_text = path.read_text(encoding="utf-8")
        else:
            print(f"[WARN] task_description.path not found: {path}")

    raw_text = str(raw_text or "").strip()
    if not raw_text:
        return {}

    if source_path:
        print(f"[INFO] Loaded task description context from: {source_path}")

    sections = _parse_sectioned_text(raw_text)
    topic = sections.get("주제") or sections.get("topic") or ""
    description = sections.get("설명") or sections.get("description") or ""
    background = sections.get("배경") or sections.get("background") or ""

    task_summary_source = description or topic or raw_text
    task_summary = _truncate_text(task_summary_source, limit=max_chars)
    metric_hint = _extract_metric_hint(raw_text)
    evaluation_hint = _extract_evaluation_hint(description or raw_text)

    background_brief = _truncate_text(background, limit=500) if background else ""

    return {
        "topic": _truncate_text(topic, limit=240),
        "task_summary": task_summary,
        "metric_hint": metric_hint,
        "evaluation_hint": evaluation_hint,
        "background_brief": background_brief,
        "source_path": source_path,
    }


def _parse_sectioned_text(raw_text: str) -> Dict[str, str]:
    text = str(raw_text or "")
    header_pattern = re.compile(r"^\[([^\]]+)\]\s*$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    if not matches:
        return {"raw": text.strip()}

    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        title = str(match.group(1)).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections[title] = body
    return sections


def _extract_metric_hint(text: str) -> str:
    candidates = [
        r"Binary\s*F1(?:\s*Score)?",
        r"\bF1(?:\s*Score)?\b",
        r"ROC[\s\-]?AUC",
        r"\bAUC\b",
        r"\bRMSE\b",
        r"\bMAE\b",
        r"\bLogLoss\b",
        r"\bAccuracy\b",
    ]
    for pattern in candidates:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def _extract_evaluation_hint(text: str) -> str:
    lines = [str(line).strip() for line in str(text or "").splitlines()]
    selected: List[str] = []
    for line in lines:
        if not line:
            continue
        if any(key in line for key in ["평가", "Public", "Private", "리더보드", "Score", "코드 검증"]):
            selected.append(line)
        if len(selected) >= 8:
            break
    return " | ".join(selected)


def _truncate_text(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if limit <= 0:
        return ""
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _build_implement_feedback(execute_result: Dict[str, Any]) -> Dict[str, Any]:
    detail = execute_result.get("detail", {}) if isinstance(execute_result, dict) else {}
    stderr_tail = str(detail.get("stderr_tail", ""))
    stdout_tail = str(detail.get("stdout_tail", ""))

    stderr_path = detail.get("stderr_path")
    if not stderr_tail and isinstance(stderr_path, str) and os.path.exists(stderr_path):
        try:
            with open(stderr_path, "r", encoding="utf-8") as file:
                stderr_tail = file.read()[-4000:]
        except Exception:
            stderr_tail = ""

    stdout_path = detail.get("stdout_path")
    if not stdout_tail and isinstance(stdout_path, str) and os.path.exists(stdout_path):
        try:
            with open(stdout_path, "r", encoding="utf-8") as file:
                stdout_tail = file.read()[-2000:]
        except Exception:
            stdout_tail = ""

    return {
        "hard_failure": bool(execute_result.get("hard_failure", True)) if isinstance(execute_result, dict) else True,
        "reason": str(execute_result.get("reason", "unknown_execute_failure")) if isinstance(execute_result, dict) else "unknown_execute_failure",
        "detail": detail,
        "stderr_tail": stderr_tail,
        "stdout_tail": stdout_tail,
    }


def _normalize_path_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _merge_unique_paths(left: List[str], right: List[str]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for raw in list(left) + list(right):
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _run_baseline_cv(config: Dict[str, Any], train_path: str, run_dir: str) -> Dict[str, Any]:
    baseline_dir = Path(run_dir) / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_result_path = baseline_dir / "baseline_result.json"
    baseline_cv_path = baseline_dir / "baseline_cv_result.json"

    try:
        train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        preprocessor = _BaselinePreprocessor()
        feature_engineering = _BaselineFeatureEngineering()
        cv_result = run_cross_validation(
            config=config,
            train_df=train_df,
            preprocessor_module=preprocessor,
            feature_module=feature_engineering,
            enabled_blocks=None,
        )
        payload = {
            "success": True,
            "train_path": train_path,
            "cv_result_path": str(baseline_cv_path),
            "cv_result": cv_result,
        }
        with baseline_cv_path.open("w", encoding="utf-8") as file:
            json.dump(cv_result, file, ensure_ascii=False, indent=2)
        with baseline_result_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(
            "=== Baseline ===\n"
            f"  objective_mean={cv_result.get('objective_mean')} "
            f"metric={cv_result.get('metric')} "
            f"mean_cv={cv_result.get('mean_cv')}"
        )
        return payload
    except Exception as exc:  # noqa: BLE001
        payload = {
            "success": False,
            "train_path": train_path,
            "error": str(exc),
            "cv_result_path": str(baseline_cv_path),
        }
        with baseline_result_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(f"=== Baseline ===\n  failed: {exc}")
        return payload


class _BaselinePreprocessor:
    def fit_preprocessor(self, train_df: pd.DataFrame, label_col: str, config: Dict[str, Any]) -> Dict[str, Any]:
        del config
        label = str(label_col)
        features = train_df.drop(columns=[label], errors="ignore").copy()
        features.columns = [str(col) for col in features.columns]
        feature_columns = list(features.columns)

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        numeric_fill: Dict[str, float] = {}
        categorical_fill: Dict[str, str] = {}

        for col in feature_columns:
            series = features[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                numeric_series = pd.to_numeric(series, errors="coerce")
                median = numeric_series.median()
                numeric_fill[col] = float(0.0 if pd.isna(median) else median)
            else:
                categorical_cols.append(col)
                string_series = series.astype("string")
                modes = string_series.dropna().mode()
                categorical_fill[col] = str(modes.iloc[0]) if len(modes) > 0 else "__MISSING__"

        return {
            "label_col": label,
            "feature_columns": feature_columns,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "numeric_fill": numeric_fill,
            "categorical_fill": categorical_fill,
        }

    def transform_preprocessor(self, df: pd.DataFrame, prep_state: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        del config
        label_col = str(prep_state.get("label_col", ""))
        feature_columns = [str(col) for col in prep_state.get("feature_columns", [])]
        numeric_cols = [str(col) for col in prep_state.get("numeric_cols", [])]
        categorical_cols = [str(col) for col in prep_state.get("categorical_cols", [])]
        numeric_fill = {
            str(key): _safe_float(value, default=0.0)
            for key, value in dict(prep_state.get("numeric_fill", {})).items()
        }
        categorical_fill = {
            str(key): str(value)
            for key, value in dict(prep_state.get("categorical_fill", {})).items()
        }

        df_in = df.copy()
        label_series = df_in[label_col].copy() if label_col and label_col in df_in.columns else None
        feature_df = df_in.drop(columns=[label_col], errors="ignore").copy()
        feature_df.columns = [str(col) for col in feature_df.columns]

        for col in feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = pd.NA
        feature_df = feature_df.reindex(columns=feature_columns)

        for col in numeric_cols:
            fill_value = numeric_fill.get(col, 0.0)
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(fill_value)

        for col in categorical_cols:
            fill_value = categorical_fill.get(col, "__MISSING__")
            feature_df[col] = feature_df[col].astype("string").fillna(fill_value)

        if label_series is not None and label_col:
            feature_df[label_col] = label_series.values
        return feature_df


class _BaselineFeatureEngineering:
    FEATURE_BLOCKS = {
        "base": {
            "description": "Baseline identity feature set",
            "enabled_by_default": True,
        }
    }

    def fit_feature_engineering(
        self,
        train_df: pd.DataFrame,
        label_col: str,
        config: Dict[str, Any],
        enabled_blocks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        del config, enabled_blocks
        feature_cols = [str(col) for col in train_df.columns if str(col) != str(label_col)]
        return {"feature_cols": feature_cols}

    def transform_feature_engineering(self, df: pd.DataFrame, fe_state: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        label_col = ""
        if isinstance(config, dict):
            data_cfg = config.get("data", {})
            if isinstance(data_cfg, dict):
                label_col = str(data_cfg.get("label_col", "") or "")
        feature_cols = [str(col) for col in fe_state.get("feature_cols", [])]
        out = df.drop(columns=[label_col], errors="ignore").copy() if label_col else df.copy()
        out.columns = [str(col) for col in out.columns]
        for col in feature_cols:
            if col not in out.columns:
                out[col] = pd.NA
        out = out.reindex(columns=feature_cols)
        return out

    def feature_registry_from_state(self, fe_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        cols = [str(col) for col in fe_state.get("feature_cols", [])]
        return [{"feature": col, "block": "base"} for col in cols]


def _build_final_selection_artifacts(
    run_dir: str,
    run_config_path: str,
    execute_cfg: Dict[str, Any],
    baseline_result: Dict[str, Any],
    best_preprocessor_path: str,
    best_preprocessor_iteration: Optional[int],
    best_preprocessor_objective: float,
    kept_feature_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    final_dir = Path(run_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_selection_path = final_dir / "final_selection.json"

    selected_feature_block_paths: List[str] = []
    for item in kept_feature_blocks:
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        if path in selected_feature_block_paths:
            continue
        selected_feature_block_paths.append(path)

    payload: Dict[str, Any] = {
        "selection_policy": "baseline_then_keep_if_objective_improves_vs_previous_success",
        "baseline_result_path": str(Path(run_dir) / "baseline" / "baseline_result.json"),
        "baseline_success": bool(baseline_result.get("success", False)) if isinstance(baseline_result, dict) else False,
        "baseline_objective_mean": (
            baseline_result.get("cv_result", {}).get("objective_mean")
            if isinstance(baseline_result, dict)
            else None
        ),
        "best_preprocessor": {
            "path": best_preprocessor_path,
            "iteration": best_preprocessor_iteration,
            "objective_mean": best_preprocessor_objective,
        },
        "kept_feature_blocks": kept_feature_blocks,
        "selected_preprocessor_module_path": best_preprocessor_path,
        "selected_feature_block_module_paths": selected_feature_block_paths,
        "selected_feature_block_count": len(selected_feature_block_paths),
    }

    final_execute_result: Dict[str, Any] = {}
    if best_preprocessor_path:
        print("=== Final Selection ===")
        print(
            "  Step Final: Execute selected modules "
            f"(preprocessor_iter={best_preprocessor_iteration}, kept_blocks={len(selected_feature_block_paths)})"
        )
        final_implement_result = {
            "pipeline_script_path": str(final_dir / "execute" / "assembled_pipeline.py"),
            "preprocessor_module_path": best_preprocessor_path,
            "feature_block_module_paths": selected_feature_block_paths,
        }
        execute_cfg_for_final = dict(execute_cfg)
        execute_cfg_for_final.setdefault("config_path", run_config_path)
        final_execute_result = execute(
            execute_cfg=execute_cfg_for_final,
            implement_result=final_implement_result,
            iteration_dir=str(final_dir),
        )
        payload["final_execute_result_path"] = str(final_dir / "execute" / "execute_result.json")
        payload["final_execute_success"] = bool(final_execute_result.get("success", False))
        payload["final_execute_reason"] = str(final_execute_result.get("reason", ""))
        payload["final_cv_result"] = final_execute_result.get("cv_result", {})
    else:
        payload["final_execute_result_path"] = ""
        payload["final_execute_success"] = False
        payload["final_execute_reason"] = "missing_best_preprocessor"
        payload["final_cv_result"] = {}

    with final_selection_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return {
        "final_selection_path": str(final_selection_path),
        "final_execute_result_path": str(payload.get("final_execute_result_path", "")),
        "final_execute_result": final_execute_result,
    }
