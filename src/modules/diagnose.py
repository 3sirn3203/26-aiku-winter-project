from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional


def diagnose(
    diagnose_cfg: Dict[str, Any],
    execute_result: Dict[str, Any],
    output_dir: str,
    iteration: int,
    best_before_iteration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diagnose_dir = os.path.join(output_dir, "diagnose")
    os.makedirs(diagnose_dir, exist_ok=True)

    max_log_tail_chars = int(diagnose_cfg.get("max_log_tail_chars", 4000))
    std_warning_threshold = float(diagnose_cfg.get("std_warning_threshold", 0.02))
    feature_count_warning_threshold = int(diagnose_cfg.get("feature_count_warning_threshold", 700))

    success = bool(execute_result.get("success", False))
    hard_failure = bool(execute_result.get("hard_failure", True))
    reason = str(execute_result.get("reason", ""))
    detail = execute_result.get("detail", {}) if isinstance(execute_result, dict) else {}
    cv_result = execute_result.get("cv_result", {}) if isinstance(execute_result, dict) else {}

    stderr_tail = _extract_log_tail(
        inline_text=detail.get("stderr_tail", ""),
        file_path=detail.get("stderr_path", ""),
        max_chars=max_log_tail_chars,
    )
    stdout_tail = _extract_log_tail(
        inline_text=detail.get("stdout_tail", ""),
        file_path=detail.get("stdout_path", ""),
        max_chars=max_log_tail_chars,
    )

    root_cause = _analyze_root_cause(
        reason=reason,
        stderr_tail=stderr_tail,
        stdout_tail=stdout_tail,
    )

    score_summary = _extract_score_summary(cv_result)
    comparison = _compare_with_previous_best(score_summary=score_summary, best_before_iteration=best_before_iteration)
    feedback = _build_feedback(
        success=success,
        hard_failure=hard_failure,
        root_cause=root_cause,
        score_summary=score_summary,
        comparison=comparison,
        std_warning_threshold=std_warning_threshold,
        feature_count_warning_threshold=feature_count_warning_threshold,
    )

    diagnose_result = {
        "iteration": iteration,
        "status": "success" if success else "failed",
        "hard_failure": hard_failure,
        "reason": reason,
        "root_cause": root_cause,
        "score_summary": score_summary,
        "comparison_to_best_before_iteration": comparison,
        "feedback_for_next_iteration": feedback,
        "artifacts": {
            "execute_stdout_path": execute_result.get("stdout_path"),
            "execute_stderr_path": execute_result.get("stderr_path"),
            "cv_result_path": execute_result.get("cv_result_path"),
        },
    }

    with open(os.path.join(diagnose_dir, "diagnose.json"), "w", encoding="utf-8") as file:
        json.dump(diagnose_result, file, ensure_ascii=False, indent=2)
    with open(os.path.join(diagnose_dir, "diagnose_logs.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "stderr_tail": stderr_tail,
                "stdout_tail": stdout_tail,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return diagnose_result


def _extract_log_tail(inline_text: Any, file_path: Any, max_chars: int) -> str:
    text = str(inline_text or "").strip()
    if text:
        return text[-max_chars:]

    path = str(file_path or "").strip()
    if not path:
        return ""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()[-max_chars:]
    except Exception:
        return ""


def _analyze_root_cause(reason: str, stderr_tail: str, stdout_tail: str) -> Dict[str, Any]:
    merged = f"{stderr_tail}\n{stdout_tail}".lower()

    if reason == "generated_module_syntax_error" or "syntaxerror" in merged or "invalid syntax" in merged:
        return {
            "category": "syntax",
            "message": "Generated module has syntax errors.",
            "confidence": "high",
        }
    if "missing required function" in merged:
        return {
            "category": "interface_contract",
            "message": "Generated module does not satisfy the required function interface.",
            "confidence": "high",
        }
    if "do not support special json characters in feature name" in merged:
        return {
            "category": "feature_name_sanitization",
            "message": "Model rejected feature names due to unsupported characters.",
            "confidence": "high",
        }
    if "none of [index" in merged and "are in the [columns]" in merged:
        return {
            "category": "schema_mismatch",
            "message": "Feature schema mismatch between train/validation/test transformations.",
            "confidence": "high",
        }
    if "timeout" in reason.lower() or "timeout" in merged:
        return {
            "category": "timeout",
            "message": "Execution exceeded configured time budget.",
            "confidence": "medium",
        }
    if "importerror" in merged or "modulenotfounderror" in merged:
        return {
            "category": "dependency",
            "message": "A required library is missing in runtime environment.",
            "confidence": "medium",
        }
    if "memory" in merged or "out of memory" in merged:
        return {
            "category": "resource",
            "message": "Execution likely failed due to memory/resource constraints.",
            "confidence": "medium",
        }

    category = "unknown"
    if reason == "validation_process_failed":
        category = "validator_runtime"
    elif reason:
        category = reason
    return {
        "category": category,
        "message": "No specific failure pattern matched. Inspect stderr logs.",
        "confidence": "low",
    }


def _extract_score_summary(cv_result: Dict[str, Any]) -> Dict[str, Any]:
    mean_cv = _safe_float(cv_result.get("mean_cv"))
    std_cv = _safe_float(cv_result.get("std_cv"))
    objective_mean = _safe_float(cv_result.get("objective_mean"))
    fold_scores = cv_result.get("fold_scores", [])
    metric = cv_result.get("metric")
    feature_registry = cv_result.get("feature_registry", [])
    feature_blocks = cv_result.get("feature_blocks", {})

    return {
        "metric": str(metric) if metric is not None else None,
        "mean_cv": mean_cv,
        "std_cv": std_cv,
        "objective_mean": objective_mean,
        "n_folds": len(fold_scores) if isinstance(fold_scores, list) else 0,
        "n_features": len(feature_registry) if isinstance(feature_registry, list) else None,
        "feature_blocks": list(feature_blocks.keys()) if isinstance(feature_blocks, dict) else [],
        "is_valid": not _is_nan(mean_cv),
    }


def _compare_with_previous_best(
    score_summary: Dict[str, Any],
    best_before_iteration: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    current_obj = _safe_float(score_summary.get("objective_mean"))
    if not best_before_iteration:
        return {
            "has_previous_best": False,
            "delta_objective_mean": None,
            "message": "No previous successful iteration.",
        }

    previous_obj = _safe_float(best_before_iteration.get("objective_mean"))
    if _is_nan(current_obj) or _is_nan(previous_obj):
        return {
            "has_previous_best": True,
            "delta_objective_mean": None,
            "message": "Comparison unavailable due to NaN objective.",
        }

    delta = float(current_obj - previous_obj)
    trend = "improved" if delta > 0 else ("degraded" if delta < 0 else "unchanged")
    return {
        "has_previous_best": True,
        "previous_best_iteration": best_before_iteration.get("iteration"),
        "previous_best_objective_mean": previous_obj,
        "delta_objective_mean": delta,
        "trend": trend,
    }


def _build_feedback(
    success: bool,
    hard_failure: bool,
    root_cause: Dict[str, Any],
    score_summary: Dict[str, Any],
    comparison: Dict[str, Any],
    std_warning_threshold: float,
    feature_count_warning_threshold: int,
) -> Dict[str, Any]:
    profile_focus: List[str] = []
    hypothesis_focus: List[str] = []
    implement_constraints: List[str] = []
    priority_actions: List[str] = []

    if not success:
        category = str(root_cause.get("category", "unknown"))
        if category == "syntax":
            implement_constraints.extend(
                [
                    "Return pure executable Python only, without markdown or prose.",
                    "Ensure generated scripts pass local syntax compilation before execute.",
                ]
            )
            priority_actions.append("Stabilize code generation syntax first.")
        elif category in {"interface_contract", "schema_mismatch"}:
            implement_constraints.extend(
                [
                    "Strictly follow required function signatures for preprocessor/feature engineering.",
                    "Keep train/validation/test feature schemas identical after transform.",
                ]
            )
            priority_actions.append("Fix module interface and schema consistency.")
        elif category == "feature_name_sanitization":
            implement_constraints.extend(
                [
                    "Sanitize generated feature names to [A-Za-z0-9_].",
                    "Deduplicate sanitized names with deterministic suffixes.",
                ]
            )
            priority_actions.append("Normalize feature names before model fitting.")
        elif category == "timeout":
            hypothesis_focus.append("Prefer lower-cost transformations and reduce expensive text/vector operations.")
            implement_constraints.append("Reduce computational complexity and feature explosion.")
            priority_actions.append("Reduce runtime and memory footprint.")
        elif category == "dependency":
            implement_constraints.append("Avoid optional dependencies not guaranteed in runtime environment.")
            priority_actions.append("Use standard library / installed package set only.")
        else:
            priority_actions.append("Inspect execute stderr and patch the failing path with explicit guardrails.")

        profile_focus.append("Focus on failure root cause from previous iteration logs.")
        if hard_failure:
            priority_actions.insert(0, "Prioritize execution stability over score improvement.")
    else:
        std_cv = _safe_float(score_summary.get("std_cv"))
        n_features = score_summary.get("n_features")
        if not _is_nan(std_cv) and std_cv > std_warning_threshold:
            hypothesis_focus.append("Prioritize robust features that generalize across folds.")
            implement_constraints.append("Avoid brittle high-variance transformations.")
            priority_actions.append("Reduce fold variance (std_cv) while preserving mean_cv.")

        if isinstance(n_features, int) and n_features > feature_count_warning_threshold:
            hypothesis_focus.append("Reduce feature dimensionality and remove weak/high-cardinality expansions.")
            implement_constraints.append("Control feature explosion with stricter caps and pruning.")
            priority_actions.append("Shrink feature set for stability and speed.")

        if comparison.get("has_previous_best") and comparison.get("delta_objective_mean") is not None:
            delta = float(comparison["delta_objective_mean"])
            if delta <= 0:
                profile_focus.append("Investigate why latest features underperformed prior best.")
                priority_actions.append("Try incremental modifications from previous best instead of full redesign.")
            else:
                profile_focus.append("Preserve high-impact transformations from current best.")
                priority_actions.append("Use current iteration as baseline and run targeted ablation next.")
        else:
            priority_actions.append("Collect one more successful iteration for reliable trend comparison.")

    if not profile_focus:
        profile_focus.append("Review missingness, cardinality, and leakage candidates for targeted next hypotheses.")
    if not hypothesis_focus:
        hypothesis_focus.append("Generate small, testable feature hypotheses with clear expected impact.")
    if not implement_constraints:
        implement_constraints.append("Keep deterministic transformations and stable output schema.")

    return {
        "profile_focus": profile_focus,
        "hypothesis_focus": hypothesis_focus,
        "implement_constraints": implement_constraints,
        "priority_actions": priority_actions,
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return True
