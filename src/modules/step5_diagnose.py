from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from jinja2 import Template
from pydantic import BaseModel, Field, ValidationError

DIAGNOSE_SYSTEM_INSTRUCTION = """You are a diagnose agent for iterative tabular feature engineering.
Return only a valid JSON object matching the schema.
Do not include markdown fences.
Do not include explanations outside JSON."""


class RootCauseResponse(BaseModel):
    category: str = "unknown"
    message: str = ""
    confidence: str = "low"


class FeedbackResponse(BaseModel):
    profile_focus: List[str] = Field(default_factory=list)
    hypothesis_focus: List[str] = Field(default_factory=list)
    implement_constraints: List[str] = Field(default_factory=list)
    priority_actions: List[str] = Field(default_factory=list)


class DiagnoseResponse(BaseModel):
    summary: str = ""
    root_cause: RootCauseResponse = Field(default_factory=RootCauseResponse)
    feedback_for_next_iteration: FeedbackResponse = Field(default_factory=FeedbackResponse)


def diagnose(
    client: genai.Client,
    diagnose_cfg: Dict[str, Any],
    execute_result: Dict[str, Any],
    output_dir: str,
    iteration: int,
    best_before_iteration: Optional[Dict[str, Any]] = None,
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diagnose_dir = os.path.join(output_dir, "diagnose")
    os.makedirs(diagnose_dir, exist_ok=True)

    max_log_tail_chars = int(diagnose_cfg.get("max_log_tail_chars", 4000))
    prompt_log_tail_chars = int(diagnose_cfg.get("prompt_log_tail_chars", min(2000, max_log_tail_chars)))
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

    score_summary = _extract_score_summary(cv_result)
    comparison = _compare_with_previous_best(score_summary=score_summary, best_before_iteration=best_before_iteration)
    fallback_root_cause = _analyze_root_cause(reason=reason, stderr_tail=stderr_tail, stdout_tail=stdout_tail)
    fallback_feedback = _build_feedback(
        success=success,
        hard_failure=hard_failure,
        root_cause=fallback_root_cause,
        score_summary=score_summary,
        comparison=comparison,
        std_warning_threshold=std_warning_threshold,
        feature_count_warning_threshold=feature_count_warning_threshold,
    )
    fallback_summary = _build_fallback_summary(
        success=success,
        reason=reason,
        score_summary=score_summary,
        comparison=comparison,
    )

    prompt_context = {
        "iteration": iteration,
        "task_context": task_context if isinstance(task_context, dict) else {},
        "execution_status": {
            "success": success,
            "hard_failure": hard_failure,
            "reason": reason,
        },
        "score_summary": score_summary,
        "comparison_to_best_before_iteration": comparison,
        "stderr_tail": stderr_tail[-prompt_log_tail_chars:],
        "stdout_tail": stdout_tail[-prompt_log_tail_chars:],
        "rule_based_hint": {
            "root_cause": fallback_root_cause,
            "feedback_for_next_iteration": fallback_feedback,
            "summary": fallback_summary,
        },
    }

    llm_output, llm_meta = _run_diagnose_llm(
        client=client,
        diagnose_cfg=diagnose_cfg,
        prompt_context=prompt_context,
        diagnose_dir=diagnose_dir,
    )

    if llm_output is None:
        summary = fallback_summary
        root_cause = fallback_root_cause
        feedback = fallback_feedback
        llm_used = False
    else:
        summary = str(llm_output.summary).strip() or fallback_summary
        root_cause = llm_output.root_cause.model_dump()
        feedback = llm_output.feedback_for_next_iteration.model_dump()
        llm_used = True

    diagnose_result = {
        "iteration": iteration,
        "status": "success" if success else "failed",
        "hard_failure": hard_failure,
        "reason": reason,
        "summary": summary,
        "root_cause": root_cause,
        "score_summary": score_summary,
        "comparison_to_best_before_iteration": comparison,
        "feedback_for_next_iteration": feedback,
        "llm": {
            "used": llm_used,
            **llm_meta,
        },
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


def _run_diagnose_llm(
    client: genai.Client,
    diagnose_cfg: Dict[str, Any],
    prompt_context: Dict[str, Any],
    diagnose_dir: str,
) -> tuple[Optional[DiagnoseResponse], Dict[str, Any]]:
    prompt_path = Path("src/prompt/5_diagnose.j2")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    template = Template(prompt_path.read_text(encoding="utf-8"))
    prompt = template.render(
        context_json=json.dumps(prompt_context, ensure_ascii=False),
        task_context_json=(
            json.dumps(prompt_context.get("task_context"), ensure_ascii=False)
            if isinstance(prompt_context.get("task_context"), dict) and prompt_context.get("task_context")
            else "null"
        ),
    )

    model = str(diagnose_cfg.get("model", "gemini-2.5-flash"))
    temperature = float(diagnose_cfg.get("temperature", 0.2))
    top_p = float(diagnose_cfg.get("top_p", 0.9))
    max_output_tokens = int(diagnose_cfg.get("max_output_tokens", diagnose_cfg.get("max_tokens", 4096)))
    max_attempts = int(diagnose_cfg.get("max_attempts", 2))
    system_instruction = str(diagnose_cfg.get("system_instruction", DIAGNOSE_SYSTEM_INSTRUCTION))

    last_raw_text = ""
    last_error = ""
    parsed_out: Optional[DiagnoseResponse] = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",
                    "response_schema": DiagnoseResponse,
                    "system_instruction": system_instruction,
                },
            )
            raw_text = str(getattr(response, "text", "") or "").strip()
            last_raw_text = raw_text
            parsed_out = _parse_diagnose_response(response=response, raw_text=raw_text)
            last_error = ""
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            raw_text = last_raw_text
            with open(os.path.join(diagnose_dir, f"diagnose_attempt_{attempt}.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "attempt": attempt,
                        "error": last_error,
                        "raw_text": raw_text,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

    with open(os.path.join(diagnose_dir, "diagnose_raw_response.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
                "max_attempts": max_attempts,
                "system_instruction": system_instruction,
                "last_error": last_error,
                "raw_text": last_raw_text,
                "prompt_context": prompt_context,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return parsed_out, {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "max_attempts": max_attempts,
        "last_error": last_error,
    }


def _parse_diagnose_response(response: Any, raw_text: str) -> DiagnoseResponse:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, DiagnoseResponse):
            return parsed
        try:
            return DiagnoseResponse.model_validate(parsed)
        except ValidationError:
            pass

    raw_obj = _parse_json_response(raw_text)
    return DiagnoseResponse.model_validate(raw_obj)


def _parse_json_response(raw_text: str) -> Any:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def _build_fallback_summary(
    success: bool,
    reason: str,
    score_summary: Dict[str, Any],
    comparison: Dict[str, Any],
) -> str:
    metric = score_summary.get("metric")
    mean_cv = score_summary.get("mean_cv")
    std_cv = score_summary.get("std_cv")
    base = f"Execute {'succeeded' if success else 'failed'}"
    if reason:
        base += f" (reason={reason})"
    if metric is not None and not _is_nan(mean_cv):
        base += f"; {metric} mean={mean_cv:.6f}, std={std_cv:.6f}" if not _is_nan(std_cv) else f"; {metric} mean={mean_cv:.6f}"

    delta = comparison.get("delta_objective_mean")
    if delta is not None and not _is_nan(delta):
        base += f"; delta objective vs previous best={float(delta):+.6f}"
    return base


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
        return {"category": "syntax", "message": "Generated module has syntax errors.", "confidence": "high"}
    if "missing required function" in merged:
        return {"category": "interface_contract", "message": "Generated module does not satisfy required functions.", "confidence": "high"}
    if "do not support special json characters in feature name" in merged:
        return {"category": "feature_name_sanitization", "message": "Model rejected unsupported feature names.", "confidence": "high"}
    if "none of [index" in merged and "are in the [columns]" in merged:
        return {"category": "schema_mismatch", "message": "Feature schema mismatch between splits.", "confidence": "high"}
    if "timeout" in reason.lower() or "timeout" in merged:
        return {"category": "timeout", "message": "Execution exceeded configured time budget.", "confidence": "medium"}
    if "importerror" in merged or "modulenotfounderror" in merged:
        return {"category": "dependency", "message": "Required package is missing in runtime.", "confidence": "medium"}
    if "memory" in merged or "out of memory" in merged:
        return {"category": "resource", "message": "Likely failed due to memory/resource constraints.", "confidence": "medium"}

    category = "validator_runtime" if reason == "validation_process_failed" else (reason or "unknown")
    return {"category": category, "message": "No specific failure pattern matched.", "confidence": "low"}


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
                    "Return pure executable Python only, without markdown/prose.",
                    "Ensure generated scripts pass syntax check before execute.",
                ]
            )
            priority_actions.append("Stabilize code syntax first.")
        elif category in {"interface_contract", "schema_mismatch"}:
            implement_constraints.extend(
                [
                    "Strictly follow required function signatures.",
                    "Keep train/validation/test feature schema identical.",
                ]
            )
            priority_actions.append("Fix module interface and schema consistency.")
        elif category == "feature_name_sanitization":
            implement_constraints.extend(
                [
                    "Sanitize feature names to [A-Za-z0-9_].",
                    "Deduplicate sanitized names deterministically.",
                ]
            )
            priority_actions.append("Normalize feature names before model fitting.")
        elif category == "timeout":
            hypothesis_focus.append("Prefer lower-cost transformations and reduce expensive vector operations.")
            implement_constraints.append("Reduce computational complexity and feature explosion.")
            priority_actions.append("Reduce runtime and memory footprint.")
        elif category == "dependency":
            implement_constraints.append("Avoid optional dependencies not guaranteed in runtime.")
            priority_actions.append("Use stable dependency set.")
        else:
            priority_actions.append("Inspect execute stderr and patch failing path explicitly.")

        profile_focus.append("Focus on previous failure root cause.")
        if hard_failure:
            priority_actions.insert(0, "Prioritize execution stability over score improvement.")
    else:
        std_cv = _safe_float(score_summary.get("std_cv"))
        n_features = score_summary.get("n_features")

        if not _is_nan(std_cv) and std_cv > std_warning_threshold:
            hypothesis_focus.append("Prioritize robust features that generalize across folds.")
            implement_constraints.append("Avoid brittle high-variance transformations.")
            priority_actions.append("Reduce fold variance while preserving mean CV.")

        if isinstance(n_features, int) and n_features > feature_count_warning_threshold:
            hypothesis_focus.append("Reduce weak/high-cardinality expansions.")
            implement_constraints.append("Control feature explosion with stricter caps.")
            priority_actions.append("Shrink feature set for stability and speed.")

        if comparison.get("has_previous_best") and comparison.get("delta_objective_mean") is not None:
            delta = float(comparison["delta_objective_mean"])
            if delta <= 0:
                profile_focus.append("Investigate why latest features underperformed prior best.")
                priority_actions.append("Use incremental modifications from previous best.")
            else:
                profile_focus.append("Preserve high-impact transformations from current best.")
                priority_actions.append("Use current best as baseline for targeted ablation.")
        else:
            priority_actions.append("Collect more successful iterations for reliable trend analysis.")

    if not profile_focus:
        profile_focus.append("Review missingness/cardinality/leakage candidates for next hypotheses.")
    if not hypothesis_focus:
        hypothesis_focus.append("Generate small, testable hypotheses with explicit expected impact.")
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
