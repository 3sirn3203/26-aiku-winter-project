from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from jinja2 import Template
from pydantic import BaseModel, Field, ValidationError

HYPOTHESIS_SYSTEM_INSTRUCTION = """You generate hypotheses for tabular ML.
Return only a valid JSON object matching the schema.
Do not include markdown fences or explanations outside JSON.
For each feature_engineering item, propose one concrete derived feature that combines multiple source columns.
Also generate direct coding instructions for Step3 implementation."""

HYPOTHESIS_WEB_RESEARCH_SYSTEM_INSTRUCTION = """You are a web research agent for tabular ML feature engineering.
Use Google Search to find practical preprocessing and feature engineering methods.
Never suggest collecting additional external tabular datasets.
Return only a valid JSON object matching the schema."""

HYPOTHESIS_WEB_RESEARCH_PROMPT = """Given the dataset profile context below, research high-impact methods for:
1) preprocessing
2) feature engineering

Requirements:
- Use web search to ground suggestions in current best practices.
- Keep methods practical for tabular competition pipelines.
- Avoid methods requiring external tabular data collection.
- Prioritize methods that can be validated with CV quickly.
- Use previous diagnose context (if provided) to avoid repeating failed directions.

Return schema:
{
  "summary": "string",
  "findings": [
    {
      "topic": "string",
      "why_relevant": "string",
      "recommended_actions": ["string", "..."],
      "source_urls": ["https://...", "..."]
    }
  ]
}

Task context JSON (nullable):
{task_context_json}

Previous diagnose context JSON (nullable):
{previous_diagnose_json}

Profile context JSON:
{profile_context_json}
"""


class HypothesisResponse(BaseModel):
    preprocessing: List[str] = Field(default_factory=list)
    feature_engineering: List[str] = Field(default_factory=list)
    preprocessing_codegen_instruction: str = ""
    feature_engineering_codegen_instructions: List[str] = Field(default_factory=list)


class WebResearchFinding(BaseModel):
    topic: str = ""
    why_relevant: str = ""
    recommended_actions: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class WebResearchResponse(BaseModel):
    summary: str = ""
    findings: List[WebResearchFinding] = Field(default_factory=list)


def generate_hypotheses(
    client: genai.Client,
    hypothesis_cfg: Dict[str, Any],
    profile_result: Dict[str, Any],
    output_dir: str,
    prev_diagnose_result: Optional[Dict[str, Any]] = None,
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hypothesis_dir = os.path.join(output_dir, "hypothesis")
    os.makedirs(hypothesis_dir, exist_ok=True)

    prompt_path = Path("src/prompt/2_hypothesis.j2")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    previous_diagnose_context = _build_previous_diagnose_context(
        prev_diagnose_result=prev_diagnose_result,
        hypothesis_cfg=hypothesis_cfg,
    )

    web_research = _run_hypothesis_web_research(
        client=client,
        hypothesis_cfg=hypothesis_cfg,
        profile_result=profile_result,
        hypothesis_dir=hypothesis_dir,
        previous_diagnose_context=previous_diagnose_context,
        task_context=task_context,
    )

    prompt_template = Template(prompt_path.read_text(encoding="utf-8"))
    feature_engineering_hypothesis_count = _resolve_feature_engineering_hypothesis_count(hypothesis_cfg)
    profile_context_for_hypothesis = _build_hypothesis_profile_context(
        profile_result=profile_result,
        hypothesis_cfg=hypothesis_cfg,
    )
    prompt = prompt_template.render(
        profile_result_json=json.dumps(profile_context_for_hypothesis, ensure_ascii=False),
        web_research_json=json.dumps(web_research.get("context"), ensure_ascii=False),
        feature_engineering_count_instruction=_build_feature_engineering_count_instruction(
            feature_engineering_hypothesis_count
        ),
        task_context_json=(
            json.dumps(task_context, ensure_ascii=False)
            if isinstance(task_context, dict) and task_context
            else "null"
        ),
        previous_diagnose_json=(
            json.dumps(previous_diagnose_context, ensure_ascii=False)
            if previous_diagnose_context is not None
            else "null"
        ),
    )

    model = str(hypothesis_cfg.get("model", "gemini-2.5-flash"))
    temperature = float(hypothesis_cfg.get("temperature", 0.3))
    max_output_tokens = int(
        hypothesis_cfg.get(
            "max_output_tokens",
            hypothesis_cfg.get("max_tokens", 2048),
        )
    )
    top_p = float(hypothesis_cfg.get("top_p", 0.95))
    max_attempts = int(hypothesis_cfg.get("max_attempts", 2))
    system_instruction = str(hypothesis_cfg.get("system_instruction", HYPOTHESIS_SYSTEM_INSTRUCTION))

    normalized: Optional[Dict[str, Any]] = None
    last_raw_text = ""
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"    [Step2:hypothesis] LLM attempt {attempt}/{max_attempts} "
                f"(model={model}, prompt_chars={len(prompt)})"
            )
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",
                    "response_schema": HypothesisResponse,
                    "system_instruction": system_instruction,
                },
            )
            response_meta = _extract_response_meta(response)
            raw_text = str(getattr(response, "text", "") or "").strip()
            last_raw_text = raw_text
            print(
                f"    [Step2:hypothesis] LLM attempt {attempt} "
                f"response_chars={len(raw_text)} finish_reason={response_meta.get('finish_reason', 'unknown')}"
            )
            parsed = _parse_hypothesis_response(response=response, raw_text=raw_text)
            normalized = parsed.model_dump()
            normalized = _enforce_feature_engineering_hypothesis_count(
                normalized=normalized,
                required_count=feature_engineering_hypothesis_count,
            )
            normalized = _normalize_codegen_instruction_fields(
                normalized=normalized,
                required_count=feature_engineering_hypothesis_count,
            )
            last_error = ""
            print(f"    [Step2:hypothesis] LLM attempt {attempt} parsed successfully")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            print(f"    [Step2:hypothesis] LLM attempt {attempt} failed: {last_error}")
            with open(os.path.join(hypothesis_dir, f"hypothesis_attempt_{attempt}.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "attempt": attempt,
                        "raw_text": last_raw_text,
                        "error": last_error,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

    if normalized is None:
        raise RuntimeError(f"Hypothesis agent failed to produce valid structured output: {last_error}")

    with open(os.path.join(hypothesis_dir, "hypothesis.json"), "w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=2)

    with open(os.path.join(hypothesis_dir, "hypothesis_raw_response.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
                "max_attempts": max_attempts,
                "system_instruction": system_instruction,
                "feature_engineering_hypothesis_count": feature_engineering_hypothesis_count,
                "feature_engineering_hypothesis_count_actual": len(
                    normalized.get("feature_engineering", [])
                ),
                "feature_engineering_codegen_instruction_count_actual": len(
                    normalized.get("feature_engineering_codegen_instructions", [])
                ),
                "raw_text": last_raw_text,
                "last_error": last_error,
                "web_research": {
                    "enabled": bool(web_research.get("enabled", False)),
                    "used": bool(web_research.get("used", False)),
                    "status": web_research.get("status", "disabled"),
                    "error": web_research.get("error", ""),
                },
                "profile_context_mode": str(
                    hypothesis_cfg.get("profile_context_mode", "compact")
                ).strip().lower(),
                "profile_context_chars": len(
                    json.dumps(profile_context_for_hypothesis, ensure_ascii=False)
                ),
                "previous_diagnose_context": previous_diagnose_context,
                "task_context": task_context,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return normalized


def _parse_json_response(raw_text: str) -> Any:
    text = raw_text.strip()
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
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return {}


def _parse_hypothesis_response(response: Any, raw_text: str) -> HypothesisResponse:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, HypothesisResponse):
            return parsed
        try:
            return HypothesisResponse.model_validate(parsed)
        except ValidationError:
            pass

    raw_obj = _parse_json_response(raw_text)
    return HypothesisResponse.model_validate(raw_obj)


def _resolve_feature_engineering_hypothesis_count(hypothesis_cfg: Dict[str, Any]) -> Optional[int]:
    keys = [
        "feature_engineering_hypothesis_count",
        "max_feature_engineering_hypotheses",
        "feature_hypotheses_count",
    ]
    for key in keys:
        value = hypothesis_cfg.get(key)
        if value is None:
            continue
        count = int(value)
        if count <= 0:
            raise ValueError(f"hypothesis.{key} must be > 0, got {count}")
        return count
    return None


def _build_feature_engineering_count_instruction(required_count: Optional[int]) -> str:
    if required_count is None:
        return "- Keep `feature_engineering` concise and practical."
    return (
        f"- Return exactly {required_count} items in `feature_engineering`."
        " Do not return fewer or more."
    )


def _enforce_feature_engineering_hypothesis_count(
    normalized: Dict[str, Any],
    required_count: Optional[int],
) -> Dict[str, Any]:
    if required_count is None:
        return normalized

    raw_items = normalized.get("feature_engineering", [])
    if not isinstance(raw_items, list):
        raise ValueError("feature_engineering must be a list")

    cleaned_items: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text:
            cleaned_items.append(text)

    if len(cleaned_items) < required_count:
        raise ValueError(
            "feature_engineering_count_mismatch: "
            f"required={required_count}, actual={len(cleaned_items)}"
        )

    output = dict(normalized)
    output["feature_engineering"] = cleaned_items[:required_count]
    return output


def _normalize_codegen_instruction_fields(
    normalized: Dict[str, Any],
    required_count: Optional[int],
) -> Dict[str, Any]:
    output = dict(normalized)
    preprocessing_hypotheses = _normalize_text_list(output.get("preprocessing"))
    feature_hypotheses = _normalize_text_list(output.get("feature_engineering"))

    preprocessing_instruction = str(output.get("preprocessing_codegen_instruction", "") or "").strip()
    if not preprocessing_instruction:
        if preprocessing_hypotheses:
            joined = "; ".join(preprocessing_hypotheses[:5])
            preprocessing_instruction = (
                "Implement preprocessing using train-only fit state and deterministic transform. "
                f"Apply these hypotheses in order: {joined}. "
                "Preserve label column when present and avoid target leakage."
            )
        else:
            preprocessing_instruction = (
                "Implement robust deterministic preprocessing with train-only fit state, "
                "stable schema alignment, missing-value handling, dtype-safe conversion, and leakage prevention."
            )

    raw_feature_guides = output.get("feature_engineering_codegen_instructions", [])
    feature_guides: List[str] = []
    if isinstance(raw_feature_guides, list):
        for item in raw_feature_guides:
            text = str(item or "").strip()
            if text:
                feature_guides.append(text)

    target_count = required_count if required_count is not None else len(feature_hypotheses)
    if target_count < 0:
        target_count = 0

    while len(feature_guides) < target_count:
        idx = len(feature_guides)
        hypothesis_text = feature_hypotheses[idx] if idx < len(feature_hypotheses) else f"feature_{idx + 1}"
        feature_guides.append(_default_feature_codegen_instruction(hypothesis_text=hypothesis_text))

    if target_count > 0:
        feature_guides = feature_guides[:target_count]

    output["preprocessing_codegen_instruction"] = preprocessing_instruction
    output["feature_engineering_codegen_instructions"] = feature_guides
    return output


def _default_feature_codegen_instruction(hypothesis_text: str) -> str:
    return (
        "Implement exactly one feature for this hypothesis. "
        f"Hypothesis: {str(hypothesis_text).strip()}. "
        "In fit(), compute only train-derived state and choose a safe FEATURE_NAME. "
        "In transform(), return one pd.Series aligned to df.index using the same logic without label leakage. "
        "Handle missing values and unseen categories deterministically."
    )


def _normalize_text_list(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    return []


def _run_hypothesis_web_research(
    client: genai.Client,
    hypothesis_cfg: Dict[str, Any],
    profile_result: Dict[str, Any],
    hypothesis_dir: str,
    previous_diagnose_context: Optional[Dict[str, Any]] = None,
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    web_cfg = hypothesis_cfg.get("web_search", {}) if isinstance(hypothesis_cfg, dict) else {}
    enabled = bool(web_cfg.get("enabled", False))

    result: Dict[str, Any] = {
        "enabled": enabled,
        "used": False,
        "status": "disabled",
        "error": "",
        "context": {
            "summary": "",
            "findings": [],
        },
    }

    if not enabled:
        print("    [Step2:web_research] skipped (disabled)")
        with open(os.path.join(hypothesis_dir, "web_research.json"), "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        return result

    model = str(web_cfg.get("model", hypothesis_cfg.get("model", "gemini-2.5-flash")))
    temperature = float(web_cfg.get("temperature", 0.2))
    top_p = float(web_cfg.get("top_p", 0.9))
    max_output_tokens = int(
        web_cfg.get(
            "max_output_tokens",
            web_cfg.get("max_tokens", 4096),
        )
    )
    max_attempts = int(web_cfg.get("max_attempts", 2))
    max_context_chars = int(web_cfg.get("max_context_chars", 4000))
    max_insight_items = int(web_cfg.get("max_insight_items", 8))
    system_instruction = str(
        web_cfg.get("system_instruction", HYPOTHESIS_WEB_RESEARCH_SYSTEM_INSTRUCTION)
    )

    profile_context = _build_hypothesis_web_profile_context(
        profile_result=profile_result,
        max_context_chars=max_context_chars,
        max_insight_items=max_insight_items,
    )
    prompt = HYPOTHESIS_WEB_RESEARCH_PROMPT.replace(
        "{profile_context_json}",
        json.dumps(profile_context, ensure_ascii=False),
    ).replace(
        "{task_context_json}",
        json.dumps(task_context, ensure_ascii=False) if isinstance(task_context, dict) and task_context else "null",
    ).replace(
        "{previous_diagnose_json}",
        json.dumps(previous_diagnose_context, ensure_ascii=False)
        if previous_diagnose_context is not None
        else "null",
    )

    last_raw_text = ""
    last_error = ""
    normalized: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"    [Step2:web_research] LLM attempt {attempt}/{max_attempts} "
                f"(model={model}, prompt_chars={len(prompt)})"
            )
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    system_instruction=system_instruction,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            response_meta = _extract_response_meta(response)
            raw_text = str(getattr(response, "text", "") or "").strip()
            last_raw_text = raw_text
            print(
                f"    [Step2:web_research] LLM attempt {attempt} "
                f"response_chars={len(raw_text)} finish_reason={response_meta.get('finish_reason', 'unknown')}"
            )
            parsed = _parse_hypothesis_web_research_response(response=response, raw_text=raw_text)
            normalized = parsed.model_dump()
            last_error = ""
            print(f"    [Step2:web_research] LLM attempt {attempt} parsed successfully")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            print(f"    [Step2:web_research] LLM attempt {attempt} failed: {last_error}")
            with open(os.path.join(hypothesis_dir, f"web_research_attempt_{attempt}.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "attempt": attempt,
                        "error": last_error,
                        "raw_text": last_raw_text,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

    if normalized is None:
        result["status"] = "failed"
        result["error"] = last_error or "web_research_failed"
    else:
        result["used"] = True
        result["status"] = "success"
        result["context"] = normalized

    with open(os.path.join(hypothesis_dir, "web_research.json"), "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    with open(os.path.join(hypothesis_dir, "web_research_raw_response.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
                "max_attempts": max_attempts,
                "system_instruction": system_instruction,
                "prompt_context": profile_context,
                "previous_diagnose_context": previous_diagnose_context,
                "task_context": task_context,
                "raw_text": last_raw_text,
                "last_error": last_error,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return result


def _build_hypothesis_web_profile_context(
    profile_result: Dict[str, Any],
    max_context_chars: int,
    max_insight_items: int,
) -> Dict[str, Any]:
    def _truncate_text(value: Any, limit: int = 200) -> str:
        text = str(value or "").strip()
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text

    def _truncate_list(value: Any, limit_items: int) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for item in value[:limit_items]:
            text = _truncate_text(item, limit=200)
            if text:
                out.append(text)
        return out

    context = {
        "summary": _truncate_text(profile_result.get("summary", ""), limit=500),
        "insights": _truncate_list(profile_result.get("insights", []), max_insight_items),
        "risks": _truncate_list(profile_result.get("risks", []), max_insight_items),
        "recommended_next_actions": _truncate_list(
            profile_result.get("recommended_next_actions", []),
            max_insight_items,
        ),
    }
    text = json.dumps(context, ensure_ascii=False)
    if len(text) <= max_context_chars:
        return context

    # If context is still too long, trim list sizes progressively.
    trimmed_items = max(2, max_insight_items // 2)
    context["insights"] = context["insights"][:trimmed_items]
    context["risks"] = context["risks"][:trimmed_items]
    context["recommended_next_actions"] = context["recommended_next_actions"][:trimmed_items]

    text = json.dumps(context, ensure_ascii=False)
    if len(text) <= max_context_chars:
        return context

    context["summary"] = _truncate_text(context.get("summary", ""), limit=max(120, max_context_chars // 4))
    return context


def _build_hypothesis_profile_context(
    profile_result: Dict[str, Any],
    hypothesis_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(profile_result, dict):
        return {}

    mode = str(hypothesis_cfg.get("profile_context_mode", "compact")).strip().lower()
    if mode == "full":
        return profile_result

    max_items = int(hypothesis_cfg.get("profile_context_max_items", 8))
    max_text_chars = int(hypothesis_cfg.get("profile_context_max_text_chars", 500))
    include_stage_stdout = bool(hypothesis_cfg.get("profile_context_include_stage_stdout", False))
    stage_stdout_max_chars = int(hypothesis_cfg.get("profile_context_stage_stdout_max_chars", 1000))

    preferred_keys = [
        "summary",
        "insights",
        "risks",
        "recommended_next_actions",
    ]

    context: Dict[str, Any] = {}
    for key in preferred_keys:
        if key not in profile_result:
            continue
        context[key] = _trim_profile_value(
            value=profile_result.get(key),
            max_items=max_items,
            max_text_chars=max_text_chars,
        )

    if include_stage_stdout:
        basic_excerpt = (
            profile_result.get("basic_profile", {}) if isinstance(profile_result.get("basic_profile"), dict) else {}
        ).get("stdout_excerpt", "")
        corr_excerpt = (
            profile_result.get("correlation_profile", {})
            if isinstance(profile_result.get("correlation_profile"), dict)
            else {}
        ).get("stdout_excerpt", "")
        context["stage_stdout_preview"] = {
            "basic": str(basic_excerpt or "")[:stage_stdout_max_chars],
            "correlation": str(corr_excerpt or "")[:stage_stdout_max_chars],
        }

    if not context:
        for idx, (key, value) in enumerate(profile_result.items()):
            if idx >= max_items:
                break
            context[str(key)] = _trim_profile_value(
                value=value,
                max_items=max_items,
                max_text_chars=max_text_chars,
            )

    context["_keys_in_original_profile"] = list(profile_result.keys())
    context["_profile_context_mode"] = "compact"
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
        return [
            _trim_profile_value(item, max_items=max_items, max_text_chars=max_text_chars)
            for item in value[:max_items]
        ]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                break
            out[str(k)] = _trim_profile_value(v, max_items=max_items, max_text_chars=max_text_chars)
        return out
    return str(value)[:max_text_chars]


def _build_previous_diagnose_context(
    prev_diagnose_result: Optional[Dict[str, Any]],
    hypothesis_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(prev_diagnose_result, dict):
        return None

    max_items = int(hypothesis_cfg.get("diagnose_focus_max_items", 3))
    max_text_chars = int(hypothesis_cfg.get("diagnose_text_max_chars", 220))

    root_cause = prev_diagnose_result.get("root_cause", {}) or {}
    score_summary = prev_diagnose_result.get("score_summary", {}) or {}
    comparison = prev_diagnose_result.get("comparison_to_best_before_iteration", {}) or {}
    feedback = prev_diagnose_result.get("feedback_for_next_iteration", {}) or {}

    def _truncate_text(value: Any) -> str:
        text = str(value or "").strip()
        if len(text) > max_text_chars:
            return text[: max_text_chars - 3] + "..."
        return text

    def _truncate_list(items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for item in items[:max_items]:
            text = _truncate_text(item)
            if text:
                out.append(text)
        return out

    def _safe_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except Exception:
            return None
        if parsed != parsed:
            return None
        return parsed

    compact = {
        "status": prev_diagnose_result.get("status"),
        "hard_failure": bool(prev_diagnose_result.get("hard_failure", False)),
        "root_cause": {
            "category": root_cause.get("category"),
            "message": _truncate_text(root_cause.get("message", "")),
        },
        "score_summary": {
            "metric": score_summary.get("metric"),
            "mean_cv": _safe_float(score_summary.get("mean_cv")),
            "std_cv": _safe_float(score_summary.get("std_cv")),
            "objective_mean": _safe_float(score_summary.get("objective_mean")),
            "n_features": score_summary.get("n_features"),
        },
        "comparison_to_previous_best": {
            "has_previous_best": comparison.get("has_previous_best"),
            "trend": comparison.get("trend"),
            "delta_objective_mean": _safe_float(comparison.get("delta_objective_mean")),
        },
        "next_iteration_hints": {
            "hypothesis_focus": _truncate_list(feedback.get("hypothesis_focus")),
            "priority_actions": _truncate_list(feedback.get("priority_actions")),
            "implement_constraints": _truncate_list(feedback.get("implement_constraints")),
        },
    }

    if compact["status"] is None and compact["root_cause"]["category"] is None:
        return None
    return compact


def _parse_hypothesis_web_research_response(response: Any, raw_text: str) -> WebResearchResponse:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, WebResearchResponse):
            return parsed
        try:
            return WebResearchResponse.model_validate(parsed)
        except ValidationError:
            pass

    raw_obj = _parse_json_response(raw_text)
    return WebResearchResponse.model_validate(raw_obj)


def _extract_response_meta(response: Any) -> Dict[str, Any]:
    candidates = getattr(response, "candidates", None)
    candidate_count = len(candidates) if isinstance(candidates, list) else 0

    finish_reason = "unknown"
    if isinstance(candidates, list) and candidates:
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is not None:
            finish_reason = str(reason)

    total_token_count: Optional[int] = None
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is not None:
        total = getattr(usage_metadata, "total_token_count", None)
        if isinstance(total, int):
            total_token_count = total

    return {
        "candidate_count": candidate_count,
        "finish_reason": finish_reason,
        "total_token_count": total_token_count,
    }
