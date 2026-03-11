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
For each feature_engineering item, propose one concrete derived feature that combines multiple source columns."""

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

Profile context JSON:
{profile_context_json}
"""


class HypothesisResponse(BaseModel):
    preprocessing: List[str] = Field(default_factory=list)
    feature_engineering: List[str] = Field(default_factory=list)


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
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hypothesis_dir = os.path.join(output_dir, "hypothesis")
    os.makedirs(hypothesis_dir, exist_ok=True)

    prompt_path = Path("src/prompt/2_hypothesis.j2")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    web_research = _run_hypothesis_web_research(
        client=client,
        hypothesis_cfg=hypothesis_cfg,
        profile_result=profile_result,
        hypothesis_dir=hypothesis_dir,
        task_context=task_context,
    )

    prompt_template = Template(prompt_path.read_text(encoding="utf-8"))
    feature_engineering_hypothesis_count = _resolve_feature_engineering_hypothesis_count(hypothesis_cfg)
    prompt = prompt_template.render(
        profile_result_json=json.dumps(profile_result, ensure_ascii=False),
        web_research_json=json.dumps(web_research.get("context"), ensure_ascii=False),
        feature_engineering_count_instruction=_build_feature_engineering_count_instruction(
            feature_engineering_hypothesis_count
        ),
        task_context_json=(
            json.dumps(task_context, ensure_ascii=False)
            if isinstance(task_context, dict) and task_context
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
                "raw_text": last_raw_text,
                "last_error": last_error,
                "web_research": {
                    "enabled": bool(web_research.get("enabled", False)),
                    "used": bool(web_research.get("used", False)),
                    "status": web_research.get("status", "disabled"),
                    "error": web_research.get("error", ""),
                },
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


def _run_hypothesis_web_research(
    client: genai.Client,
    hypothesis_cfg: Dict[str, Any],
    profile_result: Dict[str, Any],
    hypothesis_dir: str,
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
