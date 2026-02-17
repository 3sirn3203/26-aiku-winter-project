from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from jinja2 import Template
from pydantic import BaseModel, Field, ValidationError

HYPOTHESIS_SYSTEM_INSTRUCTION = """You generate hypotheses for tabular ML.
Return only a valid JSON object matching the schema.
Do not include markdown fences or explanations outside JSON."""


class HypothesisResponse(BaseModel):
    preprocessing: List[str] = Field(default_factory=list)
    feature_engineering: List[str] = Field(default_factory=list)


def generate_hypotheses(
    client: genai.Client,
    hypothesis_cfg: Dict[str, Any],
    profile_result: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    hypothesis_dir = os.path.join(output_dir, "hypothesis")
    os.makedirs(hypothesis_dir, exist_ok=True)

    prompt_path = Path("src/prompt/2_hypothesis.j2")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    prompt_template = Template(prompt_path.read_text(encoding="utf-8"))
    prompt = prompt_template.render(
        profile_result_json=json.dumps(profile_result, ensure_ascii=False),
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
        raw_text = str(getattr(response, "text", "") or "").strip()
        last_raw_text = raw_text

        try:
            parsed = _parse_hypothesis_response(response=response, raw_text=raw_text)
            normalized = parsed.model_dump()
            last_error = ""
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            with open(os.path.join(hypothesis_dir, f"hypothesis_attempt_{attempt}.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "attempt": attempt,
                        "raw_text": raw_text,
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
                "raw_text": last_raw_text,
                "last_error": last_error,
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
