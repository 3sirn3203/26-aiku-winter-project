from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from jinja2 import Template

IMPLEMENT_E2E_SYSTEM_INSTRUCTION = """You complete TODO sections in a fixed Python skeleton for tabular ML.
Return only executable Python code.
Do not include markdown fences.
Do not include explanations.
Do not remove CLI arguments or JSON output behavior."""


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
    del train_path, label_col, pipeline_config

    implement_dir = os.path.join(output_dir, "implement")
    os.makedirs(implement_dir, exist_ok=True)

    model = str(implement_cfg.get("model", "gemini-2.5-flash"))
    temperature = float(implement_cfg.get("temperature", 0.2))
    top_p = float(implement_cfg.get("top_p", 0.9))
    max_output_tokens = int(
        implement_cfg.get("max_output_tokens", implement_cfg.get("max_tokens", 8192))
    )
    max_codegen_attempts = int(implement_cfg.get("max_codegen_attempts", 2))
    syntax_check = bool(implement_cfg.get("syntax_check", True))
    prompt_path = Path(
        str(implement_cfg.get("prompt_path", "src/prompt/3_implement_e2e.j2"))
    )
    skeleton_path = Path(
        str(
            implement_cfg.get(
                "skeleton_path",
                "src/prompt/3_implement_e2e_skeleton.py",
            )
        )
    )
    system_instruction = str(
        implement_cfg.get("system_instruction", IMPLEMENT_E2E_SYSTEM_INSTRUCTION)
    )

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    if not skeleton_path.exists():
        raise FileNotFoundError(f"Skeleton template not found: {skeleton_path}")

    profile_context = _build_profile_context(profile_result=profile_result, implement_cfg=implement_cfg)
    preprocessing_hypotheses = _normalize_text_list(hypotheses.get("preprocessing"))
    feature_hypotheses = _normalize_text_list(hypotheses.get("feature_engineering"))
    skeleton_code = skeleton_path.read_text(encoding="utf-8")
    template = Template(prompt_path.read_text(encoding="utf-8"))
    output_script_path = Path(implement_dir) / "implement_pipeline.py"

    code, generation_meta = _generate_pipeline_script(
        client=client,
        template=template,
        skeleton_code=skeleton_code,
        profile_context=profile_context,
        preprocessing_hypotheses=preprocessing_hypotheses,
        feature_hypotheses=feature_hypotheses,
        output_script_path=output_script_path,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        max_codegen_attempts=max_codegen_attempts,
        syntax_check=syntax_check,
        system_instruction=system_instruction,
        external_feedback=external_feedback,
    )

    summary = {
        "pipeline_script_path": str(output_script_path),
        "meta": {
            "mode": "e2e_skeleton_todo_completion",
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "max_codegen_attempts": max_codegen_attempts,
            "syntax_check": syntax_check,
            "prompt_path": str(prompt_path),
            "skeleton_path": str(skeleton_path),
            "system_instruction": system_instruction,
            "external_feedback": external_feedback,
            "profile_context_keys": list(profile_context.keys()),
            "generation": generation_meta,
        },
    }

    with open(os.path.join(implement_dir, "implement_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with open(os.path.join(implement_dir, "implement_pipeline.py"), "w", encoding="utf-8") as file:
        file.write(code)
    return summary


def _generate_pipeline_script(
    client: genai.Client,
    template: Template,
    skeleton_code: str,
    profile_context: Dict[str, Any],
    preprocessing_hypotheses: List[str],
    feature_hypotheses: List[str],
    output_script_path: Path,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_codegen_attempts: int,
    syntax_check: bool,
    system_instruction: str,
    external_feedback: Optional[Dict[str, Any]],
) -> tuple[str, Dict[str, Any]]:
    stage_dir = output_script_path.parent
    last_raw_text = ""
    last_response_meta: Dict[str, Any] = {}
    last_error = ""
    feedback: Optional[Dict[str, Any]] = None
    final_code: Optional[str] = None

    for attempt in range(1, max_codegen_attempts + 1):
        prompt = template.render(
            skeleton_code=skeleton_code,
            profile_result_json=json.dumps(profile_context, ensure_ascii=False),
            preprocessing_hypotheses_json=json.dumps(preprocessing_hypotheses, ensure_ascii=False),
            feature_hypotheses_json=json.dumps(feature_hypotheses, ensure_ascii=False),
            generation_feedback_json=json.dumps(feedback, ensure_ascii=False) if feedback is not None else "null",
            external_feedback_json=json.dumps(external_feedback, ensure_ascii=False) if external_feedback is not None else "null",
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
            _assert_has_main(code)
            if syntax_check:
                _syntax_check(script_path=output_script_path)
            final_code = code
            last_error = ""
            feedback = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            feedback = {"attempt": attempt, "error": last_error}
            with open(stage_dir / f"implement_attempt_{attempt}.json", "w", encoding="utf-8") as file:
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
        raise RuntimeError(f"Failed to generate valid e2e implement script: {last_error}")

    with open(stage_dir / "implement_raw_response.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "raw_text": last_raw_text,
                "response_meta": last_response_meta,
                "last_error": last_error,
                "syntax_check": syntax_check,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return final_code, {"last_error": last_error}


def _assert_has_main(code: str) -> None:
    tree = ast.parse(code)
    has_main = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
        for node in ast.walk(tree)
    )
    if not has_main:
        raise ValueError("Generated script must define main().")


def _extract_python_code(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    fence_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text


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


def _extract_response_meta(response: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        finish_reason = getattr(first, "finish_reason", None)
        if finish_reason is not None:
            meta["finish_reason"] = str(finish_reason)

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
            return _to_jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _to_jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)
