from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from google import genai
from jinja2 import Template
from pydantic import BaseModel, Field, ValidationError

PROFILE_CODE_SYSTEM_INSTRUCTION = """You are a data profiling code generator for tabular machine learning.
Always return only executable Python code.
Never return JSON wrappers.
Never include markdown fences.
Never include explanations or comments outside the code.
The output must be a single complete Python script."""

PROFILE_INSIGHT_SYSTEM_INSTRUCTION = """You are a tabular profiling insight analyst.
Return only a valid JSON object matching the schema.
Do not include markdown fences.
Do not include explanations outside JSON."""


class ProfileInsightResponse(BaseModel):
    summary: str = ""
    insights: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    recommended_next_actions: List[str] = Field(default_factory=list)


def profiling(
    client: genai.Client,
    profile_cfg: Dict,
    train_path: str,
    output_dir: str,
    iteration: int,
    prev_diagnose_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile_dir = os.path.join(output_dir, "profile")
    os.makedirs(profile_dir, exist_ok=True)

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    dataset_context = _build_dataset_context(train_df)
    diagnose_prompt_context = _build_diagnose_prompt_context(
        prev_diagnose_result=prev_diagnose_result,
        profile_cfg=profile_cfg,
    )

    model = str(profile_cfg.get("model", "gemini-2.5-flash"))
    temperature = float(profile_cfg.get("temperature", 0.3))
    max_output_tokens = int(profile_cfg.get("max_output_tokens", profile_cfg.get("max_tokens", 8192)))
    top_p = float(profile_cfg.get("top_p", 0.95))
    code_system_instruction = str(profile_cfg.get("system_instruction", PROFILE_CODE_SYSTEM_INSTRUCTION))
    max_codegen_attempts = int(profile_cfg.get("max_codegen_attempts", 2))
    execution_timeout_sec = int(profile_cfg.get("execution_timeout_sec", 120))
    basic_prompt_path = Path(str(profile_cfg.get("basic_prompt_path", "src/prompt/1_profile_basic.j2")))
    correlation_prompt_path = Path(str(profile_cfg.get("correlation_prompt_path", "src/prompt/1_profile_correlation.j2")))
    insight_prompt_path = Path(str(profile_cfg.get("insight_prompt_path", "src/prompt/1_profile_insight.j2")))

    insight_model = str(profile_cfg.get("insight_model", model))
    insight_temperature = float(profile_cfg.get("insight_temperature", temperature))
    insight_max_output_tokens = int(
        profile_cfg.get(
            "insight_max_output_tokens",
            profile_cfg.get("insight_max_tokens", max_output_tokens),
        )
    )
    insight_top_p = float(profile_cfg.get("insight_top_p", top_p))
    max_insight_attempts = int(profile_cfg.get("max_insight_attempts", 2))
    insight_system_instruction = str(
        profile_cfg.get("insight_system_instruction", PROFILE_INSIGHT_SYSTEM_INSTRUCTION)
    )

    basic_stage = _run_profile_codegen_stage(
        client=client,
        stage_name="basic",
        stage_prompt_path=basic_prompt_path,
        profile_dir=profile_dir,
        train_path=train_path,
        prompt_payload={
            "dataset_context": json.dumps(dataset_context, ensure_ascii=False),
            "previous_diagnose_json": (
                json.dumps(diagnose_prompt_context, ensure_ascii=False)
                if diagnose_prompt_context is not None
                else "null"
            ),
        },
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        max_codegen_attempts=max_codegen_attempts,
        execution_timeout_sec=execution_timeout_sec,
        system_instruction=code_system_instruction,
    )

    correlation_stage = _run_profile_codegen_stage(
        client=client,
        stage_name="correlation",
        stage_prompt_path=correlation_prompt_path,
        profile_dir=profile_dir,
        train_path=train_path,
        prompt_payload={
            "dataset_context": json.dumps(dataset_context, ensure_ascii=False),
            "basic_profile_json": json.dumps(basic_stage["profile_json"], ensure_ascii=False),
            "previous_diagnose_json": (
                json.dumps(diagnose_prompt_context, ensure_ascii=False)
                if diagnose_prompt_context is not None
                else "null"
            ),
        },
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        max_codegen_attempts=max_codegen_attempts,
        execution_timeout_sec=execution_timeout_sec,
        system_instruction=code_system_instruction,
    )

    insight_stage = _run_profile_insight_stage(
        client=client,
        stage_prompt_path=insight_prompt_path,
        profile_dir=profile_dir,
        prompt_payload={
            "dataset_context": json.dumps(dataset_context, ensure_ascii=False),
            "basic_profile_json": json.dumps(basic_stage["profile_json"], ensure_ascii=False),
            "correlation_profile_json": json.dumps(correlation_stage["profile_json"], ensure_ascii=False),
            "previous_diagnose_json": (
                json.dumps(diagnose_prompt_context, ensure_ascii=False)
                if diagnose_prompt_context is not None
                else "null"
            ),
        },
        model=insight_model,
        temperature=insight_temperature,
        max_output_tokens=insight_max_output_tokens,
        top_p=insight_top_p,
        max_attempts=max_insight_attempts,
        system_instruction=insight_system_instruction,
    )

    profile_result = {
        "summary": insight_stage["insight_json"].get("summary", ""),
        "insights": insight_stage["insight_json"].get("insights", []),
        "risks": insight_stage["insight_json"].get("risks", []),
        "recommended_next_actions": insight_stage["insight_json"].get("recommended_next_actions", []),
        "basic_profile": basic_stage["profile_json"],
        "correlation_profile": correlation_stage["profile_json"],
        "stages": {
            "basic": basic_stage["profile_json"],
            "correlation": correlation_stage["profile_json"],
            "insight": insight_stage["insight_json"],
        },
    }

    # Backward-compatible artifact names
    with open(os.path.join(profile_dir, "eda_generated.py"), "w", encoding="utf-8") as file:
        file.write(basic_stage["code"])

    with open(os.path.join(profile_dir, "profile_basic_eda.py"), "w", encoding="utf-8") as file:
        file.write(basic_stage["code"])
    with open(os.path.join(profile_dir, "profile_correlation_eda.py"), "w", encoding="utf-8") as file:
        file.write(correlation_stage["code"])
    with open(os.path.join(profile_dir, "profile_insight.json"), "w", encoding="utf-8") as file:
        json.dump(insight_stage["insight_json"], file, ensure_ascii=False, indent=2)
    with open(os.path.join(profile_dir, "profile.json"), "w", encoding="utf-8") as file:
        json.dump(profile_result, file, ensure_ascii=False, indent=2)
    with open(os.path.join(profile_dir, "profile_raw_response.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "iteration": iteration,
                "basic_stage": {
                    "model": model,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": top_p,
                    "max_codegen_attempts": max_codegen_attempts,
                    "execution_timeout_sec": execution_timeout_sec,
                    "system_instruction": code_system_instruction,
                    "raw_text": basic_stage["raw_text"],
                },
                "correlation_stage": {
                    "model": model,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": top_p,
                    "max_codegen_attempts": max_codegen_attempts,
                    "execution_timeout_sec": execution_timeout_sec,
                    "system_instruction": code_system_instruction,
                    "raw_text": correlation_stage["raw_text"],
                },
                "insight_stage": {
                    "model": insight_model,
                    "temperature": insight_temperature,
                    "max_output_tokens": insight_max_output_tokens,
                    "top_p": insight_top_p,
                    "max_attempts": max_insight_attempts,
                    "system_instruction": insight_system_instruction,
                    "raw_text": insight_stage["raw_text"],
                    "last_error": insight_stage["last_error"],
                },
                "diagnose_prompt_context": diagnose_prompt_context,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return profile_result


def _run_profile_codegen_stage(
    client: genai.Client,
    stage_name: str,
    stage_prompt_path: Path,
    profile_dir: str,
    train_path: str,
    prompt_payload: Dict[str, str],
    model: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    max_codegen_attempts: int,
    execution_timeout_sec: int,
    system_instruction: str,
) -> Dict[str, Any]:
    if not stage_prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {stage_prompt_path}")
    prompt_template = Template(stage_prompt_path.read_text(encoding="utf-8"))

    execution_feedback: Optional[Dict[str, Any]] = None
    final_code = ""
    final_profile_json: Optional[Dict[str, Any]] = None
    last_raw_text = ""
    last_error = ""

    for attempt in range(1, max_codegen_attempts + 1):
        render_kwargs = dict(prompt_payload)
        render_kwargs["execution_feedback_json"] = (
            json.dumps(execution_feedback, ensure_ascii=False)
            if execution_feedback is not None
            else "null"
        )
        prompt = prompt_template.render(**render_kwargs)

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": top_p,
                    "response_mime_type": "text/plain",
                    "system_instruction": system_instruction,
                },
            )
            raw_text = str(getattr(response, "text", "") or "").strip()
            last_raw_text = raw_text
        except Exception as exc:  # noqa: BLE001
            last_error = f"llm_call_failed: {exc}"
            execution_feedback = {"attempt": attempt, "error": last_error}
            _write_json(
                os.path.join(profile_dir, f"{stage_name}_llm_attempt_{attempt}.json"),
                {"raw_text": "", "error": last_error},
            )
            continue

        eda_code = _extract_python_code(last_raw_text)
        if not eda_code:
            last_error = "empty_code"
            execution_feedback = {
                "attempt": attempt,
                "error": last_error,
                "stdout_tail": "",
                "stderr_tail": "",
            }
            _write_json(
                os.path.join(profile_dir, f"{stage_name}_llm_attempt_{attempt}.json"),
                {"raw_text": last_raw_text, "error": last_error},
            )
            continue

        eda_script_path = os.path.join(profile_dir, f"{stage_name}_eda_attempt_{attempt}.py")
        profile_json_path = os.path.join(profile_dir, f"profile_{stage_name}_attempt_{attempt}.json")
        with open(eda_script_path, "w", encoding="utf-8") as file:
            file.write(eda_code)

        exec_result = _execute_eda_script(
            script_path=eda_script_path,
            train_path=train_path,
            profile_json_path=profile_json_path,
            timeout_sec=execution_timeout_sec,
        )
        _write_json(os.path.join(profile_dir, f"{stage_name}_exec_attempt_{attempt}.json"), exec_result)
        _write_json(
            os.path.join(profile_dir, f"{stage_name}_llm_attempt_{attempt}.json"),
            {"raw_text": last_raw_text, "error": ""},
        )

        if exec_result["success"]:
            try:
                with open(profile_json_path, "r", encoding="utf-8") as file:
                    loaded_profile = json.load(file)
            except Exception as exc:  # noqa: BLE001
                last_error = f"invalid_json_output: {exc}"
                execution_feedback = {
                    "attempt": attempt,
                    "error": last_error,
                    "stdout_tail": str(exec_result.get("stdout", ""))[-2000:],
                    "stderr_tail": str(exec_result.get("stderr", ""))[-2000:],
                }
                continue

            if not isinstance(loaded_profile, dict):
                last_error = "profile_output_not_object"
                execution_feedback = {
                    "attempt": attempt,
                    "error": last_error,
                    "stdout_tail": str(exec_result.get("stdout", ""))[-2000:],
                    "stderr_tail": str(exec_result.get("stderr", ""))[-2000:],
                }
                continue

            final_code = eda_code
            final_profile_json = loaded_profile
            last_error = ""
            break

        last_error = str(exec_result.get("error", "execution_failed"))
        execution_feedback = {
            "attempt": attempt,
            "error": last_error,
            "stdout_tail": str(exec_result.get("stdout", ""))[-2000:],
            "stderr_tail": str(exec_result.get("stderr", ""))[-2000:],
        }

    if final_profile_json is None:
        raise RuntimeError(
            f"Profile {stage_name} stage failed: generated code could not be executed successfully. "
            f"Last error: {last_error}"
        )

    with open(os.path.join(profile_dir, f"{stage_name}_eda_generated.py"), "w", encoding="utf-8") as file:
        file.write(final_code)
    with open(os.path.join(profile_dir, f"profile_{stage_name}.json"), "w", encoding="utf-8") as file:
        json.dump(final_profile_json, file, ensure_ascii=False, indent=2)

    return {
        "code": final_code,
        "profile_json": final_profile_json,
        "raw_text": last_raw_text,
    }


def _run_profile_insight_stage(
    client: genai.Client,
    stage_prompt_path: Path,
    profile_dir: str,
    prompt_payload: Dict[str, str],
    model: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    max_attempts: int,
    system_instruction: str,
) -> Dict[str, Any]:
    if not stage_prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {stage_prompt_path}")
    prompt_template = Template(stage_prompt_path.read_text(encoding="utf-8"))
    prompt = prompt_template.render(**prompt_payload)

    normalized: Optional[Dict[str, Any]] = None
    last_raw_text = ""
    last_error = ""

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
                    "response_schema": ProfileInsightResponse,
                    "system_instruction": system_instruction,
                },
            )
            raw_text = str(getattr(response, "text", "") or "").strip()
            last_raw_text = raw_text

            parsed = _parse_profile_insight_response(response=response, raw_text=raw_text)
            normalized = parsed.model_dump()
            last_error = ""
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            _write_json(
                os.path.join(profile_dir, f"insight_attempt_{attempt}.json"),
                {
                    "attempt": attempt,
                    "raw_text": last_raw_text,
                    "error": last_error,
                },
            )

    if normalized is None:
        raise RuntimeError(f"Profile insight stage failed to produce valid JSON output: {last_error}")

    _write_json(
        os.path.join(profile_dir, "profile_insight_raw_response.json"),
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
    )

    return {
        "insight_json": normalized,
        "raw_text": last_raw_text,
        "last_error": last_error,
    }


def _build_dataset_context(train_df: pd.DataFrame) -> Dict[str, Any]:
    dtypes = {col: str(dtype) for col, dtype in train_df.dtypes.items()}
    missing_pct = (train_df.isna().mean() * 100).sort_values(ascending=False).head(20)
    missing_top = [{"column": str(col), "missing_pct": float(val)} for col, val in missing_pct.items() if val > 0]
    cardinality = train_df.nunique(dropna=False).sort_values(ascending=False).head(20)
    cardinality_top = [{"column": str(col), "nunique": int(val)} for col, val in cardinality.items()]
    sample_rows = train_df.head(5).fillna("__NA__").astype(str).to_dict(orient="records")

    return {
        "shape": {"rows": int(train_df.shape[0]), "cols": int(train_df.shape[1])},
        "columns": list(train_df.columns),
        "dtypes": dtypes,
        "missing_top": missing_top,
        "cardinality_top": cardinality_top,
        "sample_rows": sample_rows,
    }


def _build_diagnose_prompt_context(
    prev_diagnose_result: Optional[Dict[str, Any]],
    profile_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(prev_diagnose_result, dict):
        return None

    max_items = int(profile_cfg.get("diagnose_focus_max_items", 3))
    max_text_chars = int(profile_cfg.get("diagnose_text_max_chars", 220))

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
        if math.isnan(parsed):
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
            "profile_focus": _truncate_list(feedback.get("profile_focus")),
            "priority_actions": _truncate_list(feedback.get("priority_actions")),
            "implement_constraints": _truncate_list(feedback.get("implement_constraints")),
        },
    }

    if compact["status"] is None and compact["root_cause"]["category"] is None:
        return None
    return compact


def _parse_profile_insight_response(response: Any, raw_text: str) -> ProfileInsightResponse:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, ProfileInsightResponse):
            return parsed
        try:
            return ProfileInsightResponse.model_validate(parsed)
        except ValidationError:
            pass

    raw_obj = _parse_json_response(raw_text)
    return ProfileInsightResponse.model_validate(raw_obj)


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
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return {}


def _extract_python_code(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    # Prefer fenced python blocks if the model still emits markdown.
    fence_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text


def _execute_eda_script(
    script_path: str,
    train_path: str,
    profile_json_path: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        script_path,
        "--train-path",
        train_path,
        "--profile-json-out",
        profile_json_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "error": f"timeout_after_{timeout_sec}s",
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "returncode": None,
        }

    success = proc.returncode == 0 and os.path.exists(profile_json_path)
    error = "" if success else f"returncode_{proc.returncode}"
    if proc.returncode == 0 and not os.path.exists(profile_json_path):
        success = False
        error = "missing_profile_json_output"

    return {
        "success": success,
        "error": error,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
