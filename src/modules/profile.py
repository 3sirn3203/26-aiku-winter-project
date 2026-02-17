from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from google import genai
from jinja2 import Template

PROFILE_SYSTEM_INSTRUCTION = """You are a data profiling code generator for tabular machine learning.
Always return only executable Python code.
Never return JSON wrappers.
Never include markdown fences.
Never include explanations or comments outside the code.
The output must be a single complete Python script."""


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

    prompt_path = Path("src/prompt/1_profile.j2")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    prompt_template = Template(prompt_path.read_text(encoding="utf-8"))

    model = str(profile_cfg.get("model", "gemini-2.5-flash"))
    temperature = float(profile_cfg.get("temperature", 0.3))
    max_output_tokens = int(profile_cfg.get("max_output_tokens", profile_cfg.get("max_tokens", 8192)))
    top_p = float(profile_cfg.get("top_p", 0.95))
    system_instruction = str(profile_cfg.get("system_instruction", PROFILE_SYSTEM_INSTRUCTION))
    max_codegen_attempts = int(profile_cfg.get("max_codegen_attempts", 2))
    execution_timeout_sec = int(profile_cfg.get("execution_timeout_sec", 120))

    execution_feedback: Optional[Dict[str, Any]] = None
    final_eda_code = ""
    final_profile_json: Optional[Dict[str, Any]] = None
    last_llm_raw_text = ""
    diagnose_prompt_context = _build_diagnose_prompt_context(
        prev_diagnose_result=prev_diagnose_result,
        profile_cfg=profile_cfg,
    )

    for attempt in range(1, max_codegen_attempts + 1):
        prompt = prompt_template.render(
            dataset_context=json.dumps(dataset_context, ensure_ascii=False),
            previous_diagnose_json=json.dumps(diagnose_prompt_context, ensure_ascii=False) if diagnose_prompt_context is not None else "null",
            execution_feedback_json=json.dumps(execution_feedback, ensure_ascii=False) if execution_feedback is not None else "null",
        )

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
        last_llm_raw_text = raw_text
        eda_code = _extract_python_code(raw_text)
        if not eda_code:
            raise ValueError("Profile agent returned empty code.")

        eda_script_path = os.path.join(profile_dir, f"eda_attempt_{attempt}.py")
        profile_json_path = os.path.join(profile_dir, f"profile_attempt_{attempt}.json")
        with open(eda_script_path, "w", encoding="utf-8") as file:
            file.write(eda_code)

        exec_result = _execute_eda_script(
            script_path=eda_script_path,
            train_path=train_path,
            profile_json_path=profile_json_path,
            timeout_sec=execution_timeout_sec,
        )
        with open(os.path.join(profile_dir, f"exec_attempt_{attempt}.json"), "w", encoding="utf-8") as file:
            json.dump(exec_result, file, ensure_ascii=False, indent=2)
        with open(os.path.join(profile_dir, f"llm_attempt_{attempt}.json"), "w", encoding="utf-8") as file:
            json.dump({"raw_text": raw_text}, file, ensure_ascii=False, indent=2)

        if exec_result["success"]:
            with open(profile_json_path, "r", encoding="utf-8") as file:
                loaded_profile = json.load(file)
            if not isinstance(loaded_profile, dict):
                raise ValueError("Executed EDA output profile JSON must be an object.")
            final_eda_code = eda_code
            final_profile_json = loaded_profile
            break

        execution_feedback = {
            "attempt": attempt,
            "error": exec_result.get("error", "execution_failed"),
            "stdout_tail": str(exec_result.get("stdout", ""))[-2000:],
            "stderr_tail": str(exec_result.get("stderr", ""))[-2000:],
        }

    if final_profile_json is None:
        raise RuntimeError("Profile agent failed: generated EDA code could not be executed successfully.")

    with open(os.path.join(profile_dir, "eda_generated.py"), "w", encoding="utf-8") as file:
        file.write(final_eda_code)
    with open(os.path.join(profile_dir, "profile.json"), "w", encoding="utf-8") as file:
        json.dump(final_profile_json, file, ensure_ascii=False, indent=2)
    with open(os.path.join(profile_dir, "profile_raw_response.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "iteration": iteration,
                "model": model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "system_instruction": system_instruction,
                "max_codegen_attempts": max_codegen_attempts,
                "execution_timeout_sec": execution_timeout_sec,
                "diagnose_prompt_context": diagnose_prompt_context,
                "raw_text": last_llm_raw_text,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return final_profile_json


# 기본적인 데이터에 대한 컨텍스트 정보를 구축.
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
