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
) -> Dict[str, Any]:
    implement_dir = os.path.join(output_dir, "implement")
    os.makedirs(implement_dir, exist_ok=True)

    model = str(implement_cfg.get("model", "gemini-2.5-flash"))
    temperature = float(implement_cfg.get("temperature", 0.2))
    top_p = float(implement_cfg.get("top_p", 0.9))
    max_output_tokens = int(
        implement_cfg.get(
            "max_output_tokens",
            implement_cfg.get("max_tokens", 8192),
        )
    )
    max_codegen_attempts = int(implement_cfg.get("max_codegen_attempts", 2))
    syntax_check = bool(implement_cfg.get("syntax_check", True))

    preprocessing_hypotheses = _normalize_text_list(hypotheses.get("preprocessing"))
    feature_hypotheses = _normalize_text_list(hypotheses.get("feature_engineering"))

    prep_code, prep_meta = _generate_script(
        client=client,
        stage_name="preprocessor",
        template_path=Path("src/prompt/3_implement_prep.j2"),
        render_ctx={
            "profile_result_json": json.dumps(profile_result, ensure_ascii=False),
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
    )

    fe_code, fe_meta = _generate_script(
        client=client,
        stage_name="feature_engineering",
        template_path=Path("src/prompt/3_implement_fe.j2"),
        render_ctx={
            "profile_result_json": json.dumps(profile_result, ensure_ascii=False),
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
    )

    summary = {
        "preprocessor_path": str(Path(implement_dir) / "preprocessor.py"),
        "feature_engineering_path": str(Path(implement_dir) / "feature_engineering.py"),
        "contract": {
            "preprocessor": {
                "required_functions": ["fit_preprocessor", "transform_preprocessor"],
            },
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
            "preprocessor": prep_meta,
            "feature_engineering": fe_meta,
        },
    }

    with open(os.path.join(implement_dir, "implement_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    _ = prep_code
    _ = fe_code
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
) -> tuple[str, Dict[str, Any]]:
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    template = Template(template_path.read_text(encoding="utf-8"))
    stage_dir = output_script_path.parent
    last_raw_text = ""
    last_error = ""
    final_code: Optional[str] = None
    feedback: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_codegen_attempts + 1):
        prompt = template.render(
            **render_ctx,
            generation_feedback_json=json.dumps(feedback, ensure_ascii=False) if feedback is not None else "null",
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
        raw_text = str(getattr(response, "text", "") or "").strip()
        last_raw_text = raw_text
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
            feedback = {
                "attempt": attempt,
                "error": last_error,
            }
            with open(stage_dir / f"{stage_name}_attempt_{attempt}.json", "w", encoding="utf-8") as file:
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

    if final_code is None:
        raise RuntimeError(f"Failed to generate valid {stage_name} module: {last_error}")

    with open(stage_dir / f"{stage_name}_raw_response.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "raw_text": last_raw_text,
                "last_error": last_error,
                "required_functions": required_functions,
                "syntax_check": syntax_check,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return final_code, {
        "required_functions": required_functions,
        "last_error": last_error,
    }


def _extract_python_code(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    fence_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
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

