from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from jinja2 import Template

PREPROCESSOR_SYSTEM_INSTRUCTION = """You implement preprocessing module code for tabular ML.
Return only executable Python code.
Return exactly one class definition named GeneratedPreprocessor.
Do not include markdown fences or explanations."""

FEATURE_BLOCK_SYSTEM_INSTRUCTION = """You implement one feature block class for tabular ML.
Return only executable Python code.
Return exactly one class definition.
Do not include markdown fences or explanations."""

MODULE_HEADER = """from __future__ import annotations
from typing import Any, Dict
import pandas as pd

"""


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
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del train_path, label_col, pipeline_config, profile_result, task_context

    implement_dir = Path(output_dir) / "implement"
    implement_dir.mkdir(parents=True, exist_ok=True)

    model = str(implement_cfg.get("model", "gemini-2.5-flash"))
    temperature = float(implement_cfg.get("temperature", 0.2))
    top_p = float(implement_cfg.get("top_p", 0.9))
    max_output_tokens = int(
        implement_cfg.get("max_output_tokens", implement_cfg.get("max_tokens", 8192))
    )
    max_codegen_attempts = int(implement_cfg.get("max_codegen_attempts", 2))
    max_feature_block_attempts = int(
        implement_cfg.get("max_feature_block_attempts", max_codegen_attempts)
    )
    syntax_check = bool(implement_cfg.get("syntax_check", True))

    preprocessor_prompt_path = Path(
        str(implement_cfg.get("prompt_path", "src/prompt/3_implement_e2e.j2"))
    )
    feature_block_prompt_path = Path(
        str(
            implement_cfg.get(
                "feature_block_prompt_path",
                "src/prompt/3_implement_feature_block.j2",
            )
        )
    )
    preprocessor_system_instruction = str(
        implement_cfg.get("system_instruction", PREPROCESSOR_SYSTEM_INSTRUCTION)
    )
    feature_block_system_instruction = str(
        implement_cfg.get(
            "feature_block_system_instruction",
            FEATURE_BLOCK_SYSTEM_INSTRUCTION,
        )
    )

    if not preprocessor_prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {preprocessor_prompt_path}")
    if not feature_block_prompt_path.exists():
        raise FileNotFoundError(f"Feature block prompt template not found: {feature_block_prompt_path}")

    preprocessing_hypotheses = _normalize_text_list(hypotheses.get("preprocessing"))
    feature_hypotheses = _normalize_text_list(hypotheses.get("feature_engineering"))
    preprocessing_codegen_instruction = _resolve_preprocessing_codegen_instruction(
        hypotheses=hypotheses,
        preprocessing_hypotheses=preprocessing_hypotheses,
    )
    feature_codegen_instructions = _resolve_feature_codegen_instructions(
        hypotheses=hypotheses,
        feature_hypotheses=feature_hypotheses,
    )

    preprocessor_template = Template(preprocessor_prompt_path.read_text(encoding="utf-8"))
    feature_block_template = Template(feature_block_prompt_path.read_text(encoding="utf-8"))

    preprocessor_result = _generate_preprocessor_module(
        client=client,
        template=preprocessor_template,
        implement_dir=implement_dir,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        max_attempts=max_codegen_attempts,
        syntax_check=syntax_check,
        system_instruction=preprocessor_system_instruction,
        preprocessing_hypotheses=preprocessing_hypotheses,
        preprocessing_codegen_instruction=preprocessing_codegen_instruction,
        external_feedback=external_feedback,
    )

    feature_block_results: List[Dict[str, Any]] = []
    for index, hypothesis_text in enumerate(feature_hypotheses, start=1):
        class_name = f"GeneratedFeatureBlock{index}"
        feature_codegen_instruction = (
            feature_codegen_instructions[index - 1]
            if index - 1 < len(feature_codegen_instructions)
            else _default_feature_codegen_instruction(hypothesis_text)
        )
        block_result = _generate_feature_block_module(
            client=client,
            template=feature_block_template,
            implement_dir=implement_dir,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            max_attempts=max_feature_block_attempts,
            syntax_check=syntax_check,
            system_instruction=feature_block_system_instruction,
            class_name=class_name,
            index=index,
            hypothesis_text=hypothesis_text,
            feature_codegen_instruction=feature_codegen_instruction,
            external_feedback=external_feedback,
        )
        feature_block_results.append(block_result)

    feature_block_manifest_path = implement_dir / "feature_blocks_manifest.json"
    _write_json(
        feature_block_manifest_path,
        {
            "count": len(feature_block_results),
            "blocks": [
                {
                    "index": item.get("index"),
                    "class_name": item.get("class_name"),
                    "hypothesis": item.get("hypothesis"),
                    "module_path": item.get("path"),
                }
                for item in feature_block_results
            ],
        },
    )

    planned_pipeline_script_path = str(Path(output_dir) / "execute" / "assembled_pipeline.py")
    summary = {
        "pipeline_script_path": planned_pipeline_script_path,
        "preprocessor_module_path": preprocessor_result.get("path"),
        "feature_block_module_paths": [item.get("path") for item in feature_block_results],
        "feature_block_manifest_path": str(feature_block_manifest_path),
        "meta": {
            "mode": "module_codegen_then_execute_assembly",
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "max_codegen_attempts": max_codegen_attempts,
            "max_feature_block_attempts": max_feature_block_attempts,
            "syntax_check": syntax_check,
            "preprocessor_prompt_path": str(preprocessor_prompt_path),
            "feature_block_prompt_path": str(feature_block_prompt_path),
            "preprocessor_system_instruction": preprocessor_system_instruction,
            "feature_block_system_instruction": feature_block_system_instruction,
            "preprocessing_hypothesis_count": len(preprocessing_hypotheses),
            "feature_hypothesis_count": len(feature_hypotheses),
            "feature_codegen_instruction_count": len(feature_codegen_instructions),
        },
    }
    _write_json(implement_dir / "implement_summary.json", summary)
    return summary


def _generate_preprocessor_module(
    client: genai.Client,
    template: Template,
    implement_dir: Path,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_attempts: int,
    syntax_check: bool,
    system_instruction: str,
    preprocessing_hypotheses: List[str],
    preprocessing_codegen_instruction: str,
    external_feedback: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    output_path = implement_dir / "preprocessor_module.py"
    last_error = ""
    feedback: Optional[Dict[str, Any]] = None
    last_raw_text = ""
    last_response_meta: Dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        prompt = template.render(
            preprocessing_hypotheses_json=json.dumps(preprocessing_hypotheses, ensure_ascii=False),
            preprocessing_codegen_instruction=preprocessing_codegen_instruction,
            generation_feedback_json=json.dumps(feedback, ensure_ascii=False) if feedback is not None else "null",
            external_feedback_json=json.dumps(external_feedback, ensure_ascii=False) if external_feedback is not None else "null",
        )
        print(
            f"    [Step3:preprocessor] LLM attempt {attempt}/{max_attempts} "
            f"model={model} prompt_chars={len(prompt)}"
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
        print(
            f"    [Step3:preprocessor] LLM attempt {attempt} "
            f"response_chars={len(raw_text)} finish_reason={response_meta.get('finish_reason', 'unknown')}"
        )

        try:
            code = _extract_python_code(raw_text)
            class_code = _extract_single_class_code(code=code, expected_class_name="GeneratedPreprocessor")
            _validate_class_methods(
                class_code=class_code,
                class_name="GeneratedPreprocessor",
                required_methods=["fit_preprocessor", "transform_preprocessor"],
            )
            module_code = MODULE_HEADER + class_code.strip() + "\n"
            if syntax_check:
                _syntax_check_text(module_code, file_label="preprocessor_module")
            output_path.write_text(module_code, encoding="utf-8")
            print(f"    [Step3:preprocessor] LLM attempt {attempt} parsed successfully")
            _write_json(
                implement_dir / "preprocessor_raw_response.json",
                {
                    "raw_text": raw_text,
                    "response_meta": response_meta,
                    "last_error": "",
                },
            )
            return {
                "path": str(output_path),
                "class_name": "GeneratedPreprocessor",
                "response_meta": response_meta,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            feedback = {"attempt": attempt, "error": last_error}
            print(f"    [Step3:preprocessor] LLM attempt {attempt} failed: {last_error}")
            _write_json(
                implement_dir / f"preprocessor_attempt_{attempt}.json",
                {
                    "attempt": attempt,
                    "error": last_error,
                    "response_meta": response_meta,
                    "raw_text": raw_text,
                },
            )

    _write_json(
        implement_dir / "preprocessor_raw_response.json",
        {
            "raw_text": last_raw_text,
            "response_meta": last_response_meta,
            "last_error": last_error,
        },
    )
    raise RuntimeError(f"Failed to generate preprocessor module: {last_error}")


def _generate_feature_block_module(
    client: genai.Client,
    template: Template,
    implement_dir: Path,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_attempts: int,
    syntax_check: bool,
    system_instruction: str,
    class_name: str,
    index: int,
    hypothesis_text: str,
    feature_codegen_instruction: str,
    external_feedback: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    output_path = implement_dir / f"feature_block_{index}.py"
    last_error = ""
    feedback: Optional[Dict[str, Any]] = None
    last_raw_text = ""
    last_response_meta: Dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        prompt = template.render(
            class_name=class_name,
            feature_index=index,
            hypothesis_text=hypothesis_text,
            feature_codegen_instruction=feature_codegen_instruction,
            generation_feedback_json=json.dumps(feedback, ensure_ascii=False) if feedback is not None else "null",
            external_feedback_json=json.dumps(external_feedback, ensure_ascii=False) if external_feedback is not None else "null",
        )
        print(
            f"    [Step3:feature_block_{index}] LLM attempt {attempt}/{max_attempts} "
            f"model={model} prompt_chars={len(prompt)}"
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
        print(
            f"    [Step3:feature_block_{index}] LLM attempt {attempt} "
            f"response_chars={len(raw_text)} finish_reason={response_meta.get('finish_reason', 'unknown')}"
        )

        try:
            code = _extract_python_code(raw_text)
            class_code = _extract_single_class_code(code=code, expected_class_name=class_name)
            _validate_class_methods(
                class_code=class_code,
                class_name=class_name,
                required_methods=["fit", "transform"],
            )
            module_code = MODULE_HEADER + class_code.strip() + "\n"
            if syntax_check:
                _syntax_check_text(module_code, file_label=f"feature_block_{index}")
            output_path.write_text(module_code, encoding="utf-8")
            print(f"    [Step3:feature_block_{index}] LLM attempt {attempt} parsed successfully")
            _write_json(
                implement_dir / f"feature_block_{index}_raw_response.json",
                {
                    "raw_text": raw_text,
                    "response_meta": response_meta,
                    "last_error": "",
                    "hypothesis": hypothesis_text,
                },
            )
            return {
                "index": index,
                "class_name": class_name,
                "hypothesis": hypothesis_text,
                "path": str(output_path),
                "response_meta": response_meta,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            feedback = {"attempt": attempt, "error": last_error}
            print(f"    [Step3:feature_block_{index}] LLM attempt {attempt} failed: {last_error}")
            _write_json(
                implement_dir / f"feature_block_{index}_attempt_{attempt}.json",
                {
                    "attempt": attempt,
                    "error": last_error,
                    "response_meta": response_meta,
                    "raw_text": raw_text,
                    "class_name": class_name,
                    "hypothesis": hypothesis_text,
                },
            )

    _write_json(
        implement_dir / f"feature_block_{index}_raw_response.json",
        {
            "raw_text": last_raw_text,
            "response_meta": last_response_meta,
            "last_error": last_error,
            "hypothesis": hypothesis_text,
        },
    )
    raise RuntimeError(f"Failed to generate feature block {index}: {last_error}")


def _extract_single_class_code(code: str, expected_class_name: str) -> str:
    tree = ast.parse(code)
    class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if not class_nodes:
        raise ValueError("No class definition found.")

    target: Optional[ast.ClassDef] = None
    for node in class_nodes:
        if node.name == expected_class_name:
            target = node
            break

    if target is None:
        if len(class_nodes) != 1:
            names = [node.name for node in class_nodes]
            raise ValueError(
                "Expected one class or matching class name, "
                f"expected={expected_class_name}, found={names}"
            )
        target = class_nodes[0]
        rename = True
    else:
        rename = False

    class_code = _extract_node_source(code=code, node=target)
    if rename:
        class_code = re.sub(
            r"^(\s*class\s+)[A-Za-z_][A-Za-z0-9_]*",
            rf"\1{expected_class_name}",
            class_code,
            count=1,
            flags=re.MULTILINE,
        )
    return class_code


def _extract_node_source(code: str, node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        raise ValueError("Cannot extract class source lines.")
    lines = code.splitlines()
    return "\n".join(lines[start - 1 : end])


def _validate_class_methods(class_code: str, class_name: str, required_methods: List[str]) -> None:
    tree = ast.parse(class_code)
    class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(class_nodes) != 1:
        raise ValueError(f"{class_name} output must contain exactly one class.")

    node = class_nodes[0]
    if node.name != class_name:
        raise ValueError(f"Expected class {class_name}, got {node.name}")

    method_names = {
        item.name
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = [name for name in required_methods if name not in method_names]
    if missing:
        raise ValueError(f"{class_name} missing required methods: {missing}")


def _extract_python_code(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    fence_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text


def _syntax_check_text(code: str, file_label: str) -> None:
    try:
        ast.parse(code)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Python syntax check failed for {file_label}: {exc}") from exc


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


def _resolve_preprocessing_codegen_instruction(
    hypotheses: Dict[str, Any],
    preprocessing_hypotheses: List[str],
) -> str:
    direct = str(hypotheses.get("preprocessing_codegen_instruction", "") or "").strip()
    if direct:
        return direct

    if preprocessing_hypotheses:
        joined = "; ".join(preprocessing_hypotheses[:5])
        return (
            "Implement train-only deterministic preprocessing using these hypotheses in order: "
            f"{joined}. Preserve label column on transform and avoid target leakage."
        )

    return (
        "Implement robust deterministic preprocessing with stable schema alignment, "
        "missing-value handling, dtype-safe conversion, and no target leakage."
    )


def _resolve_feature_codegen_instructions(
    hypotheses: Dict[str, Any],
    feature_hypotheses: List[str],
) -> List[str]:
    raw = hypotheses.get("feature_engineering_codegen_instructions", [])
    instructions = _normalize_text_list(raw)
    target_count = len(feature_hypotheses)

    while len(instructions) < target_count:
        idx = len(instructions)
        hypothesis_text = feature_hypotheses[idx] if idx < len(feature_hypotheses) else f"feature_{idx + 1}"
        instructions.append(_default_feature_codegen_instruction(hypothesis_text))

    return instructions[:target_count]


def _default_feature_codegen_instruction(hypothesis_text: str) -> str:
    return (
        "Implement exactly one feature for this hypothesis. "
        f"Hypothesis: {str(hypothesis_text).strip()}. "
        "In fit(), compute train-only state. In transform(), return one pd.Series aligned to df.index "
        "without using label values and with deterministic missing-value handling."
    )


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
