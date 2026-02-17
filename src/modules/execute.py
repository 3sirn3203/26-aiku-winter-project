from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def execute(
    execute_cfg: Dict[str, Any],
    implement_result: Dict[str, Any],
    iteration_dir: str,
) -> Dict[str, Any]:
    
    execute_dir = os.path.join(iteration_dir, "execute")
    os.makedirs(execute_dir, exist_ok=True)

    preprocessor_path = str(implement_result.get("preprocessor_path", "")).strip()
    feature_engineering_path = str(implement_result.get("feature_engineering_path", "")).strip()
    if not preprocessor_path or not feature_engineering_path:
        return _hard_failure(
            execute_dir=execute_dir,
            reason="missing_generated_module_paths",
            detail={
                "preprocessor_path": preprocessor_path,
                "feature_engineering_path": feature_engineering_path,
            },
        )

    syntax_check = bool(execute_cfg.get("syntax_check_generated_modules", True))
    if syntax_check:
        ok, reason = _py_compile_check([preprocessor_path, feature_engineering_path])
        if not ok:
            return _hard_failure(
                execute_dir=execute_dir,
                reason="generated_module_syntax_error",
                detail={"error": reason},
            )

    validation_script = str(execute_cfg.get("validation_script", "src/val_wrapper.py"))
    validation_module = str(execute_cfg.get("validation_module", "")).strip()
    output_json = str(execute_cfg.get("output_json", os.path.join(execute_dir, "cv_result.json")))
    predict_test = bool(execute_cfg.get("predict_test", False))
    submission_out = str(execute_cfg.get("submission_out", os.path.join(execute_dir, "submission.csv")))
    timeout_sec = int(execute_cfg.get("timeout_sec", 1800))
    python_bin = str(execute_cfg.get("python_bin", sys.executable))

    cmd: List[str] = [python_bin]
    if validation_module:
        cmd.extend(["-m", validation_module])
    else:
        module_from_script = _script_to_module(validation_script)
        if module_from_script is not None:
            cmd.extend(["-m", module_from_script])
        else:
            cmd.append(validation_script)

    cmd.extend(
        [
            "--config",
            str(execute_cfg.get("config_path", "config/dacon.json")),
            "--preprocessor-path",
            preprocessor_path,
            "--feature-engineering-path",
            feature_engineering_path,
            "--output-json",
            output_json,
        ]
    )

    _append_optional_arg(cmd, "--cv-type", execute_cfg.get("cv_type"))
    _append_optional_arg(cmd, "--n-splits", execute_cfg.get("n_splits"))
    _append_optional_arg(cmd, "--model-type", execute_cfg.get("model_type"))
    _append_optional_arg(cmd, "--metric", execute_cfg.get("metric"))

    enabled_blocks = execute_cfg.get("enabled_blocks")
    if isinstance(enabled_blocks, list):
        enabled_blocks = ",".join([str(x).strip() for x in enabled_blocks if str(x).strip()])
    _append_optional_arg(cmd, "--enabled-blocks", enabled_blocks)

    if predict_test:
        cmd.extend(["--predict-test", "--submission-out", submission_out])

    try:
        run_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return _hard_failure(
            execute_dir=execute_dir,
            reason="validation_timeout",
            detail={
                "timeout_sec": timeout_sec,
                "command": cmd,
                "stdout_tail": str(exc.stdout or "")[-2000:],
                "stderr_tail": str(exc.stderr or "")[-2000:],
            },
        )

    stdout_path = os.path.join(execute_dir, "execute_stdout.log")
    stderr_path = os.path.join(execute_dir, "execute_stderr.log")
    Path(stdout_path).write_text(run_result.stdout or "", encoding="utf-8")
    Path(stderr_path).write_text(run_result.stderr or "", encoding="utf-8")

    if run_result.returncode != 0:
        return _hard_failure(
            execute_dir=execute_dir,
            reason="validation_process_failed",
            detail={
                "returncode": run_result.returncode,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "stdout_tail": str(run_result.stdout or "")[-2000:],
                "stderr_tail": str(run_result.stderr or "")[-2000:],
                "command": cmd,
            },
        )

    if not os.path.exists(output_json):
        return _hard_failure(
            execute_dir=execute_dir,
            reason="missing_cv_result_json",
            detail={"output_json": output_json, "command": cmd},
        )

    try:
        with open(output_json, "r", encoding="utf-8") as file:
            cv_result = json.load(file)
    except Exception as exc:  # noqa: BLE001
        return _hard_failure(
            execute_dir=execute_dir,
            reason="invalid_cv_result_json",
            detail={"output_json": output_json, "error": str(exc)},
        )

    required_keys = ["mean_cv", "std_cv", "metric", "objective_mean"]
    missing = [key for key in required_keys if key not in cv_result]
    if missing:
        return _hard_failure(
            execute_dir=execute_dir,
            reason="missing_required_cv_keys",
            detail={"missing_keys": missing, "output_json": output_json},
        )

    result = {
        "success": True,
        "hard_failure": False,
        "reason": "",
        "command": cmd,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "cv_result_path": output_json,
        "submission_path": submission_out if predict_test else None,
        "cv_result": cv_result,
    }
    with open(os.path.join(execute_dir, "execute_result.json"), "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def _append_optional_arg(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text == "":
        return
    cmd.extend([flag, text])


def _script_to_module(validation_script: str) -> Optional[str]:
    script = validation_script.strip()
    if not script:
        return None
    if script.endswith(".py"):
        normalized = script.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized.startswith("/"):
            return None
        module = normalized[:-3].replace("/", ".")
        if module:
            return module
    return None


def _py_compile_check(module_paths: List[str]) -> tuple[bool, str]:
    for path in module_paths:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            return False, f"path={path}, stderr={stderr}, stdout={stdout}"
    return True, ""


def _hard_failure(execute_dir: str, reason: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = {
        "success": False,
        "hard_failure": True,
        "reason": reason,
        "detail": detail or {},
    }
    with open(os.path.join(execute_dir, "execute_result.json"), "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result
