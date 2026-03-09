from __future__ import annotations
import os
import json
from typing import Any, Dict, List
from google import genai

from src.utils import utc_run_id
from src.modules.step1_profile import profiling
from src.modules.step2_hypothesis import generate_hypotheses
from src.modules.step3_implement import implement
from src.modules.step4_execute import execute
from src.modules.step5_diagnose import diagnose
from src.modules.final_report import make_report


def _load_llm() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def run_pipeline(config: Dict) -> Dict:

    run_id = utc_run_id()
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    run_config_path = os.path.join(run_dir, "config.json")
    with open(run_config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    data_cfg = config.get("data", {})
    fe_cfg = config.get("feature_engineering", {})

    budget_cfg = fe_cfg.get("budget", {})
    profile_cfg = fe_cfg.get("profile", {})
    hypothesis_cfg = fe_cfg.get("hypothesis", {})
    implement_cfg = fe_cfg.get("implement", {})
    execute_cfg = fe_cfg.get("execute", {})
    diagnose_cfg = fe_cfg.get("diagnose", {})
    report_cfg = fe_cfg.get("report", {})

    train_path = data_cfg.get("train_path", "data/dacon/train.csv")
    test_path = data_cfg.get("test_path", "data/dacon/test.csv")
    label_col = data_cfg.get("label_col")
    _ = test_path

    client = _load_llm()

    iter_num = budget_cfg.get("iterations", 3)
    diagnose_results: List[Dict] = []
    execute_results: List[Dict] = []
    prev_diagnose_result = None
    iteration_summaries: List[Dict] = []
    successful_iterations: List[Dict] = []

    for iteration in range(1, iter_num + 1):
        print(f"=== Iteration {iteration}/{iter_num} ===")

        iter_dir = os.path.join(run_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        # 1. Profiling
        print("  Step 1: Profiling")
        profile_result = profiling(
            client=client, 
            profile_cfg=profile_cfg, 
            train_path=train_path,
            output_dir=iter_dir,
            iteration=iteration,
            prev_diagnose_result=prev_diagnose_result
        )

        # 2. Hypothesis Generation
        print("  Step 2: Hypothesis Generation")
        hypotheses = generate_hypotheses(
            client=client, 
            hypothesis_cfg=hypothesis_cfg, 
            profile_result=profile_result,
            output_dir=iter_dir,
        )

        # 3. Implement
        print("  Step 3: Implement")
        implement_result = implement(
            client=client, 
            implement_cfg=implement_cfg, 
            profile_result=profile_result,
            hypotheses=hypotheses,
            output_dir=iter_dir,
            train_path=train_path,
            label_col=label_col,
            pipeline_config=config,
            external_feedback=None,
        )

        # 4. Execute
        print("  Step 4: Execute")
        execute_cfg_for_iter = dict(execute_cfg)
        execute_cfg_for_iter.setdefault("config_path", run_config_path)
        max_fallbacks = int(execute_cfg_for_iter.get("max_implement_fallbacks", 1))
        fallback_count = 0
        execute_attempts: List[Dict] = []
        execute_result: Dict = {}

        for attempt in range(max_fallbacks + 1):
            execute_result = execute(
                execute_cfg=execute_cfg_for_iter,
                implement_result=implement_result,
                iteration_dir=iter_dir,
            )
            execute_attempts.append(execute_result)

            if not execute_result.get("hard_failure", True):
                break

            if attempt < max_fallbacks:
                fallback_count += 1
                print(f"    Hard failure in execute. Re-running implement fallback ({fallback_count}/{max_fallbacks})")
                implement_feedback = _build_implement_feedback(execute_result)
                implement_result = implement(
                    client=client,
                    implement_cfg=implement_cfg,
                    profile_result=profile_result,
                    hypotheses=hypotheses,
                    output_dir=iter_dir,
                    train_path=train_path,
                    label_col=label_col,
                    pipeline_config=config,
                    external_feedback=implement_feedback,
                )
            else:
                break

        # 5. Diagnose
        print("  Step 5: Diagnose")
        execute_results.append({"iteration": iteration, **execute_result})
        best_before_iteration = max(successful_iterations, key=lambda x: x["objective_mean"]) if successful_iterations else None
        diagnose_result = diagnose(
            client=client,
            diagnose_cfg=diagnose_cfg,
            execute_result=execute_result,
            output_dir=iter_dir,
            iteration=iteration,
            best_before_iteration=best_before_iteration,
        )
        diagnose_results.append(diagnose_result)
        prev_diagnose_result = diagnose_result

        execute_cv = execute_result.get("cv_result", {}) if isinstance(execute_result, dict) else {}
        iteration_summaries.append(
            {
                "iteration": iteration,
                "profile_path": os.path.join(iter_dir, "profile", "profile.json"),
                "hypothesis_path": os.path.join(iter_dir, "hypothesis", "hypothesis.json"),
                "implement_summary_path": os.path.join(iter_dir, "implement", "implement_summary.json"),
                "execute_result_path": os.path.join(iter_dir, "execute", "execute_result.json"),
                "execute_attempts": len(execute_attempts),
                "implement_fallback_count": fallback_count,
                "success": bool(execute_result.get("success", False)),
                "hard_failure": bool(execute_result.get("hard_failure", True)),
                "reason": str(execute_result.get("reason", "")),
                "mean_cv": execute_cv.get("mean_cv"),
                "std_cv": execute_cv.get("std_cv"),
                "objective_mean": execute_cv.get("objective_mean"),
                "metric": execute_cv.get("metric"),
                "diagnose_path": os.path.join(iter_dir, "diagnose", "diagnose.json"),
                "diagnose_status": diagnose_result.get("status"),
            }
        )
        if execute_result.get("success", False):
            successful_iterations.append(
                {
                    "iteration": iteration,
                    "mean_cv": float(execute_cv["mean_cv"]),
                    "std_cv": float(execute_cv["std_cv"]),
                    "objective_mean": float(execute_cv["objective_mean"]),
                    "metric": str(execute_cv["metric"]),
                }
            )

    best = max(successful_iterations, key=lambda x: x["objective_mean"]) if successful_iterations else None
    report_result = make_report(
        report_cfg=report_cfg,
        diagnose_results=diagnose_results,
        iteration_summaries=iteration_summaries,
        execute_results=execute_results,
        run_dir=run_dir,
        best_summary=best,
    )

    if not successful_iterations:
        raise RuntimeError(
            "No successful iteration found. Check run artifacts for error logs. "
            f"Report: {report_result.get('report_path')}"
        )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "iterations": iteration_summaries,
        "best_summary": best,
        "report_path": report_result.get("report_path"),
        "report_json_path": report_result.get("report_json_path"),
    }


def _build_implement_feedback(execute_result: Dict[str, Any]) -> Dict[str, Any]:
    detail = execute_result.get("detail", {}) if isinstance(execute_result, dict) else {}
    stderr_tail = str(detail.get("stderr_tail", ""))
    stdout_tail = str(detail.get("stdout_tail", ""))

    stderr_path = detail.get("stderr_path")
    if not stderr_tail and isinstance(stderr_path, str) and os.path.exists(stderr_path):
        try:
            with open(stderr_path, "r", encoding="utf-8") as file:
                stderr_tail = file.read()[-4000:]
        except Exception:
            stderr_tail = ""

    stdout_path = detail.get("stdout_path")
    if not stdout_tail and isinstance(stdout_path, str) and os.path.exists(stdout_path):
        try:
            with open(stdout_path, "r", encoding="utf-8") as file:
                stdout_tail = file.read()[-2000:]
        except Exception:
            stdout_tail = ""

    return {
        "hard_failure": bool(execute_result.get("hard_failure", True)) if isinstance(execute_result, dict) else True,
        "reason": str(execute_result.get("reason", "unknown_execute_failure")) if isinstance(execute_result, dict) else "unknown_execute_failure",
        "detail": detail,
        "stderr_tail": stderr_tail,
        "stdout_tail": stdout_tail,
    }
