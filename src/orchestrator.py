from __future__ import annotations
import os
import pandas as pd
from typing import Dict
from google import genai

from src.utils import utc_run_id
from src.modules.profile import profiling
from src.modules.hypothesis import generate_hypotheses

def _load_llm() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def run_pipeline(config: Dict) -> Dict:

    run_id = utc_run_id()
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    data_cfg = config.get("data", {})
    fe_cfg = config.get("feature_engineering", {})

    budget_cfg = fe_cfg.get("budget", {})
    profile_cfg = fe_cfg.get("profile", {})
    hypothesis_cfg = fe_cfg.get("hypothesis", {})
    design_cfg = fe_cfg.get("design", {})
    implement_cfg = fe_cfg.get("implement", {})
    execute_cfg = fe_cfg.get("execute", {})
    diagnose_cfg = fe_cfg.get("diagnose", {})
    report_cfg = fe_cfg.get("report", {})

    train_path = data_cfg.get("train_path", "data/dacon/train.csv")
    test_path = data_cfg.get("test_path", "data/dacon/test.csv")

    client = _load_llm()

    iter_num = budget_cfg.get("iterations", 3)
    diagnose_result = None
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
            diagnose_result=diagnose_result
        )

        # 2. Hypothesis Generation
        print("  Step 2: Hypothesis Generation")
        hypotheses = generate_hypotheses(
            client=client, 
            hypothesis_cfg=hypothesis_cfg, 
            profile_result=profile_result,
            output_dir=iter_dir,
        )

        # 3. Design
        # design_result = design(client, design_cfg, hypotheses)

        # 4. Implement
        # implement_result = implement(client, implement_cfg, design_result)

        # 5. Execute
        # execute_result = execute(client, execute_cfg, implement_result, train_path)

        # 6. Diagnose
        # diagnose_result = diagnose(client, diagnose_cfg, execute_result)