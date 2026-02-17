from __future__ import annotations

import argparse
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.agent.graph import build_graph
from src.agent.state import AgentState


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _get_task_value(config: Dict[str, Any], key: str) -> str:
    if key in config:
        return str(config.get(key) or "")
    task = config.get("task", {}) or {}
    return str(task.get(key) or "")


def _apply_overrides(
    config: Dict[str, Any],
    input_file: Optional[str],
    target_column: Optional[str],
    problem_type: Optional[str],
    max_iters: Optional[int],
    execute_enabled: Optional[bool],
) -> Dict[str, Any]:
    if input_file:
        config["input_file"] = input_file
        config.setdefault("task", {})["input_file"] = input_file
    if target_column:
        config.setdefault("task", {})["target_column"] = target_column
    if problem_type:
        config.setdefault("task", {})["problem_type"] = problem_type
    if max_iters is not None:
        config.setdefault("agent", {})["max_iters"] = int(max_iters)
    if execute_enabled is not None:
        config.setdefault("agent", {})["execute_enabled"] = bool(execute_enabled)
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM data analysis agent.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--problem-type", default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument(
        "--execute",
        dest="execute_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    config = load_config(args.config)
    config = _apply_overrides(
        config,
        input_file=args.input_file,
        target_column=args.target_column,
        problem_type=args.problem_type,
        max_iters=args.max_iters,
        execute_enabled=args.execute_enabled,
    )
    agent_cfg = config.get("agent", {})

    state: AgentState = {
        "config": config,
        "input_file": _get_task_value(config, "input_file"),
        "target_column": _get_task_value(config, "target_column"),
        "problem_type": _get_task_value(config, "problem_type"),
        "iter_count": 0,
        "max_iters": int(agent_cfg.get("max_iters", 1)),
        "history": [],
    }

    graph = build_graph()
    result = graph.invoke(state)

    print("Agent finished.")
    print("Iterations:", result.get("iter_count"))
    print("Stop:", result.get("stop"))


if __name__ == "__main__":
    main()
