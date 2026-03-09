from __future__ import annotations
import os
import json
from argparse import ArgumentParser
from dotenv import load_dotenv

from src.orchestrator import run_pipeline


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/dacon.json", help="Path to YAML config.")
    args = parser.parse_args()

    load_dotenv()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = load_config(args.config)
    result = run_pipeline(config)

    # Print summary of results
    print("Pipeline finished.")
    print(f"Run ID: {result['run_id']}")
    print(f"Run Dir: {result['run_dir']}")
    print(f"Best Iteration: {result['best_summary']['iteration']}")
    print(f"Best Mean CV: {result['best_summary']['mean_cv']:.6f}")
    print(f"Best Std CV: {result['best_summary']['std_cv']:.6f}")


if __name__ == "__main__":
    main()
