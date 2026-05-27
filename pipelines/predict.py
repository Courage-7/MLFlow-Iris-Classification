#!/usr/bin/env -S uv run python
"""Run inference."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iris.inference.predictor import run_inference  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a logged MLflow model."
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI, e.g. models:/iris-logistic-regression/1",
    )
    args = parser.parse_args()
    run_inference(args.model_uri)
