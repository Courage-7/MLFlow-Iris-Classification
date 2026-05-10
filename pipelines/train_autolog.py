#!/usr/bin/env -S uv run python
"""Train with autologging."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iris.training.autolog import train_with_autolog  # noqa: E402

if __name__ == "__main__":
    train_with_autolog()
