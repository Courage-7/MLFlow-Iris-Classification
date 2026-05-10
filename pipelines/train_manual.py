#!/usr/bin/env python
"""Train with manual logging."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iris.training.manual import train_with_manual_logging  # noqa: E402

if __name__ == "__main__":
    train_with_manual_logging()
