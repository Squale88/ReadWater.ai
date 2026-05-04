"""Shim: dispatches to readwater.pipeline.cv.run_area.

End-to-end CV pipeline runner for an entire area. The real implementation
lives in the package; this script exists so the historical CLI invocation
(``python scripts/run_area.py --area ...``) keeps working.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

from readwater.pipeline.cv.run_area import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
