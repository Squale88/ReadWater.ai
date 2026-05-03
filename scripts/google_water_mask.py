"""Shim: dispatches to readwater.pipeline.cv.water_mask.

The real implementation lives in the package; this script exists so the
historical CLI invocation (``python scripts/google_water_mask.py --cell ...``)
keeps working. The shim also loads ``.env`` from the repo root so the
GOOGLE_MAPS_API_KEY is available — the package code itself stays free of
file-system probes at import time.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

sys.path.insert(0, str(_REPO_ROOT / "src"))

from readwater.pipeline.cv.water_mask import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
