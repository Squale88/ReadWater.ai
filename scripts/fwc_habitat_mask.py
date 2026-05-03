"""Shim: dispatches to readwater.pipeline.cv.habitat_mask.

The real implementation lives in the package; this script exists so the
historical CLI invocation (``python scripts/fwc_habitat_mask.py --cell ...``)
keeps working. The shim loads ``.env`` from the repo root for parity with
the other CV scripts; the FWC habitat fetch itself doesn't currently need
secrets, but other downstream code reached from the same process might.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

sys.path.insert(0, str(_REPO_ROOT / "src"))

from readwater.pipeline.cv.habitat_mask import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
