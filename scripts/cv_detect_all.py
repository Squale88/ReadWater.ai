"""Shim: dispatches to readwater.pipeline.cv.orchestrator.

The real implementation lives in the package; this script exists so the
historical CLI invocation (``python scripts/cv_detect_all.py --cell ...``)
keeps working.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from readwater.pipeline.cv.orchestrator import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
