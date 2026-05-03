"""Shim: dispatches to readwater.pipeline.cv.detect_drains.

The real implementation lives in the package; this script exists so the
historical CLI invocation (``python scripts/cv_detect_drains.py --cell ...``)
keeps working.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from readwater.pipeline.cv.detect_drains import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
