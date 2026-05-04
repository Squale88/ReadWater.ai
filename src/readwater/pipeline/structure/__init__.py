"""Structure phase — DEPRECATED.

Investigator agent that produces zone objects from confirmed zoom-16
cells via per-cell Claude vision calls. **Replaced by the deterministic
CV pipeline at ``readwater.pipeline.cv``.** See ``DEPRECATED.md`` at the
repo root for the full migration map.

This package is still imported by ``readwater.pipeline.cell_analyzer``
(its z16 ``run_structure_phase`` call) and by legacy validation scripts.
A future rewire will eliminate those call sites; once that's done this
whole subtree can be deleted.

Do NOT add new consumers of this package. New code should consume CV
anchor outputs via ``readwater.areas.Area("...").cell(cid).anchors_json``.
"""

from __future__ import annotations

import warnings as _warnings

from readwater.pipeline.structure.agent import run_structure_phase

_warnings.warn(
    "readwater.pipeline.structure is deprecated; use readwater.pipeline.cv "
    "instead. See DEPRECATED.md at the repo root.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["run_structure_phase"]
