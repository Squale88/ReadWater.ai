"""Computer-vision pipeline (Phase 4 architecture).

The CV pipeline replaces the previous LLM-based structure agent for anchor
discovery (deleted; see DEPRECATED.md). Each per-cell run produces:

  detect_drains, detect_islands, detect_points, detect_pockets ->
      one cv_<kind>_<ts>.json per cell with the candidate features
  orchestrator -> cv_all_<ts>.json combining the per-detector outputs
                  through dedup + clustering + parent/child linking, plus
                  a cv_all_<ts>.png review overlay.

Inputs (per cell, on disk):
  - z16 satellite image (from discovery pipeline)
  - z16 + z14 water masks (from water_mask)
  - seagrass + oyster masks (from habitat_mask)

All paths are resolved through ``readwater.storage`` — no module under here
constructs a path by hand.
"""

from __future__ import annotations
