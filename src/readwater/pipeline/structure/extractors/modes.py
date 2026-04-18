"""Single source of truth for feature-type -> extraction-mode routing.

Add new structure or subzone types here; tests assert every declared type
routes somewhere. Keep this file minimal — it should read as a table.
"""

from __future__ import annotations

# Anchor structure types -> extraction mode
STRUCTURE_TYPE_TO_MODE: dict[str, str] = {
    "oyster_bar": "region",
    "point": "region",
    "cove": "region",
    "mangrove_peninsula": "region",
    "shoreline_bend": "region",
    "current_split": "region",
    "drain": "corridor",
    "trough": "corridor",
    "pass": "corridor",
    "creek_mouth": "corridor",
    "shoreline_cut": "corridor",
    "island_edge": "edge_band",
}

# v1 subzone whitelist. Anything not in this dict is rejected by the agent
# and dropped from the pipeline with a logged reason.
SUBZONE_TYPE_TO_MODE: dict[str, str] = {
    "drain_throat": "corridor",
    "point_tip": "point_feature",
    "oyster_bar_edge": "edge_band",
    "pocket_mouth": "region",
    "island_tip_seam": "edge_band",
}

# Default mode when a structure type is unrecognized. Region is the safest
# fallback because ClickBoxExtractor in region mode handles any seed count.
DEFAULT_MODE: str = "region"


def mode_for(feature_type: str, feature_level: str) -> str:
    """Resolve a feature type to an extraction mode.

    `feature_level` is "anchor" or "subzone". Other levels (complex members)
    can call this with level="anchor" using their own feature_type vocabulary,
    which happens to reuse common sub-types like "point" or "basin".
    """
    if feature_level == "subzone":
        return SUBZONE_TYPE_TO_MODE.get(feature_type, DEFAULT_MODE)
    # Anchor or complex_member.
    if feature_type in STRUCTURE_TYPE_TO_MODE:
        return STRUCTURE_TYPE_TO_MODE[feature_type]
    # Complex members use a small extra vocabulary.
    member_fallbacks = {
        "basin": "region",
        "bar": "region",
        "channel": "corridor",
        "shoreline": "edge_band",
        "pocket": "region",
        "spit": "region",
        "mangrove_finger": "region",
    }
    return member_fallbacks.get(feature_type, DEFAULT_MODE)


def is_subzone_type_allowed(subzone_type: str) -> bool:
    """Defense-in-depth check: the prompt system message constrains this but
    the agent re-verifies in case the LLM returns something off-whitelist."""
    return subzone_type in SUBZONE_TYPE_TO_MODE
