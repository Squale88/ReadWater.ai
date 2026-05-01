"""Placeholder-and-existence checks for the cell-context prompt pair (Step 4).

The prompts are consumed by claude_vision._load_prompt + str.format in Step 5.
We lock in both the files' presence and the set of required placeholders here
so a drift in either one surfaces before runtime.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"

USER_REQUIRED_PLACEHOLDERS = {
    "{cell_id}",
    "{zoom}",
    "{center_lat}",
    "{center_lon}",
    "{coverage_miles}",
    "{ancestor_chain_block}",
    "{grid_scoring_digest}",
    "{open_thread_block}",
}


def test_cell_context_system_prompt_exists_and_nonempty():
    path = PROMPTS_DIR / "cell_context_system.txt"
    assert path.exists(), f"missing prompt: {path}"
    text = path.read_text(encoding="utf-8")
    assert len(text.strip()) > 100


def test_cell_context_user_prompt_exists_and_nonempty():
    path = PROMPTS_DIR / "cell_context_user.txt"
    assert path.exists(), f"missing prompt: {path}"
    text = path.read_text(encoding="utf-8")
    assert len(text.strip()) > 100


def test_cell_context_user_has_all_required_placeholders():
    path = PROMPTS_DIR / "cell_context_user.txt"
    text = path.read_text(encoding="utf-8")
    missing = [p for p in USER_REQUIRED_PLACEHOLDERS if p not in text]
    assert not missing, f"missing placeholders in user prompt: {missing}"


def test_cell_context_user_has_fenced_json_example():
    """The user prompt must show the expected JSON shape in a ```json block
    so the caller's _extract_json_from_response finds a template for it."""
    path = PROMPTS_DIR / "cell_context_user.txt"
    text = path.read_text(encoding="utf-8")
    assert "```json" in text
    assert "observations" in text
    assert "morphology" in text
    assert "feature_threads" in text
    assert "open_questions" in text


def test_cell_context_system_enumerates_controlled_vocabulary():
    path = PROMPTS_DIR / "cell_context_system.txt"
    text = path.read_text(encoding="utf-8")
    # Every list in the output has a declared vocabulary.
    for anchor in (
        "observations.label",
        "morphology.kind",
        "feature_threads.feature_type",
        "feature_threads.status",
    ):
        assert anchor in text, f"system prompt missing vocab section for: {anchor}"


def test_cell_context_system_flags_inherited_context_as_guidance():
    """Regression test against the 'inherited context is guidance, not truth'
    design rule — if someone rewrites the prompt we want the test to fail."""
    path = PROMPTS_DIR / "cell_context_system.txt"
    text = path.read_text(encoding="utf-8")
    assert "guidance" in text.lower()
    assert "not truth" in text.lower()
