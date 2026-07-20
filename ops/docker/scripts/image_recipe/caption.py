"""
Caption handling for v1: verbatim source prompts, no VLM rewrite pass.

These datasets are synthetic and each image ships with a source prompt the
validator's own eval prompts are derived from, so the source prompt is
already the highest-value caption signal available — rewriting or padding it
risks drifting away from the distribution the eval will actually probe.
v1 deliberately does not add a VLM captioning pass (a real, heavier
mechanism some competitors use): it's an extra model dependency, extra
per-image latency inside the training container's time budget, and an
unforced risk for a first entry. Caption *regularisation* (dropout at the
config level, trigger-word placement) is handled here in code instead of by
touching the caption text.
"""

from __future__ import annotations

import os


def apply_trigger_word(caption: str, trigger_word: str | None) -> str:
    if not trigger_word:
        return caption
    tw = trigger_word.strip()
    if not tw:
        return caption
    if tw.lower() in (caption or "").lower():
        return caption
    return f"{tw}, {caption}" if caption else tw


def apply_trigger_to_dir(image_dir: str, trigger_word: str | None) -> int:
    """Rewrite each caption file in-place with the trigger word prepended
    (idempotent — skips files that already mention it). Returns the count of
    files touched."""
    if not trigger_word or not os.path.isdir(image_dir):
        return 0
    touched = 0
    for name in os.listdir(image_dir):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(image_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                original = fh.read().strip()
        except OSError:
            continue
        updated = apply_trigger_word(original, trigger_word)
        if updated != original:
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(updated)
                touched += 1
            except OSError:
                continue
    return touched
