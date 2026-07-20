"""
Measure-and-select checkpoint finalisation.

The champion entry we studied predicts a single target step count from a
hand-fit formula and trains blind to it — whatever checkpoint exists when
training stops (by step count or by the wall-clock kill) is what gets
uploaded, with no verification it's actually the best point reached. That is
a predict-and-commit design.

This module is the different-in-kind alternative: train to a generous step
*ceiling* (recipe_table.py), let ai-toolkit periodically save checkpoints and
(if config_builder.py's sample block was accepted) periodically render
preview images from a small held-out slice of the real training images that
never went into training. After the training subprocess ends — by finishing
or by being stopped at the wall-clock reserve boundary — this module scores
each retained checkpoint's preview images against the real held-out images
using the same pixel L2 loss family the validator's own evaluator uses
(mean squared error over normalised RGB arrays), and keeps only the
best-scoring checkpoint's files in the output directory.

This only ever *removes* extra checkpoints the base pipeline would already
have discarded eventually (max_step_saves_to_keep already limits how many
accumulate) — it never invents new files, and every failure mode below
degrades to a true no-op that leaves the output directory exactly as
untouched training would have left it. A wrong guess here must never be
worse than not having this module at all.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


_STEP_RE = re.compile(r"(\d{3,})")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_WEIGHT_EXTS = {".safetensors", ".bin", ".pt"}


def _extract_step(name: str) -> int | None:
    matches = _STEP_RE.findall(name)
    if not matches:
        return None
    return int(matches[-1])


def _group_by_step(directory: Path, exts: set[str]) -> dict[int, list[Path]]:
    groups: dict[int, list[Path]] = {}
    if not directory.is_dir():
        return groups
    for entry in directory.iterdir():
        if entry.is_dir():
            continue
        if entry.suffix.lower() not in exts:
            continue
        step = _extract_step(entry.stem)
        if step is None:
            continue
        groups.setdefault(step, []).append(entry)
    return groups


def _l2_loss(a_path: Path, b_path: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(a_path) as a, Image.open(b_path) as b:
            a = a.convert("RGB")
            b = b.convert("RGB")
            if a.size != b.size:
                b = b.resize(a.size)
            arr_a = np.asarray(a, dtype=np.float64) / 255.0
            arr_b = np.asarray(b, dtype=np.float64) / 255.0
            return float(((arr_a - arr_b) ** 2).mean())
    except Exception:
        return None


@dataclass
class SelectionResult:
    chosen_step: int
    scores: dict[int, float]
    removed_steps: list[int]


def select_best(
    output_dir: str,
    held_out_image_paths: list[str],
    samples_subdir_name: str = "samples",
) -> SelectionResult | None:
    """Best-effort. Returns None (and touches nothing) on any failure or
    whenever there isn't enough information to make a confident choice."""
    try:
        out = Path(output_dir)
        if not out.is_dir() or not held_out_image_paths:
            return None

        checkpoint_groups = _group_by_step(out, _WEIGHT_EXTS)
        if len(checkpoint_groups) < 2:
            return None  # nothing to choose between; leave as-is

        samples_dir = out / samples_subdir_name
        sample_groups = _group_by_step(samples_dir, _IMAGE_EXTS)
        if not sample_groups:
            return None  # sampling wasn't produced (schema mismatch etc.) -> no-op

        held_out = [Path(p) for p in held_out_image_paths if os.path.exists(p)]
        if not held_out:
            return None

        scores: dict[int, float] = {}
        for step, sample_files in sample_groups.items():
            if step not in checkpoint_groups:
                continue  # a sample without a matching saved checkpoint is unusable
            sample_files = sorted(sample_files)[: len(held_out)]
            if not sample_files:
                continue
            pairwise = []
            for real, generated in zip(held_out, sample_files):
                loss = _l2_loss(real, generated)
                if loss is not None:
                    pairwise.append(loss)
            if pairwise:
                scores[step] = sum(pairwise) / len(pairwise)

        if not scores:
            return None  # couldn't score anything (e.g. PIL unavailable) -> no-op

        best_step = min(scores, key=scores.get)

        removed = []
        for step, files in checkpoint_groups.items():
            if step == best_step:
                continue
            for f in files:
                try:
                    f.unlink()
                except OSError:
                    pass
            removed.append(step)

        if samples_dir.is_dir():
            shutil.rmtree(samples_dir, ignore_errors=True)

        return SelectionResult(chosen_step=best_step, scores=scores, removed_steps=removed)
    except Exception:
        return None
