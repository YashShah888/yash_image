"""Non-destructive checkpoint finalisation for the G.O.D uploader.

Preview generations are stochastic and are not aligned with training images,
so raw RGB MSE is not a reliable checkpoint selector.  Challenger v2 trusts
the recipe's planned final point, validates usable weights, and guarantees the
canonical ``last.safetensors`` filename expected by diffusion evaluation.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth"}
_STEP_RE = re.compile(r"(?:step|[-_])(\d{2,})(?:\D|$)", re.IGNORECASE)
_CANONICAL_NAME = "last.safetensors"


@dataclass(frozen=True)
class SelectionResult:
    chosen_step: int
    scores: dict[int, float]
    removed_steps: list[int]
    weight_files: tuple[str, ...] = ()
    canonical_path: str | None = None


def _step(path: Path) -> int:
    for value in (path.stem, path.parent.name):
        match = _STEP_RE.search(value)
        if match:
            return int(match.group(1))
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else 0


def _candidate_weights(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _WEIGHT_EXTS
            and path.stat().st_size > 1024
        ),
        key=lambda path: (_step(path), path.stat().st_mtime_ns, path.stat().st_size),
    )


def _ensure_canonical(root: Path, weights: list[Path]) -> Path:
    canonical = root / _CANONICAL_NAME
    if canonical.is_file() and canonical.stat().st_size > 1024:
        return canonical

    safetensors = [path for path in weights if path.suffix.lower() == ".safetensors"]
    if not safetensors:
        raise RuntimeError(
            f"weights exist below {root}, but no safetensors file can become {_CANONICAL_NAME}"
        )
    source = safetensors[-1]
    if source.resolve() == canonical.resolve():
        return canonical

    temporary = root / f".{_CANONICAL_NAME}.tmp-{os.getpid()}"
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size <= 1024:
            raise RuntimeError(f"canonical checkpoint copy is unexpectedly small: {temporary}")
        os.replace(temporary, canonical)
    finally:
        temporary.unlink(missing_ok=True)
    return canonical


def validate_output(output_dir: str | Path) -> SelectionResult:
    root = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"expected output directory was not created: {root}")
    weights = _candidate_weights(root)
    if not weights:
        raise RuntimeError(f"no non-empty model weight file found below {root}")

    canonical = _ensure_canonical(root, weights)
    weights = _candidate_weights(root)
    newest = max(weights, key=lambda path: (_step(path), path.stat().st_mtime_ns))
    return SelectionResult(
        chosen_step=_step(newest),
        scores={},
        removed_steps=[],
        weight_files=tuple(str(path) for path in weights),
        canonical_path=str(canonical),
    )


def select_best(
    output_dir: str,
    held_out_image_paths: list[str] | None = None,
    samples_subdir_name: str = "samples",
) -> SelectionResult | None:
    """Compatibility API: validate/canonicalize, never MSE-rank or delete."""
    del held_out_image_paths, samples_subdir_name
    try:
        return validate_output(output_dir)
    except Exception:
        return None
