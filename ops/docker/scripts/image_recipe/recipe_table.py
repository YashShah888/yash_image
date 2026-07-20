"""Adaptive hyperparameter policy for the five current image model families."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recipe:
    rank: int
    alpha: int
    steps: int
    learning_rate: float
    caption_dropout_rate: float | None
    save_every: int
    max_saves: int
    bucket: str
    notes: list[str] = field(default_factory=list)

    @property
    def step_ceiling(self) -> int:  # compatibility with v1 logging/tests
        return self.steps


def size_bucket(n_images: int) -> str:
    if n_images <= 8:
        return "xs"
    if n_images <= 16:
        return "s"
    if n_images <= 32:
        return "m"
    return "l"


_BASE_STEPS = {
    "flux": 1650,
    "z-image": 1750,
    "qwen-image": 2350,
    "ideogram4": 1850,
    "krea2": 1850,
}

_BASE_LR = {
    "flux": 1.0e-4,
    "z-image": 9.0e-5,
    "qwen-image": 8.0e-5,
    # The upstream Ideogram template uses 4e-4.  A moderated value converges
    # materially faster than v1's 8e-5 without taking the full overfit risk.
    "ideogram4": 2.0e-4,
    "krea2": 9.0e-5,
}


def _round_to(value: float, quantum: int = 50) -> int:
    return max(quantum, int(round(value / quantum) * quantum))


def build_recipe(
    model_type: str,
    is_subject: bool,
    category: str | None,
    category_confident: bool,
    n_images: int,
    template_supports_caption_dropout: bool,
    hours_to_complete: float = 1.5,
) -> Recipe:
    model_type = (model_type or "flux").strip().lower()
    category = (category or "style").strip().lower()
    bucket = size_bucket(n_images)
    notes = [f"bucket={bucket}; images={n_images}"]

    # Rank is capacity. Identity tasks need less capacity to avoid learning
    # backgrounds; graphic/style tasks need more to carry structure and text.
    if is_subject:
        rank = {"xs": 16, "s": 20, "m": 24, "l": 24}[bucket]
    else:
        rank = {"xs": 24, "s": 32, "m": 40, "l": 48}[bucket]
    if category_confident and category in {"logo", "social", "design"}:
        rank = min(64, rank + 8)
        notes.append("graphic-format capacity boost")
    if category_confident and category == "product":
        rank = max(16, rank - 4)
        notes.append("product overfit guard")

    base_steps = _BASE_STEPS.get(model_type, _BASE_STEPS["flux"])
    # Approximate effective epochs.  Very small sets need more optimizer steps,
    # while large sets obtain more unique signal per step.
    data_scale = min(1.28, max(0.72, math.sqrt(16.0 / max(n_images, 4))))
    shape_scale = 0.68 if is_subject else 1.0
    category_scale = 1.0
    if category_confident and category in {"logo", "social", "design"}:
        category_scale = 1.10
    elif category_confident and category == "product":
        category_scale = 0.88

    # The wall-clock watchdog remains authoritative.  This only avoids asking
    # for an impossible number of steps when a short task budget is supplied.
    time_scale = min(1.20, max(0.65, max(hours_to_complete, 0.25) / 1.5))
    steps = _round_to(base_steps * data_scale * shape_scale * category_scale * time_scale)
    steps = min(3200, max(550 if is_subject else 900, steps))

    lr = _BASE_LR.get(model_type, _BASE_LR["flux"])
    if is_subject:
        lr *= 1.08
    elif category in {"style", "design"}:
        lr *= 0.90
    if n_images <= 8:
        lr *= 0.90
    lr = float(f"{lr:.8g}")

    # The evaluator mixes prompt-guided and empty-prompt img2img branches.
    # Moderate caption dropout gives the LoRA unconditional reconstruction
    # practice without discarding prompt alignment.  The same ai-toolkit
    # dataset schema is shared by all five architectures, so we intentionally
    # set this for Z/Qwen as well as templates that already declare the key.
    if is_subject:
        dropout = 0.18
    elif category == "product":
        dropout = 0.16
    elif category in {"logo", "social", "design"}:
        dropout = 0.08
    else:
        dropout = 0.12
    if model_type in {"z-image", "qwen-image"}:
        dropout = min(0.22, dropout + 0.03)
    if not template_supports_caption_dropout:
        notes.append("caption dropout injected into shared ai-toolkit dataset schema")

    save_every = max(150, _round_to(steps / 4.0, 50))
    save_every = min(save_every, 500)
    return Recipe(
        rank=rank,
        alpha=rank,
        steps=steps,
        learning_rate=lr,
        caption_dropout_rate=dropout,
        save_every=save_every,
        max_saves=4,
        bucket=bucket,
        notes=notes,
    )
