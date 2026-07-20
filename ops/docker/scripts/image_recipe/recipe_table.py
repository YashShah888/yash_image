"""
Hyperparameter policy for the ai-toolkit-driven image trainer.

Every model type this repo enters (flux, z-image, qwen-image, ideogram4,
krea2) runs through ai-toolkit's `diffusion_trainer`, so a single overlay
shape works for all five: LoRA rank/alpha, a *step ceiling* (a safe upper
bound, not a predicted exact stopping point — see checkpoint_select.py for
why), and, where the base template already exposes the field, a caption
dropout rate.

Learning rate is deliberately left at the base template's proven value for
v1: retuning LR without real tournament feedback is speculative, and a bad
LR is one of the few mistakes checkpoint selection cannot rescue you from
(an unstable run has no good checkpoint to pick). Rank and the step ceiling
are safer, more reversible knobs to make adaptive first — LR tuning is
explicitly deferred to Stage 2/3 once real results exist to calibrate
against, per the plan.

Every constant below is a first-principles starting point (small-dataset ->
more capacity headroom is dangerous -> lower rank; style/format categories
need more capacity to carry a transferable pattern -> higher rank), not a
number reverse-engineered from another miner's results. They are expected to
move once real tournament outcomes are available (Stage 2/3 of the plan).
"""

from __future__ import annotations

from dataclasses import dataclass


SIZE_BUCKETS = ("xs", "s", "m", "l")


def size_bucket(n_images: int) -> str:
    if n_images <= 8:
        return "xs"
    if n_images <= 18:
        return "s"
    if n_images <= 35:
        return "m"
    return "l"


# rank (network.linear == network.linear_alpha) by (is_subject, bucket).
# Subjects get less capacity (avoid memorising background clutter around the
# one identity); style/format categories get more (need to carry a pattern
# across diverse content).
_RANK_SUBJECT = {"xs": 12, "s": 16, "m": 20, "l": 24}
_RANK_STYLE = {"xs": 20, "s": 24, "m": 28, "l": 32}

# Per-category capacity nudges layered on top of the shape-level rank, for
# categories whose visual content has a distinct capacity need (sharp
# typography/edges want more; a single clean product shot wants less). Only
# applied when the category is confidently known.
_CATEGORY_RANK_DELTA = {
    "logo": 8,
    "social": 8,
    "design": 4,
    "product": -4,
}

# Step CEILING per model type / bucket: a safe upper bound the training
# subprocess is allowed to reach, not a target it is expected to hit exactly.
# checkpoint_select.py decides, by measurement, which point along the way to
# actually ship. Values start near each model's base-template step count and
# taper down for larger datasets (more images per epoch already means more
# gradient signal per step, so fewer total steps are needed to reach the
# same number of "effective passes").
_STEP_CEILING = {
    "flux": {"xs": 1400, "s": 1800, "m": 2000, "l": 2000},
    "z-image": {"xs": 1400, "s": 1800, "m": 2000, "l": 2000},
    "qwen-image": {"xs": 1800, "s": 2400, "m": 3000, "l": 3000},
    "ideogram4": {"xs": 1400, "s": 1800, "m": 2000, "l": 2000},
    "krea2": {"xs": 1400, "s": 1800, "m": 2000, "l": 2000},
}

# Subject tasks overfit faster and rarely need the full ceiling; give them a
# materially tighter one so checkpoint_select has a realistic best-checkpoint
# window to search, rather than searching mostly-overfit late checkpoints.
_STEP_CEILING_SUBJECT_SCALE = 0.4

# Caption dropout, only applied to templates whose datasets[0] already
# defines the key (flux/ideogram4/krea2 do; z-image/qwen-image don't -- we
# never introduce a config key a template doesn't already use).
_CAPTION_DROPOUT_SUBJECT = 0.15
_CAPTION_DROPOUT_STYLE = 0.05


@dataclass
class Recipe:
    rank: int
    step_ceiling: int
    caption_dropout_rate: float | None
    bucket: str
    notes: list[str]


def build_recipe(
    model_type: str,
    is_subject: bool,
    category: str | None,
    category_confident: bool,
    n_images: int,
    template_supports_caption_dropout: bool,
) -> Recipe:
    mt = (model_type or "").lower().strip()
    bucket = size_bucket(n_images)
    notes = [f"bucket={bucket} n={n_images}"]

    base_rank = (_RANK_SUBJECT if is_subject else _RANK_STYLE)[bucket]
    rank = base_rank
    cat = (category or "").lower().strip()
    if category_confident and cat in _CATEGORY_RANK_DELTA:
        rank = max(8, base_rank + _CATEGORY_RANK_DELTA[cat])
        notes.append(f"category delta applied ({cat}: {_CATEGORY_RANK_DELTA[cat]:+d})")

    ceiling_table = _STEP_CEILING.get(mt, _STEP_CEILING["flux"])
    ceiling = ceiling_table[bucket]
    if is_subject:
        ceiling = max(200, round(ceiling * _STEP_CEILING_SUBJECT_SCALE))
        notes.append("subject step ceiling scaled down")

    dropout = None
    if template_supports_caption_dropout:
        dropout = _CAPTION_DROPOUT_SUBJECT if is_subject else _CAPTION_DROPOUT_STYLE

    return Recipe(
        rank=rank,
        step_ceiling=ceiling,
        caption_dropout_rate=dropout,
        bucket=bucket,
        notes=notes,
    )
