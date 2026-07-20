"""
Task-shape classification for image tournament tasks.

Every image LoRA task is one of two training *shapes*:

  subject — one identity held consistent across the set (a person or a single
            product). The recipe should lock that identity in with a lower
            adapter rank so the network doesn't waste capacity on background
            variety it needs to ignore.
  style   — a shared aesthetic/format applied across diverse subjects (an art
            style, a logo system, a social template, a UI design). The recipe
            needs more adapter capacity to carry the transferable pattern
            without memorising any one example.

We first vote from the dataset's caption text (six human-readable categories
that collapse onto the two shapes below), because these datasets are
synthetic and the caption is deliberately descriptive of what the image
*is*, not just what it shows. When caption text is ambiguous (roughly tied
category scores) we optionally break the tie with a cheap, purely offline
image-content signal (perceptual-hash spread across the set): a tight
cluster of near-identical crops reads as "subject", a wide spread of visually
distinct images reads as "style". The image signal only ever resolves an
ambiguous caption vote — a confident caption vote is never overridden by it,
since the caption is the higher-trust signal for this dataset design.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field


CATEGORIES = ("person", "product", "style", "logo", "social", "design")
SUBJECT_CATEGORIES = frozenset({"person", "product"})
STYLE_CATEGORIES = frozenset({"style", "logo", "social", "design"})
FALLBACK_CATEGORY = "style"

# Independent keyword lists (not derived from any other miner's taxonomy —
# these are generic English descriptors for each category, picked to have
# low overlap with each other).
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "person": (
        "man", "woman", "person", "portrait", "face", "boy", "girl", "guy",
        "he ", "she ", "his ", "her ", "selfie", "headshot", "smiling",
        "wearing", "hair", "eyes", "model", "actor", "actress",
    ),
    "product": (
        "product", "bottle", "packaging", "box", "jar", "device", "gadget",
        "shoe", "sneaker", "watch", "bag", "container", "label", "studio shot",
        "isolated on", "white background", "e-commerce", "catalog",
    ),
    "logo": (
        "logo", "wordmark", "emblem", "brandmark", "monogram", "icon set",
        "letterhead", "insignia",
    ),
    "social": (
        "instagram", "social media", "story template", "post template",
        "carousel", "thumbnail", "banner ad", "flyer", "headline",
        "call to action", "cta",
    ),
    "design": (
        "ui", "app screen", "dashboard", "wireframe", "mockup", "website",
        "landing page", "interface", "button", "screen design", "web design",
    ),
    "style": (
        "style", "aesthetic", "art style", "painting", "illustration",
        "rendered in", "in the style of", "artstyle", "texture", "palette",
        "brushwork", "digital art", "concept art",
    ),
}


def _score_caption(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {cat: sum(lowered.count(kw) for kw in kws) for cat, kws in _KEYWORDS.items()}


def classify_from_captions(captions: list[str]) -> tuple[str, bool]:
    """Return (category, confident). `confident` is False when the top two
    categories are within one point of each other, or there is no signal at
    all — callers should treat a low-confidence result as eligible for the
    image-signal tie-break."""
    totals: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for caption in captions:
        if not caption:
            continue
        scored = _score_caption(caption)
        for cat, val in scored.items():
            totals[cat] += val

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    top_cat, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score == 0:
        return FALLBACK_CATEGORY, False
    confident = (top_score - second_score) >= 2
    return top_cat, confident


def _phash_spread(image_paths: list[str]) -> float | None:
    """Mean pairwise perceptual-hash Hamming distance across a sample of the
    dataset, normalised to [0, 1] (0 = every image looks alike, 1 = maximally
    diverse for a 64-bit hash). Returns None if PIL/imagehash aren't
    installed or there aren't enough images to compare — callers must treat
    that as "no signal", never as a hard failure."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return None

    sample = image_paths[:24]
    if len(sample) < 3:
        return None

    hashes = []
    for path in sample:
        try:
            with Image.open(path) as img:
                hashes.append(imagehash.phash(img))
        except Exception:
            continue
    if len(hashes) < 3:
        return None

    total = 0
    pairs = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            total += hashes[i] - hashes[j]
            pairs += 1
    if pairs == 0:
        return None
    return (total / pairs) / 64.0


@dataclass
class ShapeResult:
    category: str
    is_subject: bool
    confident_from_text: bool
    image_signal_used: bool = False
    notes: list[str] = field(default_factory=list)


def resolve_shape(category: str) -> bool:
    """category -> is_subject."""
    cat = (category or "").lower().strip()
    if cat in SUBJECT_CATEGORIES:
        return True
    if cat in STYLE_CATEGORIES:
        return False
    return False


def classify(captions: list[str], image_paths: list[str] | None = None) -> ShapeResult:
    category, confident = classify_from_captions(captions)
    result = ShapeResult(category=category, is_subject=resolve_shape(category), confident_from_text=confident)

    if confident or not image_paths:
        return result

    spread = _phash_spread(image_paths)
    if spread is None:
        result.notes.append("image tie-break unavailable; kept caption vote")
        return result

    # A tight cluster (low spread) under an ambiguous caption vote reads as a
    # single held-consistent subject; a wide spread reads as style/format.
    image_says_subject = spread < 0.18
    if image_says_subject != result.is_subject:
        result.is_subject = image_says_subject
        if result.category not in (SUBJECT_CATEGORIES if image_says_subject else STYLE_CATEGORIES):
            result.category = "person" if image_says_subject else "style"
        result.image_signal_used = True
        result.notes.append(f"image tie-break overrode ambiguous caption vote (spread={spread:.3f})")

    return result


def read_captions(image_dir: str) -> list[str]:
    captions = []
    if not os.path.isdir(image_dir):
        return captions
    for name in sorted(os.listdir(image_dir)):
        if name.lower().endswith(".txt"):
            try:
                with open(os.path.join(image_dir, name), encoding="utf-8") as fh:
                    captions.append(fh.read().strip())
            except OSError:
                continue
    return captions


def list_images(image_dir: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not os.path.isdir(image_dir):
        return []
    return [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if os.path.splitext(name)[1].lower() in exts
    ]
