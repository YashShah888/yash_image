"""Classify an image task without network access or heavyweight models."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CATEGORIES = ("person", "product", "style", "logo", "social", "design")
SUBJECT_CATEGORIES = frozenset({"person", "product"})
STYLE_CATEGORIES = frozenset({"style", "logo", "social", "design"})

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "person": (
        "portrait", "person", "man", "woman", "boy", "girl", "face",
        "headshot", "selfie", "hair", "eyes", "wearing", "character",
    ),
    "product": (
        "product", "bottle", "package", "packaging", "box", "jar", "shoe",
        "sneaker", "watch", "bag", "device", "gadget", "catalog", "e-commerce",
    ),
    "logo": (
        "logo", "wordmark", "brandmark", "emblem", "monogram", "insignia",
        "identity mark", "icon set",
    ),
    "social": (
        "social media", "instagram", "story", "post template", "carousel",
        "thumbnail", "banner", "flyer", "call to action", "headline",
    ),
    "design": (
        "interface", "dashboard", "app screen", "landing page", "website",
        "wireframe", "ui ", "ux ", "layout", "screen design", "mockup",
    ),
    "style": (
        "style", "aesthetic", "illustration", "painting", "brushwork",
        "rendered", "artwork", "palette", "texture", "concept art",
    ),
}


@dataclass(frozen=True)
class ShapeResult:
    category: str
    is_subject: bool
    confidence: float
    text_scores: dict[str, float]
    image_diversity: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def confident_from_text(self) -> bool:
        return self.confidence >= 0.25

    @property
    def image_signal_used(self) -> bool:
        return self.image_diversity is not None and not self.confident_from_text


def read_captions(image_dir: str | Path) -> list[str]:
    root = Path(image_dir)
    if not root.is_dir():
        return []
    captions: list[str] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() == ".txt":
            try:
                value = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value:
                captions.append(value)
    return captions


def list_images(image_dir: str | Path) -> list[str]:
    root = Path(image_dir)
    if not root.is_dir():
        return []
    return [
        str(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def _keyword_scores(captions: list[str]) -> dict[str, float]:
    scores = {category: 0.0 for category in CATEGORIES}
    joined = " ".join(captions).lower()
    joined = re.sub(r"\s+", " ", joined)
    for category, words in _KEYWORDS.items():
        for keyword in words:
            # Longer phrases are more diagnostic than single common words.
            weight = 1.0 + min(1.0, keyword.count(" ") * 0.45)
            scores[category] += joined.count(keyword) * weight
    return scores


def _phash_diversity(paths: list[str]) -> float | None:
    """Mean normalized pairwise pHash distance on at most 24 images."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return None

    hashes = []
    for path in paths[:24]:
        try:
            with Image.open(path) as image:
                hashes.append(imagehash.phash(image.convert("RGB")))
        except Exception:
            continue
    if len(hashes) < 3:
        return None

    total = 0.0
    pairs = 0
    for index, first in enumerate(hashes):
        for second in hashes[index + 1 :]:
            total += float(first - second) / 64.0
            pairs += 1
    return total / pairs if pairs else None


def classify(captions: list[str], image_paths: list[str] | None = None) -> ShapeResult:
    scores = _keyword_scores(captions)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_category, best = ranked[0]
    second = ranked[1][1]
    total = sum(scores.values())
    confidence = 0.0 if best <= 0 else min(1.0, (best - second) / max(best, 1.0))
    notes: list[str] = []

    diversity = None
    if best <= 0 or confidence < 0.25:
        diversity = _phash_diversity(image_paths or [])
        if diversity is not None:
            # Identity/product collections usually repeat a subject and therefore
            # cluster more tightly than style/layout collections.
            if diversity < 0.23:
                best_category = "person"
                notes.append(f"ambiguous text; low visual spread ({diversity:.3f}) => subject")
            else:
                best_category = "style"
                notes.append(f"ambiguous text; high visual spread ({diversity:.3f}) => style")
        elif best <= 0:
            best_category = "style"
            notes.append("no reliable category signal; conservative style fallback")

    return ShapeResult(
        category=best_category,
        is_subject=best_category in SUBJECT_CATEGORIES,
        confidence=confidence,
        text_scores=scores,
        image_diversity=diversity,
        notes=notes,
    )
