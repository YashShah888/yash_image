"""Deterministic, image-grounded caption enrichment.

The source prompt remains intact.  For format/style tasks we append only cheap,
observable image attributes (orientation, contrast and broad palette).  This
adds visual conditioning signal without a VLM download, hallucination risk or
large runtime penalty.  Subject/product tasks keep captions minimal to protect
identity and exact prompt alignment.
"""

from __future__ import annotations

import colorsys
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class CaptionStats:
    examined: int
    rewritten: int
    missing_created: int
    failures: int


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.strip(" ,.;")


def _contains_token(text: str, token: str) -> bool:
    return bool(token) and token.casefold() in text.casefold()


def _triggered(source: str, trigger_word: str | None, *, is_subject: bool) -> str:
    source = _clean(source)
    trigger = _clean(trigger_word or "")
    if not trigger or _contains_token(source, trigger):
        return source
    if is_subject:
        return f"{trigger}, {source}" if source else trigger
    return f"{source}, {trigger} style" if source else f"{trigger} style"


def _image_attributes(path: Path) -> list[str]:
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return []

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            thumb = rgb.resize((64, 64))
            stat = ImageStat.Stat(thumb)
            means = [float(value) / 255.0 for value in stat.mean[:3]]
            std = sum(float(value) for value in stat.stddev[:3]) / (3.0 * 255.0)
    except Exception:
        return []

    ratio = width / max(height, 1)
    if ratio > 1.2:
        orientation = "landscape composition"
    elif ratio < 0.83:
        orientation = "portrait composition"
    else:
        orientation = "square composition"

    hue, saturation, value = colorsys.rgb_to_hsv(*means)
    if saturation < 0.12:
        palette = "neutral palette"
    elif value < 0.28:
        palette = "dark palette"
    elif value > 0.82:
        palette = "light palette"
    else:
        hue_names = (
            "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"
        )
        palette = f"{hue_names[int(hue * len(hue_names)) % len(hue_names)]}-accented palette"

    contrast = "high contrast" if std > 0.24 else "soft contrast" if std < 0.12 else "balanced contrast"
    return [orientation, palette, contrast]


def _caption_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def enrich_directory(
    image_dir: str | Path,
    *,
    category: str,
    is_subject: bool,
    trigger_word: str | None,
    max_words: int = 75,
) -> CaptionStats:
    root = Path(image_dir)
    examined = rewritten = missing_created = failures = 0
    if not root.is_dir():
        return CaptionStats(0, 0, 0, 0)

    enrich_visual = category in {"style", "logo", "social", "design"}
    for image_path in sorted(root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        examined += 1
        cap_path = _caption_path(image_path)
        try:
            source = cap_path.read_text(encoding="utf-8").strip() if cap_path.exists() else ""
            updated = _triggered(source, trigger_word, is_subject=is_subject)
            if enrich_visual:
                attributes = _image_attributes(image_path)
                # Rotate the true attributes deterministically so every caption
                # is not forced into the same suffix order.
                if attributes:
                    digest = hashlib.blake2b(image_path.name.encode(), digest_size=1).digest()[0]
                    shift = digest % len(attributes)
                    attributes = attributes[shift:] + attributes[:shift]
                    suffix = ", ".join(attributes)
                    if suffix and suffix.casefold() not in updated.casefold():
                        updated = f"{updated}, {suffix}" if updated else suffix
            words = updated.split()
            if len(words) > max_words:
                updated = " ".join(words[:max_words]).rstrip(" ,.;")
            updated = _clean(updated)
            if not cap_path.exists():
                missing_created += 1
            if updated != source:
                cap_path.write_text(updated, encoding="utf-8")
                rewritten += 1
        except OSError:
            failures += 1
    return CaptionStats(examined, rewritten, missing_created, failures)


def apply_trigger_word(caption: str, trigger_word: str | None) -> str:
    """Compatibility helper retained for existing tests/callers."""
    return _triggered(caption, trigger_word, is_subject=True)


def apply_trigger_to_dir(image_dir: str, trigger_word: str | None) -> int:
    """Compatibility helper: add a subject-style trigger to text files."""
    root = Path(image_dir)
    touched = 0
    if not root.is_dir():
        return 0
    for path in root.glob("*.txt"):
        try:
            original = path.read_text(encoding="utf-8").strip()
            updated = _triggered(original, trigger_word, is_subject=True)
            if updated != original:
                path.write_text(updated, encoding="utf-8")
                touched += 1
        except OSError:
            continue
    return touched
