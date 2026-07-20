"""Safe dataset extraction and conservative duplicate removal."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_MEMBER_BYTES = 2_000_000_000
MAX_TOTAL_BYTES = 10_000_000_000


@dataclass(frozen=True)
class DedupResult:
    n_before: int
    n_after: int
    n_removed: int
    dup_rate: float
    note: str | None = None


@dataclass(frozen=True)
class DatasetAudit:
    images: int
    captions: int
    missing_captions: int
    corrupt_images: int
    widths: tuple[int, ...]
    heights: tuple[int, ...]


def safe_extract(zip_path: str | Path, destination: str | Path) -> None:
    destination_root = Path(destination).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination_root / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"unsafe path in dataset archive: {member.filename!r}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink is not allowed in dataset archive: {member.filename!r}")
            if member.flag_bits & 0x1:
                raise ValueError(f"encrypted member is unsupported: {member.filename!r}")
            if member.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"dataset member too large: {member.filename!r}")
            total += member.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("dataset expands beyond safety limit")
        archive.extractall(destination_root)


def list_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        return []
    return [
        path for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def _caption(path: Path) -> Path:
    return path.with_suffix(".txt")


def audit(image_dir: str | Path) -> DatasetAudit:
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore[assignment]

    images = list_images(image_dir)
    captions = missing = corrupt = 0
    widths: list[int] = []
    heights: list[int] = []
    for path in images:
        if _caption(path).exists():
            captions += 1
        else:
            missing += 1
        if Image is not None:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    widths.append(int(image.width))
                    heights.append(int(image.height))
            except Exception:
                corrupt += 1
    return DatasetAudit(
        images=len(images),
        captions=captions,
        missing_captions=missing,
        corrupt_images=corrupt,
        widths=tuple(widths),
        heights=tuple(heights),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conservative_dedup(image_dir: str | Path, *, minimum_remaining: int = 6) -> DedupResult:
    """Remove byte-identical images only, retaining the best-captioned copy.

    Tiny tournament datasets are data-starved.  Approximate pHash deletion can
    erase legitimate logo/layout variants, so v2 removes only cryptographically
    exact duplicates.  This captures accidental repeats without sacrificing
    useful supervision.
    """
    images = list_images(image_dir)
    before = len(images)
    if before <= minimum_remaining:
        return DedupResult(before, before, 0, 0.0, "small dataset: duplicate removal disabled")

    groups: dict[str, list[Path]] = {}
    for path in images:
        try:
            groups.setdefault(_sha256(path), []).append(path)
        except OSError:
            continue

    removed = 0
    for duplicates in groups.values():
        if len(duplicates) < 2:
            continue
        # Prefer the image whose paired caption carries more information.
        def quality(path: Path) -> tuple[int, int]:
            cap = _caption(path)
            try:
                text_len = len(cap.read_text(encoding="utf-8").strip())
            except OSError:
                text_len = 0
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            return text_len, size

        keep = max(duplicates, key=quality)
        for path in duplicates:
            if path == keep or before - removed <= minimum_remaining:
                continue
            try:
                path.unlink()
                cap = _caption(path)
                if cap.exists():
                    cap.unlink()
                removed += 1
            except OSError:
                continue

    after = before - removed
    return DedupResult(before, after, removed, round(removed / before, 4) if before else 0.0)


def copy_flattened_dataset(extracted_root: str | Path, target_root: str | Path) -> int:
    source_root = Path(extracted_root)
    target = Path(target_root)
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    allowed = IMAGE_EXTS | {".txt"}
    for source in source_root.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in allowed:
            continue
        destination = target / source.name
        if destination.exists():
            prefix = hashlib.blake2s(str(source.parent).encode(), digest_size=4).hexdigest()
            destination = target / f"{prefix}_{source.name}"
        shutil.copy2(source, destination)
        copied += 1
    return copied


# Backwards-compatible alias used by the prior entrypoint.
def perceptual_dedup(image_dir: str, **_: object) -> DedupResult:
    return conservative_dedup(image_dir)
