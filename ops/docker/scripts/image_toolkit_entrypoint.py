#!/usr/bin/env python3
"""G.O.D image tournament entrypoint — challenger v2.

Compatible with the current validator arguments and output contract.  The
script is deliberately self-contained at runtime and performs no network I/O.
"""

from __future__ import annotations

import argparse
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_recipe import caption, checkpoint_select, config_builder, dataset_prep, task_shape

CACHE_ROOT = Path("/cache")
DATASET_ZIP_ROOT = CACHE_ROOT / "datasets"
IMAGE_DATASET_ROOT = Path("/dataset/images")
EXTRACT_ROOT = Path("/dataset/extracted")
CONFIG_OUTPUT_ROOT = Path("/dataset/configs")
CHECKPOINTS_ROOT = Path("/app/checkpoints")
AI_TOOLKIT_ROOT = Path("/app/ai-toolkit")


def _dataset_zip(task_id: str, supplied: str) -> Path:
    candidates = [
        Path(supplied),
        DATASET_ZIP_ROOT / supplied,
        DATASET_ZIP_ROOT / f"{task_id}_tourn.zip",
        DATASET_ZIP_ROOT / f"{task_id}.zip",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not locate dataset zip; checked: {joined}")


def prepare_dataset(task_id: str, supplied_zip: str) -> Path:
    zip_path = _dataset_zip(task_id, supplied_zip)
    if IMAGE_DATASET_ROOT.exists():
        shutil.rmtree(IMAGE_DATASET_ROOT)
    IMAGE_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    extract_root = EXTRACT_ROOT / task_id
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    dataset_prep.safe_extract(zip_path, extract_root)
    copied = dataset_prep.copy_flattened_dataset(extract_root, IMAGE_DATASET_ROOT)
    if copied == 0 or not dataset_prep.list_images(IMAGE_DATASET_ROOT):
        raise RuntimeError(f"no supported images found in {zip_path}")
    return IMAGE_DATASET_ROOT


def _reserve_seconds(hours_to_complete: float) -> float:
    total = max(300.0, hours_to_complete * 3600.0)
    reserve = max(240.0, total * 0.10)
    return min(720.0, reserve)


def _signal_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.send_signal(sig)


def _graceful_stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    print("[trainer] deadline reached; requesting a graceful checkpoint", flush=True)
    _signal_group(process, signal.SIGINT)
    try:
        process.wait(timeout=90)
        return
    except subprocess.TimeoutExpired:
        pass
    print("[trainer] SIGINT grace expired; terminating process group", flush=True)
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        print("[trainer] terminate grace expired; killing process group", flush=True)
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=30)


def run_training(config_path: Path, timeout_seconds: float, log_path: Path) -> tuple[int, bool]:
    print(
        f"Starting ai-toolkit with {config_path} (optimization budget={timeout_seconds:.0f}s)",
        flush=True,
    )
    process = subprocess.Popen(
        ["python3", "run.py", str(config_path)],
        cwd=AI_TOOLKIT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    if process.stdout is None:
        raise RuntimeError("failed to capture ai-toolkit output")

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + max(60.0, timeout_seconds)
    timed_out = False
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log:
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _graceful_stop(process)
                break
            for key, _ in selector.select(timeout=min(1.0, remaining)):
                line = key.fileobj.readline()
                if line:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
        # Drain buffered output after normal or requested termination.
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
    selector.close()
    return_code = process.wait()
    return return_code, timed_out


def _fallback_caption(category: str) -> str:
    return {
        "person": "portrait of the subject",
        "product": "product photograph",
        "logo": "logo design",
        "social": "social media graphic",
        "design": "interface design",
        "style": "visual style reference",
    }.get(category, "visual reference")


def _ensure_nonempty_captions(dataset_dir: Path, category: str, trigger_word: str | None) -> None:
    fallback = _fallback_caption(category)
    for image in dataset_prep.list_images(dataset_dir):
        cap = image.with_suffix(".txt")
        try:
            current = cap.read_text(encoding="utf-8").strip() if cap.exists() else ""
            if not current:
                prefix = f"{trigger_word.strip()}, " if trigger_word and trigger_word.strip() else ""
                cap.write_text(prefix + fallback, encoding="utf-8")
        except OSError:
            continue


def main() -> None:
    started = time.monotonic()
    print("--- STARTING G.O.D IMAGE CHALLENGER V2 ---", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted(config_builder.TEMPLATE_BY_MODEL_TYPE),
    )
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--hours-to-complete", required=True, type=float)
    parser.add_argument("--trigger-word")
    args = parser.parse_args()

    dataset_dir = prepare_dataset(args.task_id, args.dataset_zip)
    before = dataset_prep.audit(dataset_dir)
    print(
        f"[dataset] images={before.images} captions={before.captions} "
        f"missing={before.missing_captions} corrupt={before.corrupt_images}",
        flush=True,
    )
    if before.corrupt_images:
        raise RuntimeError(f"dataset contains {before.corrupt_images} corrupt images")

    dedup = dataset_prep.conservative_dedup(dataset_dir)
    print(
        f"[dedup] exact duplicates removed={dedup.n_removed}; "
        f"images {dedup.n_before}->{dedup.n_after}",
        flush=True,
    )
    if dedup.note:
        print(f"[dedup] {dedup.note}", flush=True)

    raw_captions = task_shape.read_captions(dataset_dir)
    image_paths = task_shape.list_images(dataset_dir)
    initial_shape = task_shape.classify(raw_captions, image_paths)
    _ensure_nonempty_captions(dataset_dir, initial_shape.category, args.trigger_word)
    cap_stats = caption.enrich_directory(
        dataset_dir,
        category=initial_shape.category,
        is_subject=initial_shape.is_subject,
        trigger_word=args.trigger_word,
        max_words=220 if args.model_type == "flux" else 75,
    )
    print(
        f"[caption] category={initial_shape.category} subject={initial_shape.is_subject} "
        f"examined={cap_stats.examined} rewritten={cap_stats.rewritten} "
        f"created={cap_stats.missing_created} failures={cap_stats.failures}",
        flush=True,
    )
    for note in initial_shape.notes:
        print(f"[shape] {note}", flush=True)

    checkpoints_root = CHECKPOINTS_ROOT / args.task_id
    config, recipe, final_shape = config_builder.build_config(
        task_id=args.task_id,
        model=args.model,
        model_type=args.model_type,
        expected_repo_name=args.expected_repo_name,
        dataset_path=dataset_dir,
        checkpoints_root=checkpoints_root,
        trigger_word=args.trigger_word,
        hours_to_complete=args.hours_to_complete,
    )
    print(
        f"[recipe] model={args.model_type} category={final_shape.category} "
        f"subject={final_shape.is_subject} rank={recipe.rank} lr={recipe.learning_rate:g} "
        f"steps={recipe.steps} save_every={recipe.save_every} "
        f"caption_dropout={recipe.caption_dropout_rate}",
        flush=True,
    )
    for note in recipe.notes:
        print(f"[recipe] {note}", flush=True)

    config_path = CONFIG_OUTPUT_ROOT / f"{args.task_id}.yaml"
    config_builder.save_config(config, config_path)
    elapsed = time.monotonic() - started
    total = max(300.0, args.hours_to_complete * 3600.0)
    timeout = max(60.0, total - _reserve_seconds(args.hours_to_complete) - elapsed)
    log_path = CONFIG_OUTPUT_ROOT / f"{args.task_id}.training.log"
    return_code, timed_out = run_training(config_path, timeout, log_path)

    output_dir = checkpoints_root / args.expected_repo_name
    try:
        result = checkpoint_select.validate_output(output_dir)
    except Exception:
        # A normal training error must remain visible even if no checkpoint was
        # produced.  This also prevents accepting a timeout with empty output.
        raise RuntimeError(
            f"training exited with code {return_code}; no valid checkpoint at {output_dir}"
        )

    if return_code != 0 and not timed_out:
        raise RuntimeError(f"ai-toolkit failed with exit code {return_code}")
    print(
        f"[output] valid weights={len(result.weight_files)} newest_step={result.chosen_step} "
        f"canonical={result.canonical_path} path={output_dir}",
        flush=True,
    )
    print("--- IMAGE TRAINING COMPLETE ---", flush=True)


if __name__ == "__main__":
    main()
