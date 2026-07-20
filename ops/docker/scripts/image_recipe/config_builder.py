"""
ai-toolkit config construction.

Loads the existing, proven per-model-type YAML templates from
core/training_templates/ as the skeleton (they are correct — we don't
rebuild them from scratch) and overlays:

  * runtime paths (training folder, dataset folder, base model path,
    trigger word) — exactly as the base entrypoint already did;
  * the ideogram4/krea2 cached text-encoder/VAE `model_kwargs`, ported
    verbatim from ops/docker/scripts/image_toolkit_entrypoint.py so that
    behaviour is unchanged;
  * the adaptive recipe overlay from recipe_table.py (rank, step ceiling,
    caption dropout where the template already supports it);
  * a `sample` block so ai-toolkit generates preview images from the
    held-out captions at each retained checkpoint, which checkpoint_select.py
    scores to pick a checkpoint instead of trusting the step ceiling as an
    exact target.

The `sample` block's exact field names follow the standard ai-toolkit
(ostris-derived) config convention. This is the one part of this module that
cannot be verified without a GPU + the installed ai-toolkit version — see the
plan's Stage 2. If the installed ai-toolkit version rejects it, prefer
failing closed: catch that at the training-subprocess level and retry once
with sampling disabled (see image_toolkit_entrypoint.py), so a schema
mismatch degrades to "no checkpoint selection", never to a failed task.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from image_recipe import recipe_table
from image_recipe import task_shape


CACHE_ROOT = Path("/cache")
CONFIG_TEMPLATE_ROOT = Path("/workspace/core/training_templates")

TEMPLATE_BY_MODEL_TYPE = {
    "flux": "base_diffusion_flux.yaml",
    "z-image": "base_diffusion_zimage.yaml",
    "qwen-image": "base_diffusion_qwen_image.yaml",
    "ideogram4": "base_diffusion_ideogram4.yaml",
    "krea2": "base_diffusion_krea2.yaml",
}

IDEOGRAM4_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-8B-Instruct"
KREA2_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-4B-Instruct"

# Templates whose datasets[0] block already defines caption_dropout_rate.
_TEMPLATES_WITH_CAPTION_DROPOUT = {"flux", "ideogram4", "krea2"}

SAMPLE_SEED = 42


def cached_model_path(model: str) -> Path:
    return CACHE_ROOT / "models" / model.replace("/", "--")


def _apply_ideogram4_krea2_kwargs(model_type: str, model_config: dict, model_path: Path) -> None:
    """Ported verbatim (same cache paths, same required-file checks) from the
    base entrypoint's prepare_config — do not diverge from this without also
    updating the base image_toolkit_entrypoint.py's copy."""
    if model_type == "ideogram4":
        if not IDEOGRAM4_TEXT_ENCODER_CACHE.exists():
            raise FileNotFoundError(f"Expected cached Ideogram 4 text encoder at {IDEOGRAM4_TEXT_ENCODER_CACHE}")
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(IDEOGRAM4_TEXT_ENCODER_CACHE)
    elif model_type == "krea2":
        if not KREA2_TEXT_ENCODER_CACHE.exists():
            raise FileNotFoundError(f"Expected cached Krea 2 text encoder at {KREA2_TEXT_ENCODER_CACHE}")
        vae_path = model_path / "vae"
        if not vae_path.exists():
            raise FileNotFoundError(f"Expected cached Krea 2 VAE at {vae_path}")
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(KREA2_TEXT_ENCODER_CACHE)
        model_kwargs["vae_path"] = str(model_path)


def build_config(
    task_id: str,
    model: str,
    model_type: str,
    expected_repo_name: str,
    dataset_path: Path,
    checkpoints_root: Path,
    trigger_word: str | None,
    held_out_captions: list[str] | None = None,
    enable_sampling: bool = True,
) -> tuple[dict, "recipe_table.Recipe", task_shape.ShapeResult]:
    template_path = CONFIG_TEMPLATE_ROOT / TEMPLATE_BY_MODEL_TYPE[model_type]
    if not template_path.exists():
        raise FileNotFoundError(f"Missing ai-toolkit template for {model_type}: {template_path}")
    with template_path.open() as fh:
        config = yaml.safe_load(fh)

    config_body = config.setdefault("config", {})
    config_body["name"] = expected_repo_name
    process = config_body.setdefault("process", [{}])[0]
    process["training_folder"] = str(checkpoints_root)
    process["trigger_word"] = trigger_word

    datasets = process.setdefault("datasets", [{}])
    datasets[0]["folder_path"] = str(dataset_path)

    model_config = process.setdefault("model", {})
    model_path = cached_model_path(model)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected cached base model at {model_path}")
    model_config["name_or_path"] = str(model_path)
    _apply_ideogram4_krea2_kwargs(model_type, model_config, model_path)

    captions = task_shape.read_captions(str(dataset_path))
    images = task_shape.list_images(str(dataset_path))
    shape = task_shape.classify(captions, images)

    recipe = recipe_table.build_recipe(
        model_type=model_type,
        is_subject=shape.is_subject,
        category=shape.category,
        category_confident=shape.confident_from_text or shape.image_signal_used,
        n_images=len(images),
        template_supports_caption_dropout=model_type in _TEMPLATES_WITH_CAPTION_DROPOUT,
    )

    network = process.setdefault("network", {})
    network["linear"] = recipe.rank
    network["linear_alpha"] = recipe.rank

    train = process.setdefault("train", {})
    train["steps"] = recipe.step_ceiling

    if recipe.caption_dropout_rate is not None:
        datasets[0]["caption_dropout_rate"] = recipe.caption_dropout_rate

    if enable_sampling and held_out_captions:
        save = process.setdefault("save", {})
        sample_every = save.get("save_every", 250)
        train["disable_sampling"] = False
        train["skip_first_sample"] = True
        process["sample"] = {
            "sampler": "flowmatch",
            "sample_every": sample_every,
            "width": 1024,
            "height": 1024,
            "seed": SAMPLE_SEED,
            "walk_seed": False,
            "guidance_scale": 4.0,
            "sample_steps": 20,
            "prompts": held_out_captions[:4],
        }
    else:
        train.setdefault("disable_sampling", True)
        train.setdefault("skip_first_sample", True)

    return config, recipe, shape


def save_config(config: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
