"""
No-GPU tests for the adaptive image-tournament training recipe
(ops/docker/scripts/image_recipe/). These modules live outside the normal
package tree because they're COPYed standalone into the training container
image (see ops/docker/standalone-image-toolkit-trainer.dockerfile), so we add
their directory to sys.path here rather than importing them as part of the
`core`/`trainer` packages.

Run with: uv run --extra dev pytest tests/trainer/image_recipe -v
"""

import os
import sys
import tempfile
import zipfile

import pytest


_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "docker", "scripts"))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from image_recipe import caption  # noqa: E402
from image_recipe import checkpoint_select  # noqa: E402
from image_recipe import dataset_prep  # noqa: E402
from image_recipe import recipe_table  # noqa: E402
from image_recipe import task_shape  # noqa: E402


CAPTION_CASES = {
    "person": "A smiling woman wearing a red jacket, portrait, studio lighting",
    "product": "Product shot of a sneaker isolated on white background, e-commerce catalog",
    "logo": "A minimalist wordmark logo, brandmark, monogram in navy blue",
    "social": "Instagram story template with bold headline and call to action",
    "design": "App screen mockup, dashboard UI with buttons and icons",
    "style": "Digital art in a painterly art style, vibrant palette, brushwork",
}


@pytest.mark.parametrize("expected,caption", CAPTION_CASES.items())
def test_classify_captions(expected, caption):
    result = task_shape.classify([caption])
    assert result.category == expected
    assert result.confident_from_text
    assert result.is_subject == (expected in task_shape.SUBJECT_CATEGORIES)


def test_classify_no_signal_falls_back_conservatively():
    result = task_shape.classify(["", "   "])
    assert result.category == "style"
    assert not result.confident_from_text
    assert result.notes


@pytest.mark.parametrize("model_type", ["flux", "z-image", "qwen-image", "ideogram4", "krea2"])
def test_recipe_table_covers_all_five_model_types(model_type):
    """The champion repo we studied only handles sdxl/flux/qwen-image/z-image
    -- ideogram4 and krea2 are unsupported there. This is the gap our recipe
    table (and config_builder) closes; every model type must produce a
    usable recipe."""
    recipe = recipe_table.build_recipe(
        model_type=model_type, is_subject=True, category="person",
        category_confident=True, n_images=12, template_supports_caption_dropout=True,
    )
    assert recipe.rank > 0
    assert recipe.step_ceiling > 0


def test_recipe_table_subject_gets_tighter_ceiling_than_style():
    subject = recipe_table.build_recipe(
        "flux", is_subject=True, category="person", category_confident=True,
        n_images=20, template_supports_caption_dropout=True,
    )
    style = recipe_table.build_recipe(
        "flux", is_subject=False, category="style", category_confident=True,
        n_images=20, template_supports_caption_dropout=True,
    )
    assert subject.step_ceiling < style.step_ceiling


def test_recipe_table_caption_dropout_always_set_but_notes_unsupported_templates():
    """v2 no longer skips caption dropout for templates that lack a native
    key -- it injects the value into the shared ai-toolkit dataset schema
    instead and leaves a note explaining why."""
    with_support = recipe_table.build_recipe(
        "flux", is_subject=False, category="style", category_confident=True,
        n_images=20, template_supports_caption_dropout=True,
    )
    without_support = recipe_table.build_recipe(
        "z-image", is_subject=False, category="style", category_confident=True,
        n_images=20, template_supports_caption_dropout=False,
    )
    assert with_support.caption_dropout_rate is not None
    assert without_support.caption_dropout_rate is not None
    assert any("injected into shared ai-toolkit dataset schema" in note for note in without_support.notes)


def test_conservative_dedup_removes_only_byte_identical_copies():
    """v2 deliberately replaced perceptual near-duplicate removal with
    exact-only dedup: tiny tournament datasets are data-starved, and pHash
    deletion risked erasing legitimate near-duplicate logo/layout variants.
    Only true byte-identical copies should be removed, keeping the copy with
    the more informative caption."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "a.jpg"), "wb") as fh:
            fh.write(b"identical-bytes")
        with open(os.path.join(tmp, "a.txt"), "w") as fh:
            fh.write("a much more descriptive caption")
        with open(os.path.join(tmp, "a_dup.jpg"), "wb") as fh:
            fh.write(b"identical-bytes")
        with open(os.path.join(tmp, "a_dup.txt"), "w") as fh:
            fh.write("short")
        # Enough filler images to clear conservative_dedup's minimum_remaining floor.
        for i in range(8):
            with open(os.path.join(tmp, f"filler{i}.jpg"), "wb") as fh:
                fh.write(f"filler-{i}".encode())
            with open(os.path.join(tmp, f"filler{i}.txt"), "w") as fh:
                fh.write(f"filler {i}")

        result = dataset_prep.conservative_dedup(tmp)
        remaining = sorted(os.listdir(tmp))
        assert result.n_removed == 1
        assert "a.jpg" in remaining and "a.txt" in remaining
        assert "a_dup.jpg" not in remaining and "a_dup.txt" not in remaining


def test_conservative_dedup_keeps_near_duplicates_that_are_not_byte_identical():
    """A one-byte difference must survive: v2 intentionally does not run
    perceptual/near-duplicate removal."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "near1.jpg"), "wb") as fh:
            fh.write(b"almost-identical-A")
        with open(os.path.join(tmp, "near1.txt"), "w") as fh:
            fh.write("near 1")
        with open(os.path.join(tmp, "near2.jpg"), "wb") as fh:
            fh.write(b"almost-identical-B")
        with open(os.path.join(tmp, "near2.txt"), "w") as fh:
            fh.write("near 2")
        for i in range(8):
            with open(os.path.join(tmp, f"filler{i}.jpg"), "wb") as fh:
                fh.write(f"filler-{i}".encode())
            with open(os.path.join(tmp, f"filler{i}.txt"), "w") as fh:
                fh.write(f"filler {i}")

        result = dataset_prep.conservative_dedup(tmp)
        assert result.n_removed == 0
        remaining = sorted(os.listdir(tmp))
        assert "near1.jpg" in remaining and "near2.jpg" in remaining


def test_conservative_dedup_disabled_below_minimum_remaining():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            with open(os.path.join(tmp, f"img{i}.jpg"), "wb") as fh:
                fh.write(b"same-bytes")
            with open(os.path.join(tmp, f"img{i}.txt"), "w") as fh:
                fh.write(f"caption {i}")

        result = dataset_prep.conservative_dedup(tmp, minimum_remaining=6)
        assert result.n_removed == 0
        assert result.note == "small dataset: duplicate removal disabled"


def test_perceptual_dedup_alias_delegates_to_conservative_dedup():
    """The prior entrypoint imports dataset_prep.perceptual_dedup by name;
    v2 keeps that name as a compatibility alias for conservative_dedup."""
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(8):
            with open(os.path.join(tmp, f"img{i}.jpg"), "wb") as fh:
                fh.write(f"img-{i}".encode())
            with open(os.path.join(tmp, f"img{i}.txt"), "w") as fh:
                fh.write(f"caption {i}")

        result = dataset_prep.perceptual_dedup(tmp, hamming_threshold=8, thumb_threshold=20.0)
        assert result.n_removed == 0


def test_caption_enrich_appends_real_image_attributes_for_visual_categories():
    """Regression test: _image_attributes previously read PIL's ImageStat.Stat
    via a `.std` attribute that does not exist (the real attribute is
    `.stddev`), so it silently failed on every image via the broad except
    clause and never appended orientation/palette/contrast text. Must
    actually enrich style/logo/social/design captions with real signal."""
    pytest.importorskip("PIL")
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        Image.new("RGB", (512, 384), (200, 30, 30)).save(os.path.join(tmp, "img0.jpg"))
        open(os.path.join(tmp, "img0.txt"), "w").close()

        stats = caption.enrich_directory(tmp, category="design", is_subject=False, trigger_word=None)
        assert stats.failures == 0
        text = open(os.path.join(tmp, "img0.txt")).read()
        assert "composition" in text
        assert "palette" in text
        assert "contrast" in text


def test_caption_enrich_stays_minimal_for_subject_categories():
    pytest.importorskip("PIL")
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        Image.new("RGB", (256, 256), (10, 10, 10)).save(os.path.join(tmp, "img0.jpg"))
        open(os.path.join(tmp, "img0.txt"), "w").write("a person standing")

        caption.enrich_directory(tmp, category="person", is_subject=True, trigger_word="sks person")
        text = open(os.path.join(tmp, "img0.txt")).read()
        assert "composition" not in text and "palette" not in text
        assert text.startswith("sks person")


def test_safe_extract_rejects_path_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "evil.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../etc/evil.txt", "pwned")

        with pytest.raises(ValueError):
            dataset_prep.safe_extract(zip_path, os.path.join(tmp, "out"))


def test_safe_extract_allows_a_normal_archive():
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "ok.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("img1.jpg", b"fakejpegbytes")
            zf.writestr("img1.txt", "a caption")

        dest = os.path.join(tmp, "out")
        dataset_prep.safe_extract(zip_path, dest)
        assert sorted(os.listdir(dest)) == ["img1.jpg", "img1.txt"]


def test_checkpoint_select_canonicalizes_the_newest_weights_non_destructively():
    """v2 removed the pixel-MSE preview scorer (stochastic diffusion previews
    are not pixel-aligned with held-out images) in favor of trusting the
    recipe's planned final point and guaranteeing the canonical
    last.safetensors filename the diffusion evaluator expects. It must never
    delete an existing checkpoint file."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "repo_000000250.safetensors"), "wb") as fh:
            fh.write(b"0" * 2000)
        with open(os.path.join(tmp, "repo_000000500.safetensors"), "wb") as fh:
            fh.write(b"1" * 2000)

        result = checkpoint_select.select_best(tmp, None)

        assert result is not None
        assert result.chosen_step == 500
        assert result.canonical_path is not None
        assert os.path.isfile(result.canonical_path)
        remaining = sorted(os.listdir(tmp))
        assert "repo_000000250.safetensors" in remaining
        assert "repo_000000500.safetensors" in remaining
        assert "last.safetensors" in remaining


def test_checkpoint_select_returns_none_when_no_weights_exist():
    with tempfile.TemporaryDirectory() as tmp:
        assert checkpoint_select.select_best(tmp, None) is None
