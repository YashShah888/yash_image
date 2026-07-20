"""Challenger v2 image-tournament recipe.

The package intentionally keeps the public surface small.  The entrypoint uses
these modules for safe dataset handling, task classification, deterministic
caption enrichment, adaptive ai-toolkit configuration and output validation.
"""

__all__ = [
    "caption",
    "checkpoint_select",
    "config_builder",
    "dataset_prep",
    "recipe_table",
    "task_shape",
]
