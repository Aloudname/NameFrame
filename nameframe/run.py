"""Programmatic entry point for NameFrame training pipelines.

Used by both the CLI (`nameframe run`) and interactive environments
like Jupyter notebooks::

    from nameframe.run import run
    result = run(config_path="config/config.yaml")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from munch import Munch

from nameframe.config import load_config, merge_args
from nameframe.pipeline import Pipeline, PipelineResult


def run(
    config_path: Union[str, Munch, None] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    """Execute a full training pipeline.

    Args:
        config_path: Path to a YAML config file, or a pre-loaded Munch.
            Defaults to ``"config/config.yaml"``.
        config_overrides: Optional dict of dotted-key overrides merged
            on top of the loaded config.

    Returns:
        A :class:`PipelineResult` with training outcomes.

    Example:
        >>> result = run("config/config.yaml", {"train.epochs": 200})
    """
    if isinstance(config_path, Munch):
        config = config_path
    else:
        path: str = config_path if config_path else "config/config.yaml"
        config = load_config(path)

    if config_overrides:
        config = merge_args(config, config_overrides)

    pipeline: Pipeline = Pipeline(config)
    return pipeline.run()
