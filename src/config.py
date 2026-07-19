"""Project-wide configuration loading.

Loads YAML configuration files from the `config/` directory into plain
dictionaries. Kept deliberately simple (no config schema/dataclass layer yet)
so later stages can decide whether a typed schema is worth the complexity.

All paths are resolved relative to the project root, so commands work the
same whether invoked from the root or from `notebooks/`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from `config/` by name.

    Args:
        name: Config file name, with or without the `.yaml` extension
            (e.g. "jinan" or "jinan.yaml").

    Returns:
        The parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If no matching file exists under `config/`.
    """
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path = CONFIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.debug("Loading config from %s", path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def load_site_config(site: str) -> dict[str, Any]:
    """Load a site configuration by name (e.g. "jinan", "vasteras")."""
    return load_yaml_config(site)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from a config file relative to the project root."""
    return PROJECT_ROOT / relative_path
