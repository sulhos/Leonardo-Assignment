"""Stage 1 smoke test: every module in `src/` must import cleanly and the
YAML configs must load. This is the only test in the Stage-1 scaffold that
actually runs (not skipped) -- it verifies the project structure and import
paths are correct before any real logic is implemented.
"""

from __future__ import annotations

import importlib
import pkgutil

import src


def _iter_module_names(package):
    prefix = package.__name__ + "."
    for _, name, _ in pkgutil.walk_packages(package.__path__, prefix):
        yield name


def test_all_src_modules_import() -> None:
    failures = {}
    for module_name in _iter_module_names(src):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - we want to report every failure
            failures[module_name] = repr(exc)
    assert not failures, f"Modules failed to import: {failures}"


def test_all_site_configs_load() -> None:
    from src.config import load_site_config

    for site in ("jinan", "vasteras"):
        config = load_site_config(site)
        assert "site" in config
        assert "battery" in config


def test_non_site_configs_load() -> None:
    from src.config import load_yaml_config

    for name in ("scenario_generation", "ml_training"):
        config = load_yaml_config(name)
        assert config, f"{name}.yaml loaded empty"
