#!/usr/bin/env python3
"""
run_baseline_entity_linker.py

YAML-driven runner for src/entityLinker/BaselineEntityLinker.py.

This mirrors the configuration style used by run_entity_linker.py: a YAML file
contains all baseline entity-linking settings, and optional dot-notation CLI
overrides can adjust individual values without editing the file.

Usage:
    python run_baseline_entity_linker.py --config scripts/configs/entityLinker/baseline_build_inference.yaml
    python run_baseline_entity_linker.py --config scripts/configs/entityLinker/baseline_reuse_kb.yaml \
        --override linker.similarity_threshold=0.35
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required to read config files. Install it with: pip install pyyaml"
    ) from exc

def _load_baseline_entrypoint():
    """
    Loads BaselineEntityLinker.py directly.

    Importing through src.entityLinker executes that package's __init__.py, which
    imports the separate neural entity-linkings dependency. The baseline runner
    should not need that dependency just to parse config or run exact matching.
    """
    module_path = Path(__file__).resolve().parent / "src" / "entityLinker" / "BaselineEntityLinker.py"
    spec = importlib.util.spec_from_file_location("gbie_baseline_entity_linker", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load BaselineEntityLinker module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_build_and_inference


def load_config(config_path: str) -> dict[str, Any]:
    """
    Loads a YAML configuration file.

    :param config_path: Path to a .yaml / .yml file.
    :return: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML object: {config_path}")

    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """
    Applies dot-notation key=value overrides to a config dictionary in place.

    Values are parsed with yaml.safe_load, so null/bool/int/float/list values work
    naturally from the command line.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be in 'key=value' format: {override}")

        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")

        current: Any = config
        for key in keys[:-1]:
            if key not in current:
                raise ValueError(f"Missing config key: {key} (in path {key_path})")
            if not isinstance(current[key], dict):
                raise ValueError(f"Cannot descend into non-dict key: {key_path}")
            current = current[key]

        final_key = keys[-1]
        if final_key not in current:
            raise ValueError(f"Missing config key: {final_key} (in path {key_path})")
        current[final_key] = yaml.safe_load(value_str)

    return config


def _get(section: dict[str, Any], key: str, default: Any = None) -> Any:
    return section.get(key, default)


def build_baseline_namespace(config: dict[str, Any]) -> argparse.Namespace:
    """
    Converts the YAML config into the argparse.Namespace expected by
    BaselineEntityLinker.run_build_and_inference().
    """
    mode = str(config.get("mode", "build_and_inference")).lower()
    if mode not in {"build_and_inference", "inference"}:
        raise ValueError(
            f"Unsupported mode: {mode}. Use 'build_and_inference' or 'inference'."
        )

    linker = config.get("linker", {})
    if not isinstance(linker, dict):
        raise ValueError("Config key 'linker' must be a mapping.")

    use_label_for_exact_match = _get(linker, "use_label_for_exact_match", None)
    no_label_matching = _get(linker, "no_label_matching", None)
    if use_label_for_exact_match is None and no_label_matching is None:
        no_label_matching = False
    elif use_label_for_exact_match is not None and no_label_matching is None:
        no_label_matching = not bool(use_label_for_exact_match)
    elif use_label_for_exact_match is not None and no_label_matching is not None:
        expected = not bool(use_label_for_exact_match)
        if bool(no_label_matching) != expected:
            raise ValueError(
                "Conflicting exact-match settings: linker.use_label_for_exact_match "
                "and linker.no_label_matching disagree."
            )

    ns = argparse.Namespace(
        # Knowledge-base source
        knowledge_base_dir=_get(linker, "knowledge_base_dir"),
        training_data_paths=_get(linker, "training_data_paths"),
        uri_definitions_path=_get(linker, "uri_definitions_path"),
        id_to_uri_path=_get(linker, "id_to_uri_path"),

        # Inference data paths
        inference_data_path=_get(linker, "inference_data_path"),
        inference_output_path=_get(linker, "inference_output_path"),
        eval_data_path=_get(linker, "eval_data_path"),

        # Embedding and linking settings
        embeddings_model_name=_get(
            linker,
            "embeddings_model_name",
            "neuml/pubmedbert-base-embeddings",
        ),
        embeddings_index_path=_get(linker, "embeddings_index_path", "embeddings_index"),
        similarity_top_k=_get(linker, "similarity_top_k", 10),
        similarity_threshold=_get(linker, "similarity_threshold", 0.0),
        no_label_matching=bool(no_label_matching),
    )

    _validate_namespace(ns)
    return ns


def _validate_namespace(ns: argparse.Namespace) -> None:
    _require(ns, "inference_data_path", "linker.inference_data_path")
    _require(ns, "inference_output_path", "linker.inference_output_path")

    has_existing_kb = bool(ns.knowledge_base_dir) and os.path.isdir(ns.knowledge_base_dir)
    if has_existing_kb:
        return

    _require(ns, "training_data_paths", "linker.training_data_paths")
    _require(ns, "uri_definitions_path", "linker.uri_definitions_path")
    _require(ns, "id_to_uri_path", "linker.id_to_uri_path")


def _require(ns: argparse.Namespace, attr: str, yaml_key: str) -> None:
    if getattr(ns, attr, None) is None:
        print(f"[error] Required config key missing: {yaml_key}", file=sys.stderr)
        sys.exit(1)


def dispatch(config: dict[str, Any]) -> None:
    ns = build_baseline_namespace(config)
    run_build_and_inference = _load_baseline_entrypoint()
    run_build_and_inference(ns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YAML runner for the baseline GutBrainIE entity linker."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help=(
            "Optional dot-notation overrides, e.g. "
            "--override linker.similarity_threshold=0.35"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    dispatch(config)


if __name__ == "__main__":
    main()
