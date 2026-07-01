#!/usr/bin/env python3
"""
run_entity_linker.py

Unified entry point for entity linking using the entity-linkings library.

Reads a YAML configuration file, optionally applies dot-notation key=value overrides supplied on the command line, and dispatches to the appropriate entity linker run mode (training or inference).

Dispatch matrix
---------------
    mode: train -> ELEntityLinker.run_training()
    mode: inference -> ELEntityLinker.run_inference()

Usage
-----
    # Training
    python run_entity_linker.py --config scripts/configs/entityLinker/dualencoder.yaml

    # Inference
    python run_entity_linker.py --config scripts/configs/entityLinker/dualencoder.yaml --mode inference

    # With CLI overrides (dot-notation, override any YAML key)
    python run_entity_linker.py --config scripts/configs/entityLinker/dualencoder.yaml \\
                   --override linker.num_train_epochs=3 \\
                   --override linker.train_batch_size=16

Config schema
-------------
See the YAML files under scripts/configs/entityLinker/ for annotated examples of every field.
The top-level 'mode' key controls dispatch; all other parameters live under the 'linker' sub-key.
"""

import argparse
import os
import sys

# PyYAML is the only extra dependency introduced by this file
try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required to read config files.  "
        "Install it with: pip install pyyaml"
    ) from exc

from src.entityLinker.ELEntityLinker import run_training, run_inference


#################
# Config loader #
#################

def load_config(config_path: str) -> dict:
    """
    Loads and returns the YAML configuration file as a plain Python dict.

    :param config_path: Path to a .yaml / .yml config file.
    :return: Parsed configuration dict.
    :raises FileNotFoundError: If the file does not exist.
    :raises yaml.YAMLError: If the file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """
    Applies a list of dot-notation key=value override strings to 'config', mutating it in place and returning it.

    Dot notation descends into nested dicts:
        "linker.num_train_epochs=10" -> config["linker"]["num_train_epochs"] = 10
        "linker.retriever_id=bm25" -> config["linker"]["retriever_id"] = "bm25"
        "mode=inference" -> config["mode"] = "inference"

    Values are parsed with yaml.safe_load so that Python types (bool, int, float, list, null) are handled correctly without any manual casting.

    :param config: The config dict to mutate.
    :param overrides: A list of "dotted.key=value" strings.
    :return: The mutated config dict (same object).
    :raises ValueError: If an override string is malformed or a key is missing.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be in 'key=value' format: {override}")

        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")

        # Navigate to the parent dict and set the value
        current = config
        for key in keys[:-1]:
            if key not in current:
                raise ValueError(f"Missing config key: {key} (in path {key_path})")
            if not isinstance(current[key], dict):
                raise ValueError(f"Cannot descend into non-dict: {key} (in path {key_path})")
            current = current[key]

        # Parse the value as YAML so that Python types are handled correctly
        final_key = keys[-1]
        if final_key not in current:
            raise ValueError(f"Missing config key: {final_key} (in path {key_path})")
        current[final_key] = yaml.safe_load(value_str)

    return config


######################
# Namespace builders #
######################
# These functions translate the YAML config dict into the argparse.Namespace objects that ELEntityLinker expects.

def _get(section: dict, key: str, default=None):
    """Convenience getter that returns a default when a key is absent."""
    return section.get(key, default)


def build_linker_namespace(config: dict) -> argparse.Namespace:
    """
    Translates the 'linker' section of the config dict into an argparse.Namespace compatible with ELEntityLinker.run_training() and run_inference().

    All keys mirror the CLI flags documented in ELEntityLinker.parse_args().
    Missing optional keys silently fall back to the same defaults used there.

    :param config: Full config dict (top-level, not just the 'linker' sub-section).
    :return: A populated argparse.Namespace.
    :raises KeyError: If a required key is absent for the requested mode.
    :raises ValueError: If the mode is not 'train' or 'inference'.
    """
    linker = config.get("linker", {})
    mode = config.get("mode", "train").lower()

    if mode not in ("train", "inference"):
        raise ValueError(f"Mode must be 'train' or 'inference', got: {mode}")

    ns = argparse.Namespace(
        # Mode
        inference_only=(mode == "inference"),

        # Entity dictionary (required)
        entity_dict_path=_get(linker, "entity_dict_path"),

        # Retriever selection
        retriever_id=_get(linker, "retriever_id", "dualencoder"),
        reranker_id=_get(linker, "reranker_id"),

        # Model paths
        retriever_model_name_or_path=_get(linker, "retriever_model_name_or_path"),
        reranker_model_name_or_path=_get(linker, "reranker_model_name_or_path"),
        retriever_index_dir=_get(linker, "retriever_index_dir"),

        # Config files
        retriever_config=_get(linker, "retriever_config"),
        reranker_config=_get(linker, "reranker_config"),

        # Data paths (training)
        train_data_paths=_get(linker, "train_data_paths"),
        dev_data_path=_get(linker, "dev_data_path"),
        output_dir=_get(linker, "output_dir"),

        # Data paths (inference)
        inference_data_path=_get(linker, "inference_data_path"),
        inference_output_path=_get(linker, "inference_output_path"),
        eval_data_path=_get(linker, "eval_data_path"),

        # Training hyper-parameters
        num_train_epochs=_get(linker, "num_train_epochs", 5),
        train_batch_size=_get(linker, "train_batch_size", 8),
        eval_batch_size=_get(linker, "eval_batch_size", 8),
        gradient_accumulation_steps=_get(linker, "gradient_accumulation_steps", 1),
        num_hard_negatives=_get(linker, "num_hard_negatives", 0),
        num_candidates=_get(linker, "num_candidates", 30),
        candidate_retrieval_batch_size=_get(linker, "candidate_retrieval_batch_size"),
        train_retriever=_get(linker, "train_retriever"),
        train_reranker=_get(linker, "train_reranker"),

        # Misc
        cache_dir=_get(linker, "cache_dir"),
        seed=_get(linker, "seed", 42),
        remove_nil=_get(linker, "remove_nil", False),
    )

    # -- Validation (mirrors ELEntityLinker.parse_args()) --
    if ns.inference_only:
        _require(ns, "inference_data_path", "linker.inference_data_path", "inference")
        _require(ns, "inference_output_path", "linker.inference_output_path", "inference")
    else:
        _require(ns, "train_data_paths", "linker.train_data_paths", "training")
        _require(ns, "output_dir", "linker.output_dir", "training")

    _require(ns, "entity_dict_path", "linker.entity_dict_path", mode)

    return ns


def _require(ns: argparse.Namespace, attr: str, yaml_key: str, mode: str) -> None:
    """
    Asserts that a required Namespace attribute is not None, raising a descriptive error if it is missing.

    :param ns:       The Namespace to inspect.
    :param attr:     Attribute name to check.
    :param yaml_key: Dot-notation YAML key shown in the error message.
    :param mode:     Current mode label shown in the error message.
    :raises SystemExit: With a human-readable error message.
    """
    if getattr(ns, attr, None) is None:
        print(
            f"[error] Required config key missing for {mode}: {yaml_key}",
            file=sys.stderr,
        )
        sys.exit(1)


############
# Dispatch #
############

def dispatch(config: dict) -> None:
    """
    Reads the top-level 'mode' key from the config and calls the appropriate entity linker entry point.

    :param config: Fully merged and validated config dict.
    :raises ValueError: If 'mode' is not 'train' or 'inference'.
    """
    mode = config.get("mode", "train").lower()

    if mode == "train":
        ns = build_linker_namespace(config)
        run_training(ns)

    elif mode == "inference":
        ns = build_linker_namespace(config)
        run_inference(ns)

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'inference'.")


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for run_entity_linker.py.

    Only two arguments exist here: the config file path and an optional list of dot-notation overrides.
    All experiment parameters live in the YAML.

    :return: An argparse.Namespace with 'config' and 'override' attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Unified entity linking runner.  "
            "Pass a YAML config file; optionally override individual keys inline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    python run_entity_linker.py --config scripts/configs/entityLinker/dualencoder.yaml
    python run_entity_linker.py --config scripts/configs/entityLinker/dualencoder.yaml --mode inference
    python run_entity_linker.py --config scripts/configs/entityLinker/dualencoder.yaml --override linker.num_train_epochs=3
    python run_entity_linker.py --config scripts/configs/entityLinker/dualencoder.yaml --override linker.reranker_id=crossencoder linker.num_candidates=50
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the YAML experiment configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["train", "inference"],
        help=(
            "Execution mode (overrides config file if specified).  "
            "Default: read from config file (defaults to 'train' if not specified)."
        ),
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Zero or more dot-notation overrides applied on top of the config file.  "
            "Example: --override linker.num_train_epochs=5 linker.retriever_id=bm25"
        ),
    )

    return parser.parse_args()


def print_config(config: dict) -> None:
    """
    Prints the effective configuration in a human-readable YAML format.

    :param config: The config dict to print.
    """
    print("[main] Effective configuration:")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False).rstrip())
    print()


###############
# Entry point #
###############

if __name__ == "__main__":
    args = parse_args()

    # 1. Load base config from YAML
    config = load_config(args.config)

    # 2. Override mode if specified on CLI
    if args.mode is not None:
        config["mode"] = args.mode

    # 3. Apply any CLI overrides on top
    if args.override:
        apply_overrides(config, args.override)

    # 4. Echo the effective config for reproducibility
    print_config(config)

    # 5. Dispatch to the correct mode
    dispatch(config)
