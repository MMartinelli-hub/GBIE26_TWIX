#!/usr/bin/env python3
"""
run_entity_recognizer.py

Unified entry point for HF-based, LLM-based, and GLiNER-based biomedical named entity recognition (NER).

Reads a YAML configuration file, optionally applies dot-notation key=value overrides supplied on the command line, and dispatches to the appropriate recognizer module and run mode.

Dispatch matrix
---------------
    recognizer: hf + mode: train -> HFEntityRecognizer.run_training()
    recognizer: hf + mode: inference -> HFEntityRecognizer.run_inference()
    recognizer: llm -> LLMEntityRecognizer.run_inference() (LLMs are inference-only for now)
    recognizer: gliner + mode: train -> GLiNEREntityRecognizer.run_training()
    recognizer: gliner + mode: inference -> GLiNEREntityRecognizer.run_inference()

Usage
-----
    # HF training
    python run_entity_recognizer.py --config scripts/configs/entityRecognizer/hf_train.yaml

    # GLiNER inference
    python run_entity_recognizer.py --config scripts/configs/entityRecognizer/gliner_inference.yaml

    # LLM inference
    python run_entity_recognizer.py --config scripts/configs/entityRecognizer/llm_inference.yaml

    # With CLI overrides (dot-notation, override any YAML key)
    python run_entity_recognizer.py --config scripts/configs/entityRecognizer/hf_train.yaml \\
                   --override hf.num_epochs=3 \\
                   --override hf.concatenate_title_abstract=true

Config schema
-------------
See the YAML files under scripts/configs/entityRecognizer/ for annotated examples of every field.
The top-level 'recognizer' and (for hf/gliner) 'mode' keys control dispatch; all other parameters live under the 'hf', 'llm', or 'gliner' sub-key.
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
        "hf.temperature=0.5" -> config["hf"]["temperature"] = 0.5
        "gliner.num_steps=5000" -> config["gliner"]["num_steps"] = 5000
        "recognizer=llm" -> config["recognizer"] = "llm"

    Values are parsed with yaml.safe_load so that Python types (bool, int, float, list, null) are handled correctly without any manual casting.

    :param config: The config dict to mutate.
    :param overrides: A list of "dotted.key=value" strings.
    :return: The mutated config dict (same object).
    :raises ValueError: If an override string is malformed or a key is missing.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override '{override}': expected 'key=value' format."
            )

        key_path, _, raw_value = override.partition("=")
        keys = key_path.strip().split(".")

        # Parse value as YAML so booleans / numbers / lists are handled correctly
        parsed_value = yaml.safe_load(raw_value)

        # Walk down to the parent dict, creating intermediate dicts as needed
        node = config
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]

        node[keys[-1]] = parsed_value

    return config


######################
# Namespace builders #
######################
# These functions translate the YAML config dict into the argparse.Namespace objects that recognizer modules expect.

def _get(section: dict, key: str, default=None):
    """Convenience getter that returns a default when a key is absent."""
    return section.get(key, default)


def build_hf_namespace(config: dict) -> argparse.Namespace:
    """
    Translates the 'hf' section of the config dict into an argparse.Namespace compatible with HFEntityRecognizer.run_training() and run_inference().

    All keys mirror the CLI flags documented in HFEntityRecognizer.parse_args().
    Missing optional keys silently fall back to the same defaults used there.

    :param config: Full config dict (top-level, not just the 'hf' sub-section).
    :return: A populated argparse.Namespace.
    :raises KeyError: If a required key is absent for the requested mode.
    :raises ValueError: If the mode is not 'train' or 'inference'.
    """
    hf = config.get("hf", {})
    mode = config.get("mode", "train").lower()

    if mode not in ("train", "inference"):
        raise ValueError(
            f"Invalid mode '{mode}' for recognizer 'hf'.  "
            "Choose 'train' or 'inference'."
        )

    ns = argparse.Namespace(
        # -- Mode --
        inference_only=(mode == "inference"),

        # -- Data paths --
        train_data_paths =_get(hf, "train_data_paths"),
        dev_data_path    =_get(hf, "dev_data_path"),
        inference_data_path  =_get(hf, "inference_data_path"),
        inference_output_path=_get(hf, "inference_output_path"),

        # -- Model paths --
        model_name =_get(hf, "model_name", "michiyasunaga/BioLinkBERT-base"),
        output_dir =_get(hf, "output_dir"),
        model_path =_get(hf, "model_path"),

        # -- Training hyper-parameters --
        batch_size   =_get(hf, "batch_size",    8),
        num_epochs   =_get(hf, "num_epochs",    5),
        learning_rate=_get(hf, "learning_rate", 2e-5),
        max_length   =_get(hf, "max_length",    512),
        seed         =_get(hf, "seed",          42),
        num_workers  =_get(hf, "num_workers",   0),
        use_amp      =_get(hf, "use_amp",       False),

        # -- Text handling --
        separate_title_abstract  =_get(hf, "separate_title_abstract", False),
        concatenate_title_abstract=not _get(hf, "separate_title_abstract", False),
    )

    # -- Validation (mirrors HFEntityRecognizer.parse_args()) --
    if ns.inference_only:
        _require(ns, "model_path", "hf.model_path", mode)
        _require(ns, "inference_data_path", "hf.inference_data_path", mode)
        _require(ns, "inference_output_path", "hf.inference_output_path", mode)
    else:
        _require(ns, "train_data_paths", "hf.train_data_paths", mode)
        _require(ns, "output_dir", "hf.output_dir", mode)

    return ns


def build_llm_namespace(config: dict) -> argparse.Namespace:
    """
    Translates the 'llm' section of the config dict into an argparse.Namespace compatible with LLMEntityRecognizer.run_inference().

    All keys mirror the CLI flags documented in LLMEntityRecognizer.parse_args().

    :param config: Full config dict (top-level, not just the 'llm' sub-section).
    :return: A populated argparse.Namespace.
    :raises KeyError: If a required key is absent.
    """
    llm = config.get("llm", {})

    ns = argparse.Namespace(
        # -- Provider identity --
        provider =_get(llm, "provider"),
        model    =_get(llm, "model"),

        # -- Provider credentials / endpoints --
        api_key        =_get(llm, "api_key"),
        base_url       =_get(llm, "base_url"),
        azure_endpoint =_get(llm, "azure_endpoint"),
        azure_api_version=_get(llm, "azure_api_version", "2024-02-01"),

        # -- Data paths --
        inference_data_path  =_get(llm, "inference_data_path"),
        inference_output_path=_get(llm, "inference_output_path"),
        eval_data_path       =_get(llm, "eval_data_path"),
        prompts_path         =_get(llm, "prompts_path", "src/entityRecognizer/prompts.json"),
        checkpoint_path      =_get(llm, "checkpoint_path"),

        # -- Prompt selection --
        system_prompt_key=_get(llm, "system_prompt_key", "base"),
        user_prompt_key  =_get(llm, "user_prompt_key",   "base"),

        # -- Generation hyper-parameters --
        temperature   =_get(llm, "temperature",    1.0),
        max_new_tokens=_get(llm, "max_new_tokens", 512),

        # -- HuggingFace local options --
        device=_get(llm, "device"),
        
        # -- Inference mode --
        raw=_get(llm, "raw", False),
    )

    # -- Validation --
    _require(ns, "provider", "llm.provider", "inference")
    _require(ns, "model", "llm.model", "inference")
    _require(ns, "inference_data_path", "llm.inference_data_path", "inference")
    _require(ns, "inference_output_path", "llm.inference_output_path", "inference")

    return ns


def build_gliner_namespace(config: dict) -> argparse.Namespace:
    """
    Translates the 'gliner' section of the config dict into an argparse.Namespace compatible with GLiNEREntityRecognizer.run_training() and run_inference().

    All keys mirror the CLI flags documented in GLiNEREntityRecognizer.parse_args().
    Missing optional keys silently fall back to the same defaults used there.

    :param config: Full config dict (top-level, not just the 'gliner' sub-section).
    :return: A populated argparse.Namespace.
    :raises KeyError: If a required key is absent for the requested mode.
    :raises ValueError: If the mode is not 'train' or 'inference'.
    """
    gliner = config.get("gliner", {})
    mode = config.get("mode", "train").lower()

    if mode not in ("train", "inference"):
        raise ValueError(
            f"Invalid mode '{mode}' for recognizer 'gliner'.  "
            "Choose 'train' or 'inference'."
        )

    ns = argparse.Namespace(
        # -- Mode --
        inference_only=(mode == "inference"),

        # -- Data paths --
        train_data_paths =_get(gliner, "train_data_paths"),
        dev_data_path    =_get(gliner, "dev_data_path"),
        inference_data_path  =_get(gliner, "inference_data_path"),
        inference_output_path=_get(gliner, "inference_output_path"),
        eval_data_path   =_get(gliner, "eval_data_path"),

        # -- Model paths --
        model_name =_get(gliner, "model_name", "numind/NuNerZero"),
        output_dir =_get(gliner, "output_dir"),
        model_path =_get(gliner, "model_path"),
        save_directory =_get(gliner, "save_directory", "logs"),

        # -- Inference hyper-parameters --
        threshold=_get(gliner, "threshold"),

        # -- Training hyper-parameters --
        num_steps   =_get(gliner, "num_steps",     3000),
        eval_every  =_get(gliner, "eval_every",    200),
        batch_size  =_get(gliner, "batch_size",    8),
        max_len     =_get(gliner, "max_len",       384),
        warmup_ratio=_get(gliner, "warmup_ratio",  0.1),
        lr_encoder  =_get(gliner, "lr_encoder",    1e-5),
        lr_others   =_get(gliner, "lr_others",     5e-5),
        freeze_token_rep=_get(gliner, "freeze_token_rep", False),
        max_types   =_get(gliner, "max_types",     15),
        no_shuffle_types=_get(gliner, "no_shuffle_types", False),
        no_random_drop=_get(gliner, "no_random_drop", False),
        max_neg_type_ratio=_get(gliner, "max_neg_type_ratio", 1.0),

        # -- Text handling --
        separate_title_abstract  =_get(gliner, "separate_title_abstract", False),
        concatenate_title_abstract=not _get(gliner, "separate_title_abstract", False),

        # -- Other --
        seed=_get(gliner, "seed", 42),
    )

    # -- Validation (mirrors GLiNEREntityRecognizer.parse_args()) --
    if ns.inference_only:
        _require(ns, "model_path", "gliner.model_path", mode)
        _require(ns, "inference_data_path", "gliner.inference_data_path", mode)
        _require(ns, "inference_output_path", "gliner.inference_output_path", mode)
    else:
        _require(ns, "train_data_paths", "gliner.train_data_paths", mode)
        _require(ns, "output_dir", "gliner.output_dir", mode)

    # Default threshold for training mode
    if ns.threshold is None:
        ns.threshold = 0.5

    return ns


def _require(ns: argparse.Namespace, attr: str, yaml_key: str, mode: str) -> None:
    """
    Asserts that a required Namespace attribute is not None, raising a descriptive error if it is missing.

    :param ns: The Namespace to inspect.
    :param attr: Attribute name to check.
    :param yaml_key: Dot-notation YAML key shown in the error message.
    :param mode: Current mode label shown in the error message.
    :raises SystemExit: With a human-readable error message.
    """
    if getattr(ns, attr, None) is None:
        print(
            f"[ERROR] '{yaml_key}' is required for mode='{mode}' but was not set "
            f"in the config file.",
            file=sys.stderr,
        )
        sys.exit(1)


############
# Dispatch #
############

def dispatch(config: dict) -> None:
    """
    Reads the top-level 'recognizer' and 'mode' keys from the config and calls the appropriate entity recognizer entry point.

    :param config: Fully merged and validated config dict.
    :raises ValueError: If 'recognizer' is not 'hf', 'llm', or 'gliner'.
    """
    recognizer = config.get("recognizer", "").lower()

    if recognizer == "hf":
        # Import lazily: users who only use LLM or GLiNER recognition do not need torch / HF
        from src.entityRecognizer.HFEntityRecognizer import run_training, run_inference

        ns = build_hf_namespace(config)
        if ns.inference_only:
            print("[main] Dispatching -> HFEntityRecognizer / inference")
            run_inference(ns)
        else:
            print("[main] Dispatching -> HFEntityRecognizer / training")
            run_training(ns)

    elif recognizer == "llm":
        from src.entityRecognizer.LLMEntityRecognizer import run_inference

        ns = build_llm_namespace(config)
        print(
            f"[main] Dispatching -> LLMEntityRecognizer / inference "
            f"(provider={ns.provider}, model={ns.model})"
        )
        run_inference(ns)

    elif recognizer == "gliner":
        from src.entityRecognizer.GLiNEREntityRecognizer import run_training, run_inference

        ns = build_gliner_namespace(config)
        if ns.inference_only:
            print("[main] Dispatching -> GLiNEREntityRecognizer / inference")
            run_inference(ns)
        else:
            print("[main] Dispatching -> GLiNEREntityRecognizer / training")
            run_training(ns)

    else:
        raise ValueError(
            f"Unknown recognizer '{recognizer}' in config.  "
            "Valid choices: 'hf', 'llm', 'gliner'."
        )


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for run_entity_recognizer.py.

    Only two arguments exist here: the config file path and an optional list of dot-notation overrides.  
    All experiment parameters live in the YAML.

    :return: An argparse.Namespace with 'config' and 'override' attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Unified entity recognition runner (HF, LLM, or GLiNER).  "
            "Pass a YAML config file; optionally override individual keys inline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    python run_entity_recognizer.py --config scripts/configs/entityRecognizer/hf_train.yaml
    python run_entity_recognizer.py --config scripts/configs/entityRecognizer/hf_train.yaml --override hf.num_epochs=5
    python run_entity_recognizer.py --config scripts/configs/entityRecognizer/gliner_inference.yaml
    python run_entity_recognizer.py --config scripts/configs/entityRecognizer/llm_inference.yaml --override llm.temperature=0.0
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
        "--override",
        type=str,
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Zero or more dot-notation overrides applied on top of the config file.  "
            "Example: --override hf.learning_rate=1e-5 gliner.num_steps=5000"
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

    # 2. Apply any CLI overrides on top
    if args.override:
        config = apply_overrides(config, args.override)

    # 3. Echo the effective config for reproducibility
    print_config(config)

    # 4. Dispatch to the entity recognizer
    dispatch(config)
