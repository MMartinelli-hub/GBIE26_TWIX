#!/usr/bin/env python3
"""
run_term_classification.py

Unified entry point for HF-based and LLM-based biomedical term classification.

Reads a YAML configuration file, optionally applies dot-notation key=value overrides supplied on the command line, and dispatches to the appropriate classifier module and run mode.

Dispatch matrix
---------------
    classifier: hf + mode: train -> HFTermClassifier.run_training()
    classifier: hf + mode: inference -> HFTermClassifier.run_inference()
    classifier: llm -> LLMTermClassifier.run_inference() (LLMs are inference-only for now)

Usage
-----
    # Basic run
    python run_term_classification.py --config configs/hf_classify_train.yaml

    # With CLI overrides (dot-notation, override any YAML key)
    python run_term_classification.py --config configs/llm_classify_lmstudio.yaml \\
                   --override llm.temperature=0.0 \\
                   --override llm.model=gemma-3-12b-it

Config schema
-------------
See the YAML files under configs/ for annotated examples of every field.
The top-level 'classifier' and (for hf) 'mode' keys control dispatch;
all other parameters live under the 'hf' or 'llm' sub-key.
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
        "llm.temperature=0.5" -> config["llm"]["temperature"] = 0.5
        "hf.num_epochs=10" -> config["hf"]["num_epochs"] = 10
        "classifier=llm" -> config["classifier"] = "llm"

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
# These functions translate the YAML config dict into the argparse.Namespace objects that HFTermClassifier and LLMTermClassifier expect.

def _get(section: dict, key: str, default=None):
    """Convenience getter that returns a default when a key is absent."""
    return section.get(key, default)


def build_hf_namespace(config: dict) -> argparse.Namespace:
    """
    Translates the 'hf' section of the config dict into an argparse.Namespace compatible with HFTermClassifier.run_training() and run_inference().

    All keys mirror the CLI flags documented in HFTermClassifier.parse_args().
    Missing optional keys silently fall back to the same defaults used there.

    :param config: Full config dict (top-level, not just the 'hf' sub-section).
    :return: A populated argparse.Namespace.
    :raises KeyError:  If a required key is absent for the requested mode.
    :raises ValueError: If the mode is not 'train' or 'inference'.
    """
    hf = config.get("hf", {})
    mode = config.get("mode", "train").lower()

    if mode not in ("train", "inference"):
        raise ValueError(
            f"Invalid mode '{mode}' for classifier 'hf'.  "
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

        # -- Negative sampling (classification-specific) --
        negative_sample_multiplier=_get(hf, "negative_sample_multiplier", 1),
        dev_negative_sample_multiplier=_get(hf, "dev_negative_sample_multiplier") or _get(hf, "negative_sample_multiplier", 1),
        max_negative_span_words   =_get(hf, "max_negative_span_words", 5),
    )

    # -- Validation (mirrors HFTermClassifier.parse_args()) --
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
    Translates the 'llm' section of the config dict into an argparse.Namespace compatible with LLMTermClassifier.run_inference().

    All keys mirror the CLI flags documented in LLMTermClassifier.parse_args().

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
        prompts_path         =_get(llm, "prompts_path", "src/termClassifier/prompts.json"),
        checkpoint_path      =_get(llm, "checkpoint_path"),

        # -- Prompt selection --
        system_prompt_key=_get(llm, "system_prompt_key", "base"),
        user_prompt_key  =_get(llm, "user_prompt_key",   "base"),

        # -- Generation hyper-parameters --
        temperature   =_get(llm, "temperature",    1.0),
        max_new_tokens=_get(llm, "max_new_tokens", 512),

        # -- HuggingFace local options --
        device=_get(llm, "device"),
    )

    # -- Validation --
    _require(ns, "provider", "llm.provider", "inference")
    _require(ns, "model", "llm.model", "inference")
    _require(ns, "inference_data_path", "llm.inference_data_path", "inference")
    _require(ns, "inference_output_path", "llm.inference_output_path", "inference")

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
    Reads the top-level 'classifier' and 'mode' keys from the config and calls the appropriate classifier entry point.

    :param config: Fully merged and validated config dict.
    :raises ValueError: If 'classifier' is not 'hf' or 'llm'.
    """
    classifier = config.get("classifier", "").lower()

    if classifier == "hf":
        # Import lazily: users who only use LLM classification do not need torch / HF
        from src.termClassifier.HFTermClassifier import run_training, run_inference

        ns = build_hf_namespace(config)
        if ns.inference_only:
            print("[main] Dispatching -> HFTermClassifier / inference")
            run_inference(ns)
        else:
            print("[main] Dispatching -> HFTermClassifier / training")
            run_training(ns)

    elif classifier == "llm":
        from src.termClassifier.LLMTermClassifier import run_inference

        ns = build_llm_namespace(config)
        print(
            f"[main] Dispatching -> LLMTermClassifier / inference "
            f"(provider={ns.provider}, model={ns.model})"
        )
        run_inference(ns)

    else:
        raise ValueError(
            f"Unknown classifier '{classifier}' in config.  "
            "Valid choices: 'hf', 'llm'."
        )


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for run_term_classification.py.

    Only two arguments exist here: the config file path and an optional list of dot-notation overrides.  
    All experiment parameters live in the YAML.

    :return: An argparse.Namespace with 'config' and 'override' attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Unified term classification runner.  "
            "Pass a YAML config file; optionally override individual keys inline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    python run_term_classification.py --config configs/hf_classify_train.yaml
    python run_term_classification.py --config configs/llm_classify_lmstudio.yaml --override llm.temperature=0.0
    python run_term_classification.py --config configs/hf_classify_inference.yaml --override hf.model_path=runs/best
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
            "Example: --override llm.temperature=0.5 hf.num_epochs=3"
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

    # 4. Dispatch to the correct classifier
    dispatch(config)
