#!/usr/bin/env python3
"""
run_ner_ensemble.py

YAML/JSON-driven entry point for entity-level prediction ensembling.

Usage
-----
    python run_ner_ensemble.py --config scripts/configs/entityRecognizer/ensemble.yaml
    python run_ner_ensemble.py --config runs/entityRecognizer/ensembling/ensemble_config_example.json
    python run_ner_ensemble.py --config scripts/configs/entityRecognizer/ensemble.yaml --override method=union
"""

import argparse
import copy
from itertools import combinations
import json
import os
import sys

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required to read config files.  "
        "Install it with: pip install pyyaml"
    ) from exc

from src.inference.ner_ensemble import SUPPORTED_ENSEMBLE_METHODS, run_inference


METHOD_ALIASES = {
    "majority_voting": "majority",
    "weighted_voting": "weighted",
    "intersect": "intersection",
}


#################
# Config loader #
#################

def load_config(config_path: str) -> dict:
    """
    Loads and returns a YAML config or saved JSON ensemble sidecar as a config dict.

    JSON sidecars written by EnsembleNERInference are flattened back into the
    same shape accepted by the existing dispatcher. When the sidecar contains
    explicit prediction_paths, folder-discovery keys are removed so the run is
    replayed as one exact ensemble rather than expanded into batch combinations.

    :param config_path: Path to a .yaml / .yml config or .json sidecar file.
    :return: Parsed configuration dict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.lower().endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON/YAML object: {config_path}")

    if config_path.lower().endswith(".json"):
        return normalize_json_config(config)

    return config


def normalize_json_config(config: dict) -> dict:
    """
    Converts a saved ensemble JSON sidecar into a runnable flat config.

    :param config: Parsed JSON object.
    :return: Config dict accepted by build_batch_namespaces.
    """
    if not isinstance(config.get("configuration"), dict):
        return config

    run_config = copy.deepcopy(config["configuration"])
    ensemble_config = config.get("ensemble") or {}
    if not isinstance(ensemble_config, dict):
        ensemble_config = {}

    if "output_id" not in run_config and config.get("output_id") is not None:
        run_config["output_id"] = config["output_id"]
    if "method" not in run_config and ensemble_config.get("strategy") is not None:
        run_config["method"] = ensemble_config["strategy"]
    if "prediction_paths" not in run_config and ensemble_config.get("prediction_paths") is not None:
        run_config["prediction_paths"] = ensemble_config["prediction_paths"]
    if run_config.get("weights") is None and run_config.get("method") == "weighted":
        run_config["weights"] = ensemble_config.get("weights")
    if "vote_threshold" not in run_config and "vote_threshold" in ensemble_config:
        run_config["vote_threshold"] = ensemble_config["vote_threshold"]
    if "entity_key" not in run_config and ensemble_config.get("entity_key") is not None:
        run_config["entity_key"] = ensemble_config["entity_key"]
    if "output_dir" not in run_config and config.get("prediction_output_path"):
        run_config["output_dir"] = os.path.dirname(config["prediction_output_path"])

    if run_config.get("prediction_paths") is not None:
        for key in [
            "prediction_folder",
            "prediction_dir",
            "predictions_folder",
            "predictions_dir",
            "prediction_glob",
            "min_combination_size",
            "max_combination_size",
            "batch_run",
        ]:
            run_config.pop(key, None)

    return run_config


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """
    Applies dot-notation key=value overrides to a config dictionary.

    :param config: The config dict to mutate.
    :param overrides: A list of "dotted.key=value" strings.
    :return: The mutated config dict.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override '{override}': expected 'key=value' format."
            )

        key_path, _, raw_value = override.partition("=")
        keys = key_path.strip().split(".")
        parsed_value = yaml.safe_load(raw_value)

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

def _get(section: dict, key: str, default=None):
    """Convenience getter that returns a default when a key is absent."""
    return section.get(key, default)


def _first_present(section: dict, keys: list[str], default=None):
    """Returns the first present value among a list of keys."""
    for key in keys:
        if key in section:
            return section[key]
    return default


def normalize_method(method: str) -> str:
    """
    Normalizes common method aliases to the internal method names.

    :param method: Method name from the config or CLI.
    :return: Normalized method name.
    """
    normalized = str(method).lower().replace("-", "_")
    normalized = METHOD_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_ENSEMBLE_METHODS:
        raise ValueError(
            f"Unknown ensemble method '{method}'. "
            f"Valid choices: {sorted(SUPPORTED_ENSEMBLE_METHODS)}."
        )
    return normalized


def normalize_methods(config: dict) -> list[str]:
    """
    Returns the ensemble methods requested by the config.

    :param config: Full config dict.
    :return: Normalized ensemble method names.
    """
    methods = _first_present(config, ["methods", "ensemble_methods"])
    if methods is None:
        methods = [_get(config, "method", "majority")]
    elif isinstance(methods, str):
        methods = [methods]

    if not isinstance(methods, list) or not methods:
        raise ValueError("'methods' must be a non-empty string or list.")

    return [normalize_method(method) for method in methods]


def get_prediction_folder(config: dict) -> str | None:
    """
    Returns the configured folder containing prediction files, when present.

    :param config: Full config dict.
    :return: Folder path or None.
    """
    return _first_present(
        config,
        [
            "prediction_folder",
            "prediction_dir",
            "predictions_folder",
            "predictions_dir",
        ],
    )


def discover_prediction_paths(folder_path: str, pattern: str = "*.json") -> list[str]:
    """
    Lists prediction files from a folder in deterministic order.

    :param folder_path: Folder containing prediction files.
    :param pattern: Filename glob pattern. Defaults to JSON files.
    :return: Sorted prediction file paths.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Prediction folder does not exist: {folder_path}")

    paths = [
        os.path.join(folder_path, filename)
        for filename in sorted(os.listdir(folder_path))
        if os.path.isfile(os.path.join(folder_path, filename))
        and _matches_pattern(filename, pattern)
    ]
    if len(paths) < 2:
        raise ValueError(
            f"Prediction folder '{folder_path}' must contain at least two files "
            f"matching pattern '{pattern}'."
        )
    return paths


def select_combination_weights(
    weights: list[float] | None,
    combo_indices: tuple[int, ...],
    total_prediction_files: int,
    method: str,
) -> list[float] | None:
    """
    Selects weights aligned with the current prediction combination.

    :param weights: Optional configured weights.
    :param combo_indices: Indices of the current file combination.
    :param total_prediction_files: Number of discovered files in folder mode.
    :param method: Current ensemble method.
    :return: Weights for the current combination, or None.
    """
    if weights is None:
        return None

    if len(weights) == total_prediction_files:
        return [weights[index] for index in combo_indices]

    if len(weights) == len(combo_indices):
        return weights

    if method != "weighted":
        return None

    raise ValueError(
        "In folder mode, 'weights' must either match the total number of "
        "discovered prediction files or the current combination size."
    )


def _matches_pattern(filename: str, pattern: str) -> bool:
    """
    Checks the small glob surface needed for prediction discovery.

    :param filename: Candidate filename.
    :param pattern: Glob pattern.
    :return: True when filename matches pattern.
    """
    import fnmatch

    return fnmatch.fnmatch(filename, pattern)


def build_namespace(config: dict) -> argparse.Namespace:
    """
    Translates the YAML config dict into the Namespace expected by the ensemble module.

    :param config: Full config dict.
    :return: A populated argparse.Namespace.
    """
    ns = argparse.Namespace(
        prediction_paths=_get(config, "prediction_paths"),
        prediction_folder=get_prediction_folder(config),
        prediction_glob=_get(config, "prediction_glob", "*.json"),
        output_path=_get(config, "output_path"),
        output_dir=_get(config, "output_dir"),
        output_id=_get(config, "output_id"),
        method=_get(config, "method", "majority"),
        weights=_get(config, "weights"),
        vote_threshold=_get(config, "vote_threshold"),
        entity_key=_get(config, "entity_key", "entities"),
    )

    if ns.prediction_paths is None and ns.prediction_folder is None:
        print(
            "[ERROR] Either 'prediction_paths' or 'prediction_folder' is required "
            "in the config file.",
            file=sys.stderr,
        )
        sys.exit(1)

    ns.method = normalize_method(ns.method)

    return ns


def build_batch_namespaces(config: dict, config_path: str) -> list[argparse.Namespace]:
    """
    Expands a config into one or more ensemble jobs.

    Folder mode runs all combinations of size 2..N for each requested method.
    Explicit prediction_paths mode preserves the existing single-job behavior.

    :param config: Effective config dict after overrides.
    :param config_path: Source config path.
    :return: Namespaces ready for run_inference.
    """
    base_namespace = build_namespace(config)
    methods = normalize_methods(config)

    if base_namespace.prediction_folder is None:
        if len(methods) > 1:
            jobs = []
            for method in methods:
                job_config = copy.deepcopy(config)
                job_config["method"] = method
                if _get(config, "output_id"):
                    job_config["output_id"] = f"{_get(config, 'output_id')}_{method}"
                job_config.pop("methods", None)
                job_config.pop("ensemble_methods", None)
                job = build_namespace(job_config)
                job.effective_config = job_config
                job.config_path = config_path
                jobs.append(job)
            return jobs

        base_namespace.method = methods[0]
        base_namespace.effective_config = config
        base_namespace.config_path = config_path
        return [base_namespace]

    all_prediction_paths = discover_prediction_paths(
        base_namespace.prediction_folder,
        base_namespace.prediction_glob,
    )
    max_combination_size = int(_get(config, "max_combination_size", len(all_prediction_paths)))
    min_combination_size = int(_get(config, "min_combination_size", 2))
    if min_combination_size < 2:
        raise ValueError("min_combination_size must be at least 2.")
    if max_combination_size > len(all_prediction_paths):
        raise ValueError("max_combination_size cannot exceed the number of prediction files.")
    if min_combination_size > max_combination_size:
        raise ValueError("min_combination_size cannot exceed max_combination_size.")

    jobs = []
    batch_output_id = _get(config, "output_id")
    for method in methods:
        combo_index = 1
        for combination_size in range(min_combination_size, max_combination_size + 1):
            for combo_indices in combinations(range(len(all_prediction_paths)), combination_size):
                combo_paths = [all_prediction_paths[index] for index in combo_indices]
                job_config = copy.deepcopy(config)
                job_config["method"] = method
                job_config["prediction_paths"] = combo_paths
                job_config["weights"] = select_combination_weights(
                    _get(config, "weights"),
                    combo_indices,
                    len(all_prediction_paths),
                    method,
                )
                job_config["batch_run"] = {
                    "prediction_folder": base_namespace.prediction_folder,
                    "prediction_glob": base_namespace.prediction_glob,
                    "all_prediction_paths": all_prediction_paths,
                    "combination_size": combination_size,
                    "combination_index_for_method": combo_index,
                    "combination_source_indices": list(combo_indices),
                    "methods": methods,
                }
                job_config.pop("methods", None)
                job_config.pop("ensemble_methods", None)

                if batch_output_id:
                    job_config["output_id"] = (
                        f"{batch_output_id}_{method}_k{combination_size}_combo{combo_index:04d}"
                    )
                else:
                    job_config["output_id"] = f"{method}_k{combination_size}_combo{combo_index:04d}"

                job = build_namespace(job_config)
                job.effective_config = job_config
                job.config_path = config_path
                jobs.append(job)
                combo_index += 1

    return jobs


def _require(ns: argparse.Namespace, attr: str, yaml_key: str) -> None:
    """
    Asserts that a required Namespace attribute is not None.

    :param ns: The Namespace to inspect.
    :param attr: Attribute name to check.
    :param yaml_key: Dot-notation YAML key shown in the error message.
    """
    if getattr(ns, attr, None) is None:
        print(
            f"[ERROR] '{yaml_key}' is required but was not set in the config file.",
            file=sys.stderr,
        )
        sys.exit(1)


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for run_ner_ensemble.py.

    :return: An argparse.Namespace with 'config' and 'override' attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Entity ensemble runner. Pass a YAML config file or a saved JSON "
            "ensemble sidecar; optionally override individual keys inline."
        )
    )
    parser.add_argument("--config", type=str, required=True, metavar="PATH")
    parser.add_argument("--override", type=str, nargs="*", default=[], metavar="KEY=VALUE")
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

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    print_config(config)

    jobs = build_batch_namespaces(config, args.config)
    print(f"[main] Prepared {len(jobs)} entity ensemble job(s).")
    for job_index, namespace in enumerate(jobs, start=1):
        print(
            f"[main] Dispatching {job_index}/{len(jobs)} -> "
            f"EnsembleNERInference / {namespace.method} / "
            f"{len(namespace.prediction_paths)} files"
        )
        run_inference(namespace)
