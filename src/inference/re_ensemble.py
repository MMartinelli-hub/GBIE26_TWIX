"""
Relation-level ensembling utilities for GBIE relation extraction outputs.

The module combines already-generated prediction files. Each input file must use
the standard GBIE format: a dictionary keyed by paper ID, where each content
dictionary contains a relation list such as "relations".
"""

from collections import defaultdict
import copy
from datetime import datetime, timezone
import json
import os
import uuid


SUPPORTED_ENSEMBLE_METHODS = {"majority", "weighted", "intersection", "union"}
DEFAULT_ENSEMBLE_OUTPUT_DIR = os.path.join("runs", "relationExtractor", "ensembling")


def _safe_output_id(output_id: str | None = None) -> str:
    """
    Returns a filesystem-safe output ID, generating one when absent.

    :param output_id: Optional caller-provided ID.
    :return: Filesystem-safe ID string.
    """
    if output_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

    safe_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(output_id)
    ).strip("_")
    if not safe_id:
        raise ValueError("output_id must contain at least one safe filename character.")
    return safe_id


def build_ensemble_output_paths(
    output_dir: str = DEFAULT_ENSEMBLE_OUTPUT_DIR,
    output_id: str | None = None,
) -> tuple[str, str, str]:
    """
    Builds collision-free prediction and config sidecar paths.

    :param output_dir: Directory where ensemble artifacts should be saved.
    :param output_id: Optional caller-provided ID used in both filenames.
    :return: (prediction_path, config_path, output_id).
    """
    os.makedirs(output_dir, exist_ok=True)

    base_id = _safe_output_id(output_id)
    candidate_id = base_id
    suffix = 1

    while True:
        prediction_path = os.path.join(
            output_dir,
            f"ensemble_predictions_{candidate_id}.json",
        )
        config_path = os.path.join(output_dir, f"ensemble_config_{candidate_id}.json")
        if not os.path.exists(prediction_path) and not os.path.exists(config_path):
            return prediction_path, config_path, candidate_id

        if output_id is None:
            base_id = _safe_output_id()
            candidate_id = base_id
        else:
            candidate_id = f"{base_id}_{suffix}"
            suffix += 1


def load_json_data(file_path: str) -> dict:
    """
    Load JSON data from a file and return it as a dictionary.

    :param file_path: The path to the JSON file.
    :return: A dictionary containing the JSON data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_data(data: dict, file_path: str, encoding: str = "utf-8", indent: int = 4) -> None:
    """
    Save a dictionary as JSON data to a file.

    :param data: The dictionary to save as JSON.
    :param file_path: The path to the JSON file where the data will be saved.
    :param encoding: The encoding to use for the file.
    :param indent: The indentation for the JSON data.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


class EnsembleREInference:
    """
    Relation-level ensemble inference over multiple GBIE prediction files.

    Predictions are grouped by ordered subject/object mention spans. The final
    predicate and relation metadata are selected by voting over the models that
    predicted that subject/object pair.
    """

    def __init__(
        self,
        prediction_paths: list[str],
        save_path: str | None = None,
        method: str = "majority",
        weights: list[float] | None = None,
        vote_threshold: float | None = None,
        relation_key: str = "relations",
        config_save_path: str | None = None,
        output_id: str | None = None,
        run_config: dict | None = None,
        config_source: str | None = None,
    ):
        """
        :param prediction_paths: Paths to GBIE relation prediction JSON files.
        :param save_path: Optional output path for the ensembled predictions.
        :param method: One of majority, weighted, intersection, or union.
        :param weights: Model weights used by weighted voting. Must align with prediction_paths.
        :param vote_threshold: Optional minimum vote score for majority/weighted voting.
        :param relation_key: Name of the relation list field in each paper content dict.
        :param config_save_path: Optional path for the ensemble configuration sidecar.
        :param output_id: ID used to disambiguate this ensemble run's files.
        :param run_config: Effective caller configuration to persist alongside the results.
        :param config_source: Source config path, when available.
        """
        if not prediction_paths:
            raise ValueError("At least one prediction path is required.")

        self.prediction_paths = prediction_paths
        self.save_path = save_path
        self.method = self._normalize_method(method)
        self.predictions = [load_json_data(path) for path in prediction_paths]
        self.relation_key = relation_key
        self.weights = self._validate_weights(weights)
        self.vote_threshold = vote_threshold
        self.config_save_path = config_save_path
        self.output_id = output_id
        self.run_config = copy.deepcopy(run_config) if run_config is not None else None
        self.config_source = config_source

    @staticmethod
    def _normalize_method(method: str) -> str:
        """
        Normalizes common method aliases to the internal method names.

        :param method: Method name from the config or CLI.
        :return: Normalized method name.
        """
        normalized = method.lower().replace("-", "_")
        aliases = {
            "majority_voting": "majority",
            "weighted_voting": "weighted",
            "intersect": "intersection",
        }
        normalized = aliases.get(normalized, normalized)

        if normalized not in SUPPORTED_ENSEMBLE_METHODS:
            raise ValueError(
                f"Unsupported ensemble method: {method}. "
                f"Choose one of {sorted(SUPPORTED_ENSEMBLE_METHODS)}."
            )
        return normalized

    def _validate_weights(self, weights: list[float] | None) -> list[float]:
        """
        Validates and returns per-model voting weights.

        :param weights: Optional list of model weights.
        :return: Weight list aligned with self.predictions.
        """
        if weights is None:
            return [1.0] * len(self.predictions)

        if len(weights) != len(self.predictions):
            raise ValueError(
                "The number of weights must match the number of prediction paths."
            )

        parsed_weights = [float(weight) for weight in weights]
        if any(weight < 0 for weight in parsed_weights):
            raise ValueError("Model weights must be non-negative.")
        if sum(parsed_weights) <= 0:
            raise ValueError("At least one model weight must be greater than zero.")
        return parsed_weights

    @staticmethod
    def _relation_pair_key(paper_id: str, relation: dict) -> tuple:
        """
        Builds the pair-level key used to align relation predictions.

        Span offsets are preferred. Mention-level relation files that do not
        contain offsets fall back to text-span and label fields.

        :param paper_id: Paper identifier.
        :param relation: GBIE relation dictionary.
        :return: Hashable relation pair key.
        """
        span_fields = [
            "subject_start_idx",
            "subject_end_idx",
            "subject_location",
            "object_start_idx",
            "object_end_idx",
            "object_location",
        ]
        if all(field in relation for field in span_fields):
            return (
                paper_id,
                relation["subject_start_idx"],
                relation["subject_end_idx"],
                relation["subject_location"],
                relation["object_start_idx"],
                relation["object_end_idx"],
                relation["object_location"],
            )

        return (
            paper_id,
            relation.get("subject_text_span"),
            relation.get("subject_label"),
            relation.get("object_text_span"),
            relation.get("object_label"),
        )

    @staticmethod
    def _field_vote(votes: list[tuple[int, dict]], field: str, weights: list[float]):
        """
        Selects the best field value using weighted voting and deterministic ties.

        :param votes: List of (model_index, prediction_dict) votes.
        :param field: Field name to vote over.
        :param weights: Per-model weights.
        :return: Winning field value, or None when the field is absent.
        """
        scores = defaultdict(float)
        counts = defaultdict(int)

        for model_idx, prediction in votes:
            if field not in prediction:
                continue
            value = prediction[field]
            scores[value] += weights[model_idx]
            counts[value] += 1

        if not scores:
            return None

        return sorted(
            scores,
            key=lambda value: (-scores[value], -counts[value], str(value)),
        )[0]

    @staticmethod
    def _field_score(
        votes: list[tuple[int, dict]],
        field: str,
        value,
        weights: list[float],
        weighted: bool,
    ) -> float:
        """
        Computes the vote score for a selected field value.

        :param votes: List of (model_index, prediction_dict) votes.
        :param field: Field name to inspect.
        :param value: Field value whose support should be counted.
        :param weights: Per-model weights.
        :param weighted: Whether to sum weights instead of model counts.
        :return: Vote score for the selected value.
        """
        if weighted:
            return sum(
                weights[model_idx]
                for model_idx, prediction in votes
                if prediction.get(field) == value
            )
        return float(
            len({
                model_idx
                for model_idx, prediction in votes
                if prediction.get(field) == value
            })
        )

    def _threshold(self) -> float:
        """
        Computes the inclusion threshold for the selected ensemble method.

        :return: Minimum vote score required to keep a relation.
        """
        if self.vote_threshold is not None:
            return float(self.vote_threshold)

        if self.method == "intersection":
            return float(len(self.predictions))
        if self.method == "union":
            return 1.0
        if self.method == "weighted":
            return (sum(self.weights) / 2.0) + 1e-12
        return float(len(self.predictions) // 2 + 1)

    def _keep_relation(self, votes: list[tuple[int, dict]]) -> bool:
        """
        Decides whether a candidate subject/object pair should be included.

        :param votes: List of (model_index, relation_dict) votes.
        :return: True when the winning predicate passes the current ensemble rule.
        """
        if self.method == "union":
            return True

        winning_predicate = self._field_vote(votes, "predicate", self.weights)
        if winning_predicate is None:
            return False

        predicate_score = self._field_score(
            votes,
            field="predicate",
            value=winning_predicate,
            weights=self.weights,
            weighted=(self.method == "weighted"),
        )
        return predicate_score >= self._threshold()

    def _build_relation(self, pair_key: tuple, votes: list[tuple[int, dict]]) -> dict:
        """
        Builds the final relation dictionary for one ensembled pair.

        :param pair_key: Relation pair key produced by _relation_pair_key.
        :param votes: List of votes for the relation pair.
        :return: GBIE relation dictionary.
        """
        relation = {}
        index_fields = [
            "subject_start_idx",
            "subject_end_idx",
            "subject_location",
            "object_start_idx",
            "object_end_idx",
            "object_location",
        ]
        text_fields = [
            "subject_text_span",
            "subject_label",
            "subject_uri",
            "predicate",
            "object_text_span",
            "object_label",
            "object_uri",
        ]

        base_key = pair_key[:-1] if self.method == "union" else pair_key

        if len(base_key) == 7:
            (
                _,
                relation["subject_start_idx"],
                relation["subject_end_idx"],
                relation["subject_location"],
                relation["object_start_idx"],
                relation["object_end_idx"],
                relation["object_location"],
            ) = base_key
        else:
            _, subject_text_span, subject_label, object_text_span, object_label = base_key
            relation["subject_text_span"] = subject_text_span
            relation["subject_label"] = subject_label
            relation["object_text_span"] = object_text_span
            relation["object_label"] = object_label

        for field in text_fields:
            value = self._field_vote(votes, field, self.weights)
            if value is not None:
                relation[field] = value

        for field in index_fields:
            if field in relation:
                continue
            value = self._field_vote(votes, field, self.weights)
            if value is not None:
                relation[field] = value

        winning_predicate = relation.get("predicate")
        confscores = [
            vote["confscore"]
            for _, vote in votes
            if vote.get("predicate") == winning_predicate
            and isinstance(vote.get("confscore"), (int, float))
        ]
        if confscores:
            relation["confscore"] = round(sum(confscores) / len(confscores), 6)

        return relation

    def perform_relation_level_inference(self) -> dict | None:
        """
        Runs the selected relation ensemble method and optionally saves results.

        :return: Ensembled GBIE predictions when save_path is not set.
        """
        relation_votes = defaultdict(list)

        # Use the first prediction file as the template so metadata, entities,
        # and auxiliary fields stay available after ensembling.
        ensemble_results = copy.deepcopy(self.predictions[0])

        for model_idx, model_predictions in enumerate(self.predictions):
            for paper_id, content in model_predictions.items():
                seen_pairs = set()
                for relation in content.get(self.relation_key, []):
                    pair_key = self._relation_pair_key(paper_id, relation)
                    if self.method == "union":
                        pair_key = pair_key + (relation.get("predicate"),)
                    if pair_key in seen_pairs:
                        continue
                    relation_votes[pair_key].append((model_idx, relation))
                    seen_pairs.add(pair_key)

        for paper_id, content in ensemble_results.items():
            content[self.relation_key] = []

        for pair_key, votes in relation_votes.items():
            paper_id = pair_key[0]
            if not self._keep_relation(votes):
                continue
            if paper_id not in ensemble_results:
                ensemble_results[paper_id] = {self.relation_key: []}
            ensemble_results[paper_id][self.relation_key].append(
                self._build_relation(pair_key, votes)
            )

        for content in ensemble_results.values():
            content[self.relation_key] = sorted(
                content.get(self.relation_key, []),
                key=lambda relation: (
                    relation.get("subject_location", ""),
                    relation.get("subject_start_idx", -1),
                    relation.get("object_location", ""),
                    relation.get("object_start_idx", -1),
                    relation.get("predicate", ""),
                    relation.get("subject_text_span", ""),
                    relation.get("object_text_span", ""),
                ),
            )

        if self.save_path:
            save_json_data(ensemble_results, self.save_path)
            print(f"[ensemble] Saved relation ensemble predictions to {self.save_path}")
            self._save_run_config()
            return None

        return ensemble_results

    def _save_run_config(self) -> None:
        """
        Saves the effective ensemble configuration next to the prediction file.
        """
        if not self.config_save_path:
            return

        payload = {
            "output_id": self.output_id,
            "task": "relation_extraction",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_source": self.config_source,
            "prediction_output_path": self.save_path,
            "configuration": self.run_config,
            "ensemble": {
                "strategy": self.method,
                "prediction_paths": self.prediction_paths,
                "num_prediction_files": len(self.prediction_paths),
                "weights": self.weights,
                "vote_threshold": self.vote_threshold,
                "effective_threshold": self._threshold(),
                "relation_key": self.relation_key,
            },
        }
        save_json_data(payload, self.config_save_path, indent=2)
        print(f"[ensemble] Saved relation ensemble config to {self.config_save_path}")


def run_inference(args) -> dict | None:
    """
    Entry point used by run_re_ensemble.py.

    :param args: argparse.Namespace containing ensemble settings.
    :return: None after saving predictions and the configuration sidecar.
    """
    output_dir = getattr(args, "output_dir", None) or DEFAULT_ENSEMBLE_OUTPUT_DIR
    output_path, config_save_path, output_id = build_ensemble_output_paths(
        output_dir=output_dir,
        output_id=getattr(args, "output_id", None),
    )
    run_config = getattr(args, "effective_config", None)
    if run_config is None:
        run_config = {
            "prediction_paths": args.prediction_paths,
            "method": args.method,
            "weights": args.weights,
            "vote_threshold": args.vote_threshold,
            "relation_key": args.relation_key,
            "output_dir": output_dir,
            "output_id": output_id,
        }

    ensemble = EnsembleREInference(
        prediction_paths=args.prediction_paths,
        save_path=output_path,
        method=args.method,
        weights=args.weights,
        vote_threshold=args.vote_threshold,
        relation_key=args.relation_key,
        config_save_path=config_save_path,
        output_id=output_id,
        run_config=run_config,
        config_source=getattr(args, "config_path", None),
    )
    return ensemble.perform_relation_level_inference()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run relation-level ensemble inference.")
    parser.add_argument("--prediction-paths", nargs="+", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_ENSEMBLE_OUTPUT_DIR)
    parser.add_argument("--output-id")
    parser.add_argument("--output-path", help="Deprecated; ensemble outputs are saved under --output-dir.")
    parser.add_argument("--method", default="majority", choices=sorted(SUPPORTED_ENSEMBLE_METHODS))
    parser.add_argument("--weights", nargs="*", type=float)
    parser.add_argument("--vote-threshold", type=float)
    parser.add_argument("--relation-key", default="relations")
    run_inference(parser.parse_args())
