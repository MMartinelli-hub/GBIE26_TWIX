"""
Entity-level ensembling utilities for GBIE named entity recognition outputs.

The module combines already-generated prediction files. Each input file must use
the standard GBIE format: a dictionary keyed by paper ID, where each content
dictionary contains an "entities" list.
"""

from collections import defaultdict
import copy
from datetime import datetime, timezone
import json
import os
import uuid


SUPPORTED_ENSEMBLE_METHODS = {"majority", "weighted", "intersection", "union"}
DEFAULT_ENSEMBLE_OUTPUT_DIR = os.path.join("runs", "entityRecognizer", "ensembling")


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


class EnsembleNERInference:
    """
    Entity-level ensemble inference over multiple GBIE prediction files.

    Predictions are grouped by exact character span and location. The final
    entity label and text span are selected by voting over the models that
    predicted that span.
    """

    def __init__(
        self,
        prediction_paths: list[str],
        save_path: str | None = None,
        method: str = "majority",
        weights: list[float] | None = None,
        vote_threshold: float | None = None,
        entity_key: str = "entities",
        config_save_path: str | None = None,
        output_id: str | None = None,
        run_config: dict | None = None,
        config_source: str | None = None,
    ):
        """
        :param prediction_paths: Paths to GBIE NER prediction JSON files.
        :param save_path: Optional output path for the ensembled predictions.
        :param method: One of majority, weighted, intersection, or union.
        :param weights: Model weights used by weighted voting. Must align with prediction_paths.
        :param vote_threshold: Optional minimum vote score for majority/weighted voting.
        :param entity_key: Name of the entity list field in each paper content dict.
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
        self.entity_key = entity_key
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
    def _entity_span_key(paper_id: str, entity: dict) -> tuple:
        """
        Builds the span-level key used to align entity predictions.

        :param paper_id: Paper identifier.
        :param entity: GBIE entity dictionary.
        :return: Hashable entity span key.
        """
        return (
            paper_id,
            entity["start_idx"],
            entity["end_idx"],
            entity["location"],
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

        :return: Minimum vote score required to keep a span.
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

    def _span_score(self, votes: list[tuple[int, dict]]) -> float:
        """
        Computes the presence score for a candidate entity span.

        :param votes: List of (model_index, entity_dict) votes.
        :return: Vote count or weight sum, depending on the method.
        """
        if self.method == "weighted":
            return sum(self.weights[model_idx] for model_idx, _ in votes)
        return float(len({model_idx for model_idx, _ in votes}))

    def _keep_span(self, votes: list[tuple[int, dict]]) -> bool:
        """
        Decides whether a candidate entity span should be included.

        :param votes: List of (model_index, entity_dict) votes.
        :return: True when the span passes the current ensemble rule.
        """
        if self.method == "union":
            return True

        winning_label = self._field_vote(votes, "label", self.weights)
        if winning_label is None:
            return False

        label_score = self._field_score(
            votes,
            field="label",
            value=winning_label,
            weights=self.weights,
            weighted=(self.method == "weighted"),
        )
        return label_score >= self._threshold()

    def _build_entity(self, span_key: tuple, votes: list[tuple[int, dict]]) -> dict:
        """
        Builds the final entity dictionary for one ensembled span.

        :param span_key: Entity span key produced by _entity_span_key.
        :param votes: List of votes for the entity span.
        :return: GBIE entity dictionary.
        """
        _, start_idx, end_idx, location = span_key
        entity = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "location": location,
            "text_span": self._field_vote(votes, "text_span", self.weights),
            "label": self._field_vote(votes, "label", self.weights),
        }

        uri = self._field_vote(votes, "uri", self.weights)
        if uri is not None:
            entity["uri"] = uri

        confscores = [
            vote["confscore"]
            for _, vote in votes
            if isinstance(vote.get("confscore"), (int, float))
        ]
        if confscores:
            entity["confscore"] = round(sum(confscores) / len(confscores), 6)

        return entity

    def perform_entity_level_inference(self) -> dict | None:
        """
        Runs the selected entity ensemble method and optionally saves results.

        :return: Ensembled GBIE predictions when save_path is not set.
        """
        entity_votes = defaultdict(list)

        # Use the first prediction file as the template so metadata and any
        # downstream fields stay available after ensembling.
        ensemble_results = copy.deepcopy(self.predictions[0])

        for model_idx, model_predictions in enumerate(self.predictions):
            for paper_id, content in model_predictions.items():
                seen_spans = set()
                for entity in content.get(self.entity_key, []):
                    span_key = self._entity_span_key(paper_id, entity)
                    if span_key in seen_spans:
                        continue
                    entity_votes[span_key].append((model_idx, entity))
                    seen_spans.add(span_key)

        for paper_id, content in ensemble_results.items():
            content[self.entity_key] = []

        for span_key, votes in entity_votes.items():
            paper_id = span_key[0]
            if not self._keep_span(votes):
                continue
            if paper_id not in ensemble_results:
                ensemble_results[paper_id] = {self.entity_key: []}
            ensemble_results[paper_id][self.entity_key].append(
                self._build_entity(span_key, votes)
            )

        for content in ensemble_results.values():
            content[self.entity_key] = sorted(
                content.get(self.entity_key, []),
                key=lambda entity: (
                    entity.get("location", ""),
                    entity.get("start_idx", -1),
                    entity.get("end_idx", -1),
                    entity.get("label", ""),
                ),
            )

        if self.save_path:
            save_json_data(ensemble_results, self.save_path)
            print(f"[ensemble] Saved entity ensemble predictions to {self.save_path}")
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
            "task": "entity_recognition",
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
                "entity_key": self.entity_key,
            },
        }
        save_json_data(payload, self.config_save_path, indent=2)
        print(f"[ensemble] Saved entity ensemble config to {self.config_save_path}")


def run_inference(args) -> dict | None:
    """
    Entry point used by run_ner_ensemble.py.

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
            "entity_key": args.entity_key,
            "output_dir": output_dir,
            "output_id": output_id,
        }

    ensemble = EnsembleNERInference(
        prediction_paths=args.prediction_paths,
        save_path=output_path,
        method=args.method,
        weights=args.weights,
        vote_threshold=args.vote_threshold,
        entity_key=args.entity_key,
        config_save_path=config_save_path,
        output_id=output_id,
        run_config=run_config,
        config_source=getattr(args, "config_path", None),
    )
    return ensemble.perform_entity_level_inference()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run entity-level ensemble inference.")
    parser.add_argument("--prediction-paths", nargs="+", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_ENSEMBLE_OUTPUT_DIR)
    parser.add_argument("--output-id")
    parser.add_argument("--output-path", help="Deprecated; ensemble outputs are saved under --output-dir.")
    parser.add_argument("--method", default="majority", choices=sorted(SUPPORTED_ENSEMBLE_METHODS))
    parser.add_argument("--weights", nargs="*", type=float)
    parser.add_argument("--vote-threshold", type=float)
    parser.add_argument("--entity-key", default="entities")
    run_inference(parser.parse_args())
