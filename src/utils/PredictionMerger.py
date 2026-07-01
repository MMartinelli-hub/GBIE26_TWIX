"""
Prediction merging utilities for two-stage GBIE entity prediction outputs.

The module combines one termExtractor prediction file with one termClassifier
prediction file. The extractor file provides the entity spans; the classifier
file provides the semantic labels for those spans. The merged output follows
the entityRecognizer entity format.
"""

import argparse
import copy
import json
import os

import yaml


SUPPORTED_MISSING_MODES = {"strict", "ignore"}


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


class TermPredictionMerger:
    """
    Merges term extraction and term classification predictions into GBIE NER-style predictions.

    Term extraction predictions are treated as the source of truth for entity
    spans. Classification predictions are indexed by exact span and used to
    populate each entity's semantic label.
    """

    def __init__(
        self,
        term_extractor_path: str,
        term_classifier_path: str,
        output_path: str | None = None,
        missing_mode: str = "strict",
        include_na_entities: bool = False,
        entity_key: str = "entities",
    ):
        """
        :param term_extractor_path: Path to the termExtractor prediction JSON file.
        :param term_classifier_path: Path to the termClassifier prediction JSON file.
        :param output_path: Optional output path for the merged predictions.
        :param missing_mode: "strict" raises on extractor spans missing from the classifier; "ignore" drops them.
        :param include_na_entities: Whether extractor entities labelled "NA" should be retained.
        :param entity_key: Name of the entity list field in each paper content dict.
        """
        self.term_extractor_path = term_extractor_path
        self.term_classifier_path = term_classifier_path
        self.output_path = output_path
        self.missing_mode = self._normalize_missing_mode(missing_mode)
        self.include_na_entities = include_na_entities
        self.entity_key = entity_key

        self.term_extractor_predictions = load_json_data(term_extractor_path)
        self.term_classifier_predictions = load_json_data(term_classifier_path)

    @staticmethod
    def _normalize_missing_mode(missing_mode: str) -> str:
        """
        Normalizes common missing-entity mode aliases.

        :param missing_mode: Missing-entity handling mode.
        :return: Normalized mode name.
        """
        normalized = missing_mode.lower().replace("-", "_")
        aliases = {
            "error": "strict",
            "raise": "strict",
            "raise_error": "strict",
            "drop": "ignore",
            "skip": "ignore",
        }
        normalized = aliases.get(normalized, normalized)

        if normalized not in SUPPORTED_MISSING_MODES:
            raise ValueError(
                f"Unsupported missing mode: {missing_mode}. "
                f"Choose one of {sorted(SUPPORTED_MISSING_MODES)}."
            )
        return normalized

    @staticmethod
    def _entity_span_key(paper_id: str, entity: dict) -> tuple:
        """
        Builds the span-level key used to align extractor and classifier entities.

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
    def _is_na_entity(entity: dict) -> bool:
        """
        Checks whether an entity carries the NA label.

        :param entity: GBIE entity dictionary.
        :return: True when the entity label is NA.
        """
        return entity.get("label") == "NA"

    @staticmethod
    def _get_classifier_label(entity: dict) -> str | None:
        """
        Extracts the classifier label from either supported classifier output schema.

        :param entity: Entity dictionary from the termClassifier output.
        :return: Predicted label string, or None when absent.
        """
        return entity.get("predicted_label", entity.get("label"))

    def _build_classifier_index(self) -> dict:
        """
        Builds an exact-span lookup over the termClassifier predictions.

        :return: Dict mapping entity span keys to classifier entity dictionaries.
        """
        classifier_index = {}

        for paper_id, content in self.term_classifier_predictions.items():
            for entity in content.get(self.entity_key, []):
                label = self._get_classifier_label(entity)
                if label is None:
                    continue
                classifier_index[self._entity_span_key(paper_id, entity)] = entity

        return classifier_index

    def _build_entity(self, extractor_entity: dict, classifier_entity: dict) -> dict:
        """
        Builds the final entityRecognizer-style entity dictionary.

        :param extractor_entity: Entity dictionary from the termExtractor output.
        :param classifier_entity: Matching entity dictionary from the termClassifier output.
        :return: GBIE entity dictionary with span, text, label, and confscore fields.
        """
        merged_entity = {
            "start_idx": extractor_entity["start_idx"],
            "end_idx": extractor_entity["end_idx"],
            "location": extractor_entity["location"],
            "text_span": extractor_entity["text_span"],
            "label": self._get_classifier_label(classifier_entity),
        }

        if isinstance(classifier_entity.get("confscore"), (int, float)):
            merged_entity["confscore"] = classifier_entity["confscore"]
        elif isinstance(extractor_entity.get("confscore"), (int, float)):
            merged_entity["confscore"] = extractor_entity["confscore"]

        return merged_entity

    def _handle_missing_classifier_entity(self, paper_id: str, extractor_entity: dict) -> bool:
        """
        Applies the configured missing-entity policy.

        :param paper_id: Paper identifier.
        :param extractor_entity: Extractor entity missing from the classifier index.
        :return: True when the caller should keep processing by skipping this entity.
        """
        if self.missing_mode == "ignore":
            return True

        raise ValueError(
            "Entity predicted by termExtractor is missing from termClassifier: "
            f"paper_id={paper_id}, location={extractor_entity.get('location')}, "
            f"start_idx={extractor_entity.get('start_idx')}, "
            f"end_idx={extractor_entity.get('end_idx')}, "
            f"text_span={extractor_entity.get('text_span')!r}"
        )

    def merge_predictions(self, args) -> dict | None:
        """
        Runs the two-stage prediction merge and optionally saves the result.

        :return: Merged predictions when output_path is not set.
        """
        classifier_index = self._build_classifier_index()

        # Use the extractor output as the template so metadata and any
        # downstream fields stay available after merging.
        merged_predictions = copy.deepcopy(self.term_extractor_predictions)

        for paper_id, content in self.term_extractor_predictions.items():
            merged_entities = []

            for extractor_entity in content.get(self.entity_key, []):
                if self._is_na_entity(extractor_entity) and not self.include_na_entities:
                    continue

                span_key = self._entity_span_key(paper_id, extractor_entity)
                classifier_entity = classifier_index.get(span_key)

                if classifier_entity is None:
                    self._handle_missing_classifier_entity(paper_id, extractor_entity)
                    continue

                merged_entity = self._build_entity(extractor_entity, classifier_entity)
                if merged_entity["label"] == "NA" and not self.include_na_entities:
                    continue

                merged_entities.append(merged_entity)

            merged_predictions[paper_id][self.entity_key] = merged_entities

        if self.output_path:
            output_filename = generate_output_filename(args)
            output_path = os.path.join(self.output_path, output_filename)
            save_json_data(merged_predictions, output_path)
            print(f"[merge] Saved merged entity predictions to {output_path}")
            return None

        return merged_predictions
    
def generate_output_filename(args: argparse.Namespace) -> str:
    """
    Generate a standardized output filename based on configuration parameters.

    :param args: Parsed CLI arguments.
    :return: Output filename string.
    """
    extractor = os.path.splitext(os.path.basename(args.term_extractor_path))[0]
    classifier = os.path.splitext(os.path.basename(args.term_classifier_path))[0]
    missing = args.missing_mode
    include_na = "withNA" if args.include_na_entities else "noNA"
    return f"merged_{extractor}_{classifier}_{missing}_{include_na}.json"

def run_inference(args) -> dict | None:
    """
    Entry point used by command-line runners or external scripts.

    :param args: argparse.Namespace containing merge settings.
    :return: Merged predictions when no output path is provided.
    """
    merger = TermPredictionMerger(
        term_extractor_path=args.term_extractor_path,
        term_classifier_path=args.term_classifier_path,
        output_path=args.output_path,
        missing_mode=args.missing_mode,
        include_na_entities=args.include_na_entities,
        entity_key=args.entity_key,
    )
    return merger.merge_predictions(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge termExtractor spans with termClassifier labels."
    )
    # -- Config file (optional) --
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")

    # -- Args (overrides config) --
    parser.add_argument("--term-extractor-path", required=False)
    parser.add_argument("--term-classifier-path", required=False)
    parser.add_argument("--output-path", required=False)
    parser.add_argument("--missing-mode", default="strict")
    parser.add_argument("--include-na-entities", action="store_true")
    parser.add_argument("--entity-key", default="entities")
    
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    required = ["term_extractor_path", "term_classifier_path", "output_path"]
    missing = [key for key in required if not getattr(args, key, None)]
    if missing:
        parser.error(f"the following arguments are required: {', '.join('--' + key for key in missing)}")

    run_inference(parser.parse_args())
