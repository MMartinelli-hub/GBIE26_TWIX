#!/usr/bin/env python3
"""
BaselineEntityLinker.py

GBIE Baseline Named Entity Linking (NEL) module that maps entity mentions to ontology URIs using a two-stage approach combining exact matching with semantic similarity.

Organized into five sections:
    - BaselineEntityLinkerConfig : typed configuration container for the linking pipeline
    - BaselineEntityLinker       : knowledge-base construction and inference pipeline
    - NELEvaluator               : URI-based P/R/F1 evaluation (span + URI and span-only)
    - Helper functions + CLI     : save_metadata, run_inference, parse_args

Linking strategy (applied in priority order for each entity mention):
    1. Exact match on (text_span, label) against the training-data knowledge base
    2. Exact match on text_span only against the training-data knowledge base
    3. Semantic similarity match against a PubMedBERT embedding index built from URI definitions (txtai required; GPU recommended)
    4. No match URI is set to the sentinel value 'NA'

The module integrates seamlessly with the rest of the GutBrainIE pipeline:
    - Input data must follow the GBIE entity format (start_idx, end_idx, location, text_span, label, [uri]).
    - Output data mirrors the input structure, with 'uri' and 'uri_source' fields added to every entity dict.
    - NELEvaluator mirrors NERExtractionEvaluator but keys on the predicted URI rather than the predicted label, enabling direct comparison of the two tasks.

Entry point: run with --help to see CLI options.
"""

import argparse
import json
import os
from dataclasses import dataclass, field

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# Import utilities shared across the GutBrainIE pipeline
from src.utils.utils import (
    load_json_data,
    load_merge_json_data,
    save_json_data,
)


#############
# Constants #
#############

# Sentinel URI assigned when no linking strategy produces a match
_NA_URI = "NA"

# Entity field keys used throughout the pipeline
_URI_KEY        = "uri"
_URI_SOURCE_KEY = "uri_source"

# Recognised uri_source tags (stored alongside each linked entity for transparency)
_SOURCE_EXACT_TEXT_LABEL = "exact_match_text_label"
_SOURCE_EXACT_TEXT_ONLY  = "exact_match_text_only"
_SOURCE_SIMILARITY       = "similarity_match"
_SOURCE_NO_MATCH         = "no_match"


##############################
# BaselineEntityLinkerConfig #
##############################

@dataclass
class BaselineEntityLinkerConfig:
    """
    Typed configuration container for the BaselineEntityLinker pipeline.

    Fields
    ------
    embeddings_model_name : str
        HuggingFace model identifier for the PubMedBERT sentence-embedding model used by txtai.  
        Must be compatible with the txtai Embeddings API.
    embeddings_index_path : str
        Directory path where the txtai embedding index is persisted between runs.
        If the directory already exists its contents are reloaded; otherwise a new index is built from the URI definitions and saved here.
    similarity_top_k : int
        Number of candidate definitions retrieved per query during similarity matching.  
        Only the top-ranked result is used for linking; higher values are kept for optional inspection / future re-ranking strategies.
    similarity_threshold : float
        Minimum cosine-similarity score required to accept a similarity match.
        Entities whose best match falls below this threshold are assigned 'NA'.
        Set to 0.0 to accept all similarity matches (notebook default behaviour).
    use_label_for_exact_match : bool
        When True (default), the priority-1 strategy uses the composite (text_span, label) key.  
        When False the (text_span, label) step is skipped and only the text_span-only step is attempted before similarity.
    """

    embeddings_model_name:   str   = "neuml/pubmedbert-base-embeddings"
    embeddings_index_path:   str   = "embeddings_index"
    similarity_top_k:        int   = 10
    similarity_threshold:    float = 0.0
    use_label_for_exact_match: bool = True


########################
# BaselineEntityLinker #
########################

class BaselineEntityLinker:
    """
    Wraps the two-stage (exact match + semantic similarity) entity linking pipeline and provides a uniform interface consistent with the rest of the GutBrainIE inference modules (termClassifier, termExtractor, RelationExtractor, etc.).

    Knowledge-base construction:
        Call build_knowledge_base() once per experimental setup to:
            - Extract (text_span -> URI) and ((text_span, label) -> URI) mappings from gold/silver training annotations.
            - Build a txtai PubMedBERT embedding index over the provided URI definitions for fallback semantic matching.

        The knowledge base can be persisted to disk and reloaded with save_knowledge_base() / load_knowledge_base() so that the relatively expensive embedding step is only repeated when necessary.

    Inference:
        Call perform_inference() with a GBIE-format data dict whose 'entities' lists contain already-predicted entity mentions (e.g. the output of an entityRecognizer pipeline step).  
        Each entity dict is augmented in-place with 'uri' and 'uri_source' fields.
    """

    def __init__(self, config: BaselineEntityLinkerConfig = None):
        """
        Initialises the linker with an optional configuration object.

        :param config: A BaselineEntityLinkerConfig instance. Defaults to the dataclass default values when None is passed.
        """
        self.config = config if config is not None else BaselineEntityLinkerConfig()

        # Exact-match knowledge-base dictionaries populated by build_knowledge_base()
        self._text_span_to_uris:       dict[str, set]        = {}
        self._text_span_label_to_uris: dict[tuple, set]      = {}

        # ID-to-URI mapping populated together with the embedding index
        self._definition_id_to_uri:    dict[str, str]        = {}

        # txtai Embeddings instance; None until the index is built / loaded
        self._embeddings: object = None

    # -- Knowledge-base construction --

    def build_knowledge_base(self, training_data_paths: list[str], uri_definitions_path: str, id_to_uri_path: str,) -> None:
        """
        Builds the full knowledge base from training annotations and URI definitions.

        This method must be called before perform_inference() whenever a persisted knowledge base is not available.  
        For subsequent runs, prefer load_knowledge_base() to avoid rebuilding the embedding index.

        Internally it:
            1. Reads all provided training files and extracts exact-match mappings for both the text_span-only and (text_span, label) strategies.
            2. Loads the URI definitions from 'uri_definitions_path' (the split_uri_definitions.json format: {id: definition_text}).
            3. Loads the ID-to-URI mapping from 'id_to_uri_path'.
            4. Builds (or reloads from disk) the txtai PubMedBERT embedding index.

        :param training_data_paths: List of paths to GBIE-format training JSON files from which exact-match mappings are extracted.
        :param uri_definitions_path: Path to the split_uri_definitions.json file produced by generate_definitions.ipynb.
        :param id_to_uri_path: Path to the id_to_uri.json file produced by generate_definitions.ipynb.
        :return: None
        """
        print("Building exact-match knowledge base from training annotations...")
        self._build_exact_match_dicts(training_data_paths)
        print(
            f"  Text-span-only mappings : {len(self._text_span_to_uris):>6,}"
        )
        print(
            f"  Text-span+label mappings: {len(self._text_span_label_to_uris):>6,}"
        )

        print("Loading ID-to-URI mapping...")
        self._definition_id_to_uri = load_json_data(id_to_uri_path)
        print(f"  Loaded {len(self._definition_id_to_uri):,} definition-ID -> URI mappings")

        print("Building / loading PubMedBERT embedding index...")
        self._build_embeddings_index(uri_definitions_path)

    def save_knowledge_base(self, output_dir: str) -> None:
        """
        Persists the exact-match dictionaries and ID-to-URI mapping to 'output_dir'.

        The txtai embedding index is handled separately: it is saved to config.embeddings_index_path (which may differ from output_dir).
        Call this method after build_knowledge_base() to avoid rebuilding on subsequent runs.

        :param output_dir: Directory where the JSON knowledge-base files will be written.
        :return: None
        """
        os.makedirs(output_dir, exist_ok=True)

        # Serialise sets to sorted lists for reproducible JSON output
        save_json_data(
            {span: sorted(uris) for span, uris in self._text_span_to_uris.items()},
            os.path.join(output_dir, "text_span_to_uris.json"),
        )

        save_json_data(
            {
                json.dumps(list(key)): sorted(uris)
                for key, uris in self._text_span_label_to_uris.items()
            },
            os.path.join(output_dir, "text_span_label_to_uris.json"),
        )

        save_json_data(
            self._definition_id_to_uri,
            os.path.join(output_dir, "definition_id_to_uri.json"),
        )

        print(f"Knowledge base saved to '{output_dir}'.")

    def load_knowledge_base(self, knowledge_base_dir: str) -> None:
        """
        Restores the exact-match dictionaries and ID-to-URI mapping from disk and reloads the txtai embedding index from config.embeddings_index_path.

        Call this method instead of build_knowledge_base() when a persisted knowledge base is available, as it skips the expensive embedding-index construction step.

        :param knowledge_base_dir: Directory previously written by save_knowledge_base().
        :return: None
        """
        print(f"Loading knowledge base from '{knowledge_base_dir}'...")

        raw_text_span = load_json_data(
            os.path.join(knowledge_base_dir, "text_span_to_uris.json")
        )
        self._text_span_to_uris = {span: set(uris) for span, uris in raw_text_span.items()}

        raw_text_span_label = load_json_data(
            os.path.join(knowledge_base_dir, "text_span_label_to_uris.json")
        )
        self._text_span_label_to_uris = {
            tuple(json.loads(key)): set(uris)
            for key, uris in raw_text_span_label.items()
        }

        self._definition_id_to_uri = load_json_data(
            os.path.join(knowledge_base_dir, "definition_id_to_uri.json")
        )

        print(
            f"  Text-span-only mappings : {len(self._text_span_to_uris):>6,}"
        )
        print(
            f"  Text-span+label mappings: {len(self._text_span_label_to_uris):>6,}"
        )
        print(
            f"  Definition-ID -> URI    : {len(self._definition_id_to_uri):>6,}"
        )

        # Reload the txtai embedding index from its configured path
        embeddings_index_path = self.config.embeddings_index_path
        if os.path.exists(embeddings_index_path):
            print(f"Loading embedding index from '{embeddings_index_path}'...")
            self._embeddings = self._init_embeddings_model()
            self._embeddings.load(embeddings_index_path)
            print("  Embedding index loaded.")
        else:
            print(
                f"Warning: Embedding index not found at '{embeddings_index_path}'. "
                "Similarity matching will be unavailable.  Call build_knowledge_base() "
                "to create the index."
            )

    # -- Inference --
    
    def perform_inference(self, data: dict) -> dict:
        """
        Runs the entity linking pipeline over an entire dataset and returns a copy of the input data with 'uri' and 'uri_source' fields added to every entity.

        The pipeline applies three fallback strategies in priority order:
            1. Exact match on (text_span.lower(), label)
            2. Exact match on text_span.lower() alone
            3. Semantic similarity via PubMedBERT embeddings
            4. Assign sentinel 'NA' if none of the above succeeds

        :param data: A dict mapping paper IDs to GBIE content dicts.  Each content dict must contain an 'entities' list whose items follow the GBIE entity schema (start_idx, end_idx, location, text_span, label).
        :return: A dict with the same structure as 'data' but with 'uri' and 'uri_source' added to every entity dict.
        """
        result = {}
        link_stats = {
            _SOURCE_EXACT_TEXT_LABEL: 0,
            _SOURCE_EXACT_TEXT_ONLY:  0,
            _SOURCE_SIMILARITY:       0,
            _SOURCE_NO_MATCH:         0,
        }

        for paper_id, content in tqdm(data.items(), total=len(data), desc="NEL Inference"):
            linked_entities = []

            for entity in content.get("entities", []):
                linked_entity = dict(entity)
                uri, source = self._apply_linking_strategy(linked_entity)
                linked_entity[_URI_KEY]        = uri
                linked_entity[_URI_SOURCE_KEY] = source
                linked_entities.append(linked_entity)
                link_stats[source] += 1

            output_content = dict(content)
            output_content["entities"] = linked_entities
            result[paper_id] = output_content

        # Summarise linking outcome across the full dataset
        total = sum(link_stats.values())
        if total > 0:
            print(f"\n=== Entity Linking Summary ===")
            print(f"Total entities processed : {total:>6,}")
            for source, count in link_stats.items():
                pct = count / total * 100
                print(f"  {source:<32}: {count:>6,}  ({pct:5.1f}%)")
            linked_total = total - link_stats[_SOURCE_NO_MATCH]
            print(f"Overall linking rate      : {linked_total / total * 100:.1f}%")

        return result

    # -- Internal helpers --
    
    def _build_exact_match_dicts(self, training_data_paths: list[str]) -> None:
        """
        Populates _text_span_to_uris and _text_span_label_to_uris from the provided GBIE-format training annotation files.

        Both dictionaries map to URI *sets* to handle cases where the same surface form has been linked to different ontology concepts across the corpus.
        During inference, the first element of the sorted set is used (consistent with the baseline notebook strategy).

        :param training_data_paths: Ordered list of paths to GBIE JSON files. Files are processed in order; later files do not override mappings established by earlier ones -- they only extend the URI sets.
        :return: None
        """
        self._text_span_to_uris       = {}
        self._text_span_label_to_uris = {}

        for file_path in training_data_paths:
            annotation_data = load_json_data(file_path)

            for _, document_content in annotation_data.items():
                for entity in document_content.get("entities", []):
                    # Skip entities that lack a URI (e.g. predicted output files)
                    if _URI_KEY not in entity or not entity[_URI_KEY]:
                        continue

                    normalized_span = entity["text_span"].lower()
                    entity_label    = entity["label"]
                    entity_uri      = entity[_URI_KEY]

                    # Text-span-only mapping
                    if normalized_span not in self._text_span_to_uris:
                        self._text_span_to_uris[normalized_span] = set()
                    self._text_span_to_uris[normalized_span].add(entity_uri)

                    # Composite (text_span, label) mapping
                    composite_key = (normalized_span, entity_label)
                    if composite_key not in self._text_span_label_to_uris:
                        self._text_span_label_to_uris[composite_key] = set()
                    self._text_span_label_to_uris[composite_key].add(entity_uri)

    def _init_embeddings_model(self):
        """
        Instantiates and returns a txtai Embeddings object configured with the PubMedBERT model specified in config.embeddings_model_name.

        Raises ImportError with a clear install hint when txtai is absent, matching the pattern used for optional dependencies in the other modules.

        :return: A txtai.Embeddings instance (not yet indexed or loaded).
        """
        try:
            import txtai  # noqa: F401 -- imported for the side-effect check
            from txtai import Embeddings
        except ImportError as exc:
            raise ImportError(
                "The 'txtai' package is required for semantic similarity matching in "
                "BaselineEntityLinker.  Install it with: pip install txtai"
            ) from exc

        return Embeddings(path=self.config.embeddings_model_name, content=True)

    def _build_embeddings_index(self, uri_definitions_path: str) -> None:
        """
        Builds (or reloads from disk) a txtai PubMedBERT embedding index over the URI definitions file produced by generate_definitions.ipynb.

        The index maps definition IDs to their embedded representations so that an entity's text span can be compared against all known concept definitions using cosine similarity.

        If config.embeddings_index_path already exists on disk the index is loaded from there, which avoids the expensive embedding step on repeated runs.

        :param uri_definitions_path: Path to split_uri_definitions.json ({definition_id: definition_text}).
        :return: None
        """
        self._embeddings = self._init_embeddings_model()

        index_path = self.config.embeddings_index_path

        if os.path.exists(index_path):
            print(f"  Existing index found at '{index_path}'. Loading...")
            self._embeddings.load(index_path)
            print("  Embedding index loaded.")
            return

        # Build the index from scratch
        print("  No existing index found.  Building from URI definitions...")
        uri_definitions = load_json_data(uri_definitions_path)
        print(f"  Loaded {len(uri_definitions):,} URI definitions for indexing.")

        # txtai expects an iterable of (id, text) tuples
        index_data = [
            (str(def_id), def_text)
            for def_id, def_text in uri_definitions.items()
        ]

        print("  Indexing definitions (this may take several minutes on CPU)...")
        self._embeddings.index(index_data)

        print(f"  Saving index to '{index_path}'...")
        os.makedirs(index_path, exist_ok=True)
        self._embeddings.save(index_path)
        print("  Embedding index built and saved.")

    def _apply_linking_strategy(self, entity: dict) -> tuple[str, str]:
        """
        Applies the priority-ordered linking strategy to a single entity dict and returns the resolved (uri, source_tag) pair.

        Priority:
            1. Exact match on (text_span.lower(), label) -- most specific
            2. Exact match on text_span.lower() alone
            3. Semantic similarity via the PubMedBERT embedding index
            4. No match -> return ('NA', 'no_match')

        :param entity: A single GBIE entity dict with at least 'text_span' and 'label' fields.
        :return: A 2-tuple (uri_string, source_tag_string).
        """
        normalized_span = entity["text_span"].lower()
        entity_label    = entity.get("label", "")

        # -- Strategy 1: exact match on (text_span, label) --
        if self.config.use_label_for_exact_match:
            composite_key = (normalized_span, entity_label)
            if composite_key in self._text_span_label_to_uris:
                # Use the lexicographically first URI for determinism
                uri = sorted(self._text_span_label_to_uris[composite_key])[0]
                return uri, _SOURCE_EXACT_TEXT_LABEL

        # -- Strategy 2: exact match on text_span only --
        if normalized_span in self._text_span_to_uris:
            uri = sorted(self._text_span_to_uris[normalized_span])[0]
            return uri, _SOURCE_EXACT_TEXT_ONLY

        # -- Strategy 3: semantic similarity --
        if self._embeddings is not None:
            uri = self._similarity_link(entity["text_span"])
            if uri is not None:
                return uri, _SOURCE_SIMILARITY

        # -- Strategy 4: no match --
        return _NA_URI, _SOURCE_NO_MATCH

    def _similarity_link(self, text_span: str) -> str | None:
        """
        Queries the embedding index with 'text_span' and returns the URI of the most similar definition if its similarity score meets the configured threshold.

        :param text_span: The raw (non-normalised) surface form of the entity.
        :return: The resolved URI string, or None when no match meets the threshold.
        """
        search_results = self._embeddings.search(text_span, self.config.similarity_top_k)

        if not search_results:
            return None

        top_result = search_results[0]
        top_score  = top_result.get("score", 0.0)

        if top_score < self.config.similarity_threshold:
            return None

        # Resolve the definition ID to a URI via the preloaded mapping
        top_definition_id = top_result.get("id")
        return self._definition_id_to_uri.get(str(top_definition_id))


################
# NELEvaluator #
################

class NELEvaluator:
    """
    Evaluates Named Entity Linking quality by comparing predicted URIs against gold-standard annotations.

    Mirrors NERExtractionEvaluator from HFEntityRecognizer but keys the strict comparison on the predicted URI rather than the predicted entity label:

        - Strict (span + URI): an entity is a true positive only when both the character offsets AND the assigned URI match the gold annotation. This is the primary NEL metric.

        - Span-only (lenient): an entity is a true positive when only the character offsets match, regardless of the predicted URI. Matches the extraction- only baseline and decouples NER quality from linking quality.

    Precision, recall, and F1 are computed with binary averaging for both modes and returned in a single metrics dict, consistent with the NERExtractionEvaluator interface.
    """

    def evaluate(self, predictions: dict, ground_truth: dict) -> dict:
        """
        Computes strict (span + URI) and lenient (span-only) P/R/F1 across all papers.

        :param predictions:  Dict mapping paper IDs to content dicts with predicted entities (each entity must contain 'uri').
        :param ground_truth: Dict mapping paper IDs to content dicts with gold entities (each entity must contain 'uri').
        :return: A dict with keys 'strict_precision', 'strict_recall', 'strict_f1', 'span_precision', 'span_recall', 'span_f1' (all floats in [0, 1]).
        """
        strict_gt_sets   = self._build_entity_sets(ground_truth, include_uri=True)
        strict_pred_sets = self._build_entity_sets(predictions,  include_uri=True)
        span_gt_sets     = self._build_entity_sets(ground_truth, include_uri=False)
        span_pred_sets   = self._build_entity_sets(predictions,  include_uri=False)

        strict_p, strict_r, strict_f1 = self._compute_prf(strict_gt_sets, strict_pred_sets)
        span_p,   span_r,   span_f1   = self._compute_prf(span_gt_sets,   span_pred_sets)

        return {
            "strict_precision": strict_p,
            "strict_recall":    strict_r,
            "strict_f1":        strict_f1,
            "span_precision":   span_p,
            "span_recall":      span_r,
            "span_f1":          span_f1,
        }

    @staticmethod
    def _compute_prf(gt_sets: dict, pred_sets: dict) -> tuple[float, float, float]:
        """
        Computes binary precision, recall, and F1 over the union of all paper IDs.

        :param gt_sets:   Dict mapping paper IDs to sets of comparable entity tuples.
        :param pred_sets: Dict mapping paper IDs to sets of comparable entity tuples.
        :return: (precision, recall, f1) as floats in [0, 1].
        """
        y_true = []
        y_pred = []

        all_ids = set(gt_sets.keys()) | set(pred_sets.keys())
        for paper_id in all_ids:
            gt_entities   = gt_sets.get(paper_id, set())
            pred_entities = pred_sets.get(paper_id, set())
            all_entities  = gt_entities | pred_entities

            for entity in all_entities:
                y_true.append(1 if entity in gt_entities   else 0)
                y_pred.append(1 if entity in pred_entities else 0)

        if not y_true:
            return 0.0, 0.0, 0.0

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        return float(precision), float(recall), float(f1)

    @staticmethod
    def _build_entity_sets(data: dict, include_uri: bool) -> dict:
        """
        Converts entity lists to sets of comparable tuples for fast lookup.

        :param data: Dict mapping paper IDs to content dicts with 'entities' lists.
        :param include_uri: When True, each tuple is (start_idx, end_idx, location, text_span, uri); when False, the URI is omitted, yielding (start_idx, end_idx, location, text_span).
        :return: Dict mapping paper IDs to sets of tuples.
        """
        entity_sets = {}
        for paper_id, content in data.items():
            if include_uri:
                entity_sets[paper_id] = {
                    (
                        entity["start_idx"],
                        entity["end_idx"],
                        entity["location"],
                        entity["text_span"],
                        entity.get(_URI_KEY, _NA_URI),
                    )
                    for entity in content.get("entities", [])
                }
            else:
                entity_sets[paper_id] = {
                    (
                        entity["start_idx"],
                        entity["end_idx"],
                        entity["location"],
                        entity["text_span"],
                    )
                    for entity in content.get("entities", [])
                }
        return entity_sets


####################
# Metadata helpers #
####################

def save_metadata(
    output_dir: str,
    config: BaselineEntityLinkerConfig,
    args: argparse.Namespace,
) -> None:
    """
    Persists the linker configuration and CLI arguments alongside the saved knowledge base for full experimental reproducibility.

    :param output_dir: Directory where the metadata JSON files will be written.
    :param config: BaselineEntityLinkerConfig instance used during knowledge-base construction.
    :param args: Parsed argument namespace from argparse.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)

    save_json_data(
        {
            "linker_config": {
                "embeddings_model_name":    config.embeddings_model_name,
                "embeddings_index_path":    config.embeddings_index_path,
                "similarity_top_k":         config.similarity_top_k,
                "similarity_threshold":     config.similarity_threshold,
                "use_label_for_exact_match": config.use_label_for_exact_match,
            },
        },
        os.path.join(output_dir, "nel_linker_config.json"),
    )

    save_json_data(vars(args), os.path.join(output_dir, "cli_args.json"))


#############################
# Top-level entry functions #
#############################

def run_build_and_inference(args: argparse.Namespace) -> None:
    """
    Orchestrates knowledge-base construction from training data followed by NEL inference over the specified input file.

    :param args: Parsed CLI arguments.
    :return: None
    """
    config = BaselineEntityLinkerConfig(
        embeddings_model_name    = args.embeddings_model_name,
        embeddings_index_path    = args.embeddings_index_path,
        similarity_top_k         = args.similarity_top_k,
        similarity_threshold     = args.similarity_threshold,
        use_label_for_exact_match = not args.no_label_matching,
    )

    linker = BaselineEntityLinker(config)

    # -- Build or reload knowledge base --
    if args.knowledge_base_dir and os.path.isdir(args.knowledge_base_dir):
        # Fast path: load prebuilt exact-match dicts + embedding index
        linker.load_knowledge_base(args.knowledge_base_dir)
    else:
        # Slow path: build from training annotations and URI definitions
        if not args.training_data_paths:
            raise ValueError(
                "--training_data_paths is required when no prebuilt knowledge base "
                "directory is provided via --knowledge_base_dir."
            )
        linker.build_knowledge_base(
            training_data_paths  = args.training_data_paths,
            uri_definitions_path = args.uri_definitions_path,
            id_to_uri_path       = args.id_to_uri_path,
        )

        # Persist the knowledge base for future runs
        if args.knowledge_base_dir:
            linker.save_knowledge_base(args.knowledge_base_dir)
            save_metadata(args.knowledge_base_dir, config, args)

    # -- Run inference --
    print(f"\nLoading inference data from '{args.inference_data_path}'...")
    inference_data = load_json_data(args.inference_data_path)
    print(f"  Loaded {len(inference_data):,} documents.")

    linked_results = linker.perform_inference(inference_data)

    print(f"\nSaving linked predictions to '{args.inference_output_path}'...")
    save_json_data(linked_results, args.inference_output_path)
    print("Done.")

    # -- Optional evaluation against gold annotations --
    if args.eval_data_path:
        print(f"\nEvaluating against ground truth at '{args.eval_data_path}'...")
        ground_truth = load_json_data(args.eval_data_path)
        evaluator    = NELEvaluator()
        metrics      = evaluator.evaluate(linked_results, ground_truth)

        print(
            f"Strict (span + URI)  "
            f"P: {metrics['strict_precision']:.4f} | "
            f"R: {metrics['strict_recall']:.4f} | "
            f"F1: {metrics['strict_f1']:.4f}"
        )
        print(
            f"Lenient (span-only)  "
            f"P: {metrics['span_precision']:.4f} | "
            f"R: {metrics['span_recall']:.4f} | "
            f"F1: {metrics['span_f1']:.4f}"
        )


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for the baseline NEL pipeline.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Baseline Named Entity Linking for the GutBrainIE dataset.  "
            "Links predicted entity mentions to ontology URIs using exact "
            "matching and PubMedBERT-based semantic similarity."
        )
    )

    # -- Knowledge-base source --
    parser.add_argument(
        "--knowledge_base_dir",
        type=str,
        default=None,
        help=(
            "Directory containing a prebuilt knowledge base produced by a previous "
            "run (text_span_to_uris.json, text_span_label_to_uris.json, "
            "definition_id_to_uri.json).  When provided and the directory exists, "
            "the knowledge-base construction step is skipped."
        ),
    )
    parser.add_argument(
        "--training_data_paths",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more paths to GBIE-format training JSON files from which "
            "exact-match URI mappings are extracted.  Required when no prebuilt "
            "knowledge base directory is available."
        ),
    )
    parser.add_argument(
        "--uri_definitions_path",
        type=str,
        default=None,
        help=(
            "Path to split_uri_definitions.json (definition_id -> definition_text), "
            "produced by generate_definitions.ipynb.  Required when building a new "
            "knowledge base."
        ),
    )
    parser.add_argument(
        "--id_to_uri_path",
        type=str,
        default=None,
        help=(
            "Path to id_to_uri.json (definition_id -> URI), produced by "
            "generate_definitions.ipynb.  Required when building a new knowledge base."
        ),
    )

    # -- Inference data paths --
    parser.add_argument(
        "--inference_data_path",
        type=str,
        required=True,
        help=(
            "Path to a GBIE-format JSON file containing predicted entities to be "
            "linked (typically the output of an entityRecognizer inference run)."
        ),
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        required=True,
        help="Path where the linked predictions JSON will be written.",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help=(
            "Optional path to a ground-truth GBIE JSON file.  When provided, "
            "strict (span + URI) and lenient (span-only) P/R/F1 are printed after "
            "inference."
        ),
    )

    # -- Embedding model configuration --
    parser.add_argument(
        "--embeddings_model_name",
        type=str,
        default="neuml/pubmedbert-base-embeddings",
        help=(
            "HuggingFace model identifier for the sentence-embedding model used by "
            "txtai (default: neuml/pubmedbert-base-embeddings)."
        ),
    )
    parser.add_argument(
        "--embeddings_index_path",
        type=str,
        default="embeddings_index",
        help=(
            "Directory where the txtai embedding index is saved / loaded "
            "(default: embeddings_index).  Reused across runs to avoid rebuilding."
        ),
    )
    parser.add_argument(
        "--similarity_top_k",
        type=int,
        default=10,
        help=(
            "Number of candidate definitions retrieved per entity during similarity "
            "matching (default: 10).  Only the top-ranked result is used for linking."
        ),
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum cosine-similarity score required to accept a similarity match "
            "(default: 0.0 -- accept all matches, consistent with the baseline "
            "notebook).  Increase to tighten precision at the cost of recall."
        ),
    )
    parser.add_argument(
        "--no_label_matching",
        action="store_true",
        help=(
            "Disable the (text_span, label) composite exact-match step and use "
            "text-only matching as the primary strategy.  Off by default."
        ),
    )

    args = parser.parse_args()

    # -- Validation --
    if not args.knowledge_base_dir or not os.path.isdir(args.knowledge_base_dir):
        if not args.training_data_paths:
            parser.error(
                "--training_data_paths is required when --knowledge_base_dir does "
                "not point to an existing directory."
            )
        if not args.uri_definitions_path:
            parser.error(
                "--uri_definitions_path is required when building a new knowledge base."
            )
        if not args.id_to_uri_path:
            parser.error(
                "--id_to_uri_path is required when building a new knowledge base."
            )

    return args


###############
# Entry point #
###############

if __name__ == "__main__":
    args = parse_args()
    run_build_and_inference(args)
