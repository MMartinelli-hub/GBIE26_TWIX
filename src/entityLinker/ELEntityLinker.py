#!/usr/bin/env python3
"""
ELEntityLinker.py

Entity linking module that maps entity mentions (identified by character offsets) to standardized ontology URIs using the entity-linkings library from NAIST-NLP.

Organized into four classes:
    - DataConverter : bidirectional conversion between GBIE and entity-linkings formats
    - EntityLinkingData : PyTorch Dataset wrapper for entity-linkings format data
    - ELEntityLinker : main interface for inference, training, and evaluation
    - EntityLinkingEvaluator : strict-URI and span-only precision / recall / F1 evaluation

Design overview
---------------
The entity-linkings library exposes two complementary model families that can be used alone or combined in a two-stage pipeline:

    Retrievers  (candidate_retriever subpackage)
        BM25, PRIOR, TEXTEMBEDDING, E5BM25, DUALENCODER
        Accept a sentence + spans, return ranked candidate lists.

    Rerankers   (candidate_reranker subpackage)
        CROSSENCODER, CHATEL, FEVRY, EXTEND, FUSIONED
        Wrap a retriever and refine its top-k output to a single prediction per span.

ELEntityLinker accepts a retriever ID (required) and an optional reranker ID, mirrors the CLI of entity_linkings.cli.{train_retrieval,train_reranker,evaluate_pipeline}, but exposes a Python API consistent with the rest of this codebase (HFEntityRecognizer, HFRelationExtractor, etc.).

Training is supported only for models whose base class implements .train().  BM25 and PRIOR do not train; attempting to call train() on them raises NotImplementedError in the upstream library (as they have no trainable parameters).

Inference takes a GBIE-format data dict, converts each document into entity-linkings mention records, runs the retriever (+ optional reranker), and writes the predicted URIs back into a GBIE-format result dict.

Evaluation mirrors NERExtractionEvaluator (from HFEntityRecognizer) but compares predicted
URIs instead of predicted labels, at two levels:
    - Strict (span + URI)   : both character offsets and ontology URI must match.
    - Span-only (lenient)   : only character offsets are checked.

Entry point: run with --help to see CLI options.
"""

import argparse
import glob
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from tqdm import tqdm

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# entity-linkings library imports
import datasets as hf_datasets
from entity_linkings import (
    get_rerankers,
    get_retrievers,
    load_dictionary,
    get_retriever_ids,
    get_reranker_ids,
)
from entity_linkings.trainer import TrainingArguments
from entity_linkings.utils import read_yaml

# Shared project utilities
from src.utils.utils import (
    get_title_and_abstract,
    load_json_data,
    load_merge_json_data,
    save_json_data,
    seed_everything,
    print_device_info,
    get_device,
)


#############
# Constants #
#############

# Retriever IDs that do not support training (no gradient-based parameters)
_NON_TRAINABLE_RETRIEVERS = {"bm25", "prior"}

# Sentinel string written to the 'uri' field when the model predicts NIL
_NIL_URI = "-1"

# Key used in GBIE entity dicts for the ontology URI
_URI_KEY = "uri"


#################
# DataConverter #
#################

class DataConverter:
    """
    Provides static methods for bidirectional conversion between the GBIE annotation format used throughout this project and the JSONL format expected by the entity-linkings library.

    GBIE entity format (per entity in a paper's 'entities' list)
    -------------------------------------------------------
        {
            "start_idx":  <int>,   # char offset of first character in the mention
            "end_idx":    <int>,   # char offset of last character (inclusive)
            "location":   <str>,   # "title" | "abstract"
            "text_span":  <str>,   # verbatim mention string
            "label":      <str>,   # semantic category
            "uri":        <str>    # ontology URI (gold) or predicted URI (output)
        }

    entity-linkings dataset format (one record per document section)
    ----------------------------------------------------------------
        {
            "id":       <str>,       # unique record identifier  "<paper_id>-<section>"
            "text":     <str>,       # full section text (title or abstract)
            "entities": [            # list of mention dicts
                {
                    "start": <int>,  # char offset of first character (inclusive)
                    "end":   <int>,  # char offset past-the-end (exclusive, like Python slices)
                    "label": [<str>] # list containing the single gold ontology URI
                },
                ...
            ]
        }

    Offset convention
    -----------------
    GBIE uses inclusive end offsets (end_idx points at the last character of the span).
    entity-linkings uses exclusive end offsets (end is one past the last character, matching Python slice semantics).  
    The converter adds / subtracts 1 when crossing the boundary.
    """

    # -- GBIE -> entity-linkings --
    
    @staticmethod
    def gbie_to_el(gbie_data: dict) -> list[dict]:
        """
        Converts a GBIE-format data dict (paper_id -> content) to a list of
        entity-linkings records, one per (paper, section) pair.

        Papers with no entities are still included so that inference can be run
        on unannotated text; in that case the 'entities' list will be empty.

        :param gbie_data: Dict mapping paper IDs to GBIE content dicts.
        :return: List of entity-linkings format records.
        """
        records = []
        for paper_id, content in gbie_data.items():
            title, abstract = get_title_and_abstract(content)
            section_texts = {"title": title, "abstract": abstract}

            for section, text in section_texts.items():
                # Collect entities belonging to this section
                el_entities = []
                for ent in content.get("entities", []):
                    if ent.get("location") != section:
                        continue
                    # Convert GBIE inclusive end offset to exclusive (Python-slice) end
                    el_entities.append({
                        "start": ent["start_idx"],
                        "end":   ent["end_idx"] + 1,
                        "label": [ent[_URI_KEY]] if _URI_KEY in ent else [_NIL_URI],
                    })

                records.append({
                    "id":       f"{paper_id}-{section}",
                    "text":     text,
                    "entities": el_entities,
                })

        return records

    # -- entity-linkings predictions -> GBIE --

    @staticmethod
    def el_predictions_to_gbie(predictions: list[dict], gbie_data: dict,) -> dict:
        """
        Merges entity-linking predictions back into the original GBIE data structure.

        For each paper the method:
            1. Takes the gold entity list (which carries start_idx, end_idx, location, text_span, and label from an upstream NER step or the ground-truth data).
            2. Looks up the predicted URI for each gold span from the predictions index (keyed by (paper_id, section, start, end)).
            3. Writes the predicted URI into the 'uri' field of each entity dict.

        Entities whose span is not found in the predictions index (e.g. the retriever returned no candidates) receive the sentinel NIL URI.

        :param predictions: List of per-span prediction dicts produced by ELEntityLinker._run_predictions(). Each dict must have keys 'paper_id', 'section', 'start', 'end', and 'predicted_uri'.
        :param gbie_data: Original GBIE-format data dict used as the structural template.
        :return: GBIE-format result dict with 'uri' fields populated.
        """
        # Build a fast lookup: (paper_id, section, start_inclusive, end_inclusive) -> uri
        pred_index: dict[tuple, str] = {}
        for pred in predictions:
            key = (pred["paper_id"], pred["section"], pred["start"], pred["end"])
            pred_index[key] = pred["predicted_uri"]

        result = {}
        for paper_id, content in gbie_data.items():
            output_entities = []
            for ent in content.get("entities", []):
                ent_out = dict(ent)
                key = (
                    paper_id,
                    ent["location"],
                    ent["start_idx"],
                    ent["end_idx"],
                )
                ent_out[_URI_KEY] = pred_index.get(key, _NIL_URI)
                output_entities.append(ent_out)

            output_content = dict(content)
            output_content["entities"] = output_entities
            result[paper_id] = output_content

        return result

    # -- Helpers --
    
    @staticmethod
    def write_el_jsonl(records: list[dict], path: str) -> None:
        """
        Writes entity-linkings records to a JSONL file, one JSON object per line.

        :param records: List of entity-linkings format dicts.
        :param path: Destination file path.
        :return: None
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def load_el_dataset(path: str) -> hf_datasets.Dataset:
        """
        Loads a JSONL file written by write_el_jsonl() as a HuggingFace Dataset.

        :param path: Path to the JSONL file.
        :return: A HuggingFace Dataset with columns 'id', 'text', 'entities'.
        """
        return hf_datasets.load_dataset("json", data_files=path, split="train")


#####################
# EntityLinkingData #
#####################

class EntityLinkingData(Dataset):
    """
    Thin PyTorch Dataset wrapper around a list of entity-linkings format records.

    Each item is a plain dict with at least the keys 'id', 'text', and 'entities'.
    This class is mainly provided for API consistency with EntityRecognitionData and
    other Dataset wrappers in this codebase; the entity-linkings library itself works
    directly with HuggingFace Datasets rather than PyTorch Datasets.

    :param data: List of entity-linkings format record dicts.
    """

    def __init__(self, data: list[dict]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]


##################
# ELEntityLinker #
##################

@dataclass
class ELEntityLinkerConfig:
    """
    Typed configuration container for ELEntityLinker.

    Fields map directly to the YAML config keys used by the entity-linkings library.
    Default values match the recommended settings for the DUALENCODER retriever and CROSSENCODER reranker on a single mid-range GPU with a biomedical ontology of moderate size (tens of thousands of concepts).

    Retriever / reranker selection
    --------------------------------
    retriever_id : str
        One of: 'bm25', 'prior', 'textembedding', 'e5bm25', 'dualencoder'.
    reranker_id : str | None
        One of: 'crossencoder', 'chatel', 'fevry', 'extend', 'fusioned'.
        When None (default) the pipeline runs retrieval only.

    Model paths
    -----------
    retriever_model_name_or_path : str | None
        HuggingFace model hub ID or local directory for the retriever encoder.
        Overrides the 'model_name_or_path' key in the YAML config.
    reranker_model_name_or_path : str | None
        HuggingFace model hub ID or local directory for the reranker.

    Config files
    ------------
    retriever_config_path : str | None
        Path to a YAML config file whose top-level key matches retriever_id.
    reranker_config_path : str | None
        Path to a YAML config file whose top-level key matches reranker_id.

    Index
    -----
    retriever_index_dir : str | None
        Directory containing a pre-built FAISS (or BM25) index.
        When None the index is built from scratch at inference / training time.

    Training hyper-parameters
    -------------------------
    num_train_epochs : int
        Number of full passes over the training data.
    train_batch_size : int
        Per-device training batch size.
    eval_batch_size : int
        Per-device evaluation batch size.
    gradient_accumulation_steps : int
        Number of gradient accumulation steps before a parameter update.
    num_hard_negatives : int
        Number of hard-negative candidates to mine per positive during retriever training.
    num_candidates : int
        Number of retrieved candidates passed to the reranker during reranker training
        and pipeline evaluation.
    candidate_retrieval_batch_size : int | None
        Optional batch size used only when retrieving candidates for reranker
        training. When None, eval_batch_size is used.

    Misc
    ----
    cache_dir : str | None
        HuggingFace datasets cache directory.
    seed : int
        Global random seed for reproducibility.
    remove_nil : bool
        When True, NIL-labelled entities are filtered from training / evaluation sets
        before they are passed to the library.
    train_retriever : bool
        When True, train the configured retriever during train().
    train_reranker : bool
        When True, train the configured reranker during train().
    """

    retriever_id:                   str         = "dualencoder"
    reranker_id:                    Optional[str] = None

    retriever_model_name_or_path:   Optional[str] = None
    reranker_model_name_or_path:    Optional[str] = None
    retriever_config_path:          Optional[str] = None
    reranker_config_path:           Optional[str] = None
    retriever_index_dir:            Optional[str] = None

    num_train_epochs:               int   = 5
    train_batch_size:               int   = 8
    eval_batch_size:                int   = 8
    gradient_accumulation_steps:    int   = 1
    num_hard_negatives:             int   = 0
    num_candidates:                 int   = 30
    candidate_retrieval_batch_size: Optional[int] = None

    cache_dir:                      Optional[str] = None
    seed:                           int   = 42
    remove_nil:                     bool  = False
    train_retriever:                bool  = True
    train_reranker:                 bool  = False


class ELEntityLinker:
    """
    Main interface for entity linking using the entity-linkings library.

    Wraps the library's retriever (and optional reranker) in a single object that exposes train(), perform_inference(), and evaluate() methods consistent with the rest of the entityRecognizer / relationExtractor / termClassifier pipeline.

    Typical usage
    -------------
    Inference from a pre-trained checkpoint::

        config = ELEntityLinkerConfig(
            retriever_id="dualencoder",
            retriever_model_name_or_path="path/to/checkpoint",
            retriever_index_dir="path/to/index",
        )
        linker = ELEntityLinker(config)
        linker.from_pretrained(entity_dict_path="concepts.jsonl")

        gbie_data = load_json_data("data.json")
        results   = linker.perform_inference(gbie_data)

    Training a retriever from scratch::

        config = ELEntityLinkerConfig(
            retriever_id="dualencoder",
            retriever_model_name_or_path="google-bert/bert-base-uncased",
            num_train_epochs=5,
            train_batch_size=8,
        )
        linker = ELEntityLinker(config)
        linker.from_pretrained(entity_dict_path="concepts.jsonl")

        train_data = load_merge_json_data(["train1.json", "train2.json"])
        dev_data   = load_json_data("dev.json")
        linker.train(
            train_data=train_data,
            dev_data=dev_data,
            output_dir="outputs/el_model",
        )
    """

    def __init__(self, config: ELEntityLinkerConfig) -> None:
        """
        Stores the configuration.  
        The underlying model objects are not created until from_pretrained() is called so that the constructor remains lightweight.

        :param config: An ELEntityLinkerConfig instance.
        """
        self.config    = config
        self.retriever = None  # set by from_pretrained()
        self.reranker  = None  # set by from_pretrained(), only when config.reranker_id is set
        self.dictionary = None  # set by from_pretrained()

    # -- Initialization --
    
    def from_pretrained(self, entity_dict_path: str) -> None:
        """
        Loads the entity dictionary and instantiates the retriever (and optional reranker).

        When config.retriever_model_name_or_path points to a directory containing a saved checkpoint, the underlying DUALENCODER / TEXTEMBEDDING encoder is loaded from that directory automatically by the entity-linkings library.

        When config.retriever_index_dir is set, the pre-built FAISS / BM25 index is loaded from that directory, making the linker immediately ready for inference without rebuilding the index.

        :param entity_dict_path: Path to the entity dictionary JSONL file (concept definitions).
        :return: None
        """
        self.dictionary = load_dictionary(entity_dict_path, cache_dir=self.config.cache_dir)

        # -- Build retriever config dict --
        retriever_model_cfg = self._load_model_config(
            config_path=self.config.retriever_config_path,
            model_id=self.config.retriever_id,
            model_name_or_path_override=self.config.retriever_model_name_or_path,
        )

        retriever_cls       = get_retrievers(self.config.retriever_id)
        self.retriever      = retriever_cls(
            dictionary=self.dictionary,
            config=retriever_cls.Config(**retriever_model_cfg),
            index_path=self.config.retriever_index_dir,
        )

        # -- Build optional reranker --
        if self.config.reranker_id is not None:
            reranker_model_cfg = self._load_model_config(
                config_path=self.config.reranker_config_path,
                model_id=self.config.reranker_id,
                model_name_or_path_override=self.config.reranker_model_name_or_path,
            )
            reranker_cls  = get_rerankers(self.config.reranker_id)
            self.reranker = reranker_cls(
                retriever=self.retriever,
                config=reranker_cls.Config(**reranker_model_cfg),
            )

    # -- Training --                                                            
    
    def train(
        self,
        train_data:   dict,
        output_dir:   str,
        dev_data:     Optional[dict] = None,
        train_jsonl:  Optional[str]  = None,
        dev_jsonl:    Optional[str]  = None,
    ) -> None:
        """
        Runs the configured training loop for the retriever and/or reranker.

        For retrievers that do not have trainable parameters (BM25, PRIOR), this
        method raises NotImplementedError only when retriever training is enabled.

        Data conversion
        ---------------
        GBIE data dicts are converted to entity-linkings JSONL on the fly unless train_jsonl / dev_jsonl paths are explicitly supplied (in which case those pre-converted files are loaded directly, bypassing conversion).

        Retriever training
        ------------------
        When config.train_retriever is True, the retriever is trained first using
        TrainingArguments assembled from config.

        Reranker training
        -----------------
        When config.train_reranker is True, the reranker is trained using
        candidates retrieved by the configured retriever. The retriever can be a
        freshly trained model from the same call or a pre-trained model loaded by
        from_pretrained().

        :param train_data:  GBIE-format training data dict (paper_id -> content).
        :param output_dir:  Directory where the final model checkpoint(s) are saved.
        :param dev_data:    Optional GBIE-format development data dict.
        :param train_jsonl: Optional path to a pre-converted entity-linkings JSONL training file (skips automatic conversion when provided).
        :param dev_jsonl:   Optional path to a pre-converted entity-linkings JSONL development file.
        :return: None
        """
        if self.retriever is None:
            raise RuntimeError("Call from_pretrained() before train().")

        if not self.config.train_retriever and not self.config.train_reranker:
            raise ValueError("At least one of train_retriever or train_reranker must be True.")

        if self.config.train_reranker and self.reranker is None:
            raise ValueError("train_reranker=True requires config.reranker_id to be set.")

        if self.config.train_retriever and self.config.retriever_id in _NON_TRAINABLE_RETRIEVERS:
            raise NotImplementedError(
                f"Retriever '{self.config.retriever_id}' has no trainable parameters. "
                f"Choose one of the trainable retrievers: "
                + ", ".join(r for r in get_retriever_ids() if r not in _NON_TRAINABLE_RETRIEVERS)
            )

        seed_everything(self.config.seed)

        # -- Convert GBIE data to entity-linkings JSONL if needed --
        with tempfile.TemporaryDirectory() as tmpdir:
            if train_jsonl is None:
                train_jsonl = os.path.join(tmpdir, "train.jsonl")
                train_records = DataConverter.gbie_to_el(train_data)
                DataConverter.write_el_jsonl(train_records, train_jsonl)

            if dev_data is not None and dev_jsonl is None:
                dev_jsonl = os.path.join(tmpdir, "dev.jsonl")
                dev_records = DataConverter.gbie_to_el(dev_data)
                DataConverter.write_el_jsonl(dev_records, dev_jsonl)

            # Load HuggingFace datasets
            data_files = {"train": train_jsonl}
            if dev_jsonl is not None:
                data_files["validation"] = dev_jsonl
            hf_dataset = hf_datasets.load_dataset(
                "json",
                data_files=data_files,
                cache_dir=self.config.cache_dir,
            )

            if self.config.remove_nil:
                from entity_linkings.data_utils import filter_nil_entities
                hf_dataset["train"] = filter_nil_entities(hf_dataset["train"], self.dictionary)
                if "validation" in hf_dataset:
                    hf_dataset["validation"] = filter_nil_entities(hf_dataset["validation"], self.dictionary)

            # -- Optional retriever training --
            if self.config.train_retriever:
                retriever_output_dir = os.path.join(output_dir, "retriever")
                retriever_training_args = self._build_training_args(retriever_output_dir, hf_dataset)

                print(f"[ELEntityLinker] Training retriever '{self.config.retriever_id}' ...")
                self.retriever.train(
                    train_dataset=hf_dataset["train"],
                    eval_dataset=hf_dataset.get("validation"),
                    training_args=retriever_training_args,
                    num_hard_negatives=self.config.num_hard_negatives,
                )
            else:
                print(f"[ELEntityLinker] Skipping retriever training; using loaded '{self.config.retriever_id}'.")

            # -- Optional reranker training --
            if self.config.train_reranker:
                reranker_output_dir = os.path.join(output_dir, "reranker")
                reranker_training_args = self._build_training_args(reranker_output_dir, hf_dataset)
                if self.config.candidate_retrieval_batch_size is not None:
                    reranker_training_args.per_device_eval_batch_size = self.config.candidate_retrieval_batch_size

                print(f"[ELEntityLinker] Training reranker '{self.config.reranker_id}' ...")
                _configure_faiss_threads()
                self.reranker.train(
                    train_dataset=hf_dataset["train"],
                    eval_dataset=hf_dataset.get("validation"),
                    num_candidates=self.config.num_candidates,
                    training_args=reranker_training_args,
                )

    # -- Inference --
    
    def perform_inference(self, data: dict) -> dict:
        """
        Runs the entity linking pipeline over an entire GBIE-format dataset.

        For each paper the method:
            1. Extracts title and abstract texts.
            2. Collects all entity spans (start_idx, end_idx, location) from the gold / NER 'entities' list in the input data.
            3. Converts GBIE inclusive end offsets to exclusive end offsets for the library.
            4. Calls the underlying retriever (or reranker, if configured) to predict the best-matching ontology URI for each span.
            5. Writes the predicted URI back into the 'uri' field of each entity dict.

        Entities whose span receives a NIL prediction from the model are annotated with the sentinel value "-1" in the 'uri' field.

        :param data: GBIE-format data dict (paper_id -> content).  The 'entities' field of each paper must be populated (e.g. from a preceding NER step or from the gold annotations).
        :return: GBIE-format result dict with 'uri' fields populated for every entity.
        """
        if self.retriever is None:
            raise RuntimeError("Call from_pretrained() before perform_inference().")

        raw_predictions: list[dict] = []

        for paper_id, content in tqdm(data.items(), total=len(data), desc="Entity Linking"):
            title, abstract = get_title_and_abstract(content)
            section_texts = {"title": title, "abstract": abstract}

            for section, text in section_texts.items():
                # Collect spans for this section from the entity list
                section_entities = [
                    ent for ent in content.get("entities", [])
                    if ent.get("location") == section
                ]
                if not section_entities:
                    continue

                # Build (start, end_exclusive) span tuples for the library
                spans = [
                    (ent["start_idx"], ent["end_idx"] + 1)   # GBIE is inclusive → exclusive
                    for ent in section_entities
                ]

                try:
                    span_predictions = self._predict_spans(text, spans)
                except Exception as exc:  # noqa: BLE001
                    # If prediction fails for a section, assign NIL to all its spans
                    print(f"[ELEntityLinker] Warning: prediction failed for '{paper_id}/{section}': {exc}")
                    span_predictions = {span: _NIL_URI for span in spans}

                # Record each span's prediction using the GBIE-inclusive end offset as key
                for ent in section_entities:
                    span_key = (ent["start_idx"], ent["end_idx"] + 1)
                    raw_predictions.append({
                        "paper_id":      paper_id,
                        "section":       section,
                        "start":         ent["start_idx"],
                        "end":           ent["end_idx"],     # store back as inclusive
                        "predicted_uri": span_predictions.get(span_key, _NIL_URI),
                    })

        return DataConverter.el_predictions_to_gbie(raw_predictions, data)

    def _predict_spans(self, text: str, spans: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
        """
        Calls the retriever (or reranker) and returns a mapping from (start, end_exclusive) span tuples to predicted ontology URIs.

        For retrievers the library's predict() method returns a list-of-lists; we take the top-1 candidate from each inner list.
        For rerankers predict() returns a flat list with one prediction per span.

        :param text: The full section text.
        :param spans: List of (start, end_exclusive) char-offset tuples.
        :return: Dict mapping each input span to its predicted URI string.
        """
        from entity_linkings.candidate_retriever import RetrieverBase
        from entity_linkings.candidate_reranker import RerankerBase

        # Use the reranker if available, otherwise fall back to the retriever
        active_model = self.reranker if self.reranker is not None else self.retriever

        if isinstance(active_model, RerankerBase):
            # predict() returns one BaseSystemOutput per span
            preds = active_model.predict(
                sentence=text,
                spans=spans,
                num_candidates=self.config.num_candidates,
            )
            return {
                (pred.start, pred.end): pred.id
                for pred in preds
            }
        elif isinstance(active_model, RetrieverBase):
            # predict() returns a list of lists; take top-1 from each
            preds_list = active_model.predict(
                sentence=text,
                spans=spans,
                top_k=1,
            )
            result = {}
            for (start, end), candidates in zip(spans, preds_list):
                top_pred = candidates[0] if candidates else None
                result[(start, end)] = top_pred.id if top_pred is not None else _NIL_URI
            return result
        else:
            raise TypeError(
                f"Unexpected model type: {type(active_model)}.  "
                "Expected RetrieverBase or RerankerBase."
            )

    # -- Private helpers --
    
    @staticmethod
    def _load_model_config(
        config_path: Optional[str],
        model_id:    str,
        model_name_or_path_override: Optional[str],
    ) -> dict:
        """
        Reads the model sub-section from a YAML config file and optionally overrides the 'model_name_or_path' field.

        When config_path is None an empty dict is returned, relying entirely on the defaults embedded in the entity-linkings Config dataclasses.

        :param config_path: Path to a YAML file (may be None).
        :param model_id: Top-level key to extract (e.g. 'dualencoder').
        :param model_name_or_path_override: When not None, replaces the config value.
        :return: Model configuration dict.
        """
        cfg: dict = {}
        if config_path is not None:
            all_cfg = read_yaml(config_path)
            cfg = all_cfg.get(model_id.lower(), {})

        if model_name_or_path_override is not None:
            cfg["model_name_or_path"] = model_name_or_path_override

        return cfg

    def _build_training_args(
        self,
        output_dir: str,
        hf_dataset: hf_datasets.DatasetDict,
    ) -> TrainingArguments:
        """
        Assembles a TrainingArguments instance from this instance's config, adapting the eval / save strategy depending on whether a validation split is present.

        :param output_dir: Checkpoint output directory.
        :param hf_dataset: The dataset dict (used only to check for a 'validation' split).
        :return: A fully configured TrainingArguments instance.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        # If no validation split exists disable evaluation and checkpointing by epoch
        if "validation" not in hf_dataset:
            training_args.eval_strategy = "no"
            training_args.save_strategy = "no"

        return training_args


##########################
# EntityLinkingEvaluator #
##########################

class EntityLinkingEvaluator:
    """
    Evaluates entity linking quality by comparing predicted ontology URIs against ground-truth annotations at two levels of strictness.

    Strict (span + URI)
        A prediction is a true positive when both the character offsets AND the ontology URI match the gold annotation.  
        This is the primary metric because it measures the full task: locate the mention AND identify the correct concept.

    Span-only (lenient)
        A prediction is a true positive when only the character offsets match, regardless of the predicted URI.  
        This allows the entity linking quality to be compared directly with the upstream NER step (which also reports span F1).

    Precision, recall, and F1 are computed with binary averaging over the union of all paper IDs in the two input dicts.

    The evaluation mirrors NERExtractionEvaluator (from HFEntityRecognizer) and uses the same _compute_prf / _build_entity_sets internal structure to keep the codebase uniform.
    """

    def evaluate(self, predictions: dict, ground_truth: dict) -> dict:
        """
        Computes strict (span + URI) and lenient (span-only) P/R/F1 across all papers.

        :param predictions:  Dict mapping paper IDs to content with predicted 'uri' fields.
        :param ground_truth: Dict mapping paper IDs to content with gold 'uri' fields.
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
        y_true: list[int] = []
        y_pred: list[int] = []

        all_ids = set(gt_sets.keys()) | set(pred_sets.keys())
        for paper_id in all_ids:
            gt_ents   = gt_sets.get(paper_id, set())
            pred_ents = pred_sets.get(paper_id, set())
            all_ents  = gt_ents | pred_ents

            for entity in all_ents:
                y_true.append(1 if entity in gt_ents   else 0)
                y_pred.append(1 if entity in pred_ents else 0)

        if not y_true:
            return 0.0, 0.0, 0.0

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        return float(precision), float(recall), float(f1)

    @staticmethod
    def _build_entity_sets(data: dict, include_uri: bool) -> dict:
        """
        Converts entity lists to sets of comparable tuples for fast intersection / union operations.

        When include_uri is True the tuple is:
            (start_idx, end_idx, location, text_span, uri)
        When include_uri is False (span-only):
            (start_idx, end_idx, location, text_span)

        Entities without a 'uri' key are skipped (e.g. entities from an NER-only run that has not yet been linked) unless include_uri is False.

        :param data: Dict mapping paper IDs to content dicts with 'entities' lists.
        :param include_uri: Whether to include the URI in the comparison tuple.
        :return: Dict mapping paper IDs to sets of tuples.
        """
        entity_sets: dict = {}
        for paper_id, content in data.items():
            if include_uri:
                entity_sets[paper_id] = {
                    (
                        ent["start_idx"],
                        ent["end_idx"],
                        ent["location"],
                        ent["text_span"],
                        ent.get(_URI_KEY, _NIL_URI),
                    )
                    for ent in content.get("entities", [])
                    if _URI_KEY in ent
                }
            else:
                entity_sets[paper_id] = {
                    (
                        ent["start_idx"],
                        ent["end_idx"],
                        ent["location"],
                        ent["text_span"],
                    )
                    for ent in content.get("entities", [])
                }
        return entity_sets


####################
# Metadata helpers #
####################

def save_metadata(output_dir: str, args: argparse.Namespace) -> None:
    """
    Persists training arguments alongside the model checkpoint for reproducibility.

    :param output_dir: Directory where the metadata JSON file will be written.
    :param args: Parsed argument namespace from argparse.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)
    save_json_data(vars(args), os.path.join(output_dir, "training_args.json"))


def _configure_faiss_threads(num_threads: int = 1) -> None:
    """
    Limits native threading for candidate retrieval / inference.

    On macOS, FAISS + PyTorch/Transformers can occasionally terminate the
    process with a native segmentation fault or appear to hang before Python
    can raise an exception. Keeping native thread pools small is slower per
    operation but much more stable for these small per-span inference calls.
    """
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    try:
        import faiss  # type: ignore
    except ImportError:
        faiss = None

    if faiss is not None:
        try:
            faiss.omp_set_num_threads(num_threads)
        except AttributeError:
            pass

    try:
        import torch

        torch.set_num_threads(num_threads)
    except (ImportError, RuntimeError):
        pass


################
# Path helpers #
################

_RUN_NAME_MARKERS = ("retriever=", "reranker=", "retriever_model=", "reranker_model=")
_PATH_TOKEN_MAX_LEN = 80


def _sanitize_path_token(value: object) -> str:
    """
    Converts a model id / path fragment into a compact filesystem-safe token.
    """
    token = str(value).strip()
    token = token.replace("\\", "/").strip("/")
    token = token.replace("/", "__")
    token = re.sub(r"[^A-Za-z0-9._=+-]+", "-", token)
    token = re.sub(r"-{2,}", "-", token).strip("-._")
    if not token:
        token = "none"
    if len(token) > _PATH_TOKEN_MAX_LEN:
        token = token[:_PATH_TOKEN_MAX_LEN].rstrip("-._")
    return token


def _model_path_token(model_name_or_path: Optional[str]) -> Optional[str]:
    """
    Returns a readable token for a HF model id or local checkpoint path.

    Local trained checkpoints often end in '/retriever' or '/reranker'. In that
    case the parent directory is more informative than the component name.
    """
    if model_name_or_path is None:
        return None

    normalized = str(model_name_or_path).replace("\\", "/").rstrip("/")
    if not normalized:
        return None

    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return None

    component = parts[-1]
    parent = parts[-2] if len(parts) > 1 else parts[-1]
    if component == "retriever":
        marker_value = _extract_run_marker_value(parent, "retriever_model")
        if marker_value is not None:
            return marker_value
    if component == "reranker":
        marker_value = _extract_run_marker_value(parent, "reranker_model")
        if marker_value is not None:
            return marker_value

    if parts[-1] in {"retriever", "reranker", "retriever_index"} and len(parts) > 1:
        return _sanitize_path_token(parts[-2])

    return _sanitize_path_token(normalized)


def _extract_run_marker_value(run_name: str, marker: str) -> Optional[str]:
    """
    Extracts a marker value from an auto-generated run-name token.

    Example:
        marker='retriever_model' extracts 'google-bert__bert-base-uncased' from
        'retriever=textembedding_retriever_model=google-bert__bert-base-uncased'.
    """
    marker_prefix = f"{marker}="
    start = run_name.find(marker_prefix)
    if start < 0:
        return None

    value_start = start + len(marker_prefix)
    next_positions = [
        run_name.find(f"_{next_marker}=", value_start)
        for next_marker in ("retriever", "retriever_model", "reranker", "reranker_model")
    ]
    next_positions = [pos for pos in next_positions if pos >= 0]
    value_end = min(next_positions) if next_positions else len(run_name)
    value = run_name[value_start:value_end]
    return value or None


def _build_run_name(args: argparse.Namespace, include_model_names: bool = True) -> str:
    """
    Builds the shared run-name suffix used for entity-linker files/folders.
    """
    parts = [f"retriever={_sanitize_path_token(args.retriever_id)}"]

    if include_model_names:
        retriever_model = _model_path_token(getattr(args, "retriever_model_name_or_path", None))
        if retriever_model is not None:
            parts.append(f"retriever_model={retriever_model}")

    reranker_id = getattr(args, "reranker_id", None)
    if reranker_id is not None:
        parts.append(f"reranker={_sanitize_path_token(reranker_id)}")

        if include_model_names:
            reranker_model = _model_path_token(getattr(args, "reranker_model_name_or_path", None))
            if reranker_model is not None:
                parts.append(f"reranker_model={reranker_model}")

    return "_".join(parts)


def _path_has_run_name(path: Optional[str]) -> bool:
    if path is None:
        return False
    return any(marker in str(path) for marker in _RUN_NAME_MARKERS)


def _append_run_name(path: Optional[str], run_name: str) -> Optional[str]:
    """
    Appends run_name before a file extension or at the end of a directory path.

    Paths that already contain a generated run-name marker are returned unchanged
    so repeated calls do not keep growing the name.
    """
    if path is None or _path_has_run_name(path):
        return path

    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}_{run_name}{ext}"
    return f"{path}_{run_name}"


def _looks_like_hf_repo_id(value: Optional[str]) -> bool:
    """
    Heuristic for HuggingFace ids, used to avoid treating remote model ids as
    local base directories.
    """
    if value is None:
        return False

    text = str(value)
    if os.path.isabs(text) or text.startswith((".", "~")):
        return False
    if "\\" in text:
        return False
    parts = [part for part in text.split("/") if part]
    if len(parts) > 2:
        return False
    if parts and parts[0] in {"runs", "data", "scripts", "src", "outputs", "checkpoints"}:
        return False
    return bool(parts) and not text.endswith((".json", ".jsonl", ".yaml", ".yml"))


def _find_component_in_existing_run_dirs(
    base_path: str,
    args: argparse.Namespace,
    component_dir: str,
) -> Optional[str]:
    """
    Finds a saved component under an existing auto-named run directory.

    This supports configs that provide a shared base path for inference inputs,
    even when the training output suffix included model names that are not
    otherwise present in the inference config.
    """
    candidates = []
    retriever_marker = f"retriever={_sanitize_path_token(args.retriever_id)}"
    reranker_id = getattr(args, "reranker_id", None)
    reranker_marker = (
        f"reranker={_sanitize_path_token(reranker_id)}"
        if reranker_id is not None
        else None
    )
    if component_dir == "reranker" and reranker_marker is None:
        return None

    for run_dir in sorted(glob.glob(f"{base_path}_*")):
        run_name = os.path.basename(run_dir)
        if retriever_marker not in run_name:
            continue
        if component_dir == "reranker" and reranker_marker not in run_name:
            continue

        component_path = os.path.join(run_dir, component_dir)
        if os.path.exists(component_path):
            candidates.append(component_path)

    if not candidates:
        return None

    if len(candidates) > 1:
        print(
            "[ELEntityLinker] Warning: multiple auto-named input paths match "
            f"'{base_path}' for component '{component_dir}'. Using: {candidates[-1]}"
        )
    return candidates[-1]


def _resolve_component_input_path(
    path: Optional[str],
    args: argparse.Namespace,
    component_dir: str,
) -> Optional[str]:
    """
    Resolves local model/index inputs from either an explicit component path or
    a shared base directory.

    If the configured path already exists, or already ends with the component
    directory name, it is kept. Otherwise the function tries common derived
    locations such as '<base>/<component>' and '<base>_<run_name>/<component>'.
    """
    if path is None or _looks_like_hf_repo_id(path):
        return path

    normalized = path.rstrip("/\\")

    basename = os.path.basename(normalized)
    if basename == component_dir:
        return normalized

    component_candidate = os.path.join(normalized, component_dir)
    if os.path.exists(component_candidate):
        return component_candidate

    if _path_has_run_name(normalized):
        return normalized

    existing_run_component = _find_component_in_existing_run_dirs(
        normalized,
        args,
        component_dir,
    )
    if existing_run_component is not None:
        return existing_run_component

    run_name = _build_run_name(args, include_model_names=True)
    suffixed_base = _append_run_name(normalized, run_name)
    suffixed_component = os.path.join(suffixed_base, component_dir)
    if os.path.exists(suffixed_component):
        return suffixed_component
    if os.path.exists(suffixed_base):
        return suffixed_base

    if os.path.exists(normalized):
        return normalized

    return suffixed_component


#############################
# Top-level entry functions #
#############################

def run_training(args: argparse.Namespace) -> None:
    """
    Orchestrates data loading, model initialization, and training for a full entity linking training run.

    :param args: Parsed CLI arguments.
    :return: None
    """
    seed_everything(args.seed)

    device = get_device()
    print_device_info(device)
    _configure_faiss_threads()

    args.retriever_model_name_or_path = _resolve_component_input_path(
        args.retriever_model_name_or_path,
        args,
        component_dir="retriever",
    )
    if args.reranker_id is None:
        args.reranker_model_name_or_path = None
    else:
        args.reranker_model_name_or_path = _resolve_component_input_path(
            args.reranker_model_name_or_path,
            args,
            component_dir="reranker",
        )
    args.retriever_index_dir = _resolve_component_input_path(
        args.retriever_index_dir,
        args,
        component_dir="retriever_index",
    )

    run_name = _build_run_name(args, include_model_names=True)
    args.output_dir = _append_run_name(args.output_dir, run_name)
    print(f"[ELEntityLinker] Run name: {run_name}")
    print(f"[ELEntityLinker] Resolved output directory: {args.output_dir}")

    config = ELEntityLinkerConfig(
        retriever_id=args.retriever_id,
        reranker_id=args.reranker_id,
        retriever_model_name_or_path=args.retriever_model_name_or_path,
        reranker_model_name_or_path=args.reranker_model_name_or_path,
        retriever_config_path=args.retriever_config,
        reranker_config_path=args.reranker_config,
        retriever_index_dir=args.retriever_index_dir,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_hard_negatives=args.num_hard_negatives,
        num_candidates=args.num_candidates,
        candidate_retrieval_batch_size=getattr(args, "candidate_retrieval_batch_size", None),
        cache_dir=args.cache_dir,
        seed=args.seed,
        remove_nil=args.remove_nil,
        train_retriever=(
            args.train_retriever
            if getattr(args, "train_retriever", None) is not None
            else args.reranker_id is None
        ),
        train_reranker=(
            args.train_reranker
            if getattr(args, "train_reranker", None) is not None
            else args.reranker_id is not None
        ),
    )

    linker = ELEntityLinker(config)
    linker.from_pretrained(entity_dict_path=args.entity_dict_path)

    train_data = load_merge_json_data(args.train_data_paths)
    dev_data   = load_json_data(args.dev_data_path) if args.dev_data_path else None

    linker.train(
        train_data=train_data,
        output_dir=args.output_dir,
        dev_data=dev_data,
    )

    save_metadata(args.output_dir, args)
    print(f"[ELEntityLinker] Training complete.  Checkpoint saved to: {args.output_dir}")


def run_inference(args: argparse.Namespace) -> None:
    """
    Runs the entity linking inference pipeline and (optionally) evaluates against a ground-truth GBIE file.

    :param args: Parsed CLI arguments.
    :return: None
    """
    seed_everything(args.seed)

    device = get_device()
    print_device_info(device)
    _configure_faiss_threads()

    args.retriever_model_name_or_path = _resolve_component_input_path(
        args.retriever_model_name_or_path,
        args,
        component_dir="retriever",
    )
    if args.reranker_id is None:
        args.reranker_model_name_or_path = None
    else:
        args.reranker_model_name_or_path = _resolve_component_input_path(
            args.reranker_model_name_or_path,
            args,
            component_dir="reranker",
        )
    args.retriever_index_dir = _resolve_component_input_path(
        args.retriever_index_dir,
        args,
        component_dir="retriever_index",
    )

    run_name = _build_run_name(args, include_model_names=True)
    inference_output_path = _append_run_name(args.inference_output_path, run_name)
    print(f"[ELEntityLinker] Run name: {run_name}")
    print(f"[ELEntityLinker] Resolved inference output path: {inference_output_path}")

    config = ELEntityLinkerConfig(
        retriever_id=args.retriever_id,
        reranker_id=args.reranker_id,
        retriever_model_name_or_path=args.retriever_model_name_or_path,
        reranker_model_name_or_path=args.reranker_model_name_or_path,
        retriever_config_path=args.retriever_config,
        reranker_config_path=args.reranker_config,
        retriever_index_dir=args.retriever_index_dir,
        num_candidates=args.num_candidates,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    linker = ELEntityLinker(config)
    linker.from_pretrained(entity_dict_path=args.entity_dict_path)

    inference_data = load_json_data(args.inference_data_path)
    results        = linker.perform_inference(inference_data)

    save_json_data(results, inference_output_path)
    print(f"[ELEntityLinker] Inference complete.  Results saved to: {inference_output_path}")

    # Optional evaluation against a ground-truth file
    if args.eval_data_path:
        ground_truth = load_json_data(args.eval_data_path)
        evaluator    = EntityLinkingEvaluator()
        metrics      = evaluator.evaluate(results, ground_truth)

        print("\n[ELEntityLinker] Evaluation results:")
        print(
            f"  Strict  — P: {metrics['strict_precision']:.4f} | "
            f"R: {metrics['strict_recall']:.4f} | "
            f"F1: {metrics['strict_f1']:.4f}"
        )
        print(
            f"  Span    — P: {metrics['span_precision']:.4f} | "
            f"R: {metrics['span_recall']:.4f} | "
            f"F1: {metrics['span_f1']:.4f}"
        )

        if args.inference_output_path:
            metrics_path = os.path.join(
                os.path.dirname(inference_output_path),
                f"eval_results_{run_name}.json",
            )
            save_json_data(metrics, metrics_path)
            print(f"[ELEntityLinker] Metrics saved to: {metrics_path}")


#######
# CLI # 
#######

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the ELEntityLinker entry point.

    Supports two modes selected via --inference_only:
        Default (training mode)   : requires --train_data_paths, --output_dir, --entity_dict_path.
        Inference mode            : requires --inference_data_path, --inference_output_path, --entity_dict_path.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train or run inference with an entity linking model (entity-linkings library)"
    )

    # -- Mode --
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Run inference only instead of training",
    )

    # -- Entity dictionary --
    parser.add_argument(
        "--entity_dict_path",
        type=str,
        required=True,
        help="Path to the entity dictionary JSONL file (one concept per line)",
    )

    # -- Model selection --
    parser.add_argument(
        "--retriever_id",
        type=str,
        default="dualencoder",
        choices=get_retriever_ids(),
        help=(
            f"Retriever model type (default: dualencoder). "
            f"Available: {', '.join(get_retriever_ids())}"
        ),
    )
    parser.add_argument(
        "--reranker_id",
        type=str,
        default=None,
        choices=[None] + get_reranker_ids(),
        help=(
            f"Optional reranker model type (default: None = retrieval only). "
            f"Available: {', '.join(get_reranker_ids())}"
        ),
    )

    # -- Model paths --
    parser.add_argument(
        "--retriever_model_name_or_path",
        type=str,
        default=None,
        help="HuggingFace model hub ID or local path for the retriever encoder",
    )
    parser.add_argument(
        "--reranker_model_name_or_path",
        type=str,
        default=None,
        help="HuggingFace model hub ID or local path for the reranker",
    )
    parser.add_argument(
        "--retriever_index_dir",
        type=str,
        default=None,
        help="Directory containing a pre-built retriever index",
    )

    # -- Config files --
    parser.add_argument(
        "--retriever_config",
        type=str,
        default=None,
        help="Path to a YAML config file for the retriever",
    )
    parser.add_argument(
        "--reranker_config",
        type=str,
        default=None,
        help="Path to a YAML config file for the reranker",
    )

    # -- Data paths (training) --
    parser.add_argument(
        "--train_data_paths",
        type=str,
        nargs="+",
        default=None,
        help="One or more paths to GBIE-format training JSON files (merged automatically)",
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default=None,
        help="Path to GBIE-format development set JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where the final fine-tuned model checkpoint(s) will be saved",
    )

    # -- Data paths (inference) --
    parser.add_argument(
        "--inference_data_path",
        type=str,
        default=None,
        help="Path to the GBIE-format JSON file to run inference on",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        default=None,
        help="Path where the inference results JSON will be written",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help=(
            "Optional path to a ground-truth GBIE JSON file.  When provided during "
            "inference, strict (span + URI) and lenient (span-only) P/R/F1 are printed."
        ),
    )

    # -- Training hyper-parameters --
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Per-device training batch size (default: 8)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Per-device evaluation batch size (default: 8)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps before a parameter update (default: 1)",
    )
    parser.add_argument(
        "--num_hard_negatives",
        type=int,
        default=0,
        help="Hard negatives to mine per positive during retriever training (default: 0)",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=30,
        help="Top-k candidates passed to the reranker / returned by the retriever (default: 30)",
    )
    parser.add_argument(
        "--candidate_retrieval_batch_size",
        type=int,
        default=None,
        help=(
            "Optional batch size used only for candidate retrieval during "
            "reranker training. Defaults to eval_batch_size."
        ),
    )
    parser.add_argument(
        "--train_retriever",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to train the retriever in training mode. Defaults to True "
            "when no reranker is configured, otherwise False."
        ),
    )
    parser.add_argument(
        "--train_reranker",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to train the reranker in training mode. Defaults to True "
            "when reranker_id is configured, otherwise False."
        ),
    )
    parser.add_argument(
        "--remove_nil",
        action="store_true",
        default=False,
        help="Remove NIL-labelled entities from training and evaluation sets",
    )

    # -- Misc --
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42)",
    )

    args = parser.parse_args()

    # -- Validation --
    if args.inference_only:
        if not args.inference_data_path:
            parser.error("--inference_data_path is required when --inference_only is used")
        if not args.inference_output_path:
            parser.error("--inference_output_path is required when --inference_only is used")
    else:
        if not args.train_data_paths:
            parser.error("--train_data_paths is required for training")
        if not args.output_dir:
            parser.error("--output_dir is required for training")

    return args


###############
# Entry point #
###############

if __name__ == "__main__":
    args = parse_args()
    if args.inference_only:
        run_inference(args)
    else:
        run_training(args)
