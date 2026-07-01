#!/usr/bin/env python3
"""
HFRelationExtractor.py

Transformer-based relation extraction module using entity-pair markers.
Organized into five classes:
    - RelationExtractionData: Dataset wrapping for relation extraction
    - BertForRelationExtraction: custom encoder + classification head
    - RelationExtractionTrainer: training loop and inference pipeline
    - RelationExtractionEvaluator: precision / recall / F1 evaluation
    - RelationTokenizer: entity-pair-marker preprocessing (imported from Tokenizers module)

RelationExtractionData inherits from BaseTermDataset and RelationExtractionTrainer inherits from BaseTrainer (both defined in HFTermExtractor), keeping data-loading and training consistent across tasks.

Architecture:
    The subject span is wrapped with [E1] / [/E1] and the object span with [E2] / [/E2].
    The hidden states at the two opening-marker positions are concatenated to form a 2 * hidden_size relation representation.
    Representation is fed to a linear classification head that predicts the predicate (or NA for no relation).

    At inference time, a type-constraint mask derived from VALID_RELATIONS is applied to the logits before argmax so that structurally impossible predicates are never predicted.

Entry point: run with --help to see CLI options.
"""

import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.relationExtractor.losses import ATLoss

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import utilities implemented in utils.py
from src.utils.utils import (
    VALID_RELATIONS,
    get_device,
    load_json_data,
    load_merge_json_data,
    maybe_empty_cache,
    move_to_device,
    print_device_info,
    save_json_data,
    seed_everything,
)

# Import shared base classes from the term extraction module
from src.termExtractor.HFTermExtractor import BaseTermDataset, BaseTrainer

# Import the relation tokenizer
from src.termExtractor.Tokenizers import RelationTokenizer  # noqa: F401  (re-exported for callers)


#############
# Constants #
#############

from src.utils.utils import RELATION_LABEL_LIST  # noqa: F401  (re-exported for callers)

# Special tokens used to mark the subject (E1) and object (E2) entity boundaries
SPECIAL_TOKENS = {
    "additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]
}


##########################
# RelationExtractionData #
##########################

class RelationExtractionData(BaseTermDataset):
    """
    Wraps a list of pre-tokenized entity-pair-marker samples as a PyTorch Dataset.

    Each sample in 'data' must be a dictionary with the keys:
        - 'input_ids'      (torch.LongTensor, 1-D)
        - 'attention_mask' (torch.LongTensor, 1-D)
        - 'labels'         (int, scalar predicate class index)

    These are produced by RelationTokenizer.process_files().

    Overrides collate_fn to handle the scalar integer label representation: labels are
    collected with torch.tensor rather than torch.stack.
    """

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Collates a list of samples into a single batched dictionary.
        Input IDs and attention masks are stacked; labels are wrapped as a 1-D long tensor
        since they are stored as plain integers.

        :param batch: A list of sample dictionaries.
        :return: A dictionary with 'input_ids' and 'attention_mask' stacked along dim 0,
                 and 'labels' as a 1-D LongTensor.
        """
        return {
            "input_ids":      torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "labels":         torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        }

#############################
# MMForRelationExtraction #
#############################

class MMForRelationExtraction(nn.Module):
    """
    Custom sequence-classification model for biomedical relation extraction.

    Architecture:
        - A pre-trained transformer encoder (AutoModel) with [E1] / [/E1] / [E2] / [/E2]
          special tokens added to its vocabulary.
        - A linear classification head applied to the concatenation of the hidden states
          at the [E1] and [E2] marker positions, which represent the subject and object
          entities respectively. The concatenated vector has size 2 * hidden_size.

    The tokenizer is stored as an attribute of the model so that both training and
    inference share the same vocabulary.
    """

    def __init__(self, model_name: str, num_labels: int):
        """
        Initializes the encoder, resizes its embedding matrix for the four new special tokens, and attaches the classification head.

        :param model_name: HuggingFace model identifier or local checkpoint path.
        :param num_labels: Number of output classes (length of RELATION_LABEL_LIST).
        """
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Tokenizer is kept on the model for convenient access during inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        # Resize embeddings to accommodate the four new marker tokens
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        self.num_labels = num_labels

        # The head operates on the concatenation of two hidden states
        self.classifier = nn.Linear(2 * self.hidden_size, self.num_labels)

        self.loss_fnct = ATLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None,) -> dict:
        """
        Runs a forward pass and optionally computes the cross-entropy loss.

        The representations at the [E1] and [E2] token positions are extracted from the encoder's last hidden state, concatenated, and fed to the classification head.

        :param input_ids: Batch of token ID sequences (B × L).
        :param attention_mask: Batch of attention masks (B × L).
        :param labels: Optional batch of integer predicate class labels (B,). When provided, the cross-entropy loss is computed.
        :return: A dict with 'logits' (B × num_labels) and optionally 'loss'.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        """print(type(outputs))
        print(outputs)
        raise ValueError()"""
        sequence_output = outputs.last_hidden_state

        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)

        # Locate [E1] in each sequence and extract its representation
        e1_mask = (input_ids == self.e1_token_id)
        e1_position = e1_mask.float().argmax(dim=1)
        e1_repr = sequence_output[batch_indices, e1_position]

        # Locate [E2] in each sequence and extract its representation
        e2_mask = (input_ids == self.e2_token_id)
        e2_position = e2_mask.float().argmax(dim=1)
        e2_repr = sequence_output[batch_indices, e2_position]

        # Concatenate subject and object representations to form the pair embedding
        pair_repr = torch.cat([e1_repr, e2_repr], dim=-1)  # (B, 2 * hidden_size)
        # logits = self.classifier(pair_repr)
        logits = self.loss_fnct.get_label(pair_repr, num_labels=self.num_labels)
        # print(f"logits shape: {logits.shape}")
        # print(f"labels shape: {labels.shape}")

        loss = None
        if labels is not None:
            # loss_fnct = nn.CrossEntropyLoss()
            # Format labels
            labels = [torch.tensor([label]) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            # print(f"labels shape: {labels.shape}")
            loss = self.loss_fnct(logits.float(), labels.float())

        return {"loss": loss, "logits": logits}

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, type_mask: torch.Tensor = None,) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs inference for a single batch and returns the predicted predicate indices alongside their confidence scores. 
        An optional type_mask can be added to the logits before argmax to enforce schema constraints.

        :param input_ids: Batch of token ID sequences (B × L).
        :param attention_mask: Batch of attention masks (B × L).
        :param type_mask: Optional additive mask tensor of shape (num_labels,) with 0.0 for valid predicates and -inf for invalid ones. Applied before the softmax so that invalid classes are effectively excluded.
        :return: A tuple of (predictions, confscores), both 1-D tensors of shape (B,).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            if type_mask is not None:
                logits = logits + type_mask.unsqueeze(0)

            probabilities = torch.softmax(logits, dim=-1)
            confscores, predictions = torch.max(probabilities, dim=-1)
        return predictions, confscores

    def save(self, output_dir: str) -> None:
        """
        Saves the model weights and tokenizer to 'output_dir'.

        :param output_dir: Directory path for the checkpoint.
        :return: None
        """
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(output_dir)


#############################
# BertForRelationExtraction #
#############################

class BertForRelationExtraction(nn.Module):
    """
    Custom sequence-classification model for biomedical relation extraction.

    Architecture:
        - A pre-trained transformer encoder (AutoModel) with [E1] / [/E1] / [E2] / [/E2]
          special tokens added to its vocabulary.
        - A linear classification head applied to the concatenation of the hidden states
          at the [E1] and [E2] marker positions, which represent the subject and object
          entities respectively. The concatenated vector has size 2 * hidden_size.

    The tokenizer is stored as an attribute of the model so that both training and
    inference share the same vocabulary.
    """

    def __init__(self, model_name: str, num_labels: int):
        """
        Initializes the encoder, resizes its embedding matrix for the four new special tokens, and attaches the classification head.

        :param model_name: HuggingFace model identifier or local checkpoint path.
        :param num_labels: Number of output classes (length of RELATION_LABEL_LIST).
        """
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Tokenizer is kept on the model for convenient access during inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        # Resize embeddings to accommodate the four new marker tokens
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        # The head operates on the concatenation of two hidden states
        self.classifier = nn.Linear(2 * self.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None,) -> dict:
        """
        Runs a forward pass and optionally computes the cross-entropy loss.

        The representations at the [E1] and [E2] token positions are extracted from the encoder's last hidden state, concatenated, and fed to the classification head.

        :param input_ids: Batch of token ID sequences (B × L).
        :param attention_mask: Batch of attention masks (B × L).
        :param labels: Optional batch of integer predicate class labels (B,). When provided, the cross-entropy loss is computed.
        :return: A dict with 'logits' (B × num_labels) and optionally 'loss'.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)

        # Locate [E1] in each sequence and extract its representation
        e1_mask = (input_ids == self.e1_token_id)
        e1_position = e1_mask.float().argmax(dim=1)
        e1_repr = sequence_output[batch_indices, e1_position]

        # Locate [E2] in each sequence and extract its representation
        e2_mask = (input_ids == self.e2_token_id)
        e2_position = e2_mask.float().argmax(dim=1)
        e2_repr = sequence_output[batch_indices, e2_position]

        # Concatenate subject and object representations to form the pair embedding
        pair_repr = torch.cat([e1_repr, e2_repr], dim=-1)  # (B, 2 * hidden_size)
        logits = self.classifier(pair_repr)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, type_mask: torch.Tensor = None,) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs inference for a single batch and returns the predicted predicate indices alongside their confidence scores. 
        An optional type_mask can be added to the logits before argmax to enforce schema constraints.

        :param input_ids: Batch of token ID sequences (B × L).
        :param attention_mask: Batch of attention masks (B × L).
        :param type_mask: Optional additive mask tensor of shape (num_labels,) with 0.0 for valid predicates and -inf for invalid ones. Applied before the softmax so that invalid classes are effectively excluded.
        :return: A tuple of (predictions, confscores), both 1-D tensors of shape (B,).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            if type_mask is not None:
                logits = logits + type_mask.unsqueeze(0)

            probabilities = torch.softmax(logits, dim=-1)
            confscores, predictions = torch.max(probabilities, dim=-1)
        return predictions, confscores

    def save(self, output_dir: str) -> None:
        """
        Saves the model weights and tokenizer to 'output_dir'.

        :param output_dir: Directory path for the checkpoint.
        :return: None
        """
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(output_dir)


################################
# RelationExtractionTrainer    #
################################

class RelationExtractionTrainer(BaseTrainer):
    """
    Handles both the training loop and the inference pipeline for the transformer-based relation extraction model.

    Training:
        - Fine-tunes a BertForRelationExtraction with AdamW.
        - Evaluates on a raw development data dict after every epoch using full inference followed by precision / recall / F1 computation.
        - Saves the best checkpoint based on macro F1 score.
        - Supports automatic mixed precision (AMP) on CUDA.

    Inference:
        - For each paper, enumerates all ordered entity pairs within the same section.
        - Skips pairs whose (head_label, tail_label) combination is absent from VALID_RELATIONS to avoid predicting structurally impossible relations.
        - Inserts [E1] / [/E1] and [E2] / [/E2] markers, tokenizes the marked text,
          and runs a forward pass with a type-constraint mask applied to the logits.
        - If the predicted predicate is not NA, the relation is recorded in the output.
    """

    def __init__(self, model: BertForRelationExtraction, tokenizer, device: torch.device, max_length: int = 512, id2label: dict[int, str] = None, label2id: dict[str, int] = None,):
        """
        Initializes the trainer with a model, tokenizer, target device, sequence length cap, and label mappings required for inference-time type masking.

        :param model: A BertForRelationExtraction instance.
        :param tokenizer: The tokenizer stored on the model (must include all four markers).
        :param device: The torch.device to run computations on.
        :param max_length: Maximum token sequence length (default 512).
        :param id2label: Dictionary mapping integer class indices to predicate strings; required for inference.
        :param label2id: Dictionary mapping predicate strings to integer class indices; required for inference-time type masking.
        """
        super().__init__(model, tokenizer, device, max_length)
        self.id2label = id2label or {}
        self.label2id = label2id or {}

    # -- BaseTrainer abstract method implementations --

    def _forward_pass(self, batch: dict) -> torch.Tensor:
        """
        Runs a forward pass through BertForRelationExtraction and returns the cross-entropy loss computed on the concatenated entity-pair representation.

        :param batch: A dict with 'input_ids', 'attention_mask', and 'labels' tensors.
        :return: A scalar loss tensor.
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs["loss"]

    def _evaluate_on_dev(self, dev_data: dict, evaluator) -> dict:
        """
        Runs full inference over the raw development data dict and computes precision, recall, and macro F1 via the provided evaluator.

        :param dev_data: Raw development data dict (paper ID -> content).
        :param evaluator: A RelationExtractionEvaluator instance.
        :return: A dict with 'score' (macro F1) and 'log_str' (P/R/F1 summary).
        """
        dev_predictions = self.perform_inference(dev_data)
        metrics = evaluator.evaluate(dev_predictions, dev_data)
        return {
            "score": metrics["f1"],
            "log_str": (
                f"P: {metrics['precision']:.4f} | "
                f"R: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            ),
        }

    def _save_checkpoint(self, output_dir: str) -> None:
        """
        Saves the model weights and tokenizer to 'output_dir'.

        :param output_dir: Directory path for the checkpoint.
        :return: None
        """
        self.model.save(output_dir)

    # -- Inference --

    def perform_inference(self, data: dict) -> dict:
        """
        Runs the relation extraction inference pipeline over an entire dataset.

        For each paper, all ordered entity pairs within the same section whose (head_label, tail_label) combination is listed in VALID_RELATIONS are evaluated.
        A schema-constrained type mask is applied to the logits before argmax so that only structurally valid predicates (plus NA) can be predicted. Pairs predicted as NA are not included in the output relations list.

        :param data: A dict mapping paper IDs to content dicts (GBIE format). Each content dict must have an 'entities' list with 'start_idx', 'end_idx', 'location', 'text_span', and 'label' fields.
        :return: A dict with the same structure as 'data' but with a 'relations' list added to every content dict, populated with predicted relation dicts.
        """
        self.model.eval()
        result = {}

        for paper_id, content in tqdm(data.items(), total=len(data), desc="Inference"):
            predicted_relations = []

            for section in ["title", "abstract"]:
                text = content["metadata"][section]
                section_entities = [
                    entity for entity in content.get("entities", [])
                    if entity["location"] == section
                ]

                for subj_entity in section_entities:
                    for obj_entity in section_entities:
                        if subj_entity is obj_entity:
                            continue

                        # Skip type pairs that have no valid predicates in the schema
                        valid_predicates = VALID_RELATIONS.get(
                            (subj_entity["label"], obj_entity["label"]), set()
                        )
                        if not valid_predicates:
                            continue

                        # Skip overlapping spans (should not occur in clean data)
                        subj_end_excl = subj_entity["end_idx"] + 1
                        obj_end_excl  = obj_entity["end_idx"] + 1
                        if not (subj_end_excl <= obj_entity["start_idx"]
                                or obj_end_excl <= subj_entity["start_idx"]):
                            continue

                        marked_text = RelationTokenizer._insert_relation_markers(
                            text,
                            subj_entity["start_idx"],
                            subj_end_excl,
                            obj_entity["start_idx"],
                            obj_end_excl,
                        )

                        encoding = self.tokenizer(
                            marked_text,
                            return_attention_mask=True,
                            truncation=True,
                            padding="max_length",
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                        input_ids      = encoding["input_ids"].to(self.device)
                        attention_mask = encoding["attention_mask"].to(self.device)

                        # Safety check: skip if either marker was truncated
                        ids_list = input_ids[0].tolist()
                        if (self.model.e1_token_id not in ids_list
                                or self.model.e2_token_id not in ids_list):
                            continue

                        type_mask = self._build_type_mask(
                            subj_entity["label"], obj_entity["label"]
                        )
                        predictions, confscores = self.model.predict(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            type_mask=type_mask,
                        )

                        predicted_label = self.id2label[predictions.item()]
                        if predicted_label == "NA":
                            continue

                        predicted_relations.append({
                            "subject_start_idx":  subj_entity["start_idx"],
                            "subject_end_idx":    subj_entity["end_idx"],
                            "subject_location":   section,
                            "subject_text_span":  subj_entity["text_span"],
                            "subject_label":      subj_entity["label"],
                            "subject_uri":        subj_entity.get("uri", ""),
                            "predicate":          predicted_label,
                            "confscore":          round(confscores.item(), 6),
                            "object_start_idx":   obj_entity["start_idx"],
                            "object_end_idx":     obj_entity["end_idx"],
                            "object_location":    section,
                            "object_text_span":   obj_entity["text_span"],
                            "object_label":       obj_entity["label"],
                            "object_uri":         obj_entity.get("uri", ""),
                        })

            # Shallow copy of the content dict and overwrite relations field
            output_content = dict(content)
            output_content["relations"] = predicted_relations
            result[paper_id] = output_content

        return result

    def _build_type_mask(self, head_label: str, tail_label: str) -> torch.Tensor:
        """
        Builds an additive logit mask that restricts predictions to the predicates that are valid for the given (head_label, tail_label) type pair, plus NA.

        Valid entries receive 0.0 (no change to the logit), invalid entries receive -inf (effectively zero probability after softmax).

        :param head_label: Entity category string of the subject / head entity.
        :param tail_label: Entity category string of the object / tail entity.
        :return: A 1-D float tensor of shape (num_labels,) on self.device.
        """
        num_labels = len(self.label2id)
        mask = torch.full((num_labels,), float("-inf"), device=self.device)

        # NA is always a valid prediction -- the model may decide no relation exists
        mask[self.label2id["NA"]] = 0.0

        for predicate in VALID_RELATIONS.get((head_label, tail_label), set()):
            if predicate in self.label2id:
                mask[self.label2id[predicate]] = 0.0

        return mask


################################
# RelationExtractionEvaluator  #
################################

class RelationExtractionEvaluator:
    """
    Evaluates relation extraction quality by comparing predicted relations to gold annotations at the character-span level.

    A predicted relation is counted as a true positive only when the full 7-tuple (subject_start_idx, subject_end_idx, subject_location, predicate, object_start_idx, object_end_idx, object_location)
    exactly matches a gold relation in the same paper. Precision, recall, and F1 are computed in the standard micro-averaged fashion across all papers.
    """

    @staticmethod
    def _relation_key(relation: dict) -> tuple:
        """
        Extracts a hashable key from a relation dict for set-based comparison.

        :param relation: A relation dict with character-span and predicate fields.
        :return: A 7-tuple uniquely identifying the relation within its document.
        """
        return (
            relation["subject_start_idx"],
            relation["subject_end_idx"],
            relation["subject_location"],
            relation["predicate"],
            relation["object_start_idx"],
            relation["object_end_idx"],
            relation["object_location"],
        )

    def evaluate(self, predictions: dict, gold: dict) -> dict:
        """
        Computes micro-averaged precision, recall, and F1 across all papers.

        :param predictions: A dict mapping paper IDs to content dicts with a 'relations' list, as returned by RelationExtractionTrainer.perform_inference.
        :param gold: A dict mapping paper IDs to gold content dicts, also with a 'relations' list.
        :return: A dict with keys 'precision', 'recall', and 'f1' (all floats).
        """
        tp = fp = fn = 0

        for paper_id, gold_content in gold.items():
            gold_keys = {
                self._relation_key(r)
                for r in gold_content.get("relations", [])
            }
            pred_keys = {
                self._relation_key(r)
                for r in predictions.get(paper_id, {}).get("relations", [])
            }

            tp += len(gold_keys & pred_keys)
            fp += len(pred_keys - gold_keys)
            fn += len(gold_keys - pred_keys)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}


####################
# Metadata helpers #
####################

def save_metadata(
    output_dir: str,
    label2id: dict,
    id2label: dict,
    args: argparse.Namespace,
) -> None:
    """
    Persists label mapping and training arguments alongside the model checkpoint.

    :param output_dir: Directory where the metadata JSON files will be written.
    :param label2id: Dictionary mapping predicate strings to integer IDs.
    :param id2label: Dictionary mapping integer IDs to predicate strings.
    :param args: Parsed argument namespace from argparse.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full label mapping so inference can reconstruct id2label
    save_json_data(
        {
            "labels": RELATION_LABEL_LIST,
            "label2id": label2id,
            "id2label": {str(key): value for key, value in id2label.items()},
        },
        os.path.join(output_dir, "label_mapping.json"),
    )

    # Save all training hyper-parameters for reproducibility
    save_json_data(vars(args), os.path.join(output_dir, "training_args.json"))


def load_trained_model(model_dir: str) -> tuple:
    """
    Loads a trained BertForRelationExtraction model from a saved checkpoint directory.

    :param model_dir: Path to the checkpoint directory produced by save_metadata and BertForRelationExtraction.save.
    :return: A tuple of (model, id2label, training_args) where model is in eval mode, id2label maps integer IDs to predicate strings, and training_args is the original argparse namespace as a plain dict.
    """
    label_mapping  = load_json_data(os.path.join(model_dir, "label_mapping.json"))
    training_args  = load_json_data(os.path.join(model_dir, "training_args.json"))

    labels   = label_mapping["labels"]
    id2label = {int(key): value for key, value in label_mapping["id2label"].items()}
    label2id = label_mapping["label2id"]

    model = BertForRelationExtraction(
        model_name=training_args["model_name"],
        num_labels=len(labels),
    )
    # Overwrite the freshly initialised tokenizer with the saved one to preserve any vocabulary changes (e.g. added special tokens) from the original run
    model.tokenizer  = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model.e1_token_id = model.tokenizer.convert_tokens_to_ids("[E1]")
    model.e2_token_id = model.tokenizer.convert_tokens_to_ids("[E2]")

    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, id2label, label2id, training_args


#############################
# Top-level entry functions #
#############################

def run_training(args: argparse.Namespace) -> None:
    """
    Orchestrates data loading, model initialization, training, and (optional) development evaluation for a full training run.

    :param args: Parsed CLI arguments.
    :return: None
    """
    seed_everything(args.seed)

    device = get_device()
    print_device_info(device)

    # Build predicate label ↔ id mappings from the global label list
    label2id = {label: i for i, label in enumerate(RELATION_LABEL_LIST)}
    id2label = {i: label for label, i in label2id.items()}

    # -- Load raw data --
    train_data = load_merge_json_data(args.train_data_paths)
    dev_data   = load_json_data(args.dev_data_path) if args.dev_data_path else None

    # -- Initialise model (tokenizer lives on the model) --
    """model     = BertForRelationExtraction(
        model_name=args.model_name, num_labels=len(RELATION_LABEL_LIST)
    )"""
    model     = MMForRelationExtraction(
        model_name=args.model_name, num_labels=len(RELATION_LABEL_LIST)
    )
    tokenizer = model.tokenizer

    # -- Preprocess training data --
    train_tokenizer = RelationTokenizer(
        data=train_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        negative_sample_multiplier=args.negative_sample_multiplier,
        concatenate_title_abstract=args.concatenate_title_abstract,
    )
    train_samples = train_tokenizer.process_files()
    train_dataset = RelationExtractionData(train_samples)
    train_loader  = train_dataset.build_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )

    print(f"Training samples: {len(train_samples)}")

    # -- Build trainer and evaluator --
    trainer   = RelationExtractionTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        id2label=id2label,
        label2id=label2id,
    )
    evaluator = RelationExtractionEvaluator() if dev_data is not None else None

    # -- Train --
    # dev_data is passed as a raw dict; _evaluate_on_dev calls perform_inference internally so that development metrics reflect true span-level P/R/F1
    trainer.train(
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        dev_data=dev_data,
        use_amp=args.use_amp,
        evaluator=evaluator,
    )

    # Persist label mapping and hyper-parameters alongside the checkpoint
    save_metadata(args.output_dir, label2id, id2label, args)

    maybe_empty_cache(device)


def run_inference(args: argparse.Namespace) -> None:
    """
    Loads a saved checkpoint and runs inference over a JSON dataset, writing results to the specified output path.

    :param args: Parsed CLI arguments.
    :return: None
    """
    device = get_device()
    print_device_info(device)

    model, id2label, label2id, training_args = load_trained_model(args.model_path)
    model.to(device)
    model.eval()

    # CLI override takes priority over the saved value
    max_length = args.max_length if args.max_length else training_args.get("max_length", 512)

    # -- Run inference --
    data    = load_json_data(args.inference_data_path)
    trainer = RelationExtractionTrainer(
        model=model,
        tokenizer=model.tokenizer,
        device=device,
        max_length=max_length,
        id2label=id2label,
        label2id=label2id,
    )
    results = trainer.perform_inference(data)

    save_json_data(results, args.inference_output_path)
    print(f"Inference results saved to {args.inference_output_path}")

    maybe_empty_cache(device)


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for both training and inference modes.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train or run inference for a transformer-based biomedical relation extractor"
    )

    # -- Mode --
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Run inference only instead of training",
    )

    # -- Data paths --
    parser.add_argument(
        "--train_data_paths",
        type=str,
        nargs="+",
        default=None,
        help="One or more paths to training JSON files (merged automatically)",
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default=None,
        help="Path to development set JSON file",
    )
    parser.add_argument(
        "--inference_data_path",
        type=str,
        default=None,
        help="Path to JSON file used during inference",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        default=None,
        help="Path where inference results JSON will be written",
    )

    # -- Model paths --
    parser.add_argument(
        "--model_name",
        type=str,
        default="michiyasunaga/BioLinkBERT-base",
        help="HuggingFace model name or local path used to initialise the model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where the trained model checkpoint will be saved",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved model checkpoint for inference",
    )

    # -- Training hyper-parameters --
    parser.add_argument("--batch_size",     type=int,   default=16,   help="Training batch size")
    parser.add_argument("--num_epochs",     type=int,   default=1,    help="Number of training epochs")
    parser.add_argument("--learning_rate",  type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token sequence length (for both training and inference)",
    )
    parser.add_argument(
        "--negative_sample_multiplier",
        type=int,
        default=1,
        help="How many negative (NA) pair samples to generate per positive relation in the training set",
    )
    parser.add_argument("--seed",        type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes (use 0 or 2 on macOS)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision training",
    )
    parser.add_argument(
        "--concatenate_title_abstract",
        action="store_true",
        help="Concatenate title and abstract to enable cross-section relation extraction",
    )

    args = parser.parse_args()

    # -- Validation --
    if args.inference_only:
        if not args.model_path:
            parser.error("--model_path is required when --inference_only is used")
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