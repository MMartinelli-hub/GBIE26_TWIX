#!/usr/bin/env python3
"""
HFTermClassifier.py

Transformer-based term classification module using entity markers.
Organized into five classes:
    - TermClassificationData : Dataset wrapping for term classification
    - BertForTermClassification : custom encoder + classification head
    - TermClassificationTrainer : training loop and inference pipeline
    - TermClassificationEvaluator : accuracy / F1 evaluation
    - TermClassificationTokenizer : entity-marker preprocessing (imported from Tokenizers module)

TermClassificationData inherits from BaseTermDataset and TermClassificationTrainer inherits from BaseTrainer (both defined in HFTermExtractor), keeping data-loading and training consistent across tasks.

Entry point: run with --help to see CLI options.
"""

import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import utilities implemented in utils.py
from src.utils.utils import (
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

# Import the classification tokenizer
from src.termExtractor.Tokenizers import TermClassificationTokenizer  # noqa: F401  (re-exported for callers)


#############
# Constants #
#############

from src.utils.utils import LABEL_LIST  # noqa: F401  (re-exported for callers)

# Special tokens used to mark entity boundaries in the input text
SPECIAL_TOKENS = {"additional_special_tokens": ["[E1]", "[/E1]"]}


##########################
# TermClassificationData #
##########################

class TermClassificationData(BaseTermDataset):
    """
    Wraps a list of pre-tokenized entity-marker samples as a PyTorch Dataset.

    Each sample in 'data' must be a dictionary with the keys:
        - 'input_ids' (torch.LongTensor, 1-D)
        - 'attention_mask' (torch.LongTensor, 1-D)
        - 'labels' (int, scalar class index)

    These are produced by TermClassificationTokenizer.process_files().

    Overrides collate_fn to handle the scalar integer label representation: labels are collected with torch.tensor rather than torch.stack.
    """

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Collates a list of samples into a single batched dictionary.
        Input IDs and attention masks are stacked; labels are wrapped as a 1-D long tensor since they are stored as plain integers.

        :param batch: A list of sample dictionaries.
        :return: A dictionary with 'input_ids' and 'attention_mask' stacked along dim 0, and 'labels' as a 1-D LongTensor.
        """
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        }


##############################
# BertForTermClassification  #
##############################

class BertForTermClassification(nn.Module):
    """
    Custom sequence-classification model for biomedical term categorisation.

    Architecture:
        - A pre-trained transformer encoder (AutoModel) with [E1] / [/E1] special tokens added to its vocabulary.
        - A linear classification head applied to the hidden state at the [E1] marker position, which represents the target entity.

    The tokenizer is stored as an attribute of the model so that both training and inference share the same vocabulary.
    """

    def __init__(self, model_name: str, num_labels: int, encoder_config=None, tokenizer_name: str = None):
        """
        Initializes the encoder, resizes its embedding matrix for the new special tokens, and attaches the classification head.

        :param model_name: HuggingFace model identifier or local checkpoint path.
        :param num_labels: Number of output classes (length of LABEL_LIST).
        :param encoder_config: Optional already-loaded encoder config. When provided, the encoder is initialized from the config and weights are expected to be loaded from a checkpoint state dict later.
        :param tokenizer_name: Optional tokenizer source. Defaults to model_name.
        """
        super().__init__()

        if encoder_config is None:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            self.encoder = AutoModel.from_config(encoder_config)
        self.hidden_size = self.encoder.config.hidden_size

        # Tokenizer is kept on the model for convenient access during inference
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=True)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        # Resize embeddings to accommodate the two new marker tokens
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None,) -> dict:
        """
        Runs a forward pass and optionally computes the cross-entropy loss.

        The representation at the [E1] token position is extracted from the encoder's last hidden state and fed to the classification head.

        :param input_ids: Batch of token ID sequences (B × L).
        :param attention_mask: Batch of attention masks (B × L).
        :param labels: Optional batch of integer class labels (B,). When provided, the cross-entropy loss is computed.
        :return: A dict with 'logits' (B × num_labels) and optionally 'loss'.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Locate the [E1] token in each sequence and extract its representation
        e1_mask = (input_ids == self.e1_token_id)
        e1_position = e1_mask.float().argmax(dim=1)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        e1_repr = sequence_output[batch_indices, e1_position]

        logits = self.classifier(e1_repr)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

    def predict(self,input_ids: torch.Tensor,attention_mask: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs inference for a single batch and returns the predicted class indices alongside their confidence scores.

        :param input_ids: Batch of token ID sequences (B × L).
        :param attention_mask: Batch of attention masks (B × L).
        :return: A tuple of (predictions, confscores), both 1-D tensors of shape (B,).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs["logits"], dim=-1)
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
        self.encoder.config.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


##############################
# TermClassificationTrainer  #
##############################

class TermClassificationTrainer(BaseTrainer):
    """
    Handles both the training loop and the inference pipeline for the transformer-based term classification model.

    Training:
        - Fine-tunes a BertForTermClassification with AdamW.
        - Evaluates on a pre-built development DataLoader after every epoch and saves the best checkpoint based on macro F1 score.
        - Supports automatic mixed precision (AMP) on CUDA.

    Inference:
        - Iterates over all entities in the dataset.
        - Inserts [E1] / [/E1] markers around each entity span, tokenizes the marked text, and runs a forward pass to predict the entity's category.
        - Writes the predicted label and confidence score back into the entity dict.
    """

    def __init__(self, model: BertForTermClassification, tokenizer, device: torch.device, max_length: int = 512, id2label: dict[int, str] = None,):
        """
        Initializes the trainer with a model, tokenizer, target device, sequence length cap, and label mapping.

        :param model: A BertForTermClassification instance.
        :param tokenizer: The tokenizer stored on the model (must include [E1] / [/E1]).
        :param device: The torch.device to run computations on.
        :param max_length: Maximum token sequence length (default 512).
        :param id2label: Dictionary mapping integer class indices to label strings; required for inference.
        """
        super().__init__(model, tokenizer, device, max_length)
        self.id2label = id2label or {}

    # -- BaseTrainer abstract method implementations --

    def _forward_pass(self, batch: dict) -> torch.Tensor:
        """
        Runs a forward pass through BertForTermClassification and returns the cross-entropy loss at the [E1] position.

        :param batch: A dict with 'input_ids', 'attention_mask', and 'labels' tensors.
        :return: A scalar loss tensor.
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs["loss"]

    def _evaluate_on_dev(self, dev_data: DataLoader, evaluator) -> dict:
        """
        Evaluates the model on a pre-built development DataLoader and computes loss, accuracy, and macro/micro F1 via the provided evaluator.

        :param dev_data: A DataLoader over the development set.
        :param evaluator: A TermClassificationEvaluator instance.
        :return: A dict with 'score' (macro F1) and 'log_str' (metrics summary).
        """
        metrics = evaluator.evaluate(self.model, dev_data, self.device)
        return {
            "score": metrics["f1_macro"],
            "log_str": (
                f"loss: {metrics['loss']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"F1 macro: {metrics['f1_macro']:.4f} | "
                f"F1 micro: {metrics['f1_micro']:.4f}"
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
        Runs the term classification inference pipeline over an entire dataset, annotating each entity in-place with a predicted label and confidence score.

        :param data: A dict mapping paper IDs to content dicts (GBIE format). Each content dict must have an 'entities' list with 'start_idx', 'end_idx', and 'location' fields.
        :return: The same dict with 'label' and 'confscore' added to every entity.
        """
        self.model.eval()

        for _, content in tqdm(data.items(), total=len(data), desc="Inference"):
            for entity in content.get("entities", []):
                section_text = content["metadata"][entity["location"]]
                marked_text = self._insert_entity_markers(
                    section_text, entity["start_idx"], entity["end_idx"] + 1
                )

                encoding = self.tokenizer(
                    marked_text,
                    return_attention_mask=True,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                predictions, confscores = self.model.predict(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                entity["label"] = self.id2label[predictions.item()]
                entity["confscore"] = round(confscores.item(), 6)

        return data

    @staticmethod
    def _insert_entity_markers(text: str, start_idx: int, end_idx: int) -> str:
        """
        Wraps the span [start_idx, end_idx) in the source text with [E1] / [/E1] markers.

        :param text: The original section text.
        :param start_idx: Character-level start index of the span (inclusive).
        :param end_idx: Character-level end index of the span (exclusive).
        :return: The text with entity markers inserted.
        """
        return (
            f"{text[:start_idx]}"
            f"[E1]{text[start_idx:end_idx]}[/E1]"
            f"{text[end_idx:]}"
        )


##############################
# TermClassificationEvaluator#
##############################

class TermClassificationEvaluator:
    """
    Evaluates term classification quality by running a model over a DataLoader and computing loss, accuracy, and macro/micro F1.

    This class is API-compatible with TermClassificationTrainer._evaluate_on_dev and can be used standalone for benchmarking.
    """

    def evaluate(self, model: BertForTermClassification, dataloader: DataLoader, device: torch.device,) -> dict:
        """
        Runs the model over the entire DataLoader and returns aggregate metrics.

        :param model: A BertForTermClassification instance (eval mode is set internally).
        :param dataloader: A DataLoader over the evaluation set.
        :param device: The torch.device to run computations on.
        :return: A dict with keys 'loss', 'accuracy', 'f1_macro', and 'f1_micro' (all floats).
        """
        model.eval()
        y_true = []
        y_pred = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                batch = move_to_device(batch, device)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs["loss"].detach().float().item()

                predictions = torch.argmax(outputs["logits"], dim=-1)
                y_true.extend(batch["labels"].detach().cpu().tolist())
                y_pred.extend(predictions.detach().cpu().tolist())

        return {
            "loss": total_loss / max(len(dataloader), 1),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        }


####################
# Metadata helpers #
####################

def save_metadata(output_dir: str, label2id: dict, id2label: dict, args: argparse.Namespace) -> None:
    """
    Persists label mapping and training arguments alongside the model checkpoint.

    :param output_dir: Directory where the metadata JSON files will be written.
    :param label2id: Dictionary mapping label strings to integer IDs.
    :param id2label: Dictionary mapping integer IDs to label strings.
    :param args: Parsed argument namespace from argparse.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full label mapping so inference can reconstruct id2label
    save_json_data(
        {
            "labels": LABEL_LIST,
            "label2id": label2id,
            "id2label": {str(key): value for key, value in id2label.items()},
        },
        os.path.join(output_dir, "label_mapping.json"),
    )

    # Save all training hyper-parameters for reproducibility
    save_json_data(vars(args), os.path.join(output_dir, "training_args.json"))


def _checkpoint_hidden_size(state_dict: dict) -> int | None:
    """
    Reads the encoder hidden size expected by a saved classifier head.
    """
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None or classifier_weight.ndim != 2:
        return None
    return int(classifier_weight.shape[1])


def _model_name_candidates(model_dir: str, training_args: dict) -> list[str]:
    """
    Builds possible HF model IDs for older checkpoints whose training_args.json may be stale.
    """
    candidates = []

    saved_model_name = training_args.get("model_name")
    if saved_model_name:
        candidates.append(saved_model_name)

    checkpoint_name = os.path.basename(os.path.normpath(model_dir))
    if checkpoint_name.startswith("BiomedNLP-"):
        candidates.append(f"microsoft/{checkpoint_name}")
    elif checkpoint_name.startswith("BioLinkBERT-"):
        candidates.append(f"michiyasunaga/{checkpoint_name}")

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def _tokenizer_source(model_dir: str, fallback_model_name: str) -> str:
    """
    Uses the checkpoint tokenizer when present so added marker tokens are preserved.
    """
    tokenizer_files = ("tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "tokenizer.json")
    if any(os.path.exists(os.path.join(model_dir, file_name)) for file_name in tokenizer_files):
        return model_dir
    return fallback_model_name


def _matching_encoder_config(model_dir: str, training_args: dict, expected_hidden_size: int | None):
    """
    Finds an encoder config whose hidden size matches the checkpoint weights.
    """
    errors = []

    try:
        config = AutoConfig.from_pretrained(model_dir)
        config_hidden_size = getattr(config, "hidden_size", None)
        if expected_hidden_size is None or config_hidden_size == expected_hidden_size:
            return config, model_dir
        print(
            f"[load_trained_model] Ignoring checkpoint config hidden_size={config_hidden_size}; "
            f"checkpoint weights expect hidden_size={expected_hidden_size}."
        )
    except Exception as exc:
        errors.append(f"{model_dir}: {exc}")

    for candidate in _model_name_candidates(model_dir, training_args):
        try:
            config = AutoConfig.from_pretrained(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue

        config_hidden_size = getattr(config, "hidden_size", None)
        if expected_hidden_size is None or config_hidden_size == expected_hidden_size:
            return config, candidate

        print(
            f"[load_trained_model] Skipping '{candidate}' hidden_size={config_hidden_size}; "
            f"checkpoint weights expect hidden_size={expected_hidden_size}."
        )

    error_details = "\n  - ".join(errors) if errors else "no compatible candidates found"
    raise ValueError(
        "Could not reconstruct the term classifier architecture for this checkpoint. "
        f"The saved classifier head expects hidden_size={expected_hidden_size}. "
        "Check training_args.json or pass a checkpoint directory whose name maps to the original HF model ID.\n"
        f"Tried:\n  - {error_details}"
    )


def load_trained_model(model_dir: str) -> tuple:
    """
    Loads a trained BertForTermClassification model from a saved checkpoint directory.

    :param model_dir: Path to the checkpoint directory produced by save_metadata and BertForTermClassification.save.
    :return: A tuple of (model, id2label, training_args) where model is in eval mode, id2label maps integer IDs to label strings, and training_args is the original argparse namespace as a plain dict.
    """
    label_mapping = load_json_data(os.path.join(model_dir, "label_mapping.json"))
    training_args = load_json_data(os.path.join(model_dir, "training_args.json"))

    labels = label_mapping["labels"]
    id2label = {int(key): value for key, value in label_mapping["id2label"].items()}

    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu"
    )
    expected_hidden_size = _checkpoint_hidden_size(state_dict)
    encoder_config, model_source = _matching_encoder_config(
        model_dir=model_dir,
        training_args=training_args,
        expected_hidden_size=expected_hidden_size,
    )

    model = BertForTermClassification(
        model_name=model_source,
        num_labels=len(labels),
        encoder_config=encoder_config,
        tokenizer_name=_tokenizer_source(model_dir, model_source),
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, id2label, training_args


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

    # Build label↔id mappings from the global label list
    label2id = {label: i for i, label in enumerate(LABEL_LIST)}
    id2label = {i: label for label, i in label2id.items()}

    # -- Load raw data --
    train_data = load_merge_json_data(args.train_data_paths)
    dev_data = load_json_data(args.dev_data_path) if args.dev_data_path else None

    # -- Initialise model (tokenizer lives on the model) --
    model = BertForTermClassification(
        model_name=args.model_name, num_labels=len(LABEL_LIST)
    )
    tokenizer = model.tokenizer

    # -- Preprocess training data --
    train_tokenizer = TermClassificationTokenizer(
        data=train_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        negative_sample_multiplier=args.negative_sample_multiplier,
        max_negative_span_words=args.max_negative_span_words,
    )
    train_samples = train_tokenizer.process_files()
    train_dataset = TermClassificationData(train_samples)
    train_loader = train_dataset.build_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )

    print(f"Training samples: {len(train_samples)}")

    # -- Preprocess development data (if provided) --
    dev_loader = None
    if dev_data is not None:
        dev_tokenizer = TermClassificationTokenizer(
            data=dev_data,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            negative_sample_multiplier=args.dev_negative_sample_multiplier,
            max_negative_span_words=args.max_negative_span_words,
        )
        dev_samples = dev_tokenizer.process_files()
        dev_dataset = TermClassificationData(dev_samples)
        dev_loader = dev_dataset.build_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
        )
        print(f"Development samples: {len(dev_samples)}")

    # -- Build trainer and evaluator --
    trainer = TermClassificationTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        id2label=id2label,
    )
    evaluator = TermClassificationEvaluator() if dev_loader is not None else None

    # -- Train --
    trainer.train(
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        dev_data=dev_loader,
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

    model, id2label, training_args = load_trained_model(args.model_path)
    model.to(device)
    model.eval()

    # CLI override takes priority over the saved value
    max_length = args.max_length if args.max_length else training_args.get("max_length", 512)

    # -- Run inference --
    data = load_json_data(args.inference_data_path)
    trainer = TermClassificationTrainer(
        model=model,
        tokenizer=model.tokenizer,
        device=device,
        max_length=max_length,
        id2label=id2label,
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
        description="Train or run inference for a transformer-based biomedical term classifier"
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
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW learning rate")
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
        help="How many negative samples to create per positive entity in the training set",
    )
    parser.add_argument(
        "--dev_negative_sample_multiplier",
        type=int,
        default=1,
        help="How many negative samples to create per positive entity in the development set",
    )
    parser.add_argument(
        "--max_negative_span_words",
        type=int,
        default=6,
        help="Maximum number of whitespace-separated words in a sampled negative span",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
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
