#!/usr/bin/env python3
"""
GLiNEREntityRecognizer.py

GLiNER-based named entity recognition (NER) module that jointly extracts and classifies entities using the GLiNER span-prediction architecture.

Organized into five sections:
    - GLiNERTrainingConfig : typed configuration container for the training loop
    - GLiNEREntityRecognizer : training loop and inference pipeline
    - NERExtractionEvaluator : span+label P/R/F1 evaluation (re-exported from HFEntityRecognizer)
    - GLiNERTokenizer : word-level GLiNER preprocessing (re-exported from Tokenizers module)
    - Helper functions + CLI : save_metadata, run_training, run_inference, parse_args

Unlike HFEntityRecognizer and its subclasses, GLiNEREntityRecognizer does NOT inherit from BaseTrainer because GLiNER ships its own training infrastructure:
    - model.create_dataloader() for data loading
    - model.get_optimizer() for grouped learning rates (encoder vs. other parameters)
    - model.evaluate() for GLiNER-native span-level scoring
    - model.set_sampling_params() for negative-type sampling

Training therefore runs for a fixed number of gradient steps (num_steps) rather than epochs, and checkpoints are saved at configurable intervals.

Inference uses GLiNER's model.predict_entities() and converts the resulting predictions back into the GBIE entity format (start_idx, end_idx, location, text_span, label, confscore) so that all downstream evaluation utilities remain interoperable.

Entry point: run with --help to see CLI options.
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import utilities implemented in utils.py
from src.utils.utils import (
    GLINER_ENTITY_TYPES,
    get_title_and_abstract,
    get_device,
    load_json_data,
    load_merge_json_data,
    maybe_empty_cache,
    print_device_info,
    save_json_data,
    seed_everything,
)

# Import shared NER evaluator from the HF-based NER module
from src.entityRecognizer.HFEntityRecognizer import NERExtractionEvaluator  # noqa: F401  (re-exported for callers)

# Import the GLiNER-specific tokenizer
from src.termExtractor.Tokenizers import GLiNERTokenizer  # noqa: F401  (re-exported for callers)


#############
# Constants #
#############

# GLINER_ENTITY_TYPES re-exported so callers need not import from utils
from src.utils.utils import GLINER_ENTITY_TYPES  # noqa: F401  (re-exported for callers)

# Confidence score field name used by GLiNER predict_entities()
_GLINER_SCORE_KEY = "confscore"


######################
# GLiNERTrainingConfig #
######################

@dataclass
class GLiNERTrainingConfig:
    """
    Typed configuration container for the GLiNER fine-tuning loop.

    All fields mirror the SimpleNamespace config used in the upstream GLiNER training script so that users familiar with that interface can map values directly.  
    Defaults replicate the recommended settings for the NuNerZero model on a single mid-range GPU.

    Fields
    ------
    num_steps : int
        Total number of gradient update steps.  Adjust proportionally to dataset size (larger datasets allow more steps without overfitting).
    eval_every : int
        Evaluate on the development set and save a checkpoint every this many steps.
    train_batch_size : int
        Samples per training mini-batch. Reduce if GPU memory is limited.
    max_len : int
        Maximum token-sequence length fed to the model. Use 2048 for NuNerZero_long_context.
    save_directory : str
        Directory where intermediate checkpoints are written during training.
    warmup_ratio : float
        Fraction of num_steps used for linear learning-rate warm-up (< 1.0) or an absolute number of warm-up steps (>= 1.0).
    lr_encoder : float
        Learning rate for the encoder (transformer backbone) parameters.
    lr_others : float
        Learning rate for all non-encoder parameters (span head, etc.).
    freeze_token_rep : bool
        When True, the token representation layers are frozen during training.
    max_types : int
        Maximum number of entity types sampled per mini-batch.
    shuffle_types : bool
        Shuffle entity types within each batch for training diversity.
    random_drop : bool
        Randomly drop entity types to improve robustness.
    max_neg_type_ratio : float
        Ratio of negative-to-positive entity types sampled during training.
    """

    num_steps:           int   = 3000
    eval_every:          int   = 200
    train_batch_size:    int   = 8
    max_len:             int   = 384
    save_directory:      str   = "logs"
    warmup_ratio:        float = 0.1
    lr_encoder:          float = 1e-5
    lr_others:           float = 5e-5
    freeze_token_rep:    bool  = False
    max_types:           int   = 15
    shuffle_types:       bool  = True
    random_drop:         bool  = True
    max_neg_type_ratio:  float = 1.0


########################
# GLiNEREntityRecognizer #
########################

class GLiNEREntityRecognizer:
    """
    Wraps a GLiNER model to provide a unified training loop and inference pipeline for NER that is consistent with the rest of the entityRecognizer module.

    Training:
        - Accepts entity-level GLiNER samples (from GLiNERTokenizer.process_files()) and internally converts them to the token-level format required by the GLiNER training loop before calling model.create_dataloader().
        - Uses GLiNER's grouped-parameter optimizer and a cosine schedule with warm-up.
        - Saves intermediate checkpoints at configurable step intervals.
        - Optionally evaluates on a development set using GLiNER-native span-level scoring as well as the GBIE-compatible NERExtractionEvaluator (strict span+label F1).

    Inference:
        - Calls model.predict_entities() on the title and abstract of each paper separately so that the 'location' field can be set correctly on every predicted entity.
        - Converts GLiNER predictions to the GBIE entity format so that downstream relation extraction, evaluation, and data pipeline utilities are fully interoperable.
    """

    def __init__(self, model, device: torch.device, entity_types: list[str] = None, threshold: float = 0.5,):
        """
        Initializes the recognizer with a GLiNER model and inference parameters.

        :param model: A loaded GLiNER model instance (e.g. from GLiNER.from_pretrained()).
        :param device: Target torch.device for both training and inference.
        :param entity_types: List of entity type strings used during inference. Defaults to GLINER_ENTITY_TYPES from utils when None.
        :param threshold: Confidence threshold for entity predictions (default 0.5). Increase to improve precision at the cost of recall.
        """
        self.model        = model
        self.device       = device
        self.entity_types = entity_types if entity_types is not None else GLINER_ENTITY_TYPES
        self.threshold    = threshold

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self, train_samples: list, config: GLiNERTrainingConfig, eval_samples: list | None = None, eval_data: dict | None = None, evaluator: NERExtractionEvaluator | None = None,) -> None:
        """
        Runs the GLiNER fine-tuning loop for the configured number of gradient steps.

        Checkpoints are saved every config.eval_every steps.  When eval_samples and eval_data are provided, both GLiNER-native span F1 and GBIE strict F1 (span + label) are logged alongside each checkpoint.

        :param train_samples: Entity-level GLiNER samples from GLiNERTokenizer.process_files(). Internally converted to token-level format before training begins.
        :param config: A GLiNERTrainingConfig instance controlling all hyperparameters.
        :param eval_samples: Optional entity-level GLiNER samples for the development set (same format as train_samples). Used by model.evaluate() for native GLiNER F1.
        :param eval_data: Optional raw GBIE development data dict (paper_id -> content). When provided together with evaluator, GBIE strict F1 is also reported.
        :param evaluator: Optional NERExtractionEvaluator for GBIE-format evaluation.
        :return: None
        """
        self.model.to(self.device)

        # Convert entity-level samples to the token-level format expected by GLiNER
        token_level_train = GLiNERTokenizer.to_token_level(train_samples)

        # Set GLiNER sampling parameters from the config
        self.model.set_sampling_params(
            max_types=config.max_types,
            shuffle_types=config.shuffle_types,
            random_drop=config.random_drop,
            max_neg_type_ratio=config.max_neg_type_ratio,
            max_len=config.max_len,
        )

        self.model.train()

        # Build data loader using GLiNER's built-in method
        train_loader = self.model.create_dataloader(
            token_level_train,
            batch_size=config.train_batch_size,
            shuffle=True,
        )

        # Grouped optimizer: lower lr for the encoder, higher lr for the span head
        optimizer = self.model.get_optimizer(
            config.lr_encoder,
            config.lr_others,
            config.freeze_token_rep,
        )

        # Resolve warmup steps: treat values < 1 as a fraction of num_steps
        num_warmup_steps = (
            int(config.num_steps * config.warmup_ratio)
            if config.warmup_ratio < 1
            else int(config.warmup_ratio)
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=config.num_steps,
        )

        # Build the optional GLiNER-format eval structure once
        gliner_eval_dict = None
        if eval_samples is not None:
            token_level_eval = GLiNERTokenizer.to_token_level(eval_samples)
            gliner_eval_dict = {
                "entity_types": self.entity_types,
                "samples":      token_level_eval,
            }

        pbar             = tqdm(range(config.num_steps), desc="GLiNER training")
        iter_train       = iter(train_loader)

        for step in pbar:
            # Restart the iterator when the loader is exhausted (num_steps > epoch length)
            try:
                batch = next(iter_train)
            except StopIteration:
                iter_train = iter(train_loader)
                batch = next(iter_train)

            # Move tensor values to the target device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)

            loss = self.model(batch)

            # NaN loss can occur with extreme learning rates -- skip the step safely
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_description(
                f"step: {step} | epoch: {step // max(len(train_loader), 1)} | "
                f"loss: {loss.item():.4f}"
            )

            # -- Periodic evaluation and checkpoint --
            if (step + 1) % config.eval_every == 0:
                self.model.eval()
                self._evaluate_and_checkpoint(
                    step=step,
                    config=config,
                    gliner_eval_dict=gliner_eval_dict,
                    eval_data=eval_data,
                    evaluator=evaluator,
                )
                self.model.train()

        # Save the final checkpoint after training completes
        final_dir = os.path.join(config.save_directory, "finetuned_final")
        self._save_checkpoint(final_dir)
        print(f"Training complete.  Final model saved to '{final_dir}'.")

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def perform_inference(self, data: dict) -> dict:
        """
        Runs the GLiNER NER inference pipeline over an entire dataset and returns a copy of the input data with 'entities' fields populated in GBIE format.

        Title and abstract are processed independently so that each entity is tagged with the correct 'location' value ('title' or 'abstract').

        :param data: A dict mapping paper IDs to content dicts (GBIE format).
        :return: A dict with the same structure as 'data' but with predicted entities in each content dict. Each entity follows the GBIE schema: {start_idx, end_idx, location, text_span, label, confscore}.
        """
        self.model.eval()
        result = {}

        for paper_id, content in tqdm(data.items(), total=len(data), desc="GLiNER Inference"):
            title, abstract = get_title_and_abstract(content)
            entity_predictions = []

            # Process each section independently to assign the correct location tag
            for section, text in [("title", title), ("abstract", abstract)]:
                if not text:
                    continue
                raw_preds = self.model.predict_entities(
                    text,
                    self.entity_types,
                    threshold=self.threshold,
                    flat_ner=True,
                    multi_label=False,
                )
                entity_predictions.extend(
                    self._convert_predictions(raw_preds, text, section)
                )

            output_content = dict(content)
            output_content["entities"] = entity_predictions
            result[paper_id] = output_content

        return result

    # ------------------------------------------------------------------ #
    # Checkpoint management                                                #
    # ------------------------------------------------------------------ #

    def save(self, output_dir: str) -> None:
        """
        Saves the fine-tuned GLiNER model to 'output_dir' using GLiNER's built-in serializer.
        Also copies gliner_config.json to config.json so the checkpoint can be reloaded without specifying local_files_only=True explicitly.

        :param output_dir: Directory path where the model will be saved.
        :return: None
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        # GLiNER requires config.json to be present for from_pretrained() to work
        src = os.path.join(output_dir, "gliner_config.json")
        dst = os.path.join(output_dir, "config.json")
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy(src, dst)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _convert_predictions(self, raw_predictions: list, source_text: str, location: str,) -> list:
        """
        Converts GLiNER predict_entities() output to GBIE-compatible entity dicts.

        GLiNER returns exclusive character end indices (standard Python slice convention). GBIE uses inclusive end_idx, so we subtract 1.

        :param raw_predictions: List of GLiNER prediction dicts with 'start', 'end', 'text', 'label', 'score'.
        :param source_text: The original section text (used to extract the exact text span).
        :param location: Section label -- 'title' or 'abstract'.
        :return: A list of GBIE entity dicts.
        """
        entities = []
        for pred in raw_predictions:
            start = pred["start"]
            # GLiNER's 'end' is exclusive; GBIE end_idx is inclusive
            end_inclusive = pred["end"] - 1
            entities.append(
                {
                    "start_idx":  start,
                    "end_idx":    end_inclusive,
                    "location":   location,
                    # Recover the exact surface form from the source text
                    "text_span":  source_text[start : pred["end"]],
                    "label":      pred["label"],
                    "confscore":  float(pred[_GLINER_SCORE_KEY]),
                }
            )
        return entities

    def _evaluate_and_checkpoint(
        self,
        step: int,
        config: GLiNERTrainingConfig,
        gliner_eval_dict: dict | None,
        eval_data: dict | None,
        evaluator: NERExtractionEvaluator | None,
    ) -> None:
        """
        Runs evaluation (GLiNER-native and/or GBIE) and saves a checkpoint for the current step.

        :param step: Current training step (used for the checkpoint directory name).
        :param config: Training config (needed for save_directory and threshold).
        :param gliner_eval_dict: Optional token-level eval dict for model.evaluate().
        :param eval_data: Optional raw GBIE dev data for GBIE-format evaluation.
        :param evaluator: Optional NERExtractionEvaluator instance.
        :return: None
        """
        # 1. GLiNER-native span-level evaluation
        if gliner_eval_dict is not None:
            results, f1 = self.model.evaluate(
                gliner_eval_dict["samples"],
                flat_ner=True,
                threshold=self.threshold,
                batch_size=32,
                entity_types=gliner_eval_dict["entity_types"],
            )
            print(f"\nStep {step} | GLiNER native F1: {f1:.4f}\n{results}")

        # 2. GBIE strict F1 (span + label)
        if eval_data is not None and evaluator is not None:
            dev_predictions = self.perform_inference(eval_data)
            metrics = evaluator.evaluate(dev_predictions, eval_data)
            print(
                f"Step {step} | GBIE strict  P: {metrics['strict_precision']:.4f} | "
                f"R: {metrics['strict_recall']:.4f} | F1: {metrics['strict_f1']:.4f}  "
                f"(span-only F1: {metrics['span_f1']:.4f})"
            )
            # Return model to train mode after inference run
            self.model.train()

        # 3. Save checkpoint
        checkpoint_dir = os.path.join(config.save_directory, f"finetuned_{step}")
        self._save_checkpoint(checkpoint_dir)
        print(f"Checkpoint saved to '{checkpoint_dir}'.")

    def _save_checkpoint(self, output_dir: str) -> None:
        """
        Saves the current model weights to 'output_dir'.

        :param output_dir: Directory path for the checkpoint.
        :return: None
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)


####################
# Metadata helpers #
####################

def save_metadata(
    output_dir: str,
    config: GLiNERTrainingConfig,
    args: argparse.Namespace,
) -> None:
    """
    Persists the training configuration and CLI arguments alongside the final model checkpoint.

    :param output_dir: Directory where the metadata JSON files will be written.
    :param config: GLiNERTrainingConfig instance used during training.
    :param args: Parsed argument namespace from argparse.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Serialise both the dataclass config and the raw CLI args for full reproducibility
    save_json_data(
        {
            "entity_types":    GLINER_ENTITY_TYPES,
            "threshold":       args.threshold,
            "training_config": {
                "num_steps":          config.num_steps,
                "eval_every":         config.eval_every,
                "train_batch_size":   config.train_batch_size,
                "max_len":            config.max_len,
                "warmup_ratio":       config.warmup_ratio,
                "lr_encoder":         config.lr_encoder,
                "lr_others":          config.lr_others,
                "freeze_token_rep":   config.freeze_token_rep,
                "max_types":          config.max_types,
                "shuffle_types":      config.shuffle_types,
                "random_drop":        config.random_drop,
                "max_neg_type_ratio": config.max_neg_type_ratio,
            },
        },
        os.path.join(output_dir, "gliner_training_args.json"),
    )

    save_json_data(vars(args), os.path.join(output_dir, "cli_args.json"))


#############################
# Top-level entry functions #
#############################

def run_training(args: argparse.Namespace) -> None:
    """
    Orchestrates data loading, GLiNER tokenization, model initialization, fine-tuning, and (optional) development evaluation for a full training run.

    :param args: Parsed CLI arguments.
    :return: None
    """
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise ImportError(
            "The 'gliner' package is required for GLiNEREntityRecognizer.  "
            "Install it with: pip install gliner"
        ) from exc

    seed_everything(args.seed)

    device = get_device()
    print_device_info(device)

    # -- Load GLiNER model --
    print(f"Loading GLiNER model '{args.model_name}'...")
    model = GLiNER.from_pretrained(args.model_name)

    # -- Load raw GBIE data --
    print("Loading training data...")
    train_data = load_merge_json_data(args.train_data_paths)
    dev_data   = load_json_data(args.dev_data_path) if args.dev_data_path else None

    # -- Tokenize to GLiNER entity-level format --
    ner_tokenizer = GLiNERTokenizer(
        data=train_data,
        concatenate_title_abstract=args.concatenate_title_abstract,
    )
    train_samples = ner_tokenizer.process_files()
    print(f"Training samples: {len(train_samples)}")

    dev_samples = None
    if dev_data is not None:
        dev_tokenizer = GLiNERTokenizer(
            data=dev_data,
            concatenate_title_abstract=args.concatenate_title_abstract,
        )
        dev_samples = dev_tokenizer.process_files()
        print(f"Development samples: {len(dev_samples)}")

    # -- Build training config from CLI args --
    config = GLiNERTrainingConfig(
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        train_batch_size=args.batch_size,
        max_len=args.max_len,
        save_directory=args.save_directory,
        warmup_ratio=args.warmup_ratio,
        lr_encoder=args.lr_encoder,
        lr_others=args.lr_others,
        freeze_token_rep=args.freeze_token_rep,
        max_types=args.max_types,
        shuffle_types=not args.no_shuffle_types,
        random_drop=not args.no_random_drop,
        max_neg_type_ratio=args.max_neg_type_ratio,
    )

    # -- Build recognizer and evaluator --
    recognizer = GLiNEREntityRecognizer(
        model=model,
        device=device,
        entity_types=GLINER_ENTITY_TYPES,
        threshold=args.threshold,
    )
    evaluator = NERExtractionEvaluator() if dev_data is not None else None

    # -- Train --
    print("Starting training...")
    recognizer.train(
        train_samples=train_samples,
        config=config,
        eval_samples=dev_samples,
        eval_data=dev_data,
        evaluator=evaluator,
    )

    # -- Save final model --
    print(f"Saving final model to '{args.output_dir}'...")
    recognizer.save(args.output_dir)
    save_metadata(args.output_dir, config, args)
    print("Done.")

    maybe_empty_cache(device)


def run_inference(args: argparse.Namespace) -> None:
    """
    Loads a saved GLiNER checkpoint and runs NER inference over a JSON dataset, writing results in GBIE format to the specified output path.

    :param args: Parsed CLI arguments.
    :return: None
    """
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise ImportError(
            "The 'gliner' package is required for GLiNEREntityRecognizer.  "
            "Install it with: pip install gliner"
        ) from exc

    device = get_device()
    print_device_info(device)

    # Load saved training args to recover the threshold when not overridden by CLI
    saved_args_path = os.path.join(args.model_path, "gliner_training_args.json")
    threshold = args.threshold
    if threshold is None:
        if os.path.exists(saved_args_path):
            saved = load_json_data(saved_args_path)
            threshold = saved.get("threshold", 0.5)
        else:
            threshold = 0.5

    # -- Load model from checkpoint --
    print(f"Loading GLiNER model from '{args.model_path}'...")
    model = GLiNER.from_pretrained(args.model_path, local_files_only=True)

    # -- Run inference --
    data = load_json_data(args.inference_data_path)
    recognizer = GLiNEREntityRecognizer(
        model=model,
        device=device,
        entity_types=GLINER_ENTITY_TYPES,
        threshold=threshold,
    )
    results = recognizer.perform_inference(data)

    save_json_data(results, args.inference_output_path)
    print(f"Inference results saved to '{args.inference_output_path}'.")

    # -- Optional evaluation --
    if args.eval_data_path:
        ground_truth = load_json_data(args.eval_data_path)
        evaluator = NERExtractionEvaluator()
        metrics = evaluator.evaluate(results, ground_truth)
        print(
            f"Strict   (span+label)  P: {metrics['strict_precision']:.4f} | "
            f"R: {metrics['strict_recall']:.4f} | F1: {metrics['strict_f1']:.4f}"
        )
        print(
            f"Lenient  (span-only)   P: {metrics['span_precision']:.4f} | "
            f"R: {metrics['span_recall']:.4f} | F1: {metrics['span_f1']:.4f}"
        )

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
        description="Fine-tune or run inference with a GLiNER-based biomedical NER model"
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
        help="One or more paths to GBIE-format training JSON files (merged automatically)",
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default=None,
        help="Path to GBIE-format development set JSON file",
    )
    parser.add_argument(
        "--inference_data_path",
        type=str,
        default=None,
        help="Path to GBIE-format JSON file used during inference",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        default=None,
        help="Path where inference results JSON will be written",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help=(
            "Optional path to a ground-truth GBIE JSON file.  When provided during inference, "
            "strict (span+label) and lenient (span-only) P/R/F1 are printed."
        ),
    )

    # -- Model paths --
    parser.add_argument(
        "--model_name",
        type=str,
        default="numind/NuNerZero",
        help="HuggingFace model name or local path used to initialise the GLiNER model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where the final fine-tuned model checkpoint will be saved",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved GLiNER checkpoint for inference",
    )
    parser.add_argument(
        "--save_directory",
        type=str,
        default="logs",
        help="Directory where intermediate training checkpoints are written (default: logs)",
    )

    # -- Inference hyper-parameters --
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Confidence threshold for entity predictions (default: value saved in checkpoint "
            "or 0.5 when not available).  Higher values increase precision, lower values increase recall."
        ),
    )

    # -- Training hyper-parameters --
    parser.add_argument(
        "--num_steps",
        type=int,
        default=3000,
        help="Total number of gradient update steps (default: 3000)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=200,
        help="Evaluate and checkpoint every N steps (default: 200)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=384,
        help="Maximum token-sequence length (default: 384; use 2048 for long-context models)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help=(
            "Warm-up schedule: value < 1 is treated as a fraction of num_steps; "
            "value >= 1 is the absolute number of warm-up steps (default: 0.1)"
        ),
    )
    parser.add_argument(
        "--lr_encoder",
        type=float,
        default=1e-5,
        help="Learning rate for encoder (transformer backbone) parameters (default: 1e-5)",
    )
    parser.add_argument(
        "--lr_others",
        type=float,
        default=5e-5,
        help="Learning rate for non-encoder parameters (span head, etc.) (default: 5e-5)",
    )
    parser.add_argument(
        "--freeze_token_rep",
        action="store_true",
        help="Freeze token representation layers during training",
    )
    parser.add_argument(
        "--max_types",
        type=int,
        default=15,
        help="Maximum entity types sampled per mini-batch (default: 15)",
    )
    parser.add_argument(
        "--no_shuffle_types",
        action="store_true",
        help="Disable entity-type shuffling within each batch",
    )
    parser.add_argument(
        "--no_random_drop",
        action="store_true",
        help="Disable random entity-type dropping during training",
    )
    parser.add_argument(
        "--max_neg_type_ratio",
        type=float,
        default=1.0,
        help="Ratio of negative-to-positive entity types sampled per batch (default: 1.0)",
    )
    parser.add_argument(
        "--separate_title_abstract",
        action="store_true",
        help="Tokenize title and abstract as separate samples (default: concatenated)",
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

    # Derive canonical flag used throughout the module
    args.concatenate_title_abstract = not args.separate_title_abstract

    # Default threshold for training mode
    if args.threshold is None:
        args.threshold = 0.5

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