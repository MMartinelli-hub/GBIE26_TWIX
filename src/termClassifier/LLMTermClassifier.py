#!/usr/bin/env python3
"""
LLMTermClassifier.py

LLM-based term classification module. 
Supports both online providers (OpenAI, Azure OpenAI, Groq, and any other OpenAI-compatible endpoint) and local providers (LM Studio, Ollama, and HuggingFace Transformers running on the same machine).

Organized into five sections:
    - Classification constants : semantic category labels and special tokens
    - ResponseParser : converts raw LLM text output into predicted labels
    - LLMTermClassifier : zero-shot classification inference orchestrator
    - Helper functions : prompt loading and provider building (re-exported from LLMTermExtractor)
    - CLI : argument parsing and top-level entry point

Classification uses inline entity markers [E1] and [/E1] to mark the target entity within its surrounding context. 
The model predicts exactly one label from the predefined semantic category set.

Entry point: run with --help to see CLI options.
"""

import argparse
import json
import os

from tqdm import tqdm

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import utilities implemented in utils.py
from src.utils.utils import (
    get_device,
    load_json_data,
    maybe_empty_cache,
    print_device_info,
    save_json_data,
)

# Import reusable provider infrastructure from LLMTermExtractor
from src.termExtractor.LLMTermExtractor import (
    BaseLLMProvider,
    PROVIDER_REGISTRY,
    build_provider,
    load_prompts,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_NEW_TOKENS,
    LLM_CONFSCORE_PLACEHOLDER,
)


#############
# Constants #
#############

from src.utils.utils import LABEL_LIST

# Special tokens used to mark entity boundaries in the input text
ENTITY_START_TOKEN = "[E1]"
ENTITY_END_TOKEN = "[/E1]"


##################
# ResponseParser #
##################

class ResponseParser:
    """
    Converts raw LLM text output into predicted labels compatible with the format produced by HFTermClassifier.TermClassificationTrainer.perform_inference().

    The parser extracts the first token(s) from the LLM response and attempts to match it against the label set (case-insensitive). If no match is found, "NA" is assigned.
    """

    @staticmethod
    def parse(response: str) -> str:
        """
        Main entry point. Parses 'response' and extracts the predicted label.

        :param response: Raw string returned by the LLM.
        :return: The predicted label string, or "NA" if parsing fails.
        """
        # Strip whitespace and take only the first line
        text = response.strip().split('\n')[0].strip()
        
        # Attempt exact match (case-insensitive)
        for label in LABEL_LIST:
            if text.lower() == label.lower():
                return label
        
        # If no exact match, try to find a label as a substring (case-insensitive)
        text_lower = text.lower()
        for label in LABEL_LIST:
            if label.lower() in text_lower:
                return label
        
        # Default to NA if no match found
        return "NA"


#####################
# LLMTermClassifier #
#####################

class LLMTermClassifier:
    """
    Zero-shot LLM-based term classification pipeline.

    Given a dataset in the GBIE format (dict mapping paper IDs to content dicts with "entities" lists), this class:
        1. Iterates over each entity in the dataset.
        2. Marks the entity inline with [E1] / [/E1] tokens within its context.
        3. Sends the marked text through the configured LLM provider.
        4. Parses the raw response to extract the predicted label.
        5. Writes the predicted label and confidence score back into the entity dict.
        6. Returns results in the same format as HFTermClassifier.perform_inference().

    Responses are optionally cached to a JSONL checkpoint file so that long inference runs can be interrupted and resumed without re-processing already completed papers.
    """

    def __init__(self, provider: BaseLLMProvider, system_prompt: str, user_prompt_template: str, temperature: float = DEFAULT_TEMPERATURE, checkpoint_path: str | None = None,):
        """
        Initializes the classifier with an LLM provider and prompt configuration.

        :param provider: A configured BaseLLMProvider instance.
        :param system_prompt: The system-role message sent to the model for every inference call.
        :param user_prompt_template: A Python format string whose {text} placeholder will be filled with the marked entity text on each call.
        :param temperature: Sampling temperature forwarded to the provider.
        :param checkpoint_path: Optional path to a JSONL file used for resumable inference checkpointing. Already-processed paper IDs are skipped on restart.
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.checkpoint_path = checkpoint_path

    # -- Public interface --
    def perform_inference(self, data: dict) -> dict:
        """
        Runs zero-shot term classification over an entire dataset.

        :param data: A dict mapping paper IDs to content dicts (GBIE format). Each content dict must have an "entities" list and a "metadata" field with "title" and "abstract" keys.
        :return: A copy of 'data' with predicted labels and confidence scores added to every entity dict.
        """
        # Load already-processed IDs from an existing checkpoint so that a re-run after interruption skips completed papers.
        processed = self._load_checkpoint()

        result = {}

        with self._open_checkpoint() as ckpt_file:
            for paper_id, content in tqdm(data.items(), total=len(data), desc="LLM Classification"):
                # Resume: reconstruct result dict from checkpoint
                if paper_id in processed:
                    result[paper_id] = processed[paper_id]
                    continue

                output_content = dict(content)
                output_entities = []

                # Get metadata (fallback to content if metadata key doesn't exist)
                metadata = content.get("metadata", content)

                # Iterate over all entities in this paper
                entities = content.get("entities", [])
                for entity in entities:
                    # Create a copy to avoid modifying the original
                    output_entity = dict(entity)

                    # Skip entities that already have a predicted label
                    if "predicted_label" in output_entity:
                        output_entities.append(output_entity)
                        continue

                    # Extract the source text (title or abstract) from the metadata
                    location = entity.get("location", "abstract")
                    section_text = metadata.get(location, "")

                    # Get entity indices and validate they exist
                    start_idx = entity.get("start_idx")
                    end_idx = entity.get("end_idx")

                    if (not section_text or start_idx is None or end_idx is None):
                        # If we can't find context or indices, mark as unclassifiable
                        output_entity["predicted_label"] = "NA"
                        output_entity["confscore"] = LLM_CONFSCORE_PLACEHOLDER
                        output_entities.append(output_entity)
                        continue

                    # Mark the entity inline (end_idx is exclusive, but _insert_entity_markers expects inclusive)
                    marked_text = self._insert_entity_markers(section_text, start_idx, end_idx + 1)

                    # Call the LLM
                    raw_response = self._call_llm(marked_text)

                    # Parse the response to extract the predicted label
                    predicted_label = ResponseParser.parse(raw_response)

                    # Add prediction to the entity
                    output_entity["predicted_label"] = predicted_label
                    # LLMs do not expose per-entity probabilities; use sentinel
                    output_entity["confscore"] = LLM_CONFSCORE_PLACEHOLDER

                    output_entities.append(output_entity)

                output_content["entities"] = output_entities
                result[paper_id] = output_content

                # Persist immediately so progress is not lost on crash
                if ckpt_file is not None:
                    ckpt_file.write(
                        json.dumps({"id": paper_id, "content": output_content}) + "\n"
                    )
                    ckpt_file.flush()

        return result

    ####################
    # Internal helpers #
    ####################

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
            f"{ENTITY_START_TOKEN}{text[start_idx:end_idx]}{ENTITY_END_TOKEN}"
            f"{text[end_idx:]}"
        )

    def _call_llm(self, marked_text: str) -> str:
        """
        Formats the user prompt template with the marked text and calls the provider.

        :param marked_text: Marked source text with entity boundaries.
        :return: Raw string response from the LLM.
        """
        user_prompt = self.user_prompt_template.format(text=marked_text)
        return self.provider.chat_completion(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
        )

    def _load_checkpoint(self) -> dict:
        """
        Reads a structured JSONL checkpoint and returns a dict mapping paper IDs to their already-processed output content dicts.

        :return: Dict mapping paper_id -> output_content, or {} if no checkpoint exists.
        """
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return {}
        processed = {}
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed[entry["id"]] = entry["content"]
                except (json.JSONDecodeError, KeyError):
                    pass  # Skip malformed lines silently
        return processed

    def _open_checkpoint(self):
        """
        Opens the checkpoint file for appending, or returns a null context if no checkpoint path was configured.

        :return: A file object opened in append mode, or _NullContext if disabled.
        """
        if self.checkpoint_path is None:
            return _NullContext()
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        return open(self.checkpoint_path, "a", encoding="utf-8")


class _NullContext:
    """Minimal no-op context manager used when checkpointing is disabled."""

    def __enter__(self):
        return None

    def __exit__(self, *_):
        pass


####################
# Helper functions #
####################

def save_metadata(output_dir: str, args: argparse.Namespace) -> None:
    """
    Persists label mapping and inference arguments alongside the results.

    :param output_dir: Directory where metadata JSON files will be written.
    :param args: Parsed argument namespace from argparse.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save classification label list for reference
    save_json_data(
        {"labels": LABEL_LIST},
        os.path.join(output_dir, "label_mapping.json"),
    )

    # Save all inference hyper-parameters for reproducibility
    save_json_data(vars(args), os.path.join(output_dir, "inference_args.json"))


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for zero-shot LLM-based term classification.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot LLM-based biomedical term classification.  "
            "Supports OpenAI, Azure, Groq, LM Studio, Ollama, and HuggingFace backends."
        )
    )

    # -- Provider --
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=sorted(PROVIDER_REGISTRY.keys()),
        help="LLM backend to use (e.g. 'openai', 'lmstudio', 'huggingface')",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model identifier as expected by the provider "
            "(e.g. 'gpt-4o', 'medgemma-27b-text-it', 'llama3')"
        ),
    )

    # -- Provider credentials / endpoints --
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the provider.  Defaults to the relevant environment variable.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help=(
            "Override the default API base URL (OpenAI-compatible providers only).  "
            "Well-known local server defaults are applied automatically for 'lmstudio' "
            "and 'ollama'."
        ),
    )
    parser.add_argument(
        "--azure_endpoint",
        type=str,
        default=None,
        help="Azure resource endpoint URL (azure provider only).",
    )
    parser.add_argument(
        "--azure_api_version",
        type=str,
        default="2024-02-01",
        help="Azure API version string (azure provider only, default: 2024-02-01).",
    )

    # -- Data paths --
    parser.add_argument(
        "--inference_data_path",
        type=str,
        required=True,
        help="Path to the JSON file to run classification on (GBIE format with entities).",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        required=True,
        help="Path where classification results JSON will be written.",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="src/termClassifier/prompts.json",
        help="Path to the prompts JSON file (default: src/termClassifier/prompts.json).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional path to a JSONL checkpoint file.  Enables resumable classification: "
            "already-processed paper IDs are skipped on restart."
        ),
    )

    # -- Prompt selection --
    parser.add_argument(
        "--system_prompt_key",
        type=str,
        default="base",
        help="Key selecting the system prompt variant from the prompts file.",
    )
    parser.add_argument(
        "--user_prompt_key",
        type=str,
        default="base",
        help="Key selecting the user prompt variant from the prompts file.",
    )

    # -- Generation hyper-parameters --
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}; 0.0 = greedy).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum tokens the model may generate per call (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )

    # -- HuggingFace-only options --
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Target device for HuggingFace local inference ('cuda', 'mps', 'cpu').  "
            "Auto-detected when not specified."
        ),
    )

    return parser.parse_args()


###############
# Entry point #
###############

def run_inference(args: argparse.Namespace) -> None:
    """
    Orchestrates provider initialization, classification, and result persistence.

    :param args: Parsed CLI arguments.
    :return: None
    """
    device = get_device()
    print_device_info(device)

    # -- Build provider --
    provider = build_provider(
        provider_name=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        azure_endpoint=args.azure_endpoint,
        api_version=args.azure_api_version,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    # -- Load prompts --
    system_prompt, user_prompt_template = load_prompts(
        prompts_path=args.prompts_path,
        system_key=args.system_prompt_key,
        user_key=args.user_prompt_key,
    )

    # -- Build classifier --
    classifier = LLMTermClassifier(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
    )

    # -- Load inference data --
    data = load_json_data(args.inference_data_path)

    # -- Run inference --
    results = classifier.perform_inference(data)

    save_json_data(results, args.inference_output_path)
    print(f"Classification results saved to {args.inference_output_path}")

    maybe_empty_cache(device)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
