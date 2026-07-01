#!/usr/bin/env python3
"""
LLMRelationExtractor.py

LLM-based relation extraction module.
Supports both online providers (OpenAI, Azure OpenAI, Groq, and any other OpenAI-compatible endpoint) and local providers (LM Studio, Ollama, and HuggingFace Transformers running on the same machine).

Organized into five sections:
    - Relation extraction constants : predicate labels and special tokens
    - ResponseParser : converts raw LLM text output into predicted relation predicates
    - LLMRelationExtractor : zero-shot relation extraction inference orchestrator
    - Helper functions : prompt loading and provider building (re-exported from LLMTermExtractor)
    - CLI : argument parsing and top-level entry point

Relation extraction uses inline entity pair markers [E1] / [/E1] (subject) and [E2] / [/E2] (object) to identify target entities within their surrounding context.
The model predicts exactly one relation predicate from the predefined set, or NA for no relation.

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
    get_title_and_abstract,
    load_json_data,
    maybe_empty_cache,
    print_device_info,
    save_json_data,
    RELATION_LABEL_LIST,
    VALID_RELATIONS,
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

# Special tokens used to mark entity boundaries in the input text
SUBJECT_START_TOKEN = "[E1]"
SUBJECT_END_TOKEN = "[/E1]"
OBJECT_START_TOKEN = "[E2]"
OBJECT_END_TOKEN = "[/E2]"


##################
# ResponseParser #
##################

class ResponseParser:
    """
    Converts raw LLM text output into predicted relation predicates compatible with the format produced by HFRelationExtractor.RelationExtractionTrainer.perform_inference().

    The parser extracts the first token(s) from the LLM response and attempts to match it against the relation predicate set (case-insensitive). 
    If no match is found, "NA" is assigned.
    """

    @staticmethod
    def parse(response: str) -> str:
        """
        Main entry point. Parses 'response' and extracts the predicted relation predicate.

        :param response: Raw string returned by the LLM.
        :return: The predicted predicate string, or "NA" if parsing fails.
        """
        # Strip whitespace and take only the first line
        text = response.strip().split('\n')[0].strip()
        
        # Attempt exact match (case-insensitive)
        for predicate in RELATION_LABEL_LIST:
            if text.lower() == predicate.lower():
                return predicate
        
        # If no exact match, try to find a predicate as a substring (case-insensitive)
        text_lower = text.lower()
        for predicate in RELATION_LABEL_LIST:
            if predicate.lower() in text_lower:
                return predicate
        
        # Default to NA if no match found
        return "NA"


########################
# LLMRelationExtractor #
########################

class LLMRelationExtractor:
    """
    Zero-shot LLM-based relation extraction pipeline.

    Given a dataset in the GBIE format (dict mapping paper IDs to content dicts with "entities" and "relations" lists), this class:
        1. Iterates over all entity pairs within each paper section (title or abstract).
        2. Skips pairs whose (head_label, tail_label) combination is absent from VALID_RELATIONS to respect structural constraints.
        3. Marks the subject with [E1] / [/E1] and the object with [E2] / [/E2] within their shared context.
        4. Sends the marked text through the configured LLM provider.
        5. Parses the raw response to extract the predicted relation predicate.
        6. Records relations where the predicted predicate is not NA.
        7. Returns results in the same format as HFRelationExtractor.perform_inference().

    Responses are optionally cached to a JSONL checkpoint file so that long inference runs can be interrupted and resumed without re-processing already completed papers.
    """

    def __init__(self, provider: BaseLLMProvider, system_prompt: str, user_prompt_template: str, temperature: float = DEFAULT_TEMPERATURE, checkpoint_path: str | None = None, concatenate_title_abstract: bool = False,):
        """
        Initializes the extractor with an LLM provider and prompt configuration.

        :param provider: A configured BaseLLMProvider instance.
        :param system_prompt: The system-role message sent to the model for every inference call.
        :param user_prompt_template: A Python format string whose {text} placeholder will be filled with the marked entity-pair text on each call.
        :param temperature: Sampling temperature forwarded to the provider.
        :param checkpoint_path: Optional path to a JSONL file used for resumable inference checkpointing. Already-processed paper IDs are skipped on restart.
        :param concatenate_title_abstract: Whether to concatenate title and abstract to enable cross-section relations.
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.checkpoint_path = checkpoint_path
        self.concatenate_title_abstract = concatenate_title_abstract

    # -- Public interface --

    def perform_inference(self, data: dict) -> dict:
        """
        Runs zero-shot relation extraction over an entire dataset.

        :param data: A dict mapping paper IDs to content dicts (GBIE format). Each content dict must have "entities" and "relations" lists, and a "metadata" field with "title" and "abstract".
        :return: A copy of 'data' with predicted relations added.
        """
        # Load already-processed IDs from an existing checkpoint so that a re-run after interruption skips completed papers.
        processed = self._load_checkpoint()

        result = {}

        with self._open_checkpoint() as ckpt_file:
            for paper_id, content in tqdm(data.items(), total=len(data), desc="LLM Relation Extraction"):
                # Resume: reconstruct result dict from checkpoint
                if paper_id in processed:
                    result[paper_id] = processed[paper_id]
                    continue

                output_content = dict(content)
                predicted_relations = []

                # Get title and abstract
                title, abstract = get_title_and_abstract(content)

                # Determine sections to process
                if self.concatenate_title_abstract:
                    sections = [("concatenated", f"{title} {abstract}", title, abstract)]
                else:
                    sections = [
                        ("title", title, None, None),
                        ("abstract", abstract, None, None),
                    ]

                # Process each section (or concatenated)
                for section_info in sections:
                    if self.concatenate_title_abstract:
                        _, text, title_text, abstract_text = section_info
                        title_length = len(title_text)
                        separator_length = 1
                        predicted_relations.extend(
                            self._extract_relations_concatenated(
                                content, text, title_text, abstract_text, title_length, separator_length
                            )
                        )
                    else:
                        section_name, text, _, _ = section_info
                        predicted_relations.extend(
                            self._extract_relations_single_section(content, section_name, text)
                        )

                output_content["relations"] = predicted_relations
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

    def _extract_relations_single_section(self, content: dict, section: str, text: str) -> list:
        """
        Extracts relations for entity pairs within a single section (title or abstract).

        :param content: The paper content dict.
        :param section: Section name ("title" or "abstract").
        :param text: The section text.
        :return: A list of predicted relation dicts.
        """
        relations = []

        # Get all entities in this section
        section_entities = [
            entity for entity in content.get("entities", [])
            if entity["location"] == section
        ]

        # Enumerate all ordered entity pairs in this section
        for i, subj_entity in enumerate(section_entities):
            for obj_entity in section_entities[i+1:]:
                # Check if this (head_label, tail_label) combination is valid
                head_label = subj_entity.get("label", "term")
                tail_label = obj_entity.get("label", "term")
                if (head_label, tail_label) not in VALID_RELATIONS:
                    continue

                # Predict relation
                relation = self._predict_relation(
                    text=text,
                    subject_entity=subj_entity,
                    object_entity=obj_entity,
                    section=section,
                )
                if relation is not None:
                    relations.append(relation)

        return relations

    def _extract_relations_concatenated(
        self,
        content: dict,
        concatenated_text: str,
        title: str,
        abstract: str,
        title_length: int,
        separator_length: int,
    ) -> list:
        """
        Extracts relations for entity pairs when title and abstract are concatenated.
        Supports both same-section and cross-section pairs.

        :param content: The paper content dict.
        :param concatenated_text: The concatenated title + abstract text.
        :param title: The title text.
        :param abstract: The abstract text.
        :param title_length: Length of the title.
        :param separator_length: Length of the separator (typically 1 for space).
        :return: A list of predicted relation dicts.
        """
        relations = []

        # Collect all entities with adjusted indices for the concatenated text
        all_entities = []
        for entity in content.get("entities", []):
            adjusted_entity = dict(entity)
            if entity["location"] == "abstract":
                adjusted_entity["start_idx"] = entity["start_idx"] + title_length + separator_length
                adjusted_entity["end_idx"] = entity["end_idx"] + title_length + separator_length
            all_entities.append(adjusted_entity)

        # Enumerate all ordered entity pairs across the concatenated text
        for i, subj_entity in enumerate(all_entities):
            for obj_entity in all_entities[i+1:]:
                # Check if this (head_label, tail_label) combination is valid
                head_label = subj_entity.get("label", "term")
                tail_label = obj_entity.get("label", "term")
                if (head_label, tail_label) not in VALID_RELATIONS:
                    continue

                # Predict relation
                relation = self._predict_relation(
                    text=concatenated_text,
                    subject_entity=subj_entity,
                    object_entity=obj_entity,
                    section="concatenated",
                    title_length=title_length,
                    separator_length=separator_length,
                )
                if relation is not None:
                    relations.append(relation)

        return relations

    def _predict_relation(self, text: str, subject_entity: dict, object_entity: dict, section: str, title_length: int = 0, separator_length: int = 0,) -> dict | None:
        """
        Predicts a relation between a subject and object entity pair.

        :param text: The source text (either a single section or concatenated).
        :param subject_entity: The subject entity dict with start_idx, end_idx, location, label.
        :param object_entity: The object entity dict with start_idx, end_idx, location, label.
        :param section: The section name for recording in the result.
        :param title_length: Length of the title (for concatenated mode).
        :param separator_length: Length of the separator (for concatenated mode).
        :return: A relation dict if the predicted predicate is not NA, otherwise None.
        """
        # Mark both entities in the text
        marked_text = self._insert_relation_markers(
            text,
            subject_entity["start_idx"],
            subject_entity["end_idx"] + 1,  # end_idx is inclusive, but markers expect exclusive
            object_entity["start_idx"],
            object_entity["end_idx"] + 1,
        )

        # Call the LLM
        raw_response = self._call_llm(marked_text)

        # Parse the response to extract the predicted predicate
        predicted_predicate = ResponseParser.parse(raw_response)

        # Only record the relation if it's not NA
        if predicted_predicate == "NA":
            return None

        # Determine original locations
        subject_location = subject_entity["location"]
        object_location = object_entity["location"]

        # If in concatenated mode, derive original indices
        if section == "concatenated":
            subject_start = subject_entity["start_idx"]
            subject_end = subject_entity["end_idx"]
            object_start = object_entity["start_idx"]
            object_end = object_entity["end_idx"]

            # Adjust back to original locations
            if subject_location == "abstract":
                subject_start -= title_length + separator_length
                subject_end -= title_length + separator_length
            if object_location == "abstract":
                object_start -= title_length + separator_length
                object_end -= title_length + separator_length
        else:
            subject_start = subject_entity["start_idx"]
            subject_end = subject_entity["end_idx"]
            object_start = object_entity["start_idx"]
            object_end = object_entity["end_idx"]

        return {
            "subject_start_idx": subject_start,
            "subject_end_idx": subject_end,
            "subject_location": subject_location,
            "object_start_idx": object_start,
            "object_end_idx": object_end,
            "object_location": object_location,
            "predicate": predicted_predicate,
        }

    @staticmethod
    def _insert_relation_markers(text: str, subj_start: int, subj_end: int, obj_start: int, obj_end: int,) -> str:
        """
        Inserts both subject and object markers into the text.
        Assumes subject and object do not overlap.

        :param text: The original section text.
        :param subj_start: Character-level start index of the subject span (inclusive).
        :param subj_end: Character-level end index of the subject span (exclusive).
        :param obj_start: Character-level start index of the object span (inclusive).
        :param obj_end: Character-level end index of the object span (exclusive).
        :return: The text with entity markers inserted.
        """
        # Ensure the subject comes before the object in the text
        if subj_start < obj_start:
            # Subject first, then object
            marked_text = (
                f"{text[:subj_start]}"
                f"{SUBJECT_START_TOKEN}{text[subj_start:subj_end]}{SUBJECT_END_TOKEN}"
                f"{text[subj_end:obj_start]}"
                f"{OBJECT_START_TOKEN}{text[obj_start:obj_end]}{OBJECT_END_TOKEN}"
                f"{text[obj_end:]}"
            )
        else:
            # Object first, then subject
            marked_text = (
                f"{text[:obj_start]}"
                f"{OBJECT_START_TOKEN}{text[obj_start:obj_end]}{OBJECT_END_TOKEN}"
                f"{text[obj_end:subj_start]}"
                f"{SUBJECT_START_TOKEN}{text[subj_start:subj_end]}{SUBJECT_END_TOKEN}"
                f"{text[subj_end:]}"
            )

        return marked_text

    def _call_llm(self, marked_text: str) -> str:
        """
        Formats the user prompt template with the marked text and calls the provider.

        :param marked_text: Marked source text with entity pair boundaries.
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
    Persists relation predicate mapping and inference arguments alongside the results.

    :param output_dir: Directory where metadata JSON files will be written.
    :param args: Parsed argument namespace from argparse.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save relation label list for reference
    save_json_data(
        {"labels": RELATION_LABEL_LIST},
        os.path.join(output_dir, "label_mapping.json"),
    )

    # Save all inference hyper-parameters for reproducibility
    save_json_data(vars(args), os.path.join(output_dir, "inference_args.json"))


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for zero-shot LLM-based relation extraction.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot LLM-based biomedical relation extraction.  "
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
        help="Path to the JSON file to run relation extraction on (GBIE format with entities and relations).",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        required=True,
        help="Path where relation extraction results JSON will be written.",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="src/relationExtractor/prompts.json",
        help="Path to the prompts JSON file (default: src/relationExtractor/prompts.json).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional path to a JSONL checkpoint file.  Enables resumable extraction: "
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

    # -- Relation extraction specific --
    parser.add_argument(
        "--concatenate_title_abstract",
        action="store_true",
        help="Concatenate title and abstract to enable cross-section relation extraction.",
    )

    return parser.parse_args()


###############
# Entry point #
###############

def run_inference(args: argparse.Namespace) -> None:
    """
    Orchestrates provider initialization, relation extraction, and result persistence.

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

    # -- Build extractor --
    extractor = LLMRelationExtractor(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
        concatenate_title_abstract=args.concatenate_title_abstract,
    )

    # -- Load inference data --
    data = load_json_data(args.inference_data_path)

    # -- Run inference --
    results = extractor.perform_inference(data)

    save_json_data(results, args.inference_output_path)
    print(f"Relation extraction results saved to {args.inference_output_path}")

    maybe_empty_cache(device)
