#!/usr/bin/env python3
"""
LLMEntityRecognizer.py

LLM-based named entity recognition (NER) module that jointly extracts and classifies entities.
Supports both online providers (OpenAI, Azure OpenAI, Groq, and any other OpenAI-compatible endpoint) and local providers (LM Studio, Ollama, and HuggingFace Transformers running on the same machine).

Organized into four sections:
    - ResponseParser : converts raw LLM text output into structured entity lists with labels
    - LLMEntityRecognizer : zero-shot NER inference orchestrator
    - Helper functions : prompt loading and provider building (re-exported from LLMTermExtractor)
    - CLI : argument parsing and top-level entry point

All LLM provider classes and the build_provider factory are re-used from LLMTermExtractor and are not redefined here.

The key difference relative to LLMTermExtractor is the expected response format: 
instead of a plain JSON array of strings (term text only), the model is prompted to return a JSON array of {"term": ..., "label": ...} objects so that both extraction and classification are performed in a single LLM call.  
The ResponseParser handles this two-field format and falls back gracefully to the string-only format when the model does not comply.

Entry point: run with --help to see CLI options.
"""

import argparse
import json
import os
import re

from tqdm import tqdm

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import utilities implemented in utils.py
from src.utils.utils import (
    LABEL_LIST,
    get_device,
    get_title_and_abstract,
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

# Import the NER evaluator
from src.entityRecognizer.HFEntityRecognizer import NERExtractionEvaluator  # noqa: F401  (re-exported for callers)


#############
# Constants #
#############

# Valid label set (lower-cased for case-insensitive matching during response parsing)
_VALID_LABELS_LOWER = {label.lower(): label for label in LABEL_LIST if label != "NA"}


##################
# ResponseParser #
##################

class ResponseParser:
    """
    Converts raw LLM text output into a list of entity dicts compatible with the format produced by HFEntityRecognizer.EntityRecognitionTrainer.perform_inference().

    Expected response format (primary):
        A JSON array of objects, each with "term" and "label" fields:
        [{"term": "Lactobacillus rhamnosus", "label": "bacteria"}, ...]

    Fallback formats (when the model does not produce the expected structure):
        1. JSON array of strings  -- treated as term-only output; label defaults to "NA".
        2. Bullet / dash / numbered list -- one "term: label" or plain term per line.
        3. Newline-separated pairs -- "term | label" or "term: label" on each line.
        4. Last resort -- whole response treated as a single term with label "NA".

    For each extracted (term, label) pair the parser performs a case-insensitive substring search over the source text to recover character offsets, producing one entity dict per occurrence.  
    Labels that do not match any entry in LABEL_LIST are remapped to "NA".
    """

    # Bullet / dash / numbered list item prefix patterns
    _BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*", re.MULTILINE)

    @staticmethod
    def parse(response: str, source_text: str, location: str) -> list[dict]:
        """
        Main entry point. Parses 'response' into a list of entity dicts, anchoring each extracted (term, label) pair to its position(s) in 'source_text'.

        :param response: Raw string returned by the LLM.
        :param source_text: The original text that was presented to the model.
        :param location: Section label embedded in every entity dict ("title" or "abstract").
        :return: A list of entity dicts, each with keys: 'start_idx', 'end_idx', 'location', 'text_span', 'label', 'confscore'.
        """
        term_label_pairs = ResponseParser._extract_term_label_pairs(response)
        entities = []

        for term, label in term_label_pairs:
            if not term.strip():
                continue
            # Normalise the label against the allowed set; fall back to "NA" on mismatch
            normalised_label = ResponseParser._normalise_label(label)
            spans = ResponseParser._find_spans(term, source_text)
            for start, end in spans:
                entities.append(
                    {
                        "start_idx": start,
                        # HFEntityRecognizer uses inclusive end; adjust accordingly
                        "end_idx": end - 1,
                        "location": location,
                        # Recover exact casing from source text
                        "text_span": source_text[start:end],
                        "label": normalised_label,
                        # LLMs do not expose per-span probabilities; use sentinel
                        "confscore": LLM_CONFSCORE_PLACEHOLDER,
                    }
                )

        return entities

    @staticmethod
    def _extract_term_label_pairs(response: str) -> list[tuple[str, str]]:
        """
        Attempts to extract a flat list of (term, label) pairs from the LLM response, trying each format in order of likelihood.

        Formats tried, in order:
            1. JSON array of {"term": ..., "label": ...} objects  -- primary format
            2. JSON array of strings                              -- term-only fallback (label="NA")
            3. Bullet / dash list with optional ": label" suffix
            4. Newline-separated "term | label" or "term: label" pairs
            5. Comma-separated "term: label" pairs
            6. Last resort: whole response as a single term (label="NA")

        :param response: Raw LLM output string.
        :return: A (possibly empty) list of (term, label) tuples.
        """
        # 1. JSON array: look for the outermost [...] and try to parse it
        json_match = re.search(r"\[.*?\]", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and parsed:
                    # Primary format: list of {"term": ..., "label": ...} dicts
                    if isinstance(parsed[0], dict):
                        pairs = []
                        for item in parsed:
                            term  = str(item.get("term",  "")).strip()
                            label = str(item.get("label", "NA")).strip()
                            if term:
                                pairs.append((term, label))
                        if pairs:
                            return pairs
                    # Fallback: list of plain strings (term-only; label defaults to "NA")
                    if isinstance(parsed[0], str):
                        return [(str(t).strip(), "NA") for t in parsed if t]
            except json.JSONDecodeError:
                pass

        # 2. Bullet / dash / numbered list -- optional ": label" suffix
        lines = response.splitlines()
        bullet_pairs = []
        for line in lines:
            if ResponseParser._BULLET_RE.match(line):
                content = ResponseParser._BULLET_RE.sub("", line).strip()
                term, label = ResponseParser._split_term_label(content)
                if term:
                    bullet_pairs.append((term, label))
        if bullet_pairs:
            return bullet_pairs

        # 3. Newline-separated "term | label" or "term: label" pairs
        pair_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            term, label = ResponseParser._split_term_label(line)
            if term:
                pair_lines.append((term, label))
        if len(pair_lines) > 1:
            return pair_lines

        # 4. Comma-separated "term: label" pairs
        if "," in response:
            pairs = []
            for segment in response.split(","):
                term, label = ResponseParser._split_term_label(segment.strip())
                if term:
                    pairs.append((term, label))
            if pairs:
                return pairs

        # 5. Last resort: treat the entire response as a single term, label unknown
        return [(response.strip(), "NA")]

    @staticmethod
    def _split_term_label(text: str) -> tuple[str, str]:
        """
        Attempts to split a string of the form "term | label" or "term: label" into its constituent parts.  
        Returns (text, "NA") when no separator is found.

        :param text: A candidate term-label string.
        :return: A (term, label) tuple.
        """
        for sep in [" | ", ": ", " - "]:
            if sep in text:
                parts = text.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        return text.strip(), "NA"

    @staticmethod
    def _normalise_label(label: str) -> str:
        """
        Maps a raw label string from the LLM response to the closest entry in LABEL_LIST using a case-insensitive exact match.  
        Falls back to "NA" when no match is found.

        :param label: Raw label string from the LLM.
        :return: A canonical label string drawn from LABEL_LIST, or "NA".
        """
        return _VALID_LABELS_LOWER.get(label.strip().lower(), "NA")

    @staticmethod
    def _find_spans(term: str, text: str) -> list[tuple[int, int]]:
        """
        Returns all (start, end) character-offset pairs where 'term' appears in 'text',
        using a case-insensitive search.

        :param term: The term string to locate.
        :param text: The source text to search within.
        :return: A list of (start_inclusive, end_exclusive) character index tuples.
        """
        spans = []
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
        return spans


#####################
# LLMEntityRecognizer #
#####################

class LLMEntityRecognizer:
    """
    Zero-shot LLM-based NER pipeline that jointly extracts and classifies biomedical entities.

    Given a dataset in the GBIE format (dict mapping paper IDs to content dicts with "title" and "abstract" fields), this class:
        1. Formats each text with the user-supplied prompts.
        2. Sends each text through the configured LLM provider.
        3. Parses the raw response into structured entity dicts with semantic labels.
        4. Returns results in the same format as HFEntityRecognizer.perform_inference().

    Responses are optionally cached to a JSONL checkpoint file so that long inference runs can be interrupted and resumed without re-processing already completed examples.

    The NER prompt is expected to elicit a JSON array of {"term": ..., "label": ...} objects.
    The ResponseParser handles deviations from this format gracefully.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        system_prompt: str,
        user_prompt_template: str,
        temperature: float = DEFAULT_TEMPERATURE,
        checkpoint_path: str | None = None,
    ):
        """
        Initializes the recognizer with an LLM provider and prompt configuration.

        :param provider: A configured BaseLLMProvider instance.
        :param system_prompt: The system-role message sent to the model for every inference call.
        :param user_prompt_template: A Python format string whose {text} placeholder will be filled with the source text on each call.
        :param temperature: Sampling temperature forwarded to the provider.
        :param checkpoint_path: Optional path to a JSONL file used for resumable inference checkpointing.  Already-processed example IDs are skipped on restart.
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.checkpoint_path = checkpoint_path

    # -- Public interface --

    def perform_inference(self, data: dict) -> dict:
        """
        Runs zero-shot NER over an entire dataset.

        :param data: A dict mapping paper IDs to content dicts (GBIE format).  Each content dict must have "title" and "abstract" fields.
        :return: A copy of 'data' with an "entities" list populated for every paper.  Each entity follows the HFEntityRecognizer schema: {start_idx, end_idx, location, text_span, label, confscore}.
        """
        # Load already-processed IDs from an existing checkpoint so that a re-run after
        # interruption skips completed examples.
        processed = self._load_checkpoint()

        result = {}

        with self._open_checkpoint() as ckpt_file:
            for paper_id, content in tqdm(
                data.items(), total=len(data), desc="LLM NER Inference"
            ):
                # Resume: reconstruct result dict from checkpoint
                if paper_id in processed:
                    result[paper_id] = processed[paper_id]
                    continue

                title, abstract = get_title_and_abstract(content)
                entity_predictions = []

                # Title and abstract are processed independently so the location field
                # ("title" / "abstract") can be set correctly on each entity.
                for location, text in [("title", title), ("abstract", abstract)]:
                    if not text:
                        continue
                    raw_response = self._call_llm(text)
                    entities = ResponseParser.parse(raw_response, text, location)
                    entity_predictions.extend(entities)

                output_content = dict(content)
                output_content["entities"] = entity_predictions
                result[paper_id] = output_content

                # Persist immediately so progress is not lost on crash
                if ckpt_file is not None:
                    ckpt_file.write(
                        json.dumps({"id": paper_id, "content": output_content}) + "\n"
                    )
                    ckpt_file.flush()

        return result

    def run_raw_inference(self, data: dict) -> dict:
        """
        Runs inference and returns the raw LLM responses instead of parsed entities.
        Useful for debugging prompt quality or post-processing with a custom parser.

        :param data: A dict mapping paper IDs to content dicts (GBIE format).
        :return: A dict mapping paper IDs to {"title_response": str, "abstract_response": str}.
        """
        processed_raw = self._load_raw_checkpoint()
        result = {}

        with self._open_checkpoint(suffix=".raw") as ckpt_file:
            for paper_id, content in tqdm(
                data.items(), total=len(data), desc="LLM NER Raw Inference"
            ):
                if paper_id in processed_raw:
                    result[paper_id] = processed_raw[paper_id]
                    continue

                title, abstract = get_title_and_abstract(content)
                entry = {
                    "title_response":    self._call_llm(title)    if title    else "",
                    "abstract_response": self._call_llm(abstract) if abstract else "",
                }
                result[paper_id] = entry

                if ckpt_file is not None:
                    ckpt_file.write(
                        json.dumps({"id": paper_id, **entry}) + "\n"
                    )
                    ckpt_file.flush()

        return result

    ####################
    # Internal helpers #
    ####################

    def _call_llm(self, text: str) -> str:
        """
        Formats the user prompt template with the given text and calls the provider.

        :param text: Source text to embed into the user_prompt_template.
        :return: Raw string response from the LLM.
        """
        user_prompt = self.user_prompt_template.format(text=text)
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

    def _load_raw_checkpoint(self) -> dict:
        """
        Reads a raw JSONL checkpoint (title / abstract responses) and returns a dict mapping paper IDs to their response strings.

        :return: Dict mapping paper_id -> {"title_response": str, "abstract_response": str}.
        """
        raw_path = (self.checkpoint_path + ".raw") if self.checkpoint_path else None
        if not raw_path or not os.path.exists(raw_path):
            return {}
        processed = {}
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    paper_id = entry.pop("id")
                    processed[paper_id] = entry
                except (json.JSONDecodeError, KeyError):
                    pass
        return processed

    def _open_checkpoint(self, suffix: str = ""):
        """
        Opens the checkpoint file for appending, or returns a null context if no checkpoint path was configured.

        :param suffix: Optional suffix appended to self.checkpoint_path (e.g. ".raw").
        :return: A file object opened in append mode, or a _NullContext instance.
        """
        if self.checkpoint_path is None:
            return _NullContext()
        path = self.checkpoint_path + suffix
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return open(path, "a", encoding="utf-8")


class _NullContext:
    """Minimal no-op context manager used when checkpointing is disabled."""

    def __enter__(self):
        return None

    def __exit__(self, *_):
        pass


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for zero-shot LLM NER inference.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot LLM-based biomedical NER (joint entity extraction + classification).  "
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
        help="Path to the JSON file to run inference on (GBIE format).",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        required=True,
        help="Path where inference results JSON will be written.",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "prompts.json"),
        help="Path to the prompts JSON file (default: prompts.json in the same directory).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional path to a JSONL checkpoint file.  Enables resumable inference: "
            "already-processed IDs are skipped on restart."
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

    # -- Optional evaluation --
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help=(
            "Optional path to a ground-truth JSON file.  When provided, strict (span+label) "
            "and lenient (span-only) P/R/F1 are printed after inference."
        ),
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

    # -- Raw output mode --
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Save raw LLM responses instead of parsed entities.  "
            "Useful for debugging prompt quality."
        ),
    )

    return parser.parse_args()


###############
# Entry point #
###############

def run_inference(args: argparse.Namespace) -> None:
    """
    Orchestrates provider initialization, inference, optional evaluation, and result persistence.

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

    # -- Build recognizer --
    recognizer = LLMEntityRecognizer(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
    )

    # -- Load inference data --
    data = load_json_data(args.inference_data_path)

    # -- Run inference --
    if args.raw:
        results = recognizer.run_raw_inference(data)
    else:
        results = recognizer.perform_inference(data)

    save_json_data(results, args.inference_output_path)
    print(f"Inference results saved to {args.inference_output_path}")

    # -- Optional evaluation --
    if args.eval_data_path and not args.raw:
        ground_truth = load_json_data(args.eval_data_path)
        evaluator = NERExtractionEvaluator()
        metrics = evaluator.evaluate(results, ground_truth)
        print(
            f"Strict   (span+label)  P: {metrics['strict_precision']:.4f} | "
            f"R: {metrics['strict_recall']:.4f} | "
            f"F1: {metrics['strict_f1']:.4f}"
        )
        print(
            f"Lenient  (span-only)   P: {metrics['span_precision']:.4f} | "
            f"R: {metrics['span_recall']:.4f} | "
            f"F1: {metrics['span_f1']:.4f}"
        )

    maybe_empty_cache(device)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)