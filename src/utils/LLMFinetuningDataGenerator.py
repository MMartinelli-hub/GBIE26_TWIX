#!/usr/bin/env python3
"""
LLMFinetuningDataGenerator.py

Offline JSONL fine-tuning dataset generator for the four LLM-based NLP tasks: term extraction, term classification, relation extraction, and entity recognition.

Given gold-annotated data in the GBIE format, each writer class produces one or two JSONL files (train / dev) whose lines are standard chat fine-tuning examples:

    {"messages": [{"role": "system",    "content": "..."},
                  {"role": "user",      "content": "..."},
                  {"role": "assistant", "content": "..."}]}

This format is accepted by OpenAI fine-tuning, Together AI, Azure OpenAI, and any other provider that follows the OpenAI chat fine-tuning wire format.

Organised into seven sections:
    - Constants : task name strings and default prompts paths
    - Writer classes : one per task, each mirroring the iteration logic of its corresponding LLM* inference class
    - Writer registry : maps task name strings to writer classes
    - Helper functions : _build_example(), build_writer(), write_split()
    - CLI : argument parsing and top-level entry point

Design notes
------------
This is a separate module rather than an extension of the individual LLM* modules
because:
    1. Fine-tuning data generation is conceptually orthogonal to inference; mixing the two would violate the single-responsibility principle of each LLM* module.
    2. The generator draws on artefacts from *all four* LLM* modules (prompts, entity marker helpers, relation marker helpers) making it a natural cross-cutting concern that lives above the individual task modules.
    3. No provider is instantiated and no model weights are loaded, keeping the generator lightweight and runnable without GPU access.

Each writer reuses the following from its corresponding LLM* class:
    - LLMTermExtractor : load_prompts(), user_prompt_template.format(text=...)
    - LLMTermClassifier : _insert_entity_markers() (static), ENTITY_START/END_TOKEN
    - LLMRelationExtractor : _insert_relation_markers() (static), marker token constants
    - LLMEntityRecognizer : user_prompt_template.format(text=...)  (same pattern)

Adding a new task requires only:
  1. Writing a writer class with a write_examples(data, output_path) -> int method.
  2. Adding an entry to WRITER_REGISTRY.
  3. Adding a default prompts path to _DEFAULT_PROMPTS_PATHS.

Entry point: run with --help to see CLI options.
"""

import argparse
import json
import os

from tqdm import tqdm

# Import shared utilities
from src.utils.utils import (
    get_title_and_abstract,
    load_json_data,
    VALID_RELATIONS,
)

# Import prompt loading from LLMTermExtractor — single source of truth for all tasks
from src.termExtractor.LLMTermExtractor import load_prompts  # noqa: F401  (re-exported)

# Import entity marker helpers and token constants from their defining modules.
# Reusing these directly prevents any divergence between the prompts seen at
# inference time and those seen during fine-tuning.
from src.termClassifier.LLMTermClassifier import (
    LLMTermClassifier,
    ENTITY_START_TOKEN,   # noqa: F401  (exposed for callers that build custom prompts)
    ENTITY_END_TOKEN,     # noqa: F401
)
from src.relationExtractor.LLMRelationExtractor import (
    LLMRelationExtractor,
    SUBJECT_START_TOKEN,  # noqa: F401
    SUBJECT_END_TOKEN,    # noqa: F401
    OBJECT_START_TOKEN,   # noqa: F401
    OBJECT_END_TOKEN,     # noqa: F401
)


#############
# Constants #
#############

# Task name strings used in WRITER_REGISTRY and the CLI --task argument
TASK_TERM_EXTRACTION     = "term_extraction"
TASK_TERM_CLASSIFICATION = "term_classification"
TASK_RELATION_EXTRACTION = "relation_extraction"
TASK_ENTITY_RECOGNITION  = "entity_recognition"


###################################
# TermExtractionFinetuningWriter  #
###################################

class TermExtractionFinetuningWriter:
    """
    Produces JSONL fine-tuning examples for the term extraction task.

    Mirrors the per-section iteration of LLMTermExtractor.perform_inference(): title and abstract are processed independently so each generates its own example.

    For every non-empty section one training example is emitted:
        user prompt -- section text embedded in user_prompt_template.
        target response -- JSON array of unique term surface forms found in the gold entities for that section (same output format the model is prompted to produce during inference).

    Empty sections (missing title or abstract) are silently skipped, again matching the inference-time behaviour.
    """

    def __init__(self, system_prompt: str, user_prompt_template: str,):
        """
        Initializes the writer with prompt configuration.

        :param system_prompt: The system-role message used for every training example.
        :param user_prompt_template: A Python format string whose {text} placeholder is filled with the section text, identical to the template used by LLMTermExtractor at inference time.
        """
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    # -- Public interface --

    def write_examples(self, data: dict, output_path: str) -> int:
        """
        Iterates over gold-annotated data and writes one JSONL example per section.

        :param data: A dict mapping paper IDs to content dicts (GBIE format). Each content dict must have "title" and "abstract" fields (under "metadata" or at the top level) and an "entities" list whose entries carry at least "location" and "text_span" keys.
        :param output_path: Path to the output JSONL file. Parent directories are created automatically if they do not exist.
        :return: Total number of examples written.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        count = 0

        with open(output_path, "w", encoding="utf-8") as out_file:
            for paper_id, content in tqdm(data.items(), total=len(data), desc="Writing term extraction examples"):
                title, abstract = get_title_and_abstract(content)

                # Title and abstract yield separate examples, mirroring LLMTermExtractor
                for location, text in [("title", title), ("abstract", abstract)]:
                    if not text:
                        continue

                    user_prompt = self._build_user_prompt(text)
                    target = self._build_target_response(content, location)
                    example = _build_example(f"{paper_id}_{location}", self.system_prompt, user_prompt, target)

                    out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1

        return count

    ####################
    # Internal helpers #
    ####################

    def _build_user_prompt(self, text: str) -> str:
        """
        Constructs the user-role message for a single section.
        Mirrors LLMTermExtractor._call_llm() without the provider call.

        :param text: The section text to embed in the prompt template.
        :return: Formatted user prompt string.
        """
        return self.user_prompt_template.format(text=text)

    def _build_target_response(self, content: dict, location: str) -> str:
        """
        Constructs the ideal assistant response from gold annotations.

        The target is a JSON array of unique term surface forms found in the gold entities for the requested section -- the exact format LLMTermExtractor is prompted to produce.  
        Deduplication preserves first-occurrence order.

        :param content: Paper content dict carrying an "entities" list.
        :param location: Section name ("title" or "abstract") used to filter entities.
        :return: JSON-serialised array of term strings (may be "[]" for empty sections).
        """
        seen:  set[str]   = set()
        terms: list[str]  = []

        for entity in content.get("entities", []):
            if entity.get("location") != location:
                continue
            span = entity.get("text_span", "")
            if span and span not in seen:
                seen.add(span)
                terms.append(span)

        return json.dumps(terms, ensure_ascii=False)


######################################
# TermClassificationFinetuningWriter #
######################################

class TermClassificationFinetuningWriter:
    """
    Produces JSONL fine-tuning examples for the term classification task.

    Mirrors the per-entity iteration of LLMTermClassifier.perform_inference().

    For every gold entity in the dataset one training example is emitted:
        user prompt -- section text with [E1] / [/E1] markers around the entity, formatted with user_prompt_template (identical to inference).
        target response  -- the gold entity label string (e.g. "bacteria", "drug").

    Entities missing required fields (section text, start/end indices, or label) are silently skipped, matching the fallback behaviour of the inference loop.
    """

    def __init__(self, system_prompt: str, user_prompt_template: str,):
        """
        Initializes the writer with prompt configuration.

        :param system_prompt: The system-role message used for every training example.
        :param user_prompt_template: A Python format string whose {text} placeholder is filled with the marked entity text, identical to the template used by LLMTermClassifier at inference time.
        """
        self.system_prompt        = system_prompt
        self.user_prompt_template = user_prompt_template

    # -- Public interface --

    def write_examples(self, data: dict, output_path: str) -> int:
        """
        Iterates over gold-annotated data and writes one JSONL example per entity.

        :param data: A dict mapping paper IDs to content dicts (GBIE format). Each content dict must have a "metadata" field (or equivalent top-level "title" / "abstract" keys) and an "entities" list whose entries carry "location", "start_idx", "end_idx", and "label" keys.
        :param output_path: Path to the output JSONL file. Parent directories are created automatically if they do not exist.
        :return: Total number of examples written.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        count = 0

        with open(output_path, "w", encoding="utf-8") as out_file:
            for paper_id, content in tqdm(
                data.items(), total=len(data), desc="Writing term classification examples"
            ):
                # metadata may live under a "metadata" key or at the top level
                metadata = content.get("metadata", content)

                for idx, entity in enumerate(content.get("entities", [])):
                    location     = entity.get("location", "abstract")
                    section_text = metadata.get(location, "")
                    start_idx    = entity.get("start_idx")
                    end_idx      = entity.get("end_idx")
                    label        = entity.get("label")

                    # Skip entities that lack the required fields; mirrors the
                    # "unclassifiable" guard inside LLMTermClassifier.perform_inference()
                    if not section_text or start_idx is None or end_idx is None or not label:
                        continue

                    # end_idx is inclusive in the GBIE format; _insert_entity_markers
                    # expects an exclusive end, so we add 1 -- identical to inference
                    marked_text = LLMTermClassifier._insert_entity_markers(
                        section_text, start_idx, end_idx + 1
                    )
                    user_prompt = self._build_user_prompt(marked_text)
                    example     = _build_example(f"{paper_id}_{idx}", self.system_prompt, user_prompt, label)

                    out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1

        return count

    ####################
    # Internal helpers #
    ####################

    def _build_user_prompt(self, marked_text: str) -> str:
        """
        Constructs the user-role message for a single entity.
        Mirrors LLMTermClassifier._call_llm() without the provider call.

        :param marked_text: Section text with [E1] / [/E1] entity markers inserted.
        :return: Formatted user prompt string.
        """
        return self.user_prompt_template.format(text=marked_text)


############################################
# RelationExtractionFinetuningWriter       #
############################################

class RelationExtractionFinetuningWriter:
    """
    Produces JSONL fine-tuning examples for the relation extraction task.

    Mirrors the pair-enumeration logic of LLMRelationExtractor.perform_inference() (both single-section and concatenated-section modes).

    For every ordered entity pair whose (head_label, tail_label) combination appears in
    VALID_RELATIONS, one training example is emitted:
        user prompt -- section text (or concatenated title+abstract) with [E1]/[/E1] and [E2]/[/E2] markers for the pair, formatted with user_prompt_template (identical to inference).
        target response -- the gold predicate string if a relation exists between the pair in the annotations (e.g. "interact", "located in"), or "NA" if no relation is annotated for this structurally valid pair.

    The include_negatives flag controls whether "NA" examples are written (default: True).
    Setting it to False retains only positive (non-NA) examples, which can be useful when the negative-to-positive ratio is too high.
    """

    def __init__(self, system_prompt: str, user_prompt_template: str, include_negatives: bool = True, concatenate_title_abstract: bool = False,):
        """
        Initializes the writer with prompt configuration and relation-extraction options.

        :param system_prompt: The system-role message used for every training example.
        :param user_prompt_template: A Python format string whose {text} placeholder is filled with the marked entity-pair text, identical to the template used by LLMRelationExtractor at inference time.
        :param include_negatives: When True (default), structurally valid pairs that have no gold relation are included as "NA" examples.  Set to False to emit only positive (non-NA) examples.
        :param concatenate_title_abstract: When True, title and abstract are concatenated before enumerating pairs, enabling cross-section entity pairs. Mirrors LLMRelationExtractor with the same flag.
        """
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.include_negatives = include_negatives
        self.concatenate_title_abstract = concatenate_title_abstract

    # -- Public interface --

    def write_examples(self, data: dict, output_path: str) -> int:
        """
        Iterates over gold-annotated data and writes one JSONL example per valid entity pair.

        :param data: A dict mapping paper IDs to content dicts (GBIE format). Each content dict must have "metadata" (with "title" and "abstract"), an "entities" list, and a "relations" list.
        :param output_path: Path to the output JSONL file. Parent directories are created automatically if they do not exist.
        :return: Total number of examples written.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        count = 0

        with open(output_path, "w", encoding="utf-8") as out_file:
            for paper_id, content in tqdm( data.items(), total=len(data), desc="Writing relation extraction examples"):
                title, abstract = get_title_and_abstract(content)

                if self.concatenate_title_abstract:
                    candidate_examples = self._examples_concatenated(
                        content, title, abstract
                    )
                else:
                    candidate_examples = (
                        self._examples_single_section(content, "title",    title)
                        + self._examples_single_section(content, "abstract", abstract)
                    )

                pos_idx, neg_idx = 0, 0
                for system_p, user_p, target in candidate_examples:
                    if target == "NA":
                        if not self.include_negatives:
                            continue
                        else:
                            neg_idx += 1
                            idx = f"neg{neg_idx}"
                    else:
                        pos_idx += 1
                        idx = f"pos{pos_idx}"
                    out_file.write(
                        json.dumps(
                            _build_example(f"{paper_id}_{idx}", system_p, user_p, target),
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    count += 1

        return count

    ####################
    # Internal helpers #
    ####################

    def _examples_single_section(self, content: dict, section: str, text: str,) -> list[tuple[str, str, str]]:
        """
        Generates (system_prompt, user_prompt, target) tuples for all valid entity pairs within a single section.
        Mirrors LLMRelationExtractor._extract_relations_single_section() exactly.

        :param content: Paper content dict.
        :param section: Section name ("title" or "abstract").
        :param text: Section text.
        :return: List of (system_prompt, user_prompt, gold_predicate_or_"NA") tuples.
        """
        if not text:
            return []

        results: list[tuple[str, str, str]] = []

        # Build a fast lookup for gold-annotated relations in this section
        gold_lookup = self._build_gold_lookup(
            content.get("relations", []), section=section
        )

        section_entities = [
            e for e in content.get("entities", []) if e.get("location") == section
        ]

        # Enumerate all ordered pairs; outer entity is treated as subject ([E1]), inner entity as object ([E2]) -- identical to LLMRelationExtractor
        for i, subj in enumerate(section_entities):
            for obj in section_entities[i + 1:]:
                head_label = subj.get("label", "term")
                tail_label = obj.get("label", "term")
                if (head_label, tail_label) not in VALID_RELATIONS:
                    continue

                marked_text = LLMRelationExtractor._insert_relation_markers(
                    text,
                    subj["start_idx"],
                    subj["end_idx"] + 1,   # end_idx inclusive → exclusive
                    obj["start_idx"],
                    obj["end_idx"]  + 1,
                )
                user_prompt = self._build_user_prompt(marked_text)
                target      = gold_lookup.get(
                    (subj["start_idx"], subj["end_idx"],
                     obj["start_idx"],  obj["end_idx"]),
                    "NA",
                )
                results.append((self.system_prompt, user_prompt, target))

        return results

    def _examples_concatenated(self, content: dict, title: str, abstract: str,) -> list[tuple[str, str, str]]:
        """
        Generates (system_prompt, user_prompt, target) tuples for all valid entity pairs in the concatenated title + abstract text, supporting cross-section pairs.
        Mirrors LLMRelationExtractor._extract_relations_concatenated() exactly.

        :param content: Paper content dict.
        :param title: Title text.
        :param abstract: Abstract text.
        :return: List of (system_prompt, user_prompt, gold_predicate_or_"NA") tuples.
        """
        concatenated_text = f"{title} {abstract}"
        title_length      = len(title)
        separator_length  = 1   # single space between title and abstract

        results: list[tuple[str, str, str]] = []

        # Gold lookup with no section filter to cover cross-section relations
        gold_lookup = self._build_gold_lookup(
            content.get("relations", []), section=None
        )

        # Adjust abstract entity indices for the concatenated text.
        # The original location is preserved so we can reverse the offset when computing gold lookup keys (which use original, per-section indices).
        all_entities: list[dict] = []
        for entity in content.get("entities", []):
            adj = dict(entity)
            if entity.get("location") == "abstract":
                offset           = title_length + separator_length
                adj["start_idx"] = entity["start_idx"] + offset
                adj["end_idx"]   = entity["end_idx"]   + offset
            all_entities.append(adj)

        for i, subj in enumerate(all_entities):
            for obj in all_entities[i + 1:]:
                head_label = subj.get("label", "term")
                tail_label = obj.get("label", "term")
                if (head_label, tail_label) not in VALID_RELATIONS:
                    continue

                marked_text = LLMRelationExtractor._insert_relation_markers(
                    concatenated_text,
                    subj["start_idx"],
                    subj["end_idx"] + 1,
                    obj["start_idx"],
                    obj["end_idx"]  + 1,
                )
                user_prompt = self._build_user_prompt(marked_text)

                # The gold lookup keys use original (per-section) indices, so we
                # reverse the concatenation offset for abstract entities
                offset = title_length + separator_length
                orig_subj_start = subj["start_idx"] - (offset if subj.get("location") == "abstract" else 0)
                orig_subj_end   = subj["end_idx"]   - (offset if subj.get("location") == "abstract" else 0)
                orig_obj_start  = obj["start_idx"]  - (offset if obj.get("location")  == "abstract" else 0)
                orig_obj_end    = obj["end_idx"]     - (offset if obj.get("location")  == "abstract" else 0)

                target = gold_lookup.get(
                    (orig_subj_start, orig_subj_end, orig_obj_start, orig_obj_end),
                    "NA",
                )
                results.append((self.system_prompt, user_prompt, target))

        return results

    @staticmethod
    def _build_gold_lookup(relations: list[dict], section: str | None,) -> dict[tuple[int, int, int, int], str]:
        """
        Builds a (subj_start, subj_end, obj_start, obj_end) → predicate lookup from gold-annotated relations.

        :param relations: List of gold relation dicts from the GBIE format.
        :param section: When not None, only relations where both subject and object are in this section are included. Pass None to include all relations (used in concatenated mode where cross-section pairs are valid).
        :return: Dict mapping 4-tuple entity-pair index keys to their gold predicate.
        """
        lookup: dict[tuple[int, int, int, int], str] = {}

        for rel in relations:
            # In single-section mode both endpoints must reside in the requested section
            if section is not None:
                if (rel.get("subject_location") != section
                        or rel.get("object_location") != section):
                    continue

            key = (
                rel["subject_start_idx"],
                rel["subject_end_idx"],
                rel["object_start_idx"],
                rel["object_end_idx"],
            )
            lookup[key] = rel["predicate"]

        return lookup

    def _build_user_prompt(self, marked_text: str) -> str:
        """
        Constructs the user-role message for a single entity pair.
        Mirrors LLMRelationExtractor._call_llm() without the provider call.

        :param marked_text: Text with [E1]/[/E1] and [E2]/[/E2] markers inserted.
        :return: Formatted user prompt string.
        """
        return self.user_prompt_template.format(text=marked_text)


########################################
# EntityRecognitionFinetuningWriter    #
########################################

class EntityRecognitionFinetuningWriter:
    """
    Produces JSONL fine-tuning examples for the joint NER task (entity recognition).

    Mirrors the per-section iteration of LLMEntityRecognizer.perform_inference().

    For every non-empty section one training example is emitted:
        user prompt -- section text embedded in user_prompt_template.
        target response -- JSON array of {"term": ..., "label": ...} objects drawn from gold entities for that section, deduplicated by (surface_form, label) pair and preserving first-occurrence order -- the exact output format the model is prompted to produce during inference.
    """

    def __init__(self, system_prompt: str, user_prompt_template: str,):
        """
        Initializes the writer with prompt configuration.

        :param system_prompt: The system-role message used for every training example.
        :param user_prompt_template: A Python format string whose {text} placeholder is filled with the section text, identical to the template used by LLMEntityRecognizer at inference time.
        """
        self.system_prompt        = system_prompt
        self.user_prompt_template = user_prompt_template

    # -- Public interface --

    def write_examples(self, data: dict, output_path: str) -> int:
        """
        Iterates over gold-annotated data and writes one JSONL example per section.

        :param data: A dict mapping paper IDs to content dicts (GBIE format). Each content dict must have "title" and "abstract" fields and an "entities" list whose entries carry "location", "text_span", and "label" keys.
        :param output_path: Path to the output JSONL file. Parent directories are created automatically if they do not exist.
        :return: Total number of examples written.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        count = 0

        with open(output_path, "w", encoding="utf-8") as out_file:
            for paper_id, content in tqdm(
                data.items(), total=len(data), desc="Writing NER examples"
            ):
                title, abstract = get_title_and_abstract(content)

                # Title and abstract yield separate examples, mirroring LLMEntityRecognizer
                for location, text in [("title", title), ("abstract", abstract)]:
                    if not text:
                        continue

                    user_prompt = self._build_user_prompt(text)
                    target      = self._build_target_response(content, location)
                    example     = _build_example(f"{paper_id}_{location}", self.system_prompt, user_prompt, target)

                    out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1

        return count

    ####################
    # Internal helpers #
    ####################

    def _build_user_prompt(self, text: str) -> str:
        """
        Constructs the user-role message for a single section.
        Mirrors LLMEntityRecognizer._call_llm() without the provider call.

        :param text: The section text to embed in the prompt template.
        :return: Formatted user prompt string.
        """
        return self.user_prompt_template.format(text=text)

    def _build_target_response(self, content: dict, location: str) -> str:
        """
        Constructs the ideal assistant response from gold annotations.

        The target is a JSON array of {"term": ..., "label": ...} objects -- the exact format LLMEntityRecognizer is prompted to produce.  
        Deduplication is performed on (term, label) pairs to avoid redundant entries from repeated spans.

        :param content: Paper content dict carrying an "entities" list.
        :param location: Section name ("title" or "abstract") used to filter entities.
        :return: JSON-serialised array of {"term": ..., "label": ...} objects (may be "[]" for sections with no annotated entities).
        """
        seen:  set[tuple[str, str]]     = set()
        pairs: list[dict[str, str]]     = []

        for entity in content.get("entities", []):
            if entity.get("location") != location:
                continue
            term  = entity.get("text_span", "")
            label = entity.get("label", "NA")
            key   = (term, label)
            if term and key not in seen:
                seen.add(key)
                pairs.append({"term": term, "label": label})

        return json.dumps(pairs, ensure_ascii=False)


###################
# Writer registry #
###################

# Maps task name strings (as used in the CLI) to their writer classes.
# To add a new task, insert an entry here; no other code needs changing.
WRITER_REGISTRY: dict[str, type] = {
    TASK_TERM_EXTRACTION:     TermExtractionFinetuningWriter,
    TASK_TERM_CLASSIFICATION: TermClassificationFinetuningWriter,
    TASK_RELATION_EXTRACTION: RelationExtractionFinetuningWriter,
    TASK_ENTITY_RECOGNITION:  EntityRecognitionFinetuningWriter,
}

# Default prompts.json paths for each task.
# Mirrors the --prompts_path defaults in the corresponding LLM* modules.
_DEFAULT_PROMPTS_PATHS: dict[str, str] = {
    TASK_TERM_EXTRACTION:     "src/termExtractor/prompts.json",
    TASK_TERM_CLASSIFICATION: "src/termClassifier/prompts.json",
    TASK_RELATION_EXTRACTION: "src/relationExtractor/prompts.json",
    TASK_ENTITY_RECOGNITION:  "src/entityRecognizer/prompts.json",
}


####################
# Helper functions #
####################

def _build_example(id:str, system_prompt: str, user_prompt: str, target_response: str,) -> dict:
    """
    Assembles a single fine-tuning example in the standard chat messages format.

    The format is accepted by OpenAI fine-tuning, Together AI, Azure OpenAI, and any other provider that follows the OpenAI chat fine-tuning wire format.

    :param id: The ID of the example.
    :param system_prompt: System-role message string.
    :param user_prompt: User-role message string.
    :param target_response: The ideal assistant response (gold label, predicate, or JSON array string depending on the task).
    :return: A dict with a single "messages" key containing the three-turn conversation.
    """
    return {
        "id": id,
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": target_response},
        ]
    }


def build_writer(task: str, system_prompt: str, user_prompt_template: str, include_negatives: bool = True, concatenate_title_abstract: bool = False,) -> object:
    """
    Factory function that instantiates the correct writer class from WRITER_REGISTRY.

    Relation-extraction-specific arguments (include_negatives, concatenate_title_abstract) are forwarded only to RelationExtractionFinetuningWriter and are silently ignored by the other three writers.

    :param task: One of the keys in WRITER_REGISTRY (e.g. "term_extraction", "relation_extraction").
    :param system_prompt: System-role prompt string.
    :param user_prompt_template: User-role prompt template (with {text} placeholder).
    :param include_negatives: When True, structurally valid entity pairs with no gold relation are written as "NA" training examples (relation extraction only).
    :param concatenate_title_abstract: When True, title and abstract are joined before pair enumeration (relation extraction only).
    :return: A configured writer instance.
    :raises ValueError: If task is not in WRITER_REGISTRY.
    """
    if task not in WRITER_REGISTRY:
        raise ValueError(
            f"Unknown task '{task}'.  "
            f"Available tasks: {sorted(WRITER_REGISTRY.keys())}"
        )

    cls = WRITER_REGISTRY[task]

    # RelationExtractionFinetuningWriter accepts two extra keyword arguments that the other writers do not expose; route them only when appropriate.
    if task == TASK_RELATION_EXTRACTION:
        return cls(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            include_negatives=include_negatives,
            concatenate_title_abstract=concatenate_title_abstract,
        )

    return cls(
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )


def write_split(writer: object, data_path: str, output_path: str, split_name: str = "",) -> int:
    """
    Loads a gold-annotated JSON file and writes fine-tuning examples for one data split.

    This is a convenience wrapper that combines data loading with the writer's write_examples() call, producing the same progress reporting style as the run_inference() functions in the LLM* modules.

    :param writer: A configured writer instance (any of the four writer classes).
    :param data_path: Path to the gold-annotated JSON input file (GBIE format).
    :param output_path: Path to the output JSONL file.
    :param split_name: Human-readable label printed in progress messages (e.g. "train", "dev").  May be empty.
    :return: Total number of examples written.
    """
    label = f"[{split_name}] " if split_name else ""
    print(f"{label}Loading data from:             {data_path}")
    data  = load_json_data(data_path)
    print(f"{label}Writing fine-tuning examples to: {output_path}")
    count = writer.write_examples(data, output_path)
    print(f"{label}Done -- {count} examples written.")
    return count


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for JSONL fine-tuning data generation.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate JSONL fine-tuning datasets from gold-annotated GBIE data.  "
            "Produces chat-formatted examples compatible with OpenAI fine-tuning, "
            "Together AI, Azure OpenAI, and any provider that accepts the OpenAI "
            "chat fine-tuning wire format."
        )
    )

    # -- Task selection --
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=sorted(WRITER_REGISTRY.keys()),
        help=(
            "NLP task for which to generate fine-tuning data.  "
            "Each task uses the prompts and input format of its corresponding "
            "LLM* inference module."
        ),
    )

    # -- Data paths --
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the gold-annotated training JSON file (GBIE format).",
    )
    parser.add_argument(
        "--train_output_path",
        type=str,
        required=True,
        help="Path where the training JSONL file will be written.",
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default=None,
        help="Optional path to a gold-annotated development JSON file (GBIE format).",
    )
    parser.add_argument(
        "--dev_output_path",
        type=str,
        default=None,
        help=(
            "Path where the development JSONL file will be written.  "
            "Required when --dev_data_path is provided."
        ),
    )

    # -- Prompt configuration --
    parser.add_argument(
        "--prompts_path",
        type=str,
        default=None,
        help=(
            "Path to the prompts JSON file.  "
            "When omitted, the standard prompts.json for the selected task is used."
        ),
    )
    parser.add_argument(
        "--system_prompt_key",
        type=str,
        default="base",
        help="Key selecting the system prompt variant from the prompts file (default: base).",
    )
    parser.add_argument(
        "--user_prompt_key",
        type=str,
        default="base",
        help="Key selecting the user prompt variant from the prompts file (default: base).",
    )

    # -- Relation extraction specific --
    parser.add_argument(
        "--include_negatives",
        action="store_true",
        default=True,
        help=(
            "[relation_extraction only] Include structurally valid entity pairs that "
            "have no gold relation as 'NA' training examples (default: True)."
        ),
    )
    parser.add_argument(
        "--exclude_negatives",
        dest="include_negatives",
        action="store_false",
        help=(
            "[relation_extraction only] Suppress 'NA' examples; write only positive "
            "(non-NA) relation training examples."
        ),
    )
    parser.add_argument(
        "--concatenate_title_abstract",
        action="store_true",
        help=(
            "[relation_extraction only] Concatenate title and abstract before "
            "enumerating entity pairs, enabling cross-section relation examples."
        ),
    )

    return parser.parse_args()


###############
# Entry point #
###############

def run(args: argparse.Namespace) -> None:
    """
    Orchestrates prompt loading, writer construction, and JSONL generation.

    :param args: Parsed CLI arguments from parse_args().
    :return: None
    """
    # Resolve prompts path: explicit CLI argument takes precedence over per-task default
    prompts_path = args.prompts_path or _DEFAULT_PROMPTS_PATHS[args.task]

    # Load system and user prompts from the JSON file
    system_prompt, user_prompt_template = load_prompts(
        prompts_path=prompts_path,
        system_key=args.system_prompt_key,
        user_key=args.user_prompt_key,
    )

    # Validate dev arguments before any I/O work starts
    if args.dev_data_path and not args.dev_output_path:
        raise ValueError(
            "--dev_output_path is required when --dev_data_path is provided."
        )

    # Build the writer for the selected task
    writer = build_writer(
        task=args.task,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        include_negatives=args.include_negatives,
        concatenate_title_abstract=args.concatenate_title_abstract,
    )

    # Write training split
    write_split(
        writer=writer,
        data_path=args.train_data_path,
        output_path=args.train_output_path,
        split_name="train",
    )

    # Optionally write development split using the same writer configuration
    if args.dev_data_path:
        write_split(
            writer=writer,
            data_path=args.dev_data_path,
            output_path=args.dev_output_path,
            split_name="dev",
        )


if __name__ == "__main__":
    args = parse_args()
    run(args)