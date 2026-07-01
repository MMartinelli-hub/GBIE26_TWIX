"""
Tokenizers.py

Tokenizer classes for transformer-based NLP tasks, organized into four classes:
    - BaseTokenizer : abstract base defining the shared preprocessing contract
    - BIOTokenizer : BIO-tag preprocessing for term extraction
    - TermClassificationTokenizer : marker-based preprocessing for term classification
    - RelationTokenizer : entity-pair-marker preprocessing for relation extraction

Both concrete tokenizers share the same outer loop (process_files / _process_paper delegation) via BaseTokenizer; only the per-paper logic differs.
"""

import random
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from src.utils.utils import get_title_and_abstract, VALID_RELATIONS

import re

#################
# BaseTokenizer #
#################

class BaseTokenizer(ABC):
    """
    Abstract base class for all task-specific tokenizers.

    Provides the shared outer preprocessing loop (process_files) and enforces a uniform interface via the abstract _process_paper hook. 
    Concrete subclasses implement only the task-specific per-paper logic.

    Subclasses:
        - BIOTokenizer : BIO tagging for term extraction
        - TermClassificationTokenizer : entity-marker encoding for term classification
        - RelationTokenizer : entity-pair-marker encoding for relation extraction
    """

    def __init__(self, data: dict, tokenizer, label2id: dict[str, int], max_length: int = 512,):
        """
        Initializes the tokenizer with dataset, HuggingFace tokenizer, label mapping, and sequence length cap.

        :param data: A dictionary of paper IDs -> content dicts (GBIE format).
        :param tokenizer: A HuggingFace tokenizer instance.
        :param label2id: A dictionary mapping label strings to integer IDs.
        :param max_length: Maximum token sequence length (default 512).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def process_files(self) -> list:
        """
        Iterates over all papers in the dataset and delegates per-paper
        processing to _process_paper.

        :return: A flat list of sample dictionaries ready for use in a Dataset.
        """
        samples = []
        for _, content in tqdm(self.data.items(), total=len(self.data), desc="Preprocessing"):
            samples.extend(self._process_paper(content))
        return samples

    @abstractmethod
    def _process_paper(self, content: dict) -> list:
        """
        Produces a list of tokenized samples for a single paper.

        :param content: A dictionary containing the paper's metadata and annotations, in the GBIE format.
        :return: A list of sample dicts, each containing at minimum 'input_ids', 'attention_mask', and 'labels'.
        """


################
# BIOTokenizer #
################

class BIOTokenizer(BaseTokenizer):
    """
    Preprocesses raw GBIE data into BIO-tagged sequences for term extraction.

    Extracts the title and abstract from each paper, tokenizes the text, and assigns BIO tags to each token based on the entity annotations. 
    Optionally concatenates the title and abstract into a single sequence.

    The output is a list of dictionaries containing the input IDs, attention masks, and label IDs for each token in the sequence.
    """

    def __init__(self, data: dict, tokenizer, label2id: dict[str, int], max_length: int = 512, concatenate_title_abstract: bool = True,):
        """
        Initializes the BIOTokenizer with the given data, tokenizer, label mapping, and configuration options.

        :param data: A dictionary containing the raw input data, in the GBIE format.
        :param tokenizer: A tokenizer instance from the HuggingFace Transformers library.
        :param label2id: A dictionary mapping BIO tags to their corresponding label IDs.
        :param max_length: The maximum sequence length for tokenization (default is 512).
        :param concatenate_title_abstract: Whether to concatenate the title and abstract into a single sequence (default is True). If False, each section is processed separately.
        """
        super().__init__(data, tokenizer, label2id, max_length)
        self.concatenate_title_abstract = concatenate_title_abstract

    def _process_paper(self, content: dict) -> list:
        """
        Produces BIO-tagged samples for all text sections in a single paper.

        :param content: A dictionary containing the paper's metadata and annotations, in the GBIE format.
        :return: A list of sample dicts, each with 'input_ids', 'attention_mask', and 'labels' tensors.
        """
        processed = []
        text_lst, entities_lst = self._extract_entities(content)

        for text, entities in zip(text_lst, entities_lst):
            label_ids, input_ids, attention_mask = self._tokenize_with_bio(text, entities)
            processed.append(
                {
                    "labels": torch.as_tensor(label_ids, dtype=torch.long),
                    "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.as_tensor(attention_mask, dtype=torch.long),
                }
            )

        return processed

    def _extract_entities(self, content: dict) -> tuple[list, list]:
        """
        Extracts the title, abstract, and entities from the paper's content.
        Depending on the configuration, it either concatenates the title and abstract or processes them separately.

        :param content: A dictionary containing the paper's metadata and annotations, in the GBIE format.
        :return: A tuple containing a list of text sections (title and/or abstract) and a list of entity annotation lists.
        """
        title, abstract = get_title_and_abstract(content)
        text_lst = []
        entities_lst = []

        if self.concatenate_title_abstract:
            entities_lst.append(
                [
                    {
                        **entity,
                        "label": "term",
                        "start_idx": entity["start_idx"] + len(title) + 1,
                        "end_idx": entity["end_idx"] + len(title) + 1,
                    }
                    if entity["location"] == "abstract"
                    else {**entity, "label": "term"}
                    for entity in content.get("entities", [])
                ]
            )
            text_lst.append(f"{title} {abstract}")
        else:
            for section in ["title", "abstract"]:
                entities_lst.append(
                    [
                        {**entity, "label": "term"}
                        for entity in content.get("entities", [])
                        if entity["location"] == section
                    ]
                )
                text_lst.append(title if section == "title" else abstract)

        return text_lst, entities_lst

    def _tokenize_with_bio(self, text: str, entities: list) -> tuple[list, list, list]:
        """
        Tokenizes the input text and assigns BIO tags to each token based on the provided entity annotations.

        :param text: The input text to be tokenized.
        :param entities: A list of entity annotations, where each entity is a dictionary containing 'start_idx', 'end_idx', and 'label' keys.
        :return: A tuple of (label_ids, input_ids, attention_mask), each a plain list.
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        offsets = encoding["offset_mapping"]
        bio_tags = ["O"] * len(offsets)

        for entity in entities:
            first_token_assigned = False
            for i, (token_start, token_end) in enumerate(offsets):
                if i >= 1 and (token_start, token_end) == (0, 0):
                    break

                if token_start >= entity["start_idx"] and token_end <= entity["end_idx"] + 1:
                    if not first_token_assigned:
                        bio_tags[i] = "B-term"
                        first_token_assigned = True
                    else:
                        bio_tags[i] = "I-term"

        label_ids = []
        for offset, tag in zip(offsets, bio_tags):
            if offset == (0, 0):
                label_ids.append(-100)
            else:
                label_ids.append(self.label2id[tag])

        return label_ids, encoding["input_ids"], encoding["attention_mask"]


################################
# TermClassificationTokenizer  #
################################

class TermClassificationTokenizer(BaseTokenizer):
    """
    Preprocesses raw GBIE data into entity-marker-encoded sequences for term classification.

    For each annotated entity, the corresponding text span is wrapped with [E1] / [/E1] markers and the resulting sequence is tokenized. 
    The representation at the [E1] position is later used by the classifier head.

    Optionally generates negative (NA-labelled) samples by sampling random non-overlapping spans from the same text to balance the training set.
    """

    def __init__(self, data: dict, tokenizer, label2id: dict[str, int], max_length: int = 512, negative_sample_multiplier: int = 1, max_negative_span_words: int = 6,):
        """
        Initializes the TermClassificationTokenizer with dataset, tokenizer, label mapping, and negative-sampling configuration.

        :param data: A dictionary containing the raw input data, in the GBIE format.
        :param tokenizer: A HuggingFace tokenizer instance (must already have [E1] / [/E1] added to its vocabulary).
        :param label2id: A dictionary mapping category label strings to integer IDs.
        :param max_length: Maximum token sequence length (default 512).
        :param negative_sample_multiplier: Number of negative (NA) samples to generate per positive entity (default 1).
        :param max_negative_span_words: Maximum number of whitespace-separated words in a sampled negative span (default 6).
        """
        super().__init__(data, tokenizer, label2id, max_length)
        self.negative_sample_multiplier = negative_sample_multiplier
        self.max_negative_span_words = max_negative_span_words

    def _process_paper(self, content: dict) -> list:
        """
        Produces entity-marker-encoded samples for all sections in a single paper, including any generated negative samples.

        :param content: A dictionary containing the paper's metadata and annotations, in the GBIE format.
        :return: A list of sample dicts, each with 'input_ids', 'attention_mask', and 'labels' (as a plain integer).
        """
        samples = []

        for section in ["title", "abstract"]:
            text = content["metadata"][section]
            section_entities = [
                entity for entity in content.get("entities", [])
                if entity["location"] == section
            ]

            # -- Positive samples: one per annotated entity --
            for entity in section_entities:
                marked_text = self._insert_entity_markers(
                    text, entity["start_idx"], entity["end_idx"] + 1
                )
                input_ids, attention_mask = self._tokenize(marked_text)
                samples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": self.label2id[entity["label"]],
                    }
                )

            # -- Negative samples: randomly sampled non-overlapping spans --
            negative_spans = self._sample_negative_spans(text, section_entities)
            for start_idx, end_idx in negative_spans:
                marked_text = self._insert_entity_markers(text, start_idx, end_idx + 1)
                input_ids, attention_mask = self._tokenize(marked_text)
                samples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": self.label2id["NA"],
                    }
                )

        return samples

    def _tokenize(self, marked_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes a single entity-marked text string.

        :param marked_text: Input text with [E1] / [/E1] markers inserted.
        :return: A tuple of (input_ids, attention_mask) as 1-D LongTensors.
        """
        encoding = self.tokenizer(
            marked_text,
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

    def _insert_entity_markers(self, text: str, start_idx: int, end_idx: int) -> str:
        """
        Wraps the span [start_idx, end_idx) in the source text with [E1] / [/E1]
        markers.

        :param text: The original section text.
        :param start_idx: Character-level start index of the span (inclusive).
        :param end_idx: Character-level end index of the span (exclusive).
        :return: The text with entity markers inserted.
        """
        return f"{text[:start_idx]}[E1]{text[start_idx:end_idx]}[/E1]{text[end_idx:]}"

    def _sample_negative_spans(self, text: str, entities: list) -> list[tuple[int, int]]:
        """
        Randomly samples non-overlapping character spans to serve as negative (NA-labelled) examples.

        :param text: The source section text.
        :param entities: List of positive entity dicts for this section.
        :return: A list of (start_idx, end_idx) tuples for the selected spans.
        """
        number_negative_samples = len(entities) * self.negative_sample_multiplier
        if number_negative_samples == 0:
            return []

        positive_spans = [(entity["start_idx"], entity["end_idx"]) for entity in entities]
        candidate_spans = self._generate_candidate_negative_spans(text, positive_spans)

        random.shuffle(candidate_spans)
        return candidate_spans[:number_negative_samples]

    def _generate_candidate_negative_spans(self, text: str, positive_spans: list[tuple[int, int]],) -> list[tuple[int, int]]:
        """
        Enumerates all valid negative candidate spans up to max_negative_span_words words in length, filtering out any span that overlaps with a positive entity.

        :param text: The source section text.
        :param positive_spans: List of (start_idx, end_idx) tuples for positive entities.
        :return: A deduplicated list of (start_idx, end_idx) candidate negative spans.
        """
        candidates = []
        words = list(self._iter_word_spans(text))

        for start_word_idx in range(len(words)):
            for span_length in range(1, self.max_negative_span_words + 1):
                end_word_idx = start_word_idx + span_length - 1
                if end_word_idx >= len(words):
                    break

                start_idx = words[start_word_idx][0]
                end_idx = words[end_word_idx][1] - 1

                if self._is_valid_negative_span(text, start_idx, end_idx, positive_spans):
                    candidates.append((start_idx, end_idx))

        # Preserve order while deduplicating
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _iter_word_spans(text: str):
        """
        Yields (start, end) character-index pairs for each whitespace-separated word in the text.

        :param text: The source text.
        :return: A generator of (start_idx, end_idx) tuples where end_idx is exclusive.
        """
        start_idx = None
        for idx, char in enumerate(text):
            if not char.isspace() and start_idx is None:
                start_idx = idx
            elif char.isspace() and start_idx is not None:
                yield (start_idx, idx)
                start_idx = None

        if start_idx is not None:
            yield (start_idx, len(text))

    @staticmethod
    def _is_valid_negative_span(text: str, start_idx: int, end_idx: int, positive_spans: list[tuple[int, int]],) -> bool:
        """
        Determines whether a candidate span is a valid negative sample: 
        - it must be non-empty, 
        - not start with '(' or end with ')', 
        - not overlap with any positive entity.

        :param text: The source section text.
        :param start_idx: Character-level start index (inclusive).
        :param end_idx: Character-level end index (inclusive).
        :param positive_spans: List of (start_idx, end_idx) positive entity spans.
        :return: True if the span is a valid negative sample, False otherwise.
        """
        span_text = text[start_idx : end_idx + 1].strip()
        if not span_text:
            return False

        if span_text.startswith("(") or span_text.endswith(")"):
            return False

        for positive_start, positive_end in positive_spans:
            if not (end_idx < positive_start or start_idx > positive_end):
                return False

        return True
    

#######################
# RelationTokenizer   #
#######################

class RelationTokenizer(BaseTokenizer):
    """
    Preprocesses raw GBIE data into entity-pair-marker-encoded sequences for relation extraction.

    For each annotated relation, the subject span is wrapped with [E1] / [/E1] and the object span is wrapped with [E2] / [/E2] in the same section text. 
    The resulting sequence is tokenized and stored alongside the integer predicate label.

    Optionally generates negative (NA-labelled) samples by selecting entity pairs that co-occur in the same section but have no annotated relation between them. 
    Only pairs whose (head_label, tail_label) combination is present in VALID_RELATIONS are considered as negatives, which keeps the label space structurally consistent.

    Samples where either marker is absent after truncation are silently discarded to avoid feeding degenerate inputs to the model.
    """

    def __init__(self, data: dict, tokenizer, label2id: dict[str, int], max_length: int = 512, negative_sample_multiplier: int = 1, concatenate_title_abstract: bool = False,):
        """
        Initializes the RelationTokenizer with dataset, tokenizer, label mapping, and negative-sampling configuration.

        :param data: A dictionary containing the raw input data, in the GBIE format.
        :param tokenizer: A HuggingFace tokenizer instance (must already have [E1] / [/E1] / [E2] / [/E2] added to its vocabulary).
        :param label2id: A dictionary mapping predicate label strings to integer IDs.
        :param max_length: Maximum token sequence length (default 512).
        :param negative_sample_multiplier: Number of negative (NA) pair samples to generate per positive relation (default 1).
        :param concatenate_title_abstract: Whether to concatenate title and abstract into a single sequence (default False). If True, enables cross-section relations.
        """
        super().__init__(data, tokenizer, label2id, max_length)
        self.negative_sample_multiplier = negative_sample_multiplier
        self.concatenate_title_abstract = concatenate_title_abstract

        # Cache the token IDs for both opening markers once at init time
        self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

    def _process_paper(self, content: dict) -> list:
        """
        Produces entity-pair-marker-encoded samples for all sections in a single paper, including any generated negative samples.

        Can process in two modes:
        - concatenate_title_abstract=False: Only same-section pairs are processed (default).
        - concatenate_title_abstract=True: Title and abstract are concatenated, enabling cross-section relations.

        :param content: A dictionary containing the paper's metadata and annotations, in the GBIE format.
        :return: A list of sample dicts, each with 'input_ids', 'attention_mask', and 'labels' (plain integer predicate IDs).
        """
        samples = []
        text_lst, entities_lst, relations_lst = self._extract_entities_and_relations(content)

        for text, entities, relations in zip(text_lst, entities_lst, relations_lst):
            # Build a set of annotated (subject_start, object_start) pairs for fast lookup during negative sampling
            annotated_pairs = {
                (r["subject_start_idx"], r["object_start_idx"])
                for r in relations
            }

            # -- Positive samples: one per annotated relation --
            for relation in relations:
                sample = self._build_sample(
                    text=text,
                    subj_start=relation["subject_start_idx"],
                    subj_end=relation["subject_end_idx"] + 1,  # convert to exclusive
                    obj_start=relation["object_start_idx"],
                    obj_end=relation["object_end_idx"] + 1,    # convert to exclusive
                    label=self.label2id[relation["predicate"]],
                )
                if sample is not None:
                    samples.append(sample)

            # -- Negative samples: entity pairs with no annotated relation --
            negative_pairs = self._sample_negative_pairs(entities, annotated_pairs)
            for subj_entity, obj_entity in negative_pairs:
                sample = self._build_sample(
                    text=text,
                    subj_start=subj_entity["start_idx"],
                    subj_end=subj_entity["end_idx"] + 1,
                    obj_start=obj_entity["start_idx"],
                    obj_end=obj_entity["end_idx"] + 1,
                    label=self.label2id["NA"],
                )
                if sample is not None:
                    samples.append(sample)

        return samples

    def _extract_entities_and_relations(self, content: dict) -> tuple[list, list, list]:
        """
        Extracts entities and relations from the paper's content, with optional title-abstract concatenation.
        Adjusts indices when concatenating to account for the separator.

        :param content: A dictionary containing the paper's metadata and annotations, in the GBIE format.
        :return: A tuple of (text_list, entities_list, relations_list) where each element is a list corresponding to each section(s).
        """
        title, abstract = get_title_and_abstract(content)
        text_lst = []
        entities_lst = []
        relations_lst = []

        if self.concatenate_title_abstract:
            # Concatenate title and abstract with a space separator
            concatenated_text = f"{title} {abstract}"
            title_length = len(title)
            separator_length = 1  # space

            # Adjust entities: title entities keep original indices, abstract entities get offset
            all_entities = []
            for entity in content.get("entities", []):
                adjusted_entity = dict(entity)
                if entity["location"] == "abstract":
                    adjusted_entity["start_idx"] = entity["start_idx"] + title_length + separator_length
                    adjusted_entity["end_idx"] = entity["end_idx"] + title_length + separator_length
                all_entities.append(adjusted_entity)

            # Adjust relations: title relations keep original indices, abstract relations get offset
            all_relations = []
            #for relation in content.get("relations", []):
            for relation in content.get("relations", []):
                adjusted_relation = dict(relation)
                # Adjust subject indices if in abstract
                if relation["subject_location"] == "abstract":
                    adjusted_relation["subject_start_idx"] = (
                        relation["subject_start_idx"] + title_length + separator_length
                    )
                    adjusted_relation["subject_end_idx"] = (
                        relation["subject_end_idx"] + title_length + separator_length
                    )
                # Adjust object indices if in abstract
                if relation["object_location"] == "abstract":
                    adjusted_relation["object_start_idx"] = (
                        relation["object_start_idx"] + title_length + separator_length
                    )
                    adjusted_relation["object_end_idx"] = (
                        relation["object_end_idx"] + title_length + separator_length
                    )
                all_relations.append(adjusted_relation)

            text_lst.append(concatenated_text)
            entities_lst.append(all_entities)
            relations_lst.append(all_relations)
        else:
            # Process each section separately
            for section in ["title", "abstract"]:
                text = title if section == "title" else abstract
                section_entities = [
                    entity for entity in content.get("entities", [])
                    if entity["location"] == section
                ]
                section_relations = [
                    #relation for relation in content.get("relations", [])
                    relation for relation in content.get("relations", [])
                    if relation["subject_location"] == section
                    and relation["object_location"] == section
                ]

                text_lst.append(text)
                entities_lst.append(section_entities)
                relations_lst.append(section_relations)

        return text_lst, entities_lst, relations_lst


    def _build_sample(self, text: str, subj_start: int, subj_end: int, obj_start: int, obj_end: int, label: int,) -> dict | None:
        """
        Inserts entity markers into the text, tokenizes the result, and returns a sample dict. 
        Returns None if either [E1] or [E2] is absent after truncation, which discards degenerate samples that would corrupt the representation.

        :param text: The source section text.
        :param subj_start: Character-level start index of the subject span (inclusive).
        :param subj_end: Character-level end index of the subject span (exclusive).
        :param obj_start: Character-level start index of the object span (inclusive).
        :param obj_end: Character-level end index of the object span (exclusive).
        :param label: Integer predicate label ID.
        :return: A sample dict with 'input_ids', 'attention_mask', and 'labels', or None if the sample is invalid after truncation.
        """
        # Overlapping spans would corrupt the marker insertion -- skip silently
        if not (subj_end <= obj_start or obj_end <= subj_start):
            return None

        marked_text = self._insert_relation_markers(
            text, subj_start, subj_end, obj_start, obj_end
        )
        input_ids, attention_mask = self._tokenize(marked_text)

        # Verify that both opening markers survived truncation
        if (self.e1_token_id not in input_ids) or (self.e2_token_id not in input_ids):
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }

    def _tokenize(self, marked_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes a single entity-pair-marked text string.

        :param marked_text: Input text with [E1] / [/E1] and [E2] / [/E2] markers inserted.
        :return: A tuple of (input_ids, attention_mask) as 1-D LongTensors.
        """
        encoding = self.tokenizer(
            marked_text,
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

    @staticmethod
    def _insert_relation_markers(
        text: str,
        subj_start: int,
        subj_end: int,
        obj_start: int,
        obj_end: int,
    ) -> str:
        """
        Wraps the subject span with [E1] / [/E1] and the object span with [E2] / [/E2].
        Tags are inserted right-to-left (highest position first) so that earlier insertions do not shift the indices of later ones.

        :param text: The source section text.
        :param subj_start: Character-level start of the subject span (inclusive).
        :param subj_end: Character-level end of the subject span (exclusive).
        :param obj_start: Character-level start of the object span (inclusive).
        :param obj_end: Character-level end of the object span (exclusive).
        :return: The text with both entity-pair markers inserted.
        """
        # Build (position, tag) pairs and sort descending so insertions are idempotent
        insertions = sorted(
            [
                (subj_start, "[E1]"),
                (subj_end,   "[/E1]"),
                (obj_start,  "[E2]"),
                (obj_end,    "[/E2]"),
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        for pos, tag in insertions:
            text = text[:pos] + tag + text[pos:]
        return text

    def _sample_negative_pairs(
        self,
        entities: list,
        annotated_pairs: set[tuple[int, int]],
    ) -> list[tuple[dict, dict]]:
        """
        Selects a random subset of entity pairs that co-occur in the section but have no annotated relation, capped at negative_sample_multiplier times the number of annotated relations already processed.

        Only pairs whose (head_label, tail_label) combination is present in VALID_RELATIONS are considered, ensuring that all NA-labelled samples represent structurally plausible but unannotated interactions.

        :param entities: List of entity dicts for the current section.
        :param annotated_pairs: Set of (subject_start_idx, object_start_idx) tuples for already-annotated positive relations in the section.
        :return: A list of (subject_entity, object_entity) tuples to use as negatives.
        """
        num_negatives = len(annotated_pairs) * self.negative_sample_multiplier
        if num_negatives == 0 or len(entities) < 2:
            return []

        candidates = []
        for i, subj in enumerate(entities):
            for j, obj in enumerate(entities):
                if i == j:
                    continue
                # Only consider type combinations that have at least one valid predicate
                if (subj["label"], obj["label"]) not in VALID_RELATIONS:
                    continue
                # Exclude already-annotated pairs (both directions are treated independently)
                if (subj["start_idx"], obj["start_idx"]) in annotated_pairs:
                    continue
                candidates.append((subj, obj))

        random.shuffle(candidates)
        return candidates[:num_negatives]
    

####################
# NERBIOTokenizer  #
####################

class NERBIOTokenizer(BIOTokenizer):
    """
    Extends BIOTokenizer to produce entity-type-specific BIO tags for joint term extraction and classification (NER).

    Unlike BIOTokenizer, which collapses all entities to a single 'term' class, NERBIOTokenizer preserves the original entity label from the annotation and generates entity-type-specific tags such as B-drug, I-chemical, B-DDF, I-anatomical location, etc.

    This allows an AutoModelForTokenClassification to jointly predict both entity boundaries and semantic categories in a single forward pass, combining the extraction and classification tasks that BIOTokenizer and TermClassificationTokenizer address separately.

    The only structural change relative to BIOTokenizer is the tag generation logic:
        - _extract_entities : entity labels are preserved as-is (not replaced with "term")
        - _tokenize_with_bio : uses B-{label} / I-{label} instead of B-term / I-term
    """

    def _extract_entities(self, content: dict) -> tuple[list, list]:
        """
        Extracts text sections and entity annotations while preserving each entity's original label, enabling entity-type-specific BIO tagging.

        Unlike BIOTokenizer._extract_entities, entity labels are NOT overwritten with "term".
        Index adjustments for title-abstract concatenation are applied identically to the parent class.

        :param content: A dictionary containing the paper's metadata and annotations, in the GBIE format.
        :return: A tuple of (text_lst, entities_lst), where each entity dict retains its original 'label' field.
        """
        title, abstract = get_title_and_abstract(content)
        text_lst = []
        entities_lst = []

        if self.concatenate_title_abstract:
            entities_lst.append(
                [
                    {
                        **entity,
                        "start_idx": entity["start_idx"] + len(title) + 1,
                        "end_idx": entity["end_idx"] + len(title) + 1,
                    }
                    if entity["location"] == "abstract"
                    else dict(entity)
                    for entity in content.get("entities", [])
                ]
            )
            text_lst.append(f"{title} {abstract}")
        else:
            for section in ["title", "abstract"]:
                entities_lst.append(
                    [
                        dict(entity)
                        for entity in content.get("entities", [])
                        if entity["location"] == section
                    ]
                )
                text_lst.append(title if section == "title" else abstract)

        return text_lst, entities_lst

    def _tokenize_with_bio(self, text: str, entities: list) -> tuple[list, list, list]:
        """
        Tokenizes the input text and assigns entity-type-specific BIO tags to each token.

        Tags have the form B-{label} / I-{label} using each entity's 'label' field, rather than the generic B-term / I-term used by BIOTokenizer.  
        Entity labels containing hyphens (e.g. 'anatomical location') are handled correctly because the tag is constructed by string concatenation rather than by splitting the final tag.

        :param text: The input text to be tokenized.
        :param entities: A list of entity annotations with 'start_idx', 'end_idx', and 'label' keys.
        :return: A tuple of (label_ids, input_ids, attention_mask), each a plain list.
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        offsets = encoding["offset_mapping"]
        bio_tags = ["O"] * len(offsets)

        for entity in entities:
            first_token_assigned = False
            entity_label = entity["label"]
            for i, (token_start, token_end) in enumerate(offsets):
                if i >= 1 and (token_start, token_end) == (0, 0):
                    break

                if token_start >= entity["start_idx"] and token_end <= entity["end_idx"] + 1:
                    if not first_token_assigned:
                        bio_tags[i] = f"B-{entity_label}"
                        first_token_assigned = True
                    else:
                        bio_tags[i] = f"I-{entity_label}"

        label_ids = []
        for offset, tag in zip(offsets, bio_tags):
            if offset == (0, 0):
                label_ids.append(-100)
            else:
                label_ids.append(self.label2id[tag])

        return label_ids, encoding["input_ids"], encoding["attention_mask"]
    
####################
# GLiNERTokenizer  #
####################

class GLiNERTokenizer:
    """
    Converts GBIE-format data into the word-level tokenized format expected by GLiNER for fine-tuning and evaluation.

    GLiNER does not use HuggingFace subword tokenization.  
    Instead, it operates on a flat list of whitespace-aware word tokens together with entity annotations expressed as (start_token_idx, end_token_idx, label) triples -- all indices inclusive and 0-based.
    Labels are always lower-cased to match GLiNER's convention.

    Unlike the HuggingFace-based tokenizers, GLiNERTokenizer is a standalone class that does NOT inherit from BaseTokenizer: it requires neither a HuggingFace tokenizer instance nor a label2id mapping, so the BaseTokenizer constructor signature would be misleading.
    The public interface mirrors BaseTokenizer however -- process_files() / _process_paper() -- so callers can use it in the same way.

    Output format per document (entity-level):
        {
            "tokenized_text": ["word1", "word2", ...],
            "ner": [[start_token_idx, end_token_idx, "label"], ...]
        }

    For training, the entity-level output must be further converted to token-level (one NER entry per token of each entity span) via the static method to_token_level().  
    This two-step design allows the entity-level samples to be used directly for evaluation while the token-level variant feeds the GLiNER training loop.
    """

    def __init__(self, data: dict, concatenate_title_abstract: bool = True):
        """
        Initializes the tokenizer with the dataset and concatenation configuration.

        :param data: A dictionary of paper IDs -> content dicts (GBIE format).
        :param concatenate_title_abstract: When True (default), the title and abstract are joined with a single space into one token sequence per document. When False, each section is returned as a separate sample.
        """
        self.data = data
        self.concatenate_title_abstract = concatenate_title_abstract

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def process_files(self) -> list:
        """
        Iterates over all papers in the dataset and delegates per-paper processing to _process_paper(), collecting all resulting samples into a flat list.

        :return: A flat list of GLiNER-format sample dicts, one per document (or one per section when concatenate_title_abstract=False).
        """
        samples = []
        for _, content in tqdm(self.data.items(), total=len(self.data), desc="GLiNER Preprocessing"):
            samples.extend(self._process_paper(content))
        return samples

    @staticmethod
    def to_token_level(samples: list) -> list:
        """
        Converts entity-level GLiNER samples to the token-level format required by the GLiNER training loop.

        In entity-level format each NER entry is [start_tok, end_tok, label] representing a multi-token span.  
        In token-level format each entry becomes a series of single-token annotations [tok_idx, tok_idx, label] -- one per token in the original span.

        This conversion is applied to training data only; evaluation uses the entity-level format so that standard span-level P/R/F1 can be computed.

        :param samples: A list of entity-level GLiNER-format dicts produced by process_files().
        :return: A new list of token-level GLiNER-format dicts.
        """
        token_level_samples = []
        for sample in samples:
            token_ner = []
            for start, end, label in sample["ner"]:
                for idx in range(start, end + 1):
                    token_ner.append((idx, idx, label.lower()))
            token_level_samples.append(
                {
                    "tokenized_text": sample["tokenized_text"],
                    "ner": token_ner,
                }
            )
        return token_level_samples

    # ------------------------------------------------------------------ #
    # Per-paper processing                                                 #
    # ------------------------------------------------------------------ #

    def _process_paper(self, content: dict) -> list:
        """
        Produces GLiNER-format sample(s) for a single paper.

        When concatenate_title_abstract=True, title and abstract tokens are merged into a single sequence with token-index offsets adjusted for the abstract.  
        When False, each section yields an independent sample.

        :param content: A dictionary containing the paper's metadata and annotations, in GBIE format.
        :return: A list of one or two GLiNER-format sample dicts.
        """
        title, abstract = get_title_and_abstract(content)
        entities = content.get("entities", [])

        if self.concatenate_title_abstract:
            return [self._build_sample_concatenated(title, abstract, entities)]
        else:
            samples = []
            for section in ["title", "abstract"]:
                text = title if section == "title" else abstract
                section_entities = [e for e in entities if e.get("location") == section]
                samples.append(self._build_sample_single(text, section_entities))
            return samples

    def _build_sample_concatenated(self, title: str, abstract: str, entities: list) -> dict:
        """
        Builds a single GLiNER sample from the concatenated title and abstract, adjusting token indices for abstract entities to account for the title's token count.

        :param title: Title text.
        :param abstract: Abstract text.
        :param entities: Combined entity list for the document.
        :return: A GLiNER-format sample dict.
        """
        title_tokens, title_spans = self._tokenize_text_with_positions(title)
        abstract_tokens, abstract_spans = self._tokenize_text_with_positions(abstract)
        title_token_count = len(title_tokens)

        # Concatenate token lists; abstract token indices are offset by the title length
        all_tokens = title_tokens + abstract_tokens
        ner = []

        for section, tokens_spans, offset in [
            ("title",    title_spans,    0),
            ("abstract", abstract_spans, title_token_count),
        ]:
            section_entities = [e for e in entities if e.get("location") == section]
            ner.extend(
                self._map_entities_to_tokens(section_entities, tokens_spans, offset)
            )

        # Sort by start token index for deterministic ordering
        ner.sort(key=lambda x: x[0])

        return {"tokenized_text": all_tokens, "ner": ner}

    def _build_sample_single(self, text: str, entities: list) -> dict:
        """
        Builds a single GLiNER sample for one text section (title or abstract).

        :param text: Section text.
        :param entities: Entity list for this section only.
        :return: A GLiNER-format sample dict.
        """
        tokens, token_spans = self._tokenize_text_with_positions(text)
        ner = self._map_entities_to_tokens(entities, token_spans, token_offset=0)
        ner.sort(key=lambda x: x[0])
        return {"tokenized_text": tokens, "ner": ner}

    # ------------------------------------------------------------------ #
    # Character-to-token alignment                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _map_entities_to_tokens(
        entities: list,
        token_spans: list,
        token_offset: int,
    ) -> list:
        """
        Maps each entity from character-level indices to token-level indices using the precomputed (start_char, end_char) spans for each token.

        Entities that do not align to any token are skipped with a warning.

        :param entities: List of GBIE entity dicts with 'start_idx', 'end_idx', and 'label'.
        :param token_spans: List of (start_char, end_char) pairs -- end_char is exclusive.
        :param token_offset: Integer offset added to every resulting token index (for abstract entities in a concatenated sequence).
        :return: A list of [start_token_idx, end_token_idx, label] triples.
        """
        ner = []
        for entity in entities:
            entity_start_char = entity["start_idx"]
            entity_end_char   = entity["end_idx"] + 1  # convert inclusive end to exclusive

            start_tok = None
            end_tok   = None

            for i, (tok_start, tok_end) in enumerate(token_spans):
                if tok_end <= entity_start_char:
                    continue  # Token is entirely before the entity
                if tok_start >= entity_end_char:
                    break     # Token is entirely after the entity
                # Token overlaps with the entity span
                if start_tok is None:
                    start_tok = i
                end_tok = i

            if start_tok is not None and end_tok is not None:
                ner.append([
                    start_tok + token_offset,
                    end_tok   + token_offset,
                    entity["label"].lower(),
                ])
            else:
                # Diagnostic only -- does not raise to allow partial annotation files
                print(
                    f"GLiNERTokenizer warning: entity '{entity.get('text_span', '?')}' "
                    f"(chars {entity_start_char}-{entity_end_char}) "
                    f"could not be aligned to any token."
                )
        return ner

    # ------------------------------------------------------------------ #
    # Word-level tokenizer                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize_text_with_positions(text: str) -> tuple[list, list]:
        """
        Splits text into word-level tokens, preserving punctuation as separate tokens but keeping hyphenated (e.g. 'dose-response') and underscored (e.g. 'IL_6') expressions intact.  
        Whitespace tokens are discarded.

        Contraction splitting (e.g. "don't" -> ["don", "'", "t"]) is applied so that subword boundaries match the tokenization assumed during model pre-training.

        :param text: The input text to tokenize.
        :return: A tuple of (tokens, token_spans) where token_spans contains (start_char, end_char) pairs with end_char exclusive, one per token.
        """
        tokens      = []
        token_spans = []

        # Match word-characters-with-hyphens/underscores, punctuation, whitespace, or
        # any single non-whitespace character (catch-all for Unicode punctuation, etc.)
        pattern = re.compile(r"\w+|[.,!?;:\'\"()\[\]{}<>]|[\s]+|\S")

        for match in pattern.finditer(text):
            token     = match.group()
            start_pos = match.start()

            # Discard pure whitespace tokens
            if token.isspace():
                continue

            # Keep hyphenated or underscored compounds intact (e.g. 'dose-response')
            if re.match(r"\w+-\w+", token) or re.match(r"\w+_\w+", token):
                tokens.append(token)
                token_spans.append((start_pos, match.end()))
                continue

            # Split contractions (e.g. "don't" -> "don", "'", "t")
            contraction_match = re.match(r"(\w+)(')(\w+)", token)
            if contraction_match:
                for group in contraction_match.groups():
                    end_pos = start_pos + len(group)
                    tokens.append(group)
                    token_spans.append((start_pos, end_pos))
                    start_pos = end_pos
                continue

            tokens.append(token)
            token_spans.append((start_pos, match.end()))

        return tokens, token_spans


########################
# EntityLinkingsTokenizer #
########################

class EntityLinkingsTokenizer:
    """
    Converts GBIE-format data into the entity-linkings format for entity linking and disambiguation tasks.

    The entity-linkings format is a list of dicts, each containing 'id', 'text', and 'entities' keys:
        {
            "id": "doc-001",
            "text": "She graduated from NAIST.",
            "entities": [{"start": 19, "end": 24, "label": ["000011"]}],
        }

    Entity indices are character-level (0-based, with end index exclusive).  
    Labels are stored as lists to accommodate multiple possible URIs/IDs per entity.

    Like GLiNERTokenizer, this is a standalone class that does NOT inherit from BaseTokenizer, as it requires neither a HuggingFace tokenizer instance nor a label2id mapping.  
    However, the public interface mirrors BaseTokenizer (process_files / _process_paper) for consistency.

    When concatenate_title_abstract=True, title and abstract are merged into a single sequence with entity indices adjusted accordingly. 
    When False, title and abstract are processed as separate documents.
    """

    def __init__(self, data: dict, concatenate_title_abstract: bool = True):
        """
        Initializes the tokenizer with the dataset and concatenation configuration.

        :param data: A dictionary of paper IDs -> content dicts (GBIE format).
        :param concatenate_title_abstract: When True (default), the title and abstract are joined with a single space into one sequence per document. When False, each section is returned as a separate sample with distinct IDs.
        """
        self.data = data
        self.concatenate_title_abstract = concatenate_title_abstract

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def process_files(self) -> list:
        """
        Iterates over all papers in the dataset and delegates per-paper processing to _process_paper(), collecting all resulting samples into a flat list.

        :return: A flat list of entity-linkings format sample dicts, one per document (or one per section when concatenate_title_abstract=False).
        """
        samples = []
        for pmid, content in tqdm(self.data.items(), total=len(self.data), desc="Entity-Linkings Preprocessing"):
            samples.extend(self._process_paper(pmid, content))
        return samples

    # ------------------------------------------------------------------ #
    # Per-paper processing                                                 #
    # ------------------------------------------------------------------ #

    def _process_paper(self, pmid: str, content: dict) -> list:
        """
        Produces entity-linkings format sample(s) for a single paper.

        When concatenate_title_abstract=True, title and abstract are merged into a single sequence.  
        When False, each section yields an independent sample with adjusted IDs (e.g., "{pmid}_title", "{pmid}_abstract").

        :param pmid: Paper ID from the dataset key.
        :param content: A dictionary containing the paper's metadata and annotations, in GBIE format.
        :return: A list of one or two entity-linkings format sample dicts.
        """
        title, abstract = get_title_and_abstract(content)
        entities = content.get("entities", [])

        if self.concatenate_title_abstract:
            return [self._build_sample_concatenated(pmid, title, abstract, entities)]
        else:
            samples = []
            for section in ["title", "abstract"]:
                text = title if section == "title" else abstract
                section_id = f"{pmid}_{section}"
                section_entities = [e for e in entities if e.get("location") == section]
                samples.append(self._build_sample_single(section_id, text, section_entities))
            return samples

    def _build_sample_concatenated(self, pmid: str, title: str, abstract: str, entities: list) -> dict:
        """
        Builds a single entity-linkings sample from the concatenated title and abstract, adjusting entity indices for abstract entities to account for the title length and separator.

        :param pmid: Paper ID.
        :param title: Title text.
        :param abstract: Abstract text.
        :param entities: Combined entity list for the document.
        :return: An entity-linkings format sample dict.
        """
        concatenated_text = f"{title} {abstract}".strip()
        title_length = len(title)
        separator_length = 1  # space

        entity_list = []
        for section, offset in [("title", 0), ("abstract", title_length + separator_length)]:
            section_entities = [e for e in entities if e.get("location") == section]
            for entity in section_entities:
                adjusted_entity = self._build_entity_dict(entity, offset)
                if adjusted_entity is not None:
                    entity_list.append(adjusted_entity)

        # Sort by start index for deterministic ordering
        entity_list.sort(key=lambda x: x["start"])

        return {
            "id": pmid,
            "text": concatenated_text,
            "entities": entity_list,
        }

    def _build_sample_single(self, sample_id: str, text: str, entities: list) -> dict:
        """
        Builds a single entity-linkings sample for one text section (title or abstract).

        :param sample_id: Document ID (may include section suffix).
        :param text: Section text.
        :param entities: Entity list for this section only.
        :return: An entity-linkings format sample dict.
        """
        entity_list = []
        for entity in entities:
            adjusted_entity = self._build_entity_dict(entity, offset=0)
            if adjusted_entity is not None:
                entity_list.append(adjusted_entity)

        # Sort by start index for deterministic ordering
        entity_list.sort(key=lambda x: x["start"])

        return {
            "id": sample_id,
            "text": text,
            "entities": entity_list,
        }

    # ------------------------------------------------------------------ #
    # Entity dict builder                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_entity_dict(entity: dict, offset: int = 0) -> dict | None:
        """
        Converts a single GBIE entity dict to entity-linkings format.

        Extracts start_idx, end_idx, and label from the GBIE entity, applies the character-level offset,
        and wraps the label in a list (as per entity-linkings convention where labels are lists of possible URIs/IDs).

        :param entity: A GBIE entity dict with at minimum 'start_idx', 'end_idx', and 'label' keys.
        :param offset: Character index offset to apply (for abstract entities when concatenating).
        :return: A dict with 'start', 'end', and 'label' keys for entity-linkings format, or None if the entity is invalid.
        """
        start_idx = entity.get("start_idx")
        end_idx = entity.get("end_idx")
        label = entity.get("label")

        # Skip invalid entities
        if start_idx is None or end_idx is None or label is None:
            return None

        # Convert end_idx from inclusive to exclusive (entity-linkings convention)
        # and apply offset
        return {
            "start": start_idx + offset,
            "end": end_idx + 1 + offset,
            "label": [label] if isinstance(label, str) else label,
        }
