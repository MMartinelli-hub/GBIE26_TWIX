import json
import re

import nltk
print('Nltk version: {}.'.format(nltk.__version__))

from nltk.tokenize import TreebankWordTokenizer as twt
from nltk.tokenize import WordPunctTokenizer as wpt
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
nltk.download('punkt_tab')

import os

PREDICTIONS_DIR = "runs/entityRecognizer/dev"
OUTPUT_DIR = "runs/entityRecognizer/dev_atlop"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for filename in os.listdir(PREDICTIONS_DIR):
    if filename.endswith(".json"):
        print(f"Found prediction file: {filename}")
        PATH_NER_PREDICTIONS = os.path.join(PREDICTIONS_DIR, filename)
        PATH_OUTPUT_NER_PREDICTIONS = os.path.join(OUTPUT_DIR, filename.replace(".json", "_atlop_format.json"))

    with open(PATH_NER_PREDICTIONS, 'r', encoding='utf-8') as file:
        ner_predictions = json.load(file)

    def get_articles_with_less_than_5_entities(data: dict):
        pmid_list = []
        for pmid, article in data.items():
            entities = article['entities']
            if len(entities) < 5:
                pmid_list.append(pmid)
        return pmid_list

    def remove_articles_with_less_than_5_entities(data: dict, data_name: str):
        pmid_list = get_articles_with_less_than_5_entities(data)
        for pmid in pmid_list:
            del data[pmid]
        print(f'{data_name} - {len(pmid_list)} articles removed.')

    remove_articles_with_less_than_5_entities(ner_predictions, "ner_predictions")

    def get_entity_boundary_offsets(article: dict, location: str, text: str, pmid: str) -> set:
        """Return character offsets where tokens must be split for exact entity alignment."""
        boundaries = set()
        for entity in article['entities']:
            if entity.get('location') != location:
                continue
            start = entity.get('start_idx')
            end = entity.get('end_idx')
            if start is None or end is None:
                continue
            if start < 0 or end < start or end >= len(text):
                raise Exception(f'{pmid} - Entity has illegal offsets for {location}: {entity}')
            boundaries.add(start)
            boundaries.add(end + 1)  # Source end_idx is inclusive; token spans are half-open.
        return boundaries

    def tokenize_text_with_entity_boundaries(text: str, boundaries: set) -> list:
        """Tokenize text and split tokenizer tokens wherever an entity boundary falls inside them."""
        tokens = []
        for token_start, token_end in wpt().span_tokenize(text):
            internal_boundaries = sorted(
                boundary for boundary in boundaries
                if token_start < boundary < token_end
            )
            piece_boundaries = [token_start] + internal_boundaries + [token_end]
            for piece_start, piece_end in zip(piece_boundaries, piece_boundaries[1:]):
                if piece_start < piece_end:
                    tokens.append((text[piece_start:piece_end], piece_start, piece_end))
        return tokens

    def tokenize_docs(data: dict, data_name: str):
        print(f"Tokenizing articles in set {data_name}...")

        for pmid, article in data.items():
            title = article['metadata']['title']
            abstract = article['metadata']['abstract']

            title_boundaries = get_entity_boundary_offsets(article, 'title', title, pmid)
            abstract_boundaries = get_entity_boundary_offsets(article, 'abstract', abstract, pmid)
            article['tokenized_title'] = tokenize_text_with_entity_boundaries(title, title_boundaries)
            article['tokenized_abstract'] = tokenize_text_with_entity_boundaries(abstract, abstract_boundaries)
            
    tokenize_docs(ner_predictions, "ner_predictions")

    def map_entities_to_tokens(data: dict, data_name: str):
        print(f"Mapping entities to tokens in set {data_name}...")
        
        for pmid, article in data.items():
            for entity in article['entities']:
                location = entity['location']
                start = entity['start_idx']
                end = entity['end_idx']
                start_token = None
                end_token = None
                if location == 'title':
                    for idx, token in enumerate(article['tokenized_title']):
                        if start == token[1] and start is not None:
                            start_token = idx
                        if end == token[2]-1 and end is not None:
                            end_token = idx
                elif location == 'abstract':
                    for idx, token in enumerate(article['tokenized_abstract']):
                        if start == token[1] and start is not None:
                            start_token = idx
                        if end == token[2]-1 and end is not None:
                            end_token = idx
                else:
                    raise Exception(f'{pmid} - Unrecognized Location: {location}')
                if start_token is not None and end_token is not None:
                    entity['start_token'] = start_token
                    entity['end_token'] = end_token
                else:
                    print (data[pmid]['tokenized_title'])
                    print(data[pmid]['tokenized_abstract'])
                    raise Exception(f'{pmid} - Not able to assign token(s) to entity: {entity}')
                
    map_entities_to_tokens(ner_predictions, "ner_predictions")

    sentence_splitter = PunktSentenceTokenizer()

    def get_abstract_entity_spans(article: dict, abstract: str, title: str, pmid: str) -> list:
        """Return abstract entity spans as half-open offsets, validating source indices."""
        spans = []
        for entity in article['entities']:
            location = entity['location']
            start = entity['start_idx']
            end = entity['end_idx']
            if location == 'title':
                if start < 0 or end < start or end >= len(title):
                    raise Exception(f'{pmid} - Found title entity having illegal offsets: {entity}')
                continue
            if location == 'abstract':
                if start < 0 or end < start or end >= len(abstract):
                    raise Exception(f'{pmid} - Found abstract entity having illegal offsets: {entity}')
                spans.append((start, end + 1, entity))
                continue
            raise Exception(f'{pmid} - Unrecognized Location: {location}')
        return spans

    def merge_sentence_spans_around_entities(sentences: list, entity_spans: list) -> list:
        """Merge adjacent sentence spans whenever their boundary cuts through an entity."""
        if not sentences:
            return []

        merge_next = [False] * len(sentences)
        for sent_idx in range(len(sentences) - 1):
            left_end = sentences[sent_idx][1]
            right_start = sentences[sent_idx + 1][0]
            for entity_start, entity_end, _ in entity_spans:
                if entity_start < left_end and right_start < entity_end:
                    merge_next[sent_idx] = True
                    break

        merged_spans = []
        sent_idx = 0
        while sent_idx < len(sentences):
            start_val = sentences[sent_idx][0]
            end_val = sentences[sent_idx][1]
            while sent_idx < len(sentences) - 1 and merge_next[sent_idx]:
                sent_idx += 1
                end_val = sentences[sent_idx][1]
            merged_spans.append((start_val, end_val))
            sent_idx += 1
        return merged_spans

    def find_containing_sentence(sentences: list, entity_start: int, entity_end: int):
        for sent_idx, (sent_start, sent_end) in enumerate(sentences):
            if sent_start <= entity_start and entity_end <= sent_end:
                return sent_idx
        return None

    def get_sentence_spans(data: dict, data_name: str):
        print(f"Getting sentence spans in set {data_name}...")

        for pmid, article in data.items():
            title = article['metadata']['title']
            abstract = article['metadata']['abstract']
            sentences = list(sentence_splitter.span_tokenize(abstract))
            entity_spans = get_abstract_entity_spans(article, abstract, title, pmid)
            article['sentences'] = merge_sentence_spans_around_entities(sentences, entity_spans)

            for entity_start, entity_end, entity in entity_spans:
                if find_containing_sentence(article['sentences'], entity_start, entity_end) is None:
                    raise Exception(
                        f'{pmid} - Entity is not contained in one sentence after entity-aware merging: {entity}\n'
                        f'Entity text: "{entity["text_span"]}"\n'
                        f'Entity location: {entity_start}-{entity_end - 1}\n'
                        f'Sentence spans: {article["sentences"]}'
                    )

    get_sentence_spans(ner_predictions, "ner_predictions")

    def check_sentence_spans(data: dict, data_name: str):
        print(f"Checking sentence spans in set {data_name}...")

        for pmid, article in data.items():
            # Process each entity in the article.
            for entity in article['entities']:
                location = entity['location']
                start = entity['start_idx']
                end = entity['end_idx']
                # For title entities, we do nothing regarding sentence spans.
                if location == 'title':
                    continue

                # Only process abstract entities.
                if location == 'abstract':
                    start_sentence = None
                    end_sentence = None

                    # Iterate over the original sentence spans to determine in which sentences the entity start and end fall.
                    for idx, s in enumerate(article['sentences']):
                        # Using >= and <= to include boundaries.
                        if start >= s[0] and start <= s[1] and start_sentence is None:
                            start_sentence = idx
                            #print(f'Start sentence assigned: {idx}')
                        if end >= s[0] and end <= s[1] and end_sentence is None:
                            end_sentence = idx
                            #print(f'End sentence assigned: {idx}')

                    if start_sentence is None:
                        raise Exception(f'{pmid} - Start sentence not assigned for entity: {entity}')
                    if end_sentence is None:
                        raise Exception(f'{pmid} - End sentence not assigned for entity: {entity}')
                    
                    # If the entity falls in two different sentences, raise Exception.
                    if start_sentence != end_sentence:
                        raise Exception(f'{pmid} - Entity assigned to two different sentences ({start_sentence}, {end_sentence}): {entity}')

    check_sentence_spans(ner_predictions, "ner_predictions")

    def map_tokens_to_sentences(data: dict, data_name: str):
        """
        For each article, map tokens in the 'tokenized abstract' to the sentence in which they are located.
        Uses the 'sentences' field in the article, which is assumed to be a list of (start, end) tuples.
        
        The mapping is stored as a dictionary where the key is the token index (its position in the tokenized abstract)
        and the value is the sentence index. For example, if the first token belongs to sentence 0 and the third token
        belongs to sentence 1, the mapping will include entries {0: 0, 2: 1}.
        
        Raises an Exception if a token does not fall within any of the sentence spans.
        """
        print(f"Mapping tokens to sentences in set {data_name}...")

        for pmid, article in data.items():
            # Retrieve the tokenized abstract and the sentence spans.
            tokens = article.get('tokenized_abstract')
            sentences = article.get('sentences')
            
            if tokens is None:
                raise Exception(f"Article {pmid} is missing 'tokenized abstract'.")
            if sentences is None:
                raise Exception(f"Article {pmid} is missing 'sentences'. Make sure to run get_sentence_spans first.")
            
            token_to_sentence = {}
            
            # Iterate over each token and determine which sentence it belongs to.
            for token_index, token_entry in enumerate(tokens):
                # Each token_entry is assumed to be a tuple: (token_text, start_offset, end_offset)
                token_text, token_start, token_end = token_entry
                assigned_sentence = None
                
                # Check each sentence span to see if the token falls within it.
                for sentence_index, (sent_start, sent_end) in enumerate(sentences):
                    # We assume a token belongs to a sentence if its start is >= sentence start and its end is <= sentence end.
                    if token_start >= sent_start and token_end <= sent_end:
                        assigned_sentence = sentence_index
                        break  # Stop once we find the sentence that contains the token.
                
                if assigned_sentence is None:
                    raise Exception(
                        f"Token '{token_text}' (index {token_index}, offsets {token_start}-{token_end}) "
                        f"in article {pmid} does not fall within any sentence span: {sentences}"
                    )
                
                token_to_sentence[token_index] = assigned_sentence
            
            # Add the mapping to the article dictionary.
            article['tokens_to_sentences_map'] = token_to_sentence

    map_tokens_to_sentences(ner_predictions, "ner_predictions")

    def map_entities_to_tokens_within_sentences(data: dict, data_name: str) -> dict:
        """
        For each article, this function maps each entity (assumed to be in the abstract)
        to the token positions within the sentence that contains it.
        
        For each entity in article['entities'] (with location 'abstract'), it adds:
        - 'located_in_sentence': the sentence index in which the entity's tokens are located,
        - 'start_token_in_sentence': the position of the entity's start token within that sentence,
        - 'end_token_in_sentence': the position of the entity's end token within that sentence.
        
        This function relies on:
        - article['tokenized abstract']: a list of tokens of the form (token_text, start_offset, end_offset)
        - article['tokens_to_sentences_map']: a mapping { token_index -> sentence_index }
        - article['sentences']: a list of (start, end) sentence spans for the abstract.
        """
        print(f"Mapping entities to tokens within sentences in set {data_name}...")
        
        for pmid, article in data.items():
            # Retrieve required fields.
            tokens = article.get('tokenized_abstract')
            token_to_sentence = article.get('tokens_to_sentences_map')
            sentences = article.get('sentences')
            
            if tokens is None:
                raise Exception(f"Article {pmid} is missing 'tokenized abstract'.")
            if token_to_sentence is None:
                raise Exception(f"Article {pmid} is missing 'tokens_to_sentences_map'. Run map_tokens_to_sentences first.")
            if sentences is None:
                raise Exception(f"Article {pmid} is missing 'sentences'. Run get_sentence_spans first.")
            
            # Build a helper mapping: for each sentence index, list the token indices that fall into that sentence.
            sentence_to_token_indices = {}
            for token_index in range(len(tokens)):
                sent_idx = token_to_sentence.get(token_index)
                if sent_idx is None:
                    raise Exception(
                        f"In article {pmid}, token index {token_index} is not mapped to any sentence. Tokens: {tokens[token_index]}"
                    )
                sentence_to_token_indices.setdefault(sent_idx, []).append(token_index)
            
            # Now process each entity.
            for entity in article.get('entities', []):
                if entity.get('location') != 'abstract': # Only process entities in the abstract.
                    continue
                
                # Retrieve the token indices for this entity.
                entity_start_token = entity.get('start_token')
                entity_end_token = entity.get('end_token')
                
                if entity_start_token is None or entity_end_token is None:
                    raise Exception(
                        f"Entity in article {pmid} is missing start_token or end_token: {entity}"
                    )
                
                # Determine the sentence in which the entity's tokens are located.
                sentence_for_start = token_to_sentence.get(entity_start_token)
                sentence_for_end = token_to_sentence.get(entity_end_token)
                
                if sentence_for_start is None or sentence_for_end is None:
                    raise Exception(
                        f"Entity in article {pmid} has tokens not mapped to any sentence: {entity}"
                    )
                
                if sentence_for_start != sentence_for_end:
                    raise Exception(
                        f"Entity in article {pmid} spans multiple sentences (start in {sentence_for_start}, end in {sentence_for_end}): {entity}"
                    )
                
                located_sentence = sentence_for_start  # or sentence_for_end, both are same.
                
                # Get the list of token indices for the sentence.
                tokens_in_sentence = sentence_to_token_indices.get(located_sentence)
                if tokens_in_sentence is None:
                    raise Exception(
                        f"Sentence {located_sentence} not found in helper mapping for article {pmid}."
                    )
                
                # Find the position within the sentence for the start token.
                try:
                    start_token_in_sentence = tokens_in_sentence.index(entity_start_token)
                except ValueError:
                    raise Exception(
                        f"Entity start token {entity_start_token} not found in sentence tokens {tokens_in_sentence} for article {pmid}."
                    )
                
                # And the position within the sentence for the end token.
                try:
                    end_token_in_sentence = tokens_in_sentence.index(entity_end_token)
                except ValueError:
                    raise Exception(
                        f"Entity end token {entity_end_token} not found in sentence tokens {tokens_in_sentence} for article {pmid}."
                    )
                
                # Add the new fields to the entity.
                entity['located_in_sentence'] = located_sentence
                entity['start_token_in_sentence'] = start_token_in_sentence
                entity['end_token_in_sentence'] = end_token_in_sentence

    map_entities_to_tokens_within_sentences(ner_predictions, "ner_predictions")

    def convert_to_docred_format(data: dict, data_name: str, is_test=False) -> list:
        """
        Converts articles (in our intermediate format) to the DocRED format.
        
        For each article, a new dictionary is produced with the following keys:
        - "vertexSet": a list of entity mentions (each entity becomes a list with one mention).
            Each mention is a dict with:
                "pos": [start_token_in_sentence, end_token_in_sentence],
                "type": entity label,
                "sent_id": sentence id (0 for title; abstract sentences are numbered starting at 1),
                "name": the entity text span.
        - "title": the title string of the article.
        - "sents": a list of lists of tokens. The first entry is the title tokenization and subsequent
                    entries are the tokenizations of the abstract sentences.
        
        For abstract sentence tokenization, we use the sentence spans in article['sentences'] (a list of (start, end) offsets)
        and the tokenized abstract (article['tokenized abstract'], where each token is a tuple (token_text, start, end)).
        
        Returns a list of DocRED-formatted document dictionaries.
        """
        print(f"Converting articles to DocRED format for set {data_name}...")

        docred_docs = []
        for pmid, article in data.items():
            # 1. Build the vertexSet.
            # Each entity becomes a single mention. We assume that the entities in article['entities'] 
            # are ordered with title entities first, then abstract entities.
            vertexSet = []
            for entity in article.get('entities', []):
                # Determine the sentence id according to DocRED.
                # Title entities are assigned to sentence 0.
                if entity.get('location') == 'title':
                    sent_id = 0
                else:
                    # For abstract entities, we expect a field 'located_in_sentence' computed earlier.
                    # In our intermediate format abstract sentences are numbered 0,1,... but in DocRED the title is sentence 0.
                    # So add 1.
                    sent_id = entity.get('located_in_sentence', 0) + 1

                # The token offsets of the entity within its sentence.
                if entity['location'] == 'title':
                    pos = [entity.get('start_token'), entity.get('end_token')+1]
                else:
                    pos = [entity.get('start_token_in_sentence'), entity.get('end_token_in_sentence')+1]
                mention = {
                    "pos": pos,
                    "type": entity.get("label").upper(),
                    "sent_id": sent_id,
                    "name": entity.get("text_span")
                }
                # Each vertexSet entry is a list of mentions (we have one per entity).
                vertexSet.append([mention])
            
            # 2. Build the sents field.
            # The first sentence is the tokenization of the title.
            
            title_tokens = article.get("tokenized_title", [])
            tokens_in_title = []
            for token in title_tokens:
                tokens_in_title.append(token[0])
            sents = [tokens_in_title]
                
            # For the abstract sentences, we use article['sentences'] (list of (start, end) spans)
            # and article['tokenized abstract'] (list of tokens, each as (token_text, start, end)).
            abstract_tokens = article.get("tokenized_abstract", [])
            abstract_sents = []
            for span in article.get("sentences", []):
                s_start, s_end = span
                tokens_in_sentence = []
                for token, t_start, t_end in abstract_tokens:
                    # If a token falls completely within the sentence span, add it.
                    if t_start >= s_start and t_end <= s_end:
                        tokens_in_sentence.append(token)
                abstract_sents.append(tokens_in_sentence)
            sents.extend(abstract_sents)
            
            # 3. The title string (a concatenation of pmid + '||' + title).
            doc_title = f'{pmid}||{article["metadata"]["title"]}'
            
            # 4. Build the final document dictionary.
            doc = {
                "vertexSet": vertexSet,
                "title": doc_title,
                "sents": sents
            }
            docred_docs.append(doc)
        
        return docred_docs

    docred_ner_predictions = convert_to_docred_format(ner_predictions, "ner_predictions")

    def dump_to_json(docred_dict, output_file_path):
        dict_with_double_quotes = json.dumps(docred_dict, ensure_ascii=False)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(dict_with_double_quotes)

    dump_to_json(docred_ner_predictions, PATH_OUTPUT_NER_PREDICTIONS)