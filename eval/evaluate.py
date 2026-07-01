#%%
import json
from argparse import ArgumentParser
import os
import pandas as pd

argParse = ArgumentParser()
argParse.add_argument('--TE_folder', type=str, help='Path to the folder containing the predictions for the term extraction subtask')
argParse.add_argument('--NER_folder', type=str, help='Path to the folder containing the predictions for the named entity recognition subtask')
argParse.add_argument('--NERD_folder', type=str, help='Path to the folder containing the predictions for the named entity disambiguation subtask')
argParse.add_argument('--M_RE_folder', type=str, help='Path to the folder containing the predictions for the mention-level relation extraction subtask')
argParse.add_argument('--C_RE_folder', type=str, help='Path to the folder containing the predictions for the concept-level relation extraction subtask')

arguments = argParse.parse_args()

# DEFINE HERE FOR WHICH SUBTASK(S) YOU WANT TO EVAL YOUR PREDICTIONS
EVAL_TE = arguments.TE_folder is not None
EVAL_NER = arguments.NER_folder is not None
EVAL_NERD = arguments.NERD_folder is not None
EVAL_MENTION_LEVEL_RE = arguments.M_RE_folder is not None
EVAL_CONCEPT_LEVEL_RE = arguments.C_RE_folder is not None

GROUND_TRUTH_PATH = "../data/Annotations/Dev/json_format/dev.json"
GROUND_TRUTH_PATH_TE = "../data/Annotations/Dev/json_format/dev_term_only.json"

try:
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as file:
        ground_truth = json.load(file)
except OSError:
    raise OSError(f'Error in opening the specified json file: {GROUND_TRUTH_PATH}')

try:
    with open(GROUND_TRUTH_PATH_TE, 'r', encoding='utf-8') as file:
        ground_truth_TE = json.load(file)
except OSError:
    raise OSError(f'Error in opening the specified json file: {GROUND_TRUTH_PATH_TE}')


LEGAL_ENTITY_LABELS = [
    "anatomical location",
    "animal",
    "bacteria",
    "biomedical technique",
    "chemical",
    "DDF",
    "dietary supplement",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique"
]

LEGAL_RELATION_LABELS = [
    "administered",
    "affect",
    "change abundance",
    "change effect",
    "change expression",
    "compared to",
    "impact",
    "influence",
    "interact",
    "is a",
    "is linked to",
    "located in",
    "part of",
    "produced by",
    "strike",
    "target",
    "used by"
]



def remove_duplicated_entities(predictions: dict) -> None:
    removed_count = 0
    for pmid in list(predictions.keys()):
        seen = set()
        deduped = []
        for ent in predictions[pmid]["entities"]:
            key = (ent["start_idx"], ent["end_idx"], ent["location"])
            if key not in seen:
                seen.add(key)
                deduped.append(ent)
            else:
                removed_count += 1
        predictions[pmid]["entities"] = deduped
    
    if removed_count > 0:
        print(f"=== Removed {removed_count} duplicated entities from predictions ===")
    else:
        #print("=== No duplicated entities found in predictions ===")
        pass

def remove_illegal_entities(predictions: dict) -> None:
    removed_count = 0
    for pmid in list(predictions.keys()):
        seen = set()
        parsed = []
        for ent in predictions[pmid]["entities"]:
            if ent["label"] not in LEGAL_ENTITY_LABELS:
                removed_count += 1
                continue
            parsed.append(ent)
        predictions[pmid]["entities"] = parsed
    
    if removed_count > 0:
        print(f"=== Removed {removed_count} illegal entities from predictions ===")
    else:
        #print("=== No duplicated entities found in predictions ===")
        pass

def remove_overlapping_entities(predictions: dict) -> None:
    removed_count = 0

    # Iterate over PMIDs
    for pmid in list(predictions.keys()):
        original_len = len(predictions[pmid]['entities'])
        
        # Group entities by location
        groups = {'title': [], 'abstract': []}
        for ent in predictions[pmid]['entities']:
            loc = ent["location"]
            groups[loc].append(ent)

        # For each location, build overlap clusters and select the longest
        keepers = set()
        for loc in groups:
            group = groups[loc]
            # sort by start_idx so we have overlapping entities contiguous
            group = sorted(group, key=lambda e: e["start_idx"])

            clusters = []
            cluster = []
            current_end = None

            for ent in group:
                if not cluster:
                    # start the first cluster
                    cluster = [ent]
                    current_end = ent["end_idx"]
                else:
                    # check overlap: ent.start_idx < current_end
                    if ent["start_idx"] < current_end:
                        cluster.append(ent)
                        # extend cluster span if needed
                        if ent["end_idx"] > current_end:
                            current_end = ent["end_idx"]
                    else:
                        clusters.append(cluster)
                        cluster = [ent]
                        current_end = ent["end_idx"]
            if cluster:
                clusters.append(cluster)

            # pick the longest entity in each cluster
            for clust in clusters:
                # initialize with first entity
                longest = clust[0]
                max_len = longest["end_idx"] - longest["start_idx"]
                # compare with the rest
                for ent in clust[1:]:
                    length = ent["end_idx"] - ent["start_idx"]
                    if length > max_len:
                        longest = ent
                        max_len = length
                # track by (start, end, loc)
                keepers.add((longest["start_idx"],
                             longest["end_idx"],
                             longest["location"]))

        # Rebuild the entity list in original order, keeping only the keepers
        deduped = []
        for ent in predictions[pmid]['entities']:
            key = (ent["start_idx"], ent["end_idx"], ent["location"])
            if key in keepers:
                deduped.append(ent)
                keepers.remove(key)  # avoid duplicates

        predictions[pmid]["entities"] = deduped

        # count how many overlapping entities have been removed for this document
        removed_count += (original_len - len(deduped))

    if removed_count > 0:
        print(f"=== Removed {removed_count} overlapping entities ===")
    else:
        #print("=== No overlapping entity found ===")
        pass

def eval_submission_TE(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            predictions = json.load(file)
    except OSError:
        raise OSError(f'Error in opening the specified json file: {path}')

    # Remove duplicated and overlapping entities
    remove_duplicated_entities(predictions)
    remove_overlapping_entities(predictions)
    
    ground_truth_NER = dict()
    count_annotated_entities_per_label = {}
    
    for pmid, article in ground_truth_TE.items():
        if pmid not in ground_truth_NER:
            ground_truth_NER[pmid] = []
        for entity in article['entities']:
            start_idx = int(entity["start_idx"])
            end_idx = int(entity["end_idx"])
            location = str(entity["location"])
            text_span = str(entity["text_span"])
            label = str(entity["label"]) 
            
            entry = (start_idx, end_idx, location, text_span, label)
            ground_truth_NER[pmid].append(entry)
            
            if label not in count_annotated_entities_per_label:
                count_annotated_entities_per_label[label] = 0
            count_annotated_entities_per_label[label] += 1

    count_predicted_entities_per_label = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    count_true_positives_per_label = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    count_predicted_entities = 0
    count_true_positives = 0

    for pmid in predictions.keys():
        try:
            entities = predictions[pmid]['entities']
        except KeyError:
            raise KeyError(f'{pmid} - Not able to find field \"entities\" within article')
        
        for entity in entities:
            try:
                start_idx = int(entity["start_idx"])
                end_idx = int(entity["end_idx"])
                location = str(entity["location"])
                text_span = str(entity["text_span"])
                label = str(entity["label"]) 
            except KeyError:
                raise KeyError(f'{pmid} - Not able to find one or more of the expected fields for entity: {entity}')
            
            if label != "term":
                raise NameError(f'{pmid} - Illegal label {label} for entity: {entity}')

            count_predicted_entities += 1
            if label in count_predicted_entities_per_label:
                count_predicted_entities_per_label[label] += 1

            entry = (start_idx, end_idx, location, text_span, label)
            if entry in ground_truth_NER[pmid]:
                count_true_positives += 1
                count_true_positives_per_label[label] += 1

    count_annotated_entities = sum(count_annotated_entities_per_label[label] for label in list(count_annotated_entities_per_label.keys()))
    #count_predicted_entities = sum(count_predicted_entities_per_label[label] for label in list(count_annotated_entities_per_label.keys()))
    #count_true_positives = sum(count_true_positives_per_label[label] for label in list(count_annotated_entities_per_label.keys()))
    
    micro_precision = count_true_positives / (count_predicted_entities + 1e-10)
    micro_recall = count_true_positives / (count_annotated_entities + 1e-10)
    micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10))

    precision, recall, f1 = 0, 0, 0
    n = 0
    for label in list(count_annotated_entities_per_label.keys()):
        n += 1
        current_precision = count_true_positives_per_label[label] / (count_predicted_entities_per_label[label] + 1e-10) 
        current_recall = count_true_positives_per_label[label] / (count_annotated_entities_per_label[label] + 1e-10) 
        
        precision += current_precision
        recall += current_recall
        f1 += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))
    
    precision = precision / n
    recall = recall / n
    f1 = f1 / n

    return precision, recall, f1, micro_precision, micro_recall, micro_f1


def eval_submission_NER(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            predictions = json.load(file)
    except OSError:
        raise OSError(f'Error in opening the specified json file: {path}')

    # Remove duplicated and overlapping entities
    remove_duplicated_entities(predictions)
    remove_overlapping_entities(predictions)
    remove_illegal_entities(predictions)
    
    ground_truth_NER = dict()
    count_annotated_entities_per_label = {}
    
    for pmid, article in ground_truth.items():
        if pmid not in ground_truth_NER:
            ground_truth_NER[pmid] = []
        for entity in article['entities']:
            start_idx = int(entity["start_idx"])
            end_idx = int(entity["end_idx"])
            location = str(entity["location"])
            text_span = str(entity["text_span"])
            label = str(entity["label"]) 
            
            entry = (start_idx, end_idx, location, text_span, label)
            ground_truth_NER[pmid].append(entry)
            
            if label not in count_annotated_entities_per_label:
                count_annotated_entities_per_label[label] = 0
            count_annotated_entities_per_label[label] += 1

    count_predicted_entities_per_label = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    count_true_positives_per_label = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    count_predicted_entities = 0
    count_true_positives = 0

    for pmid in predictions.keys():
        try:
            entities = predictions[pmid]['entities']
        except KeyError:
            raise KeyError(f'{pmid} - Not able to find field \"entities\" within article')
        
        for entity in entities:
            try:
                start_idx = int(entity["start_idx"])
                end_idx = int(entity["end_idx"])
                location = str(entity["location"])
                text_span = str(entity["text_span"])
                label = str(entity["label"]) 
            except KeyError:
                raise KeyError(f'{pmid} - Not able to find one or more of the expected fields for entity: {entity}')
            
            if label not in LEGAL_ENTITY_LABELS:
                raise NameError(f'{pmid} - Illegal label {label} for entity: {entity}')

            count_predicted_entities += 1
            if label in count_predicted_entities_per_label:
                count_predicted_entities_per_label[label] += 1

            entry = (start_idx, end_idx, location, text_span, label)
            if entry in ground_truth_NER[pmid]:
                count_true_positives += 1
                count_true_positives_per_label[label] += 1

    count_annotated_entities = sum(count_annotated_entities_per_label[label] for label in list(count_annotated_entities_per_label.keys()))
    #count_predicted_entities = sum(count_predicted_entities_per_label[label] for label in list(count_annotated_entities_per_label.keys()))
    #count_true_positives = sum(count_true_positives_per_label[label] for label in list(count_annotated_entities_per_label.keys()))

    micro_precision = count_true_positives / (count_predicted_entities + 1e-10)
    micro_recall = count_true_positives / (count_annotated_entities + 1e-10)
    micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10))

    precision, recall, f1 = 0, 0, 0
    n = 0
    for label in list(count_annotated_entities_per_label.keys()):
        n += 1
        current_precision = count_true_positives_per_label[label] / (count_predicted_entities_per_label[label] + 1e-10) 
        current_recall = count_true_positives_per_label[label] / (count_annotated_entities_per_label[label] + 1e-10) 
        
        precision += current_precision
        recall += current_recall
        f1 += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))
    
    precision = precision / n
    recall = recall / n
    f1 = f1 / n

    return precision, recall, f1, micro_precision, micro_recall, micro_f1

def eval_submission_NERD(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            predictions = json.load(file)
    except OSError:
        raise OSError(f'Error in opening the specified json file: {path}')

    # Remove duplicated and overlapping entities
    remove_duplicated_entities(predictions)
    remove_overlapping_entities(predictions)
    remove_illegal_entities(predictions)
    ground_truth_NERD = dict()
    count_annotated_entities_per_label = {}
    
    for pmid, article in ground_truth.items():
        if pmid not in ground_truth_NERD:
            ground_truth_NERD[pmid] = []
        for entity in article['entities']:
            start_idx = int(entity["start_idx"])
            end_idx = int(entity["end_idx"])
            location = str(entity["location"])
            text_span = str(entity["text_span"])
            label = str(entity["label"]) 
            uri = str(entity["uri"])

            entry = (start_idx, end_idx, location, text_span, label, uri)
            ground_truth_NERD[pmid].append(entry)
            
            if label not in count_annotated_entities_per_label:
                count_annotated_entities_per_label[label] = 0
            count_annotated_entities_per_label[label] += 1

    count_predicted_entities_per_label = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    count_true_positives_per_label = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    count_predicted_entities = 0
    count_true_positives = 0

    for pmid in predictions.keys():
        try:
            entities = predictions[pmid]['entities']
        except KeyError:
            raise KeyError(f'{pmid} - Not able to find field \"entities\" within article')
        
        for entity in entities:
            try:
                start_idx = int(entity["start_idx"])
                end_idx = int(entity["end_idx"])
                location = str(entity["location"])
                text_span = str(entity["text_span"])
                label = str(entity["label"]) 
                uri = str(entity["uri"])
            except KeyError:
                raise KeyError(f'{pmid} - Not able to find one or more of the expected fields for entity: {entity}')
            
            if label not in LEGAL_ENTITY_LABELS:
                raise NameError(f'{pmid} - Illegal label {label} for entity: {entity}')

            count_predicted_entities += 1
            if label in count_predicted_entities_per_label:
                count_predicted_entities_per_label[label] += 1

            entry = (start_idx, end_idx, location, text_span, label, uri)
            if entry in ground_truth_NERD[pmid]:
                count_true_positives += 1
                count_true_positives_per_label[label] += 1

    count_annotated_entities = sum(count_annotated_entities_per_label[label] for label in list(count_annotated_entities_per_label.keys()))
    #count_predicted_entities = sum(count_predicted_entities_per_label[label] for label in list(count_annotated_entities_per_label.keys()))
    #count_true_positives = sum(count_true_positives_per_label[label] for label in list(count_annotated_entities_per_label.keys()))

    micro_precision = count_true_positives / (count_predicted_entities + 1e-10)
    micro_recall = count_true_positives / (count_annotated_entities + 1e-10)
    micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10))

    precision, recall, f1 = 0, 0, 0
    n = 0
    for label in list(count_annotated_entities_per_label.keys()):
        n += 1
        current_precision = count_true_positives_per_label[label] / (count_predicted_entities_per_label[label] + 1e-10) 
        current_recall = count_true_positives_per_label[label] / (count_annotated_entities_per_label[label] + 1e-10) 
        
        precision += current_precision
        recall += current_recall
        f1 += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))
    
    precision = precision / n
    recall = recall / n
    f1 = f1 / n

    return precision, recall, f1, micro_precision, micro_recall, micro_f1


def remove_duplicated_mention_level_relations(predictions: dict) -> None:
    removed_count = 0
    for pmid in list(predictions.keys()):
        seen = set()
        deduped = []
        for rel in predictions[pmid]["mention_level_relations"]:
            key = (rel['subject_text_span'], rel["subject_label"], rel["predicate"], rel['object_text_span'], rel["object_label"])
            if key not in seen:
                seen.add(key)
                deduped.append(rel)
            else:
                removed_count += 1
        predictions[pmid]["mention_level_relations"] = deduped

    if removed_count > 0:
        print(f"=== Removed {removed_count} duplicated mention-level relations from predictions ===")
    else:
        #print("=== No duplicated mention-level relations found in predictions ===")
        pass

def remove_illegal_mention_level_relations(predictions: dict) -> None:
    removed_count = 0
    for pmid in list(predictions.keys()):
        seen = set()
        parsed = []
        for rel in predictions[pmid]["mention_level_relations"]:
            if rel["subject_label"] not in LEGAL_ENTITY_LABELS or rel["object_label"] not in LEGAL_ENTITY_LABELS or rel["predicate"] not in LEGAL_RELATION_LABELS:
                removed_count += 1
                continue
            parsed.append(rel)
        predictions[pmid]["mention_level_relations"] = parsed

    if removed_count > 0:
        print(f"=== Removed {removed_count} illegal mention-level relations from predictions ===")
    else:
        #print("=== No illegal mention-level relations found in predictions ===")
        pass


def eval_submission_mention_level_RE(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            predictions = json.load(file)
    except OSError:
        raise OSError(f'Error in opening the specified json file: {path}')

    # Remove duplicated mention-level relations
    remove_duplicated_mention_level_relations(predictions)
    remove_illegal_mention_level_relations(predictions)

    ground_truth_mention_level_RE = dict()
    count_annotated_relations_per_label = {}

    for pmid, article in ground_truth.items():
        if pmid not in ground_truth_mention_level_RE:
            ground_truth_mention_level_RE[pmid] = []
        for relation in article['mention_level_relations']:
            subject_text_span = str(relation["subject_text_span"])
            subject_label = str(relation["subject_label"])
            predicate = str(relation["predicate"])
            object_text_span = str(relation["object_text_span"])
            object_label = str(relation["object_label"]) 

            entry = (subject_text_span, subject_label, predicate, object_text_span, object_label)
            ground_truth_mention_level_RE[pmid].append(entry)
            label = (subject_label, predicate, object_label)
            if label not in count_annotated_relations_per_label:
                count_annotated_relations_per_label[label] = 0
            count_annotated_relations_per_label[label] += 1

    count_predicted_relations_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}
    count_true_positives_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}
    count_predicted_relations = 0
    count_true_positives = 0

    for pmid in predictions.keys():
        try:
            relations = predictions[pmid]['mention_level_relations']
        except KeyError:
            raise KeyError(f'{pmid} - Not able to find field \"mention_level_relations\" within article')

        for relation in relations:
            try:
                subject_text_span = str(relation["subject_text_span"])
                subject_label = str(relation["subject_label"])
                predicate = str(relation["predicate"])
                object_text_span = str(relation["object_text_span"])
                object_label = str(relation["object_label"]) 
            except KeyError:
                raise KeyError(f'{pmid} - Not able to find one or more of the expected fields for relation: {relation}')
            
            if subject_label not in LEGAL_ENTITY_LABELS:
                raise NameError(f'{pmid} - Illegal subject entity label {subject_label} for relation: {relation}')
            
            if object_label not in LEGAL_ENTITY_LABELS:
                raise NameError(f'{pmid} - Illegal object entity label {object_label} for relation: {relation}')
            
            if predicate not in LEGAL_RELATION_LABELS:
                raise NameError(f'{pmid} - Illegal predicate {predicate} for relation: {relation}')
                        
            entry = (subject_text_span, subject_label, predicate, object_text_span, object_label)
            label = (subject_label, predicate, object_label) 
            
            count_predicted_relations += 1
            if label in count_predicted_relations_per_label:
                count_predicted_relations_per_label[label] += 1
            
            if entry in ground_truth_mention_level_RE[pmid]:
                count_true_positives += 1
                count_true_positives_per_label[label] += 1
    
    count_annotated_relations = sum(count_annotated_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys()))
    #count_predicted_relations = sum(count_predicted_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys()))
    #count_true_positives = sum(count_true_positives_per_label[label] for label in list(count_annotated_relations_per_label.keys()))

    micro_precision = count_true_positives / (count_predicted_relations + 1e-10)
    micro_recall = count_true_positives / (count_annotated_relations + 1e-10)
    micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10))

    precision, recall, f1 = 0, 0, 0
    n = 0
    for label in list(count_annotated_relations_per_label.keys()):
        n += 1
        current_precision = count_true_positives_per_label[label] / (count_predicted_relations_per_label[label] + 1e-10) 
        current_recall = count_true_positives_per_label[label] / (count_annotated_relations_per_label[label] + 1e-10) 
        
        precision += current_precision
        recall += current_recall
        f1 += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))
    
    precision = precision / n
    recall = recall / n
    f1 = f1 / n

    return precision, recall, f1, micro_precision, micro_recall, micro_f1


def remove_duplicated_concept_level_relations(predictions: dict) -> None:
    removed_count = 0
    for pmid in list(predictions.keys()):
        seen = set()
        deduped = []
        for rel in predictions[pmid]["concept_level_relations"]:
            key = (rel['subject_uri'], rel["subject_label"], rel["predicate"], rel['object_uri'], rel["object_label"])
            if key not in seen:
                seen.add(key)
                deduped.append(rel)
            else:
                removed_count += 1
        predictions[pmid]["concept_level_relations"] = deduped

    if removed_count > 0:
        print(f"=== Removed {removed_count} duplicated concept-level relations from predictions ===")
    else:
        #print("=== No duplicated concept-level relations found in predictions ===")
        pass

def remove_illegal_concept_level_relations(predictions: dict) -> None:
    removed_count = 0
    for pmid in list(predictions.keys()):
        seen = set()
        parsed = []
        for rel in predictions[pmid]["concept_level_relations"]:
            if rel["subject_label"] not in LEGAL_ENTITY_LABELS or rel["object_label"] not in LEGAL_ENTITY_LABELS or rel["predicate"] not in LEGAL_RELATION_LABELS:
                removed_count += 1
                continue
            parsed.append(rel)
        predictions[pmid]["concept_level_relations"] = parsed

    if removed_count > 0:
        print(f"=== Removed {removed_count} illegal concept-level relations from predictions ===")
    else:
        #print("=== No illegal concept-level relations found in predictions ===")
        pass

    if removed_count > 0:
        print(f"=== Removed {removed_count} illegal mention-level relations from predictions ===")
    else:
        #print("=== No illegal mention-level relations found in predictions ===")
        pass

def eval_submission_concept_level_RE(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            predictions = json.load(file)
    except OSError:
        raise OSError(f'Error in opening the specified json file: {path}')
    
    # Remove duplicated concept-level relations
    remove_duplicated_concept_level_relations(predictions)
    remove_illegal_concept_level_relations(predictions)

    ground_truth_concept_level_RE = dict()
    count_annotated_relations_per_label = {}

    for pmid, article in ground_truth.items():
        if pmid not in ground_truth_concept_level_RE:
            ground_truth_concept_level_RE[pmid] = []
        for relation in article['concept_level_relations']:
            subject_uri = str(relation["subject_uri"])
            subject_label = str(relation["subject_label"])
            predicate = str(relation["predicate"])
            object_uri = str(relation["object_uri"])
            object_label = str(relation["object_label"])

            entry = (subject_uri, subject_label, predicate, object_uri, object_label)
            ground_truth_concept_level_RE[pmid].append(entry)
            label = (subject_label, predicate, object_label)
            if label not in count_annotated_relations_per_label:
                count_annotated_relations_per_label[label] = 0
            count_annotated_relations_per_label[label] += 1

    count_predicted_relations_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}
    count_true_positives_per_label = {label: 0 for label in list(count_annotated_relations_per_label.keys())}
    count_predicted_relations = 0
    count_true_positives = 0

    for pmid in predictions.keys():
        try:
            relations = predictions[pmid]['concept_level_relations']
        except KeyError:
            raise KeyError(f'{pmid} - Not able to find field \"concept_level_relations\" within article')

        for relation in relations:
            try:
                subject_uri = str(relation["subject_uri"])
                subject_label = str(relation["subject_label"])
                predicate = str(relation["predicate"])
                object_uri = str(relation["object_uri"])
                object_label = str(relation["object_label"]) 
            except KeyError:
                raise KeyError(f'{pmid} - Not able to find one or more of the expected fields for relation: {relation}')
            
            if subject_label not in LEGAL_ENTITY_LABELS:
                raise NameError(f'{pmid} - Illegal subject entity label {subject_label} for relation: {relation}')
            
            if object_label not in LEGAL_ENTITY_LABELS:
                raise NameError(f'{pmid} - Illegal object entity label {object_label} for relation: {relation}')
            
            if predicate not in LEGAL_RELATION_LABELS:
                raise NameError(f'{pmid} - Illegal predicate {predicate} for relation: {relation}')

            entry = (subject_uri, subject_label, predicate, object_uri, object_label)
            label = (subject_label, predicate, object_label)

            count_predicted_relations += 1
            if label in count_predicted_relations_per_label:
                count_predicted_relations_per_label[label] += 1

            if entry in ground_truth_concept_level_RE[pmid]:
                count_true_positives += 1
                count_true_positives_per_label[label] += 1
    
    count_annotated_relations = sum(count_annotated_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys()))
    #count_predicted_relations = sum(count_predicted_relations_per_label[label] for label in list(count_annotated_relations_per_label.keys()))
    #count_true_positives = sum(count_true_positives_per_label[label] for label in list(count_annotated_relations_per_label.keys()))

    micro_precision = count_true_positives / (count_predicted_relations + 1e-10)
    micro_recall = count_true_positives / (count_annotated_relations + 1e-10)
    micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10))

    precision, recall, f1 = 0, 0, 0
    n = 0
    for label in list(count_annotated_relations_per_label.keys()):
        n += 1
        current_precision = count_true_positives_per_label[label] / (count_predicted_relations_per_label[label] + 1e-10) 
        current_recall = count_true_positives_per_label[label] / (count_annotated_relations_per_label[label] + 1e-10) 
        
        precision += current_precision
        recall += current_recall
        f1 += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))
    
    precision = precision / n
    recall = recall / n
    f1 = f1 / n

    return precision, recall, f1, micro_precision, micro_recall, micro_f1


def main_TE(predictions_folder, round_to_decimal_position=4):
    results = {}
    for filename in os.listdir(predictions_folder):
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(predictions_folder, filename)
        precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_TE(file_path)
        parsed_filename = filename.replace('.json', '').replace('_test_predictions', '').replace('_dev_predictions', '')
        results[parsed_filename] = {
            'precision': round(precision, round_to_decimal_position),
            'recall': round(recall, round_to_decimal_position),
            'f1': round(f1, round_to_decimal_position),
            'micro_precision': round(micro_precision, round_to_decimal_position),
            'micro_recall': round(micro_recall, round_to_decimal_position),
            'micro_f1': round(micro_f1, round_to_decimal_position)
        }
        print(f"\n\n=== TE {parsed_filename}===")
        print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
        print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
        print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
        print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
        print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
        print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")
    
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.sort_values(by='micro_f1', ascending=False, inplace=True)
    return results_df

def main_NER(predictions_folder, round_to_decimal_position=4):
    results = {}
    for filename in os.listdir(predictions_folder):
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(predictions_folder, filename)
        precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_NER(file_path)
        parsed_filename = filename.replace('.json', '').replace('_test_predictions', '').replace('_dev_predictions', '')
        #results[parsed_filename] = {
        results[filename] = {
            'precision': round(precision, round_to_decimal_position),
            'recall': round(recall, round_to_decimal_position),
            'f1': round(f1, round_to_decimal_position),
            'micro_precision': round(micro_precision, round_to_decimal_position),
            'micro_recall': round(micro_recall, round_to_decimal_position),
            'micro_f1': round(micro_f1, round_to_decimal_position)
        }
        print(f"\n\n=== NER {parsed_filename}===")
        print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
        print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
        print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
        print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
        print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
        print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.sort_values(by='micro_f1', ascending=False, inplace=True)
    return results_df

def main_NERD(predictions_folder, round_to_decimal_position=4):
    results = {}
    for filename in os.listdir(predictions_folder):
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(predictions_folder, filename)
        precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_NERD(file_path)
        parsed_filename = filename.replace('.json', '').replace('_test_predictions', '').replace('_dev_predictions', '')
        results[parsed_filename] = {
            'precision': round(precision, round_to_decimal_position),
            'recall': round(recall, round_to_decimal_position),
            'f1': round(f1, round_to_decimal_position),
            'micro_precision': round(micro_precision, round_to_decimal_position),
            'micro_recall': round(micro_recall, round_to_decimal_position),
            'micro_f1': round(micro_f1, round_to_decimal_position)
        }
        print(f"\n\n=== NERD {parsed_filename}===")
        print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
        print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
        print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
        print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
        print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
        print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.sort_values(by='micro_f1', ascending=False, inplace=True)
    return results_df

def main_MENTION_LEVEL_RE(predictions_folder, round_to_decimal_position=4):
    results = {}
    for filename in os.listdir(predictions_folder):
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(predictions_folder, filename)
        precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_mention_level_RE(file_path)
        parsed_filename = filename.replace('.json', '').replace('_test_predictions', '').replace('_dev_predictions', '')
        results[parsed_filename] = {
            'precision': round(precision, round_to_decimal_position),
            'recall': round(recall, round_to_decimal_position),
            'f1': round(f1, round_to_decimal_position),
            'micro_precision': round(micro_precision, round_to_decimal_position),
            'micro_recall': round(micro_recall, round_to_decimal_position),
            'micro_f1': round(micro_f1, round_to_decimal_position)
        }
        print(f"\n\n=== Mention-level RE {parsed_filename}===")
        print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
        print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
        print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
        print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
        print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
        print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.sort_values(by='micro_f1', ascending=False, inplace=True)
    return results_df

def main_CONCEPT_LEVEL_RE(predictions_folder, round_to_decimal_position=4):
    results = {}
    for filename in os.listdir(predictions_folder):
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(predictions_folder, filename)
        precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_concept_level_RE(file_path)
        parsed_filename = filename.replace('.json', '').replace('_test_predictions', '').replace('_dev_predictions', '')
        results[parsed_filename] = {
            'precision': round(precision, round_to_decimal_position),
            'recall': round(recall, round_to_decimal_position),
            'f1': round(f1, round_to_decimal_position),
            'micro_precision': round(micro_precision, round_to_decimal_position),
            'micro_recall': round(micro_recall, round_to_decimal_position),
            'micro_f1': round(micro_f1, round_to_decimal_position)
        }
        print(f"\n\n=== Concept-level RE {parsed_filename}===")
        print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
        print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
        print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
        print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
        print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
        print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.sort_values(by='micro_f1', ascending=False, inplace=True)
    return results_df

if __name__ == '__main__':
    round_to_decimal_position = 4

    if EVAL_TE:
        results = main_TE(arguments.TE_folder, round_to_decimal_position)
        results.to_csv('TE_evaluation_results.csv')

    if EVAL_NER:
        results = main_NER(arguments.NER_folder, round_to_decimal_position)
        results.to_csv('NER_evaluation_results.csv')

    if EVAL_NERD:
        results = main_NERD(arguments.NERD_folder, round_to_decimal_position)
        results.to_csv('NERD_evaluation_results.csv')

    if  EVAL_MENTION_LEVEL_RE:
        results = main_MENTION_LEVEL_RE(arguments.M_RE_folder, round_to_decimal_position)
        results.to_csv('MENTION_LEVEL_RE_evaluation_results.csv')

    if EVAL_CONCEPT_LEVEL_RE:
        results = main_CONCEPT_LEVEL_RE(arguments.C_RE_folder, round_to_decimal_position)
        results.to_csv('CONCEPT_LEVEL_RE_evaluation_results.csv')