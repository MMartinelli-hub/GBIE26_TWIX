# %%
import json
import pandas as pd
import os
import torch
import random
import numpy as np

###################################
# Constants Shared Across Modules #
###################################

# Full set of entity labels used throughout the termClassifier module
LABEL_LIST = [
    "NA",
    "anatomical location",
    "animal",
    "biomedical technique",
    "bacteria",
    "chemical",
    "dietary supplement",
    "DDF",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique",
]

# Full NER BIO label set for the joint extraction + classification task.
# "O" is always first; B- and I- variants are generated for every non-NA entity class drawn from LABEL_LIST, preserving the original ordering.
NER_BIO_LABELS = ["O"] + [
    f"{prefix}-{label}"
    for prefix in ["B", "I"]
    for label in LABEL_LIST
    if label != "NA"
]

# Entity type list for GLiNER -- all LABEL_LIST entries except "NA", lower-cased to match the convention enforced by GLiNER during both fine-tuning and inference.
# The ordering mirrors LABEL_LIST so that the two constants stay in sync.
GLINER_ENTITY_TYPES = [label.lower() for label in LABEL_LIST if label != "NA"]


# Full set of relation predicate labels used throughout the relationExtractor module.
# "NA" is always the first entry and acts as the negative / no-relation class.
RELATION_LABEL_LIST = [
    "NA",
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
    "used by",
]
 
# Schema-constrained valid predicates for each (head_entity_label, tail_entity_label) pair.
# Keys are 2-tuples of entity category strings drawn from LABEL_LIST; values are sets of
# predicate strings drawn from RELATION_LABEL_LIST. Only pairs that have at least one
# valid predicate are listed -- unlisted (head, tail) combinations are structurally
# impossible and are excluded from both training sample generation and inference.
VALID_RELATIONS: dict[tuple[str, str], set[str]] = {
    # Anatomical Location as head
    ("anatomical location", "human"):               {"located in"},
    ("anatomical location", "animal"):              {"located in"},
    # Bacteria as head
    ("bacteria", "bacteria"):                       {"interact"},
    ("bacteria", "chemical"):                       {"interact"},
    ("bacteria", "drug"):                           {"interact"},
    ("bacteria", "DDF"):                            {"influence"},
    ("bacteria", "gene"):                           {"change expression"},
    ("bacteria", "human"):                          {"located in"},
    ("bacteria", "animal"):                         {"located in"},
    ("bacteria", "microbiome"):                     {"part of"},
    # Chemical as head
    ("chemical", "anatomical location"):            {"located in"},
    ("chemical", "human"):                          {"located in", "administered"},
    ("chemical", "animal"):                         {"located in", "administered"},
    ("chemical", "chemical"):                       {"interact", "part of"},
    ("chemical", "microbiome"):                     {"impact", "produced by"},
    ("chemical", "bacteria"):                       {"impact"},
    ("chemical", "DDF"):                            {"influence"},
    ("chemical", "gene"):                           {"change expression"},
    # Dietary Supplement as head
    ("dietary supplement", "bacteria"):             {"impact"},
    ("dietary supplement", "microbiome"):           {"impact"},
    ("dietary supplement", "DDF"):                  {"influence"},
    ("dietary supplement", "gene"):                 {"change expression"},
    ("dietary supplement", "human"):                {"administered"},
    ("dietary supplement", "animal"):               {"administered"},
    # Drug as head
    ("drug", "bacteria"):                           {"impact"},
    ("drug", "microbiome"):                         {"impact"},
    ("drug", "gene"):                               {"change expression"},
    ("drug", "human"):                              {"administered"},
    ("drug", "animal"):                             {"administered"},
    ("drug", "chemical"):                           {"interact"},
    ("drug", "drug"):                               {"interact"},
    ("drug", "DDF"):                                {"change effect"},
    # Food as head
    ("food", "bacteria"):                           {"impact"},
    ("food", "microbiome"):                         {"impact"},
    ("food", "DDF"):                                {"influence"},
    ("food", "gene"):                               {"change expression"},
    ("food", "human"):                              {"administered"},
    ("food", "animal"):                             {"administered"},
    # DDF as head
    ("DDF", "anatomical location"):                 {"strike"},
    ("DDF", "bacteria"):                            {"change abundance"},
    ("DDF", "microbiome"):                          {"change abundance"},
    ("DDF", "chemical"):                            {"interact"},
    ("DDF", "DDF"):                                 {"affect", "is a"},
    ("DDF", "human"):                               {"target"},
    ("DDF", "animal"):                              {"target"},
    # Human / Animal as head
    ("human", "biomedical technique"):              {"used by"},
    ("animal", "biomedical technique"):             {"used by"},
    # Microbiome as head
    ("microbiome", "biomedical technique"):         {"used by"},
    ("microbiome", "anatomical location"):          {"located in"},
    ("microbiome", "human"):                        {"located in"},
    ("microbiome", "animal"):                       {"located in"},
    ("microbiome", "gene"):                         {"change expression"},
    ("microbiome", "DDF"):                          {"is linked to"},
    ("microbiome", "microbiome"):                   {"compared to"},
}


##########################
# Data Loading Functions #
##########################

def load_json_data(file_path: str) -> dict:
    """
    Load JSON data from a file and return it as a dictionary.

    :param file_path: The path to the JSON file.
    :return: A dictionary containing the JSON data.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data from a file and return it as a pandas DataFrame.

    :param file_path: The path to the CSV file.
    :return: A pandas DataFrame containing the CSV data.
    """

    return pd.read_csv(file_path)

def save_json_data(data: dict, file_path: str, encoding: str = "utf-8", indent: int = 4) -> None:
    """
    Save a dictionary as JSON data to a file. If the directory does not exist, it will be created.

    :param data: The dictionary to save as JSON.
    :param file_path: The path to the JSON file where the data will be saved.
    :param encoding: The encoding to use for the file.
    :param indent: The indentation for the JSON data.
    :return: None
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_merge_json_data(file_paths: list) -> dict:
    """
    Load and merge multiple JSON files into a single dictionary.

    :param file_paths: A list of paths to the JSON files to be merged.
    :return: A dictionary containing the merged JSON data.
    """

    merged_data = {}
    for file_path in file_paths:
        data = load_json_data(file_path)
        merged_data.update(data)
    return merged_data

def load_merge_csv_data(file_paths: list) -> pd.DataFrame:
    """
    Load and merge multiple CSV files into a single pandas DataFrame.

    :param file_paths: A list of paths to the CSV files to be merged.
    :return: A pandas DataFrame containing the merged CSV data.
    """

    data_frames = [load_csv_data(file_path) for file_path in file_paths]
    return pd.concat(data_frames, ignore_index=True)


###############################
# Device Management Functions #
###############################

def get_device() -> torch.device:
    """
    Get the available device (GPU or CPU) for PyTorch.

    :return: A torch.device object representing the available device.
    """

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def move_to_device(data, device: torch.device):
    """
    Move data to the specified device.

    :param data: The data to be moved (can be a tensor, list, or dictionary).
    :param device: The target device to move the data to.
    :return: The data moved to the specified device.
    """

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    
def print_device_info(device: torch.device) -> None:
    """
    Print information about the specified device.

    :param device: The device to print information about.
    :return: None
    """

    print(f"Device: {device}")
    if device.type == "cuda":
        try:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
        except Exception:
            pass
    elif device.type == "mps":
        print("MPS backend is active")
        try:
            print(f"MPS recommended max memory: {torch.mps.recommended_max_memory()}")
            print(f"MPS current allocated memory: {torch.mps.current_allocated_memory()}")
            print(f"MPS driver allocated memory: {torch.mps.driver_allocated_memory()}")
        except Exception:
            pass
    
def maybe_empty_cache(device: torch.device) -> None:
    """
    Empty the GPU cache if the device is a GPU.

    :param device: The device to check for GPU and potentially empty the cache.
    :return: None
    """

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    
    
#####################
# Seeding Functions #
#####################

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


########################################
# GutBrainIE Data Processing Functions #
########################################

def get_title_and_abstract(content: dict, metadata_key: str = "metadata") -> tuple:
    """
    Extract the title and abstract from the content dictionary.
    The content dictionary should be in the format of the GutBrainIE dataset entries, where the title and abstract are located in the "metadata" field.

    :param content: A dictionary containing the content of a paper, including a "metadata" field with "title" and "abstract".
    :return: A tuple containing the title and abstract as strings.
    """
    metadata = content.get("metadata", content)
    return metadata["title"], metadata["abstract"]

def map_predicted_relations_to_entities(content: dict) -> list:
    """
    Map predicted relations to their corresponding head and tail entities in the content dictionary.
    This function assumes that the content dictionary contains a "relations" field, which is a list of dictionaries with "subject_start_idx", "subject_end_idx", "subject_location", "object_start_idx", "object_end_idx", "object_location", and "predicate" keys.
    It also assumes that there is an "entities" field, which is a list of dictionaries in GBIE format.
    It also assumes there is a "metadata" field containing the "title" and "abstract" of the paper, which are used to determine the text spans for mapping entities.
    
    :param content: A dictionary containing the content of a paper, including "relations" and "entities".
    :return: A list of dictionaries, representing parsed relations in GBIE format.
    """
    ret = {}
    for pmid, data in content.items():
        try:
            ret[pmid] = []
            for relation in data["relations"]:
                subject_start_idx, subject_end_idx, subject_location = relation["subject_start_idx"], relation["subject_end_idx"], relation["subject_location"]
                object_start_idx, object_end_idx, object_location = relation["object_start_idx"], relation["object_end_idx"], relation["object_location"]
                predicate = relation["predicate"]

                if subject_location == "title":
                    subject_text = data["metadata"]["title"][subject_start_idx:subject_end_idx+1]
                elif subject_location == "abstract":
                    subject_text = data["metadata"]["abstract"][subject_start_idx:subject_end_idx+1]
                else:
                    raise ValueError(f"Invalid subject location: {subject_location}")
                if object_location == "title":
                    object_text = data["metadata"]["title"][object_start_idx:object_end_idx+1]
                elif object_location == "abstract":
                    object_text = data["metadata"]["abstract"][object_start_idx:object_end_idx+1]
                else:
                    raise ValueError(f"Invalid object location: {object_location}")
                
                # Find the corresponding head and tail entities in the "entities" field
                head_entity = None
                tail_entity = None
                for entity in data["entities"]:
                    if entity["start_idx"] == subject_start_idx and entity["end_idx"] == subject_end_idx and entity["location"] == subject_location:
                        head_entity = entity
                    if entity["start_idx"] == object_start_idx and entity["end_idx"] == object_end_idx and entity["location"] == object_location:
                        tail_entity = entity
                    if head_entity is not None and tail_entity is not None:
                        break
                if head_entity is None:
                    raise ValueError(f"Could not find head entity for relation with subject span ({subject_start_idx}, {subject_end_idx}) in {subject_location}")
                if tail_entity is None:
                    raise ValueError(f"Could not find tail entity for relation with object span ({object_start_idx}, {object_end_idx}) in {object_location}")
                ret[pmid].append({
                    "subject_start_idx": subject_start_idx,
                    "subject_end_idx": subject_end_idx,
                    "subject_location": subject_location,
                    "subject_text_span": subject_text,
                    "subject_label": head_entity["label"],
                    "subject_uri": head_entity["uri"],
                    "predicate": predicate,
                    "confscore": -1.0,
                    "object_start_idx": object_start_idx,
                    "object_end_idx": object_end_idx,
                    "object_location": object_location,
                    "object_text_span": object_text,
                    "object_label": tail_entity["label"],
                    "object_uri": tail_entity["uri"]
                    })
        except Exception as e:
            print(f"Error processing PMID {pmid}: {e}")
    return ret

def add_mention_level_relations_to_release_dict(release_dict):
    for pmid, article in release_dict.items():
        tuples = set()
        for relation in article["relations"]:
            tuples.add((relation["subject_text_span"], relation["subject_label"], relation["predicate"], relation["object_text_span"], relation["object_label"]))
        if "mention_level_relations" not in release_dict[pmid]:
            release_dict[pmid]["mention_level_relations"] = []		
        for entry in tuples:
            release_dict[pmid]["mention_level_relations"].append({"subject_text_span": entry[0], "subject_label": entry[1], "predicate": entry[2], "object_text_span": entry[3], "object_label": entry[4]})


def add_concept_level_relations_to_release_dict(release_dict):
    for pmid, article in release_dict.items():
        tuples = set()
        for relation in article["relations"]:
            tuples.add((relation["subject_uri"], relation["subject_label"], relation["predicate"], relation["object_uri"], relation["object_label"]))
        if "concept_level_relations" not in release_dict[pmid]:
            release_dict[pmid]["concept_level_relations"] = []
        for entry in tuples:
            release_dict[pmid]["concept_level_relations"].append({"subject_uri": entry[0], "subject_label": entry[1], "predicate": entry[2], "object_uri": entry[3], "object_label": entry[4]})

def process_relation_extraction_output(input_file: str, output_file: str):
    """
    Process the output from the LLMRelationExtraction module by mapping predicted relations to their corresponding head and tail entities, and then adding mention-level and concept-level relations to the release dictionary. 
    The processed data is then saved to a new JSON file.

    :param input_file: The path to the input JSON file containing the raw output from the relation extraction module.
    :param output_file: The path to the output JSON file where the processed data will be saved.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Processing {input_file}...")
    ret = map_predicted_relations_to_entities(data)
    for pmid, relations in ret.items():
        data[pmid]['relations'] = relations
        data[pmid]['mention_level_relations'] = []
        data[pmid]['concept_level_relations'] = []
    add_mention_level_relations_to_release_dict(data)
    add_concept_level_relations_to_release_dict(data)
    with open(output_file, "w", encoding="utf-8") as f:            
        json.dump(data, f, ensure_ascii=False, indent=4)

def main_process_relation_extraction_outputs():
    import json
    import os
    base_path = "runs/relationExtractor/dev/raw"
    for filename in os.listdir(base_path):
        if filename.endswith(".json"):
            print(f"Found file: {filename}")
            input_file = os.path.join(base_path, filename)
            os.makedirs(os.path.join(base_path, "parsed"), exist_ok=True)
            output_file = os.path.join(base_path, "parsed", f"{filename}")
            process_relation_extraction_output(input_file, output_file)
            print(f"Processed {input_file} and saved to {output_file}")