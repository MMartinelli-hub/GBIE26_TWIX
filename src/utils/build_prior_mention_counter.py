#!/usr/bin/env python3
"""
Build the mention-counter file required by entity-linkings PRIOR retrieval.

The upstream PRIOR indexer expects a JSON object shaped as:

    {
      "surface mention": {
        "Entity_Title": 3,
        "Other_Entity": 1
      }
    }

For this GBIE project, annotations carry ontology URIs, while the PRIOR indexer
matches counter entries against the dictionary's ``name`` field. This script
bridges that gap by mapping each annotation URI to its dictionary name.

The entity-linkings PRIOR indexer normalizes entity names by replacing spaces
with underscores before dictionary lookup. For dictionaries whose names contain
spaces, use ``--prior-dictionary-output`` and point inference
``entity_dict_path`` at the generated dictionary copy.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NIL_URIS = {"", "-1", "NIL", "nil", "None", "none", "null"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def normalize_entity_name_for_prior(name: str) -> str:
    """Mirror entity-linkings PRIOR entity-name normalization."""
    return name.replace(" ", "_")


def load_dictionary(dictionary_path: Path) -> tuple[dict[str, str], list[dict[str, Any]], int]:
    entries = list(iter_jsonl(dictionary_path))
    uri_to_name: dict[str, str] = {}
    duplicate_ids = 0

    for entry in entries:
        entity_id = entry.get("id")
        entity_name = entry.get("name")
        if not entity_id or not entity_name:
            continue
        if entity_id in uri_to_name:
            duplicate_ids += 1
            continue
        uri_to_name[entity_id] = entity_name

    return uri_to_name, entries, duplicate_ids


def mention_from_annotation(entity: dict[str, Any]) -> str | None:
    mention = entity.get("text_span")
    if mention is None:
        return None
    mention = str(mention).strip()
    return mention or None


def build_counter(
    train_paths: list[Path],
    uri_to_name: dict[str, str],
    normalize_entity_names: bool,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
    stats = {
        "documents": 0,
        "entities": 0,
        "counted": 0,
        "skipped_nil_uri": 0,
        "skipped_missing_uri": 0,
        "skipped_unknown_uri": 0,
        "skipped_missing_mention": 0,
    }

    for train_path in train_paths:
        data = load_json(train_path)
        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level object in {train_path}")

        stats["documents"] += len(data)
        for content in data.values():
            for entity in content.get("entities", []):
                stats["entities"] += 1

                uri = entity.get("uri")
                if uri is None:
                    stats["skipped_missing_uri"] += 1
                    continue
                uri = str(uri).strip()
                if uri in NIL_URIS:
                    stats["skipped_nil_uri"] += 1
                    continue

                mention = mention_from_annotation(entity)
                if mention is None:
                    stats["skipped_missing_mention"] += 1
                    continue

                entity_name = uri_to_name.get(uri)
                if entity_name is None:
                    stats["skipped_unknown_uri"] += 1
                    continue

                if normalize_entity_names:
                    entity_name = normalize_entity_name_for_prior(entity_name)

                counter[mention][entity_name] += 1
                stats["counted"] += 1

    serializable = {
        mention: dict(sorted(entity_counts.items()))
        for mention, entity_counts in sorted(counter.items())
    }
    return serializable, stats


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_prior_dictionary(path: Path, entries: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            output_entry = dict(entry)
            name = output_entry.get("name")
            if isinstance(name, str):
                normalized_name = normalize_entity_name_for_prior(name)
                if normalized_name != name:
                    changed += 1
                output_entry["name"] = normalized_name
            fh.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build entity-linkings PRIOR mention_counter.json from GBIE annotations."
    )
    parser.add_argument(
        "--train-data-paths",
        nargs="+",
        required=True,
        type=Path,
        help="One or more GBIE training JSON files with entities containing text_span and uri.",
    )
    parser.add_argument(
        "--entity-dict-path",
        required=True,
        type=Path,
        help="Entity-linkings dictionary JSONL with id/name/description fields.",
    )
    parser.add_argument(
        "--output-path",
        default=Path("runs/entityLinker/prior/mention_counter.json"),
        type=Path,
        help="Where to write the PRIOR mention counter JSON.",
    )
    parser.add_argument(
        "--prior-dictionary-output",
        default=Path("runs/entityLinker/prior/entity_dictionary_prior.jsonl"),
        type=Path,
        help=(
            "Where to write a PRIOR-compatible dictionary copy with spaces in names "
            "replaced by underscores. Use this path as linker.entity_dict_path for PRIOR."
        ),
    )
    parser.add_argument(
        "--no-prior-dictionary",
        action="store_true",
        help="Only write the mention counter; do not write a normalized dictionary copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    missing_inputs = [
        str(path)
        for path in [args.entity_dict_path, *args.train_data_paths]
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing_inputs))

    write_prior_dict = not args.no_prior_dictionary
    uri_to_name, dictionary_entries, duplicate_ids = load_dictionary(args.entity_dict_path)
    counter, stats = build_counter(
        train_paths=args.train_data_paths,
        uri_to_name=uri_to_name,
        normalize_entity_names=write_prior_dict,
    )

    write_json(args.output_path, counter)

    changed_dictionary_names = 0
    if write_prior_dict:
        changed_dictionary_names = write_prior_dictionary(
            args.prior_dictionary_output,
            dictionary_entries,
        )

    print("[build_prior_mention_counter] Done")
    print(f"  Mention counter: {args.output_path}")
    print(f"  Mentions: {len(counter)}")
    print(f"  Counted annotations: {stats['counted']}")
    print(f"  Skipped unknown URIs: {stats['skipped_unknown_uri']}")
    print(f"  Skipped NIL URIs: {stats['skipped_nil_uri']}")
    print(f"  Skipped missing mentions/URIs: {stats['skipped_missing_mention'] + stats['skipped_missing_uri']}")
    if duplicate_ids:
        print(f"  Duplicate dictionary ids ignored: {duplicate_ids}")
    if write_prior_dict:
        print(f"  PRIOR dictionary: {args.prior_dictionary_output}")
        print(f"  Dictionary names normalized: {changed_dictionary_names}")


if __name__ == "__main__":
    main()
