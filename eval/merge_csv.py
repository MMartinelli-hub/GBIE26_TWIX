#!/usr/bin/env python3
"""
Merge multiple CSV files with the same structure into a single aggregated file.
Supports sorting by a specified column in ascending or descending order.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional


def merge_csv_files(
    csv_paths: List[str],
    output_path: str,
    sort_column: str = "micro_f1",
    ascending: bool = False
) -> None:
    """
    Merge multiple CSV files into a single file with optional sorting.
    
    Args:
        csv_paths: List of paths to CSV files to merge
        output_path: Path where the merged CSV will be saved
        sort_column: Column name to sort by (default: "micro_f1")
        ascending: If True, sort in ascending order; if False, descending (default: False)
    """
    try:
        # Read all CSV files
        dataframes = []
        for path in csv_paths:
            print(f"Reading: {path}")
            df = pd.read_csv(path)
            dataframes.append(df)
        
        # Merge all dataframes
        merged_df = pd.concat(dataframes, ignore_index=False)
        
        # Sort by specified column
        if sort_column in merged_df.columns:
            print(f"Sorting by '{sort_column}' (ascending={ascending})")
            merged_df = merged_df.sort_values(by=sort_column, ascending=ascending)
        else:
            print(f"Warning: Column '{sort_column}' not found. Available columns: {list(merged_df.columns)}")
        
        # Save merged file
        merged_df.to_csv(output_path)
        print(f"Merged CSV saved to: {output_path}")
        print(f"Total rows: {len(merged_df)}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    # Example usage:
    csv_files = [
        "results/dev/MENTION_LEVEL_RE_dev_finetune_g10_sb2.csv",
        "results/dev/MENTION_LEVEL_RE_dev_finetune_g50_sb5.csv",
        "results/dev/MENTION_LEVEL_RE_dev_finetune_g100_sb10.csv",
        "results/dev/MENTION_LEVEL_RE_dev_finetune_gs10_b2.csv",
        "results/dev/MENTION_LEVEL_RE_dev_finetune_gs100_b10.csv",
        "results/dev/MENTION_LEVEL_RE_dev_finetune_gs50_b5.csv"
    ]
    
    output_file = "results/dev/MENTION_LEVEL_RE_dev_finetune_merged.csv"
    
    # Merge with default sorting (by micro_f1, descending)
    merge_csv_files(csv_files, output_file)
    
    # Example: sort by 'f1' in ascending order
    # merge_csv_files(csv_files, output_file, sort_column="f1", ascending=True)
