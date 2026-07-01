# cd <path_to_your_project_root>

# Relation extraction fine-tuning data
python -m src.utils.LLMFinetuningDataGenerator \
    --task relation_extraction \
    --train_data_path data/Annotations/Train/merged_quality/json_format/train_merged.json \
    --train_output_path data/finetune/re_train.jsonl \
    --dev_data_path data/Annotations/Dev/json_format/dev.json \
    --dev_output_path data/finetune/re_dev.jsonl \
    --concatenate_title_abstract