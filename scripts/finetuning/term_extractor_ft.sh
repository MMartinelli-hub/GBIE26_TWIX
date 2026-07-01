# cd <path_to_your_project_root>

# Term extraction fine-tuning data
python -m src.utils.LLMFinetuningDataGenerator \
    --task term_extraction \
    --train_data_path data/Annotations/Train/merged_quality/json_format/train_merged.json \
    --train_output_path data/finetune/te_train.jsonl \
    --dev_data_path data/Annotations/Dev/json_format/dev.json \
    --dev_output_path data/finetune/te_dev.jsonl