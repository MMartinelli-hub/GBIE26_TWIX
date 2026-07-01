# cd <path_to_your_project_root>

# Entity recognition fine-tuning data
python -m src.utils.LLMFinetuningDataGenerator \
    --task entity_recognition \
    --train_data_path data/Annotations/Train/merged_quality/json_format/train_merged.json \
    --train_output_path data/finetune/ner_train.jsonl \
    --dev_data_path data/Annotations/Dev/json_format/dev.json \
    --dev_output_path data/finetune/ner_dev.jsonl \