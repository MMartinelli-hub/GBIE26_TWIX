# List of filenames to iterate over
filenames=("test_biolink.json" "test_biomed_abs.json" "test_biomed_full.json" 
"merged_BiomedNLP-BiomedBERT-large-uncased-abstract_test_predictions_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_ignore_noNA.json" 
"merged_BiomedNLP-BiomedBERT-large-uncased-abstract_test_predictions_BiomedNLP-BiomedElectra-base-uncased-abstract_ignore_noNA_atlop_format.json" 
"merged_BiomedNLP-BiomedBERT-large-uncased-abstract_test_predictions_BioLinkBERT-base_ignore_noNA_atlop_format.json" 
"merged_BiomedNLP-BiomedBERT-large-uncased-abstract_test_predictions_BioLinkBERT-large_ignore_noNA_atlop_format.json" 
"merged_BiomedNLP-BiomedBERT-large-uncased-abstract_test_predictions_BiomedNLP-BiomedBERT-base-uncased-abstract_ignore_noNA_atlop_format.json" 
"merged_BiomedNLP-BiomedBERT-large-uncased-abstract_test_predictions_BiomedNLP-BiomedBERT-large-uncased-abstract_ignore_noNA_atlop_format.json"
)

# Iterate over the list
for filename in "${filenames[@]}"
do
    echo "Running script for: $filename"
    # Execute the python script with the filename as an argument
    python atlop_interface.py --data_dir ./data \
    --transformer_type roberta \
    --model_name_or_path roberta-large  \
    --train_file train_annotated_g.json \
    --save_path outputs/finetune_g100_sb10 \
    --load_path outputs/finetune_g100_sb10 \
    --load_checkpoint best.ckpt \
    --dev_file "$filename" \
    --test_file "$filename" \
    --train_batch_size 4 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_labels 1 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 2.0 \
    --seed 66 \
    --num_class 18
done

