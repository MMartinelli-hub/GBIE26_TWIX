# TWIX Team @ GutBrainIE 2026

This repo contains the codebase developed by team TWIX for the GutBrainIE challenge at CLEF 2026. 

The repository supports both component-wise experiments and composed pipelines for:

- term extraction (span detection);
- term classification (semantic typing of candidate spans);
- two-stage term extraction + classification;
- joint entity recognition;
- entity linking to ontology URIs;
- relation extraction; and
- entity- and relation-level ensembling.

The same GBIE JSON representation is used throughout the pipeline, so predictions from one stage can be passed to the next. Hugging Face (HF) encoders, prompted/fine-tuned LLMs, and GLiNER models are supported where applicable. Experiment outputs are written under `runs/`, model checkpoints under `checkpoints/`, and timestamped launcher logs under `scripts/logs/`.

**IMPORTANT!** Run commands from the repository root. In this way relative paths should work as intended. If you encounter runtime problems related to paths, try to include in the problematic bash script a first line `$ cd <path_to_your_project_root>`

## Reference

If you use this code in your work, please kindly cite the following paper:

```bibtex
@inproceedings{martinelli_etal-2026,
    title={{TWIX: a Two-Stage Approach for End-To-End Named Entity Recognition and Relation Extraction}},
    author={Martinelli, Marco and Menotti, Laura},
    booktitle = {Working Notes of CLEF 2026 -- Conference and Labs of the Evaluation Forum, CEUR Workshop Proceedings},
    editor = {Sanchez Salido, Eva and Barrón-Cedeño, Alberto and Seco de Herrera, Alba García and MacAvaney, Sean and Struß, Julia Maria},
    year={2026},
    organization={CEUR-WS},
}
```

## 1. Environment setup

The launchers are Bash scripts. Use Linux, macOS, WSL, or Git Bash.

```bash
git clone <REPOSITORY_URL>
cd GBIE26_TWIX

conda env create -f condaEnv.yml
conda activate twix
```

`condaEnv.yml` is the full environment and includes the spaCy English model used by the entity linking pipeline. `condaEnv_no-encore.yml` is the same environment without `en-core-web-sm`.

Two optional backends imported by the code are not listed in the captured environment file. Install the ones required by your experiments:

```bash
pip install gliner
pip install entity-linkings
```

The second package is the [NAIST-NLP entity-linkings library](https://github.com/naist-nlp/entity-linkings). If its pipeline reports a missing spaCy model, run:

```bash
python -m spacy download en_core_web_sm
```

Verify the runners without starting an experiment:

```bash
python run_term_extractor.py --help
python run_term_classifier.py --help
python run_entity_recognizer.py --help
python run_entity_linker.py --help
python run_relation_extractor.py --help
```

### Run with yaml configs

Every main launcher accepts YAML overrides without changing the checked-in file:

```bash
bash scripts/run_term_extractor.sh \
  --config scripts/configs/termExtractor/hf_train.yaml \
  --override hf.num_epochs=10 hf.batch_size=16 hf.seed=42
```

Values are parsed as YAML, so numbers, booleans, lists, and `null` retain their types.

### Data and output layout
Within `data` folder, place the folders `Annotations` and `Articles` from the challenge data. Place the [NEL data provided by organizers](https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline/tree/main/Train/NEL/definitions) within folder `data/Concepts`. Within `data/Concepts`, also place the `uris.csv` file provided in the challenge data.

The default HF training configs use platinum + gold + silver for term extraction and entity recognition, platinum + gold + silver for term classification, and platinum + gold for relation extraction. 

## 2. Common experiment interface

Each task has:

1. a Python dispatcher in the repository root;
2. a Bash launcher in `scripts/run_*.sh` that captures logs; and
3. task-specific YAML files in `scripts/configs/<task>/`.

Use the launcher for normal runs:

```bash
bash scripts/run_<task>.sh --config path/to/config.yaml \
  --override section.key=value
```

The `batch_*.sh` scripts reproduce model grids by training/inferencing each active model in their hard-coded arrays. Comment or uncomment array entries to match the exact experimental grid you want to reproduce.

## 3. Term extraction

Term extraction predicts mention spans without assigning the final biomedical entity type.

### HF training and inference

Configs:

- `scripts/configs/termExtractor/hf_train.yaml`
- `scripts/configs/termExtractor/hf_inference.yaml`

```bash
# Train one model
bash scripts/run_term_extractor.sh \
  --config scripts/configs/termExtractor/hf_train.yaml

# Run its checkpoint on the configured inference set
bash scripts/run_term_extractor.sh \
  --config scripts/configs/termExtractor/hf_inference.yaml
```

To train and infer the complete active HF model grid:

```bash
bash scripts/batch_term_extractor_hf.sh
```

The batch script varies `hf.model_name`, saves checkpoints to `checkpoints/termExtractor/<model>`, and predictions to `runs/termExtractor/<model>_inference_predictions.json`.

### LLM fine-tuning data and inference

LLMs are inference-only inside this repository. “Training” consists of generating chat-formatted JSONL, submitting it to the selected external fine-tuning service/framework, and then placing the resulting model identifier in the inference config.

Generate term-extraction fine-tuning data:

```bash
python -m src.utils.LLMFinetuningDataGenerator \
  --task term_extraction \
  --train_data_path data/Annotations/Train/merged_quality/json_format/train_merged.json \
  --train_output_path data/finetune/te_train.jsonl \
  --dev_data_path data/Annotations/Dev/json_format/dev.json \
  --dev_output_path data/finetune/te_dev.jsonl
```

This is the portable equivalent of `scripts/finetuning/term_extractor_ft.sh`.

Run one prompted or externally fine-tuned model:

```bash
bash scripts/run_term_extractor.sh \
  --config scripts/configs/termExtractor/llm_inference.yaml \
  --override llm.provider=lmstudio llm.model=medgemma-4b-it-mlx
```

Run the LLM grid defined in the repository:

```bash
bash scripts/batch_term_extractor_llm.sh
```

Supported provider names are `openai`, `azure`, `groq`, `together`, `lmstudio`, `ollama`, and `huggingface`. LM Studio defaults to `http://localhost:1234/v1`; Ollama defaults to `http://localhost:11434/v1`. Start the corresponding local server and load the configured model before running. For hosted providers, set credentials through the provider's normal environment variables or the YAML fields `api_key`, `base_url`, `azure_endpoint`, and `azure_api_version`.

Prompts are selected from `src/termExtractor/prompts.json` with `llm.system_prompt_key` and `llm.user_prompt_key`. LLM checkpoint JSON files allow interrupted inference to resume.

## 4. Term classification

Term classification assigns an entity type (or `NA`) to candidate spans.

### HF training and inference

Configs:

- `scripts/configs/termClassifier/hf_train.yaml`
- `scripts/configs/termClassifier/hf_inference.yaml`

```bash
bash scripts/run_term_classifier.sh \
  --config scripts/configs/termClassifier/hf_train.yaml

bash scripts/run_term_classifier.sh \
  --config scripts/configs/termClassifier/hf_inference.yaml
```

Run the active HF model grid:

```bash
bash scripts/batch_term_classifier_hf.sh
```

The classifier configs expose `negative_sample_multiplier`, `dev_negative_sample_multiplier`, and `max_negative_span_words` in addition to the common HF hyperparameters.

### LLM fine-tuning data and inference

```bash
python -m src.utils.LLMFinetuningDataGenerator \
  --task term_classification \
  --train_data_path data/Annotations/Train/merged_quality/json_format/train_merged.json \
  --train_output_path data/finetune/tc_train.jsonl \
  --dev_data_path data/Annotations/Dev/json_format/dev.json \
  --dev_output_path data/finetune/tc_dev.jsonl
```

This is the portable equivalent of `scripts/finetuning/term_classifier_ft.sh`.

```bash
# One LLM
bash scripts/run_term_classifier.sh \
  --config scripts/configs/termClassifier/llm_inference.yaml

# All configured LLMs
bash scripts/batch_term_classifier_llm.sh
```

Prompts live in `src/termClassifier/prompts.json`.

## 5. Merge term extractor + term classifier predictions

The merger treats extractor spans as the source of truth and looks up the classifier label by exact `(paper_id, start_idx, end_idx, location)`. The result follows the joint entity-recognizer schema and can therefore be passed to NER ensembling, entity linking, or evaluation.

Merge one pair:

```bash
mkdir -p runs/termMerged

python -m src.utils.PredictionMerger \
  --term-extractor-path runs/termExtractor/<EXTRACTOR_PREDICTIONS>.json \
  --term-classifier-path runs/termClassifier/<CLASSIFIER_PREDICTIONS>.json \
  --output-path runs/termMerged \
  --missing-mode ignore
```

`--missing-mode strict` raises if an extracted span has no classifier prediction; `ignore` drops it. Add `--include-na-entities` to retain `NA` labels. Output names record both source models and merge settings.

Create the Cartesian product used for merged-term experiments:

```bash
mkdir -p runs/termMerged
for extractor in runs/termExtractor/*.json; do
  for classifier in runs/termClassifier/*.json; do
    python -m src.utils.PredictionMerger \
      --term-extractor-path "$extractor" \
      --term-classifier-path "$classifier" \
      --output-path runs/termMerged \
      --missing-mode ignore
  done
done
```

## 6. Entity recognition

Joint entity recognition predicts both spans and biomedical entity labels.

### HF

```bash
bash scripts/run_entity_recognizer.sh \
  --config scripts/configs/entityRecognizer/hf_train.yaml

bash scripts/run_entity_recognizer.sh \
  --config scripts/configs/entityRecognizer/hf_inference.yaml

# Full active HF grid
bash scripts/batch_entity_recognizer_hf.sh
```

### LLM

Generate external fine-tuning data:

```bash
python -m src.utils.LLMFinetuningDataGenerator \
  --task entity_recognition \
  --train_data_path data/Annotations/Train/merged_quality/json_format/train_merged.json \
  --train_output_path data/finetune/ner_train.jsonl \
  --dev_data_path data/Annotations/Dev/json_format/dev.json \
  --dev_output_path data/finetune/ner_dev.jsonl
```

This corresponds to `scripts/finetuning/entity_extractor_ft.sh`.

```bash
bash scripts/run_entity_recognizer.sh \
  --config scripts/configs/entityRecognizer/llm_inference.yaml

bash scripts/batch_entity_recognizer_llm.sh
```

Prompts live in `src/entityRecognizer/prompts.json`.

### GLiNER

```bash
bash scripts/run_entity_recognizer.sh \
  --config scripts/configs/entityRecognizer/gliner_train.yaml

bash scripts/run_entity_recognizer.sh \
  --config scripts/configs/entityRecognizer/gliner_inference.yaml

# NuNerZero model grid
bash scripts/batch_entity_recognizer_gliner.sh
```

Training settings such as `num_steps`, `eval_every`, `max_len`, encoder/other learning rates, and negative type sampling are defined in `gliner_train.yaml`; inference threshold is defined in `gliner_inference.yaml`.

## 7. Entity linking

Entity linking expects GBIE documents whose `entities` contain mention spans. Point `linker.inference_data_path` at gold entities, a joint NER prediction, a merged-term prediction, or an NER ensemble. The output adds predicted URI information to each entity.

The concept dictionary used by neural retrieval is `data/Concepts/uri_retrieved_definitions_el_format.jsonl`.

### Baseline: exact match + semantic fallback

Build the mention knowledge base and embedding index, then infer:

```bash
python run_baseline_entity_linker.py \
  --config scripts/configs/entityLinker/baseline_build_inference.yaml
```

For end-to-end inference, override `linker.inference_data_path` with the chosen NER or merged-term output.

Reuse the saved knowledge base/index with a different similarity threshold:

```bash
python run_baseline_entity_linker.py \
  --config scripts/configs/entityLinker/baseline_reuse_kb.yaml \
  --override linker.similarity_threshold=0.35
```

The baseline uses `(text_span, label)` exact matching first when `use_label_for_exact_match: true`, then semantic similarity over URI definitions via `txtai`.

### Train and run a retriever

Train one text-embedding retriever:

```bash
bash scripts/run_entity_linker.sh \
  --config scripts/configs/entityLinker/train_retriever.yaml
```

Train the retriever model grid:

```bash
bash scripts/batch_train_retriever.sh
```

Run retriever-only inference by setting the reranker to `null` and pointing all paths at the produced run directory:

```bash
bash scripts/run_entity_linker.sh \
  --config scripts/configs/entityLinker/inference.yaml \
  --override \
    linker.retriever_id=textembedding \
    linker.reranker_id=null \
    linker.retriever_model_name_or_path=runs/entityLinker/<RETRIEVER_RUN> \
    linker.reranker_model_name_or_path=null \
    linker.retriever_index_dir=runs/entityLinker/<RETRIEVER_RUN> \
    linker.inference_data_path=runs/entityRecognizer/<NER_PREDICTIONS>.json
```

The runner resolves the `retriever` and `retriever_index` component subdirectories automatically. Supported retrievers are `bm25`, `prior`, `textembedding`, `e5bm25`, and `dualencoder`. `scripts/configs/entityLinker/train_dualencoder.yaml` and `infer_dualencoder_crossencoder.yaml` provide the dedicated dual-encoder examples.

For PRIOR retrieval, first build its mention counter and normalized dictionary:

```bash
python scripts/build_prior_mention_counter.py \
  --train-data-paths \
    data/Annotations/Train/platinum_quality/json_format/train_platinum.json \
    data/Annotations/Train/gold_quality/json_format/train_gold.json \
  --entity-dict-path data/Concepts/uri_retrieved_definitions_el_format.jsonl \
  --output-path runs/entityLinker/prior/mention_counter.json \
  --prior-dictionary-output runs/entityLinker/prior/entity_dictionary_prior.jsonl
```

Then use the counter as `linker.retriever_model_name_or_path` and the normalized dictionary as `linker.entity_dict_path`.

### Train and run retriever + reranker

Train a reranker on candidates from a trained retriever:

```bash
bash scripts/run_entity_linker.sh \
  --config scripts/configs/entityLinker/train_reranker.yaml

# Full configured retriever/reranker grid
bash scripts/batch_train_reranker.sh
```

Run inference with both stages:

```bash
bash scripts/run_entity_linker.sh \
  --config scripts/configs/entityLinker/inference.yaml \
  --override \
    linker.retriever_model_name_or_path=runs/entityLinker/<RETRIEVER_RUN> \
    linker.reranker_model_name_or_path=runs/entityLinker/<RERANKER_RUN> \
    linker.retriever_index_dir=runs/entityLinker/<RETRIEVER_RUN> \
    linker.inference_data_path=runs/entityRecognizer/<NER_PREDICTIONS>.json
```

Run the complete inference grid encoded in the repository:

```bash
bash scripts/batch_entity_linker.sh
```

Supported rerankers are `crossencoder`, `chatel`, `fevry`, `extend`, and `fusioned`. Component-level defaults for all retriever/reranker families are retained under `scripts/configs/entityLinker/original/`.

## 8. Relation extraction
The relation extractor module implemented in this codebase frames the task as a sequence classification problem leveraging the transformers library.

However, our submitted relation extraction runs use ATLOP, following the approach introduced in the organizers’ [baseline system](https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline). We extend this approach with a two-stage training strategy consisting of an initial training on lower quality annotations followed by a fine-tuning on higher quality annotations. Further details are available in our Working Notes paper *TWIX: A Two-Stage Approach for End-to-End Named Entity Recognition and Relation Extraction*.

To reproduce these experiments, first follow the setup instructions in the [baseline repository](https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline). Then run the scripts under `scripts/configs/atlopRelationExtraction/` in the following order:

1. `pretrain_sb10.sh`
2. `finetune_g100_psb10.sh`
3. `eval_test_finetune_g100_psb10.sh`

Because the organizers’ conversion script presented several issues when transforming annotations and predictions into the ATLOP input format, we provide a reimplementation in `NER_predictions_to_atlop_format.py`.

## 9. Ensembling

NER and RE ensembling support `majority`, `weighted`, `intersection`, and `union`. For weighted voting, set `weights` in the same order as `prediction_paths`. `vote_threshold` can override the default inclusion threshold.

Each ensemble writes both predictions and an exact configuration sidecar, making the run replayable.

### NER ensembling

Edit `prediction_paths` in `scripts/configs/entityRecognizer/ensemble.yaml`, then run:

```bash
python run_ner_ensemble.py \
  --config scripts/configs/entityRecognizer/ensemble.yaml
```

The paper-selected four-model example is preserved as:

```bash
python run_ner_ensemble.py \
  --config scripts/configs/entityRecognizer/ensemble_combo0194.yaml
```

Outputs default to `runs/entityRecognizer/ensembling/`. A saved JSON sidecar in that folder can be passed back to `--config` to replay the exact NER ensemble.

### Relation ensembling

Edit `prediction_paths` in `scripts/configs/relationExtractor/ensemble.yaml`, then run:

```bash
python run_re_ensemble.py \
  --config scripts/configs/relationExtractor/ensemble.yaml
```

Outputs default to `runs/relationExtractor/ensembling/`.

### Merged-term ensembling

Merged term predictions already have NER-compatible `entities`, so use the NER ensemble runner on selected files from `runs/termMerged`:

```bash
python run_ner_ensemble.py \
  --config scripts/configs/entityRecognizer/ensemble.yaml \
  --override \
    prediction_paths='[runs/termMerged/<MERGE_1>.json, runs/termMerged/<MERGE_2>.json, runs/termMerged/<MERGE_3>.json]' \
    output_dir=runs/termMerged/ensembling \
    output_id=term_merged_majority \
    method=majority
```

Both ensemble runners also support folder mode. Add `prediction_folder`, optional `prediction_glob`, `methods`, `min_combination_size`, and `max_combination_size` to a copied config to enumerate all file combinations. Be careful: this grows combinatorially.

## 10. Evaluation

Most task runners evaluate automatically when their config provides `eval_data_path`. The challenge-style bulk evaluator is `eval/evaluate.py`:

```bash
python eval/evaluate.py \
  --TE_folder runs/termExtractor \
  --NER_folder runs/entityRecognizer \
  --NERD_folder runs/entityLinker \
  --M_RE_folder runs/relationExtractor \
  --C_RE_folder runs/relationExtractor
```

Pass only the folders for subtasks you want to score. Before using this script, set `GROUND_TRUTH_PATH` and `GROUND_TRUTH_PATH_TE` near its top to the desired dev or test JSON files; the checked-in values are author-specific development-set paths. CSV summaries are written under `eval/`.

## 11. Suggested end-to-end reproduction order

After fixing config paths and selecting the model arrays that correspond to the paper table:

```bash
# A. Component predictions
bash scripts/batch_term_extractor_hf.sh
bash scripts/batch_term_classifier_hf.sh
bash scripts/batch_entity_recognizer_hf.sh

# B. Merge TE + TC with the all-pairs loop from Section 5

# C. Entity and merged-term ensembles
python run_ner_ensemble.py --config scripts/configs/entityRecognizer/ensemble.yaml

# D. Entity linking
bash scripts/batch_train_retriever.sh
bash scripts/batch_train_reranker.sh
bash scripts/batch_entity_linker.sh

# E. Relation extraction and ensembling
Refer to section 9
```
