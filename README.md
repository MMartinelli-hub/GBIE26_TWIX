# TWIX at GutBrainIE 2026

> **Research use only. Not a medical device.**
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

## 1. Overview

TWIX is the research codebase developed by team TWIX for the [GutBrainIE challenge at CLEF 2026](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/). It supports component experiments and composed pipelines for term extraction, term classification, two-stage entity recognition, joint entity recognition, entity linking, relation extraction, and ensembling.

The full collection used for training TWIX can be found at: [https://zenodo.org/records/23492330](https://zenodo.org/records/23492330)

If you use this system fully or in part, please cite the following paper:

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

## 2. Intended use

**Primary intended use.** Develop, reproduce, and compare biomedical information extraction approaches on the GutBrainIE benchmark. The system recognizes entity mentions, assigns biomedical categories, links mentions to reference concepts, and extracts mention-level and concept-level relations from scientific literature.

**Intended users.** Biomedical NLP researchers, data scientists, software developers, and clinicians conducting research on literature analysis.

**Out-of-scope uses.** This release must not be used for:

- clinical diagnosis, prognosis, treatment, monitoring, or other decisions about individual patients;
- decisions about individuals, such as insurance, employment, or access to care;
- re-identification of annotators or other individuals, including linkage of pseudonymized annotation identifiers to identities;
- processing identifiable patient records or other personal records as part of the documented benchmark workflow.

Research on languages, domains, full-text documents, or input types outside the benchmark requires separate evaluation. Extracted entities, concept links, and relations must be verified against their source publications and must not be interpreted as clinical recommendations or evidence of causation.

## 3. Data provenance

TWIX uses the GutBrainIE 2026 collection of publicly available PubMed titles and abstracts and associated annotations. The benchmark collection does not include full-text articles or restricted-access documents. 

## 4. Method

- **Algorithm or architecture.** TWIX supports separate span extraction and term classification, joint entity recognition, GLiNER-based recognition, and prompted or externally fine-tuned LLM alternatives. Entity linking supports exact matching with a semantic fallback and retrieval with optional reranking. The repository’s RE runner uses sequence classification, while the submitted ATLOP experiments use initial training on lower-quality annotations followed by fine-tuning on higher-quality annotations. Entity and relation ensembles combine component predictions.
- **Training setting.** Centralized model training on supplied files. HF encoder and GLiNER experiments run locally. LLM inference can use local endpoints or hosted providers; LLM fine-tuning is performed separately with the selected service or framework. Hosted requests and fine-tuning uploads transfer data outside the local environment.

The GBIE JSON representation connects pipeline stages. Experiment predictions are written under `runs/`, model checkpoints under `checkpoints/`, and timestamped launcher logs under `scripts/logs/`. Generated prompts, intermediate outputs, logs, and checkpoints may retain input text or derived information.

Run the commands below from the repository root. For paths interpreted relative to the working directory, use explicit paths when running from another location.

### Environment setup

The launchers are Bash scripts. Use Linux, macOS, WSL, or Git Bash.

```bash
git clone https://github.com/MMartinelli-hub/GBIE26_TWIX.git
cd GBIE26_TWIX

conda env create -f _env/condaEnv.yml
conda activate twix
```

`_env/condaEnv.yml` is the full environment and includes the spaCy English model used by the entity linking pipeline. `_env/condaEnv_no-encore.yml` is the same environment without `en-core-web-sm`.

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

#### Run with yaml configs

Every main launcher accepts YAML overrides without changing the checked-in file:

```bash
bash scripts/run_term_extractor.sh \
  --config scripts/configs/termExtractor/hf_train.yaml \
  --override hf.num_epochs=10 hf.batch_size=16 hf.seed=42
```

Values are parsed as YAML, so numbers, booleans, lists, and `null` retain their types.

#### Data and output layout
Within `data` folder, place the folders `Annotations` and `Articles` from the challenge data. Place the [NEL data provided by organizers](https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline/tree/main/Train/NEL/definitions) within folder `data/Concepts`. Within `data/Concepts`, also place the `uris.csv` file provided in the challenge data.

The checked-in HF training configs refer to Platinum, Gold, and Silver paths for term extraction, term classification, and entity recognition, and Platinum and Gold paths for the sequence-classification RE runner. Verify these paths against the exact dataset release used in your experiment; map legacy folder names explicitly rather than assuming they match the current 2026 distribution. The submitted ATLOP experiments use a separate training sequence described under “Relation extraction.” 

### Common experiment interface

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

### Term extraction

Term extraction predicts mention spans without assigning the final biomedical entity type.

#### HF training and inference

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

#### LLM fine-tuning data and inference

This repository generates chat-formatted JSONL for LLM fine-tuning and runs inference with prompted or fine-tuned models. LLM optimization itself is performed separately using the selected service or framework. For hosted fine-tuning, uploading the generated JSONL transfers its contents to the provider; a local fine-tuning framework may be used separately where supported. Place the resulting model identifier in the inference config.

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

Supported provider names are `openai`, `azure`, `groq`, `together`, `lmstudio`, `ollama`, and `huggingface`. LM Studio defaults to `http://localhost:1234/v1`; Ollama defaults to `http://localhost:11434/v1`. Start the corresponding local server and load the configured model before running. For hosted providers, configure credentials through the provider’s supported environment variables or a private configuration file. The available fields include `api_key`, `base_url`, `azure_endpoint`, and `azure_api_version`; do not commit credentials in shared YAML files. Hosted execution transmits prompts and their included source text to the selected endpoint. Processing location and retention depend on the provider and deployment. LM Studio and Ollama remain local only when configured to use local endpoints.

Prompts are selected from `src/termExtractor/prompts.json` with `llm.system_prompt_key` and `llm.user_prompt_key`. LLM checkpoint JSON files allow interrupted inference to resume.

### Term classification

Term classification assigns an entity type (or `NA`) to candidate spans.

#### HF training and inference

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

#### LLM fine-tuning data and inference

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

### Merge term extractor + term classifier predictions

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

### Entity recognition

Joint entity recognition predicts both spans and biomedical entity labels.

#### HF

```bash
bash scripts/run_entity_recognizer.sh \
  --config scripts/configs/entityRecognizer/hf_train.yaml

bash scripts/run_entity_recognizer.sh \
  --config scripts/configs/entityRecognizer/hf_inference.yaml

# Full active HF grid
bash scripts/batch_entity_recognizer_hf.sh
```

#### LLM

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

#### GLiNER

```bash
bash scripts/run_entity_recognizer.sh \
  --config scripts/configs/entityRecognizer/gliner_train.yaml

bash scripts/run_entity_recognizer.sh \
  --config scripts/configs/entityRecognizer/gliner_inference.yaml

# NuNerZero model grid
bash scripts/batch_entity_recognizer_gliner.sh
```

Training settings such as `num_steps`, `eval_every`, `max_len`, encoder/other learning rates, and negative type sampling are defined in `gliner_train.yaml`; inference threshold is defined in `gliner_inference.yaml`.

### Entity linking

Entity linking expects GBIE documents whose `entities` contain mention spans. Point `linker.inference_data_path` at gold entities, a joint NER prediction, a merged-term prediction, or an NER ensemble. The output adds predicted URI information to each entity.

The concept dictionary used by neural retrieval is `data/Concepts/uri_retrieved_definitions_el_format.jsonl`.

#### Baseline: exact match + semantic fallback

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

#### Train and run a retriever

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

#### Train and run retriever + reranker

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

### Relation extraction
The relation extractor module implemented in this codebase frames the task as a sequence classification problem leveraging the transformers library.

However, our submitted relation extraction runs use ATLOP, following the approach introduced in the organizers’ [baseline system](https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline). We extend this approach with a two-stage training strategy consisting of an initial training on lower quality annotations followed by a fine-tuning on higher quality annotations. Further details are available in our Working Notes paper *TWIX: A Two-Stage Approach for End-to-End Named Entity Recognition and Relation Extraction*.

To reproduce these experiments, first follow the setup instructions in the [baseline repository](https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline). Then run the scripts under `scripts/configs/atlopRelationExtraction/` in the following order:

1. `pretrain_sb10.sh`
2. `finetune_g100_psb10.sh`
3. `eval_test_finetune_g100_psb10.sh`

Use `NER_predictions_to_atlop_format.py` for the TWIX conversion of annotations and predictions into ATLOP input format. Keep entity spans and ordering consistent across the NER, linking, and RE stages.

### Ensembling

NER and RE ensembling support `majority`, `weighted`, `intersection`, and `union`. For weighted voting, set `weights` in the same order as `prediction_paths`. `vote_threshold` can override the default inclusion threshold.

Each ensemble writes both predictions and an exact configuration sidecar, making the run replayable.

#### NER ensembling

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

#### Relation ensembling

Edit `prediction_paths` in `scripts/configs/relationExtractor/ensemble.yaml`, then run:

```bash
python run_re_ensemble.py \
  --config scripts/configs/relationExtractor/ensemble.yaml
```

Outputs default to `runs/relationExtractor/ensembling/`.

#### Merged-term ensembling

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

### Suggested end-to-end reproduction order

After selecting the dataset paths, model arrays, and configurations for the experiment:

1. Generate component predictions:

   ```bash
   bash scripts/batch_term_extractor_hf.sh
   bash scripts/batch_term_classifier_hf.sh
   bash scripts/batch_entity_recognizer_hf.sh
   ```

2. Merge term-extractor and term-classifier predictions using the pairwise or all-pairs commands under “Merge term extractor + term classifier predictions.”
3. Configure and run entity ensembling:

   ```bash
   python run_ner_ensemble.py --config scripts/configs/entityRecognizer/ensemble.yaml
   ```

4. Train and run the selected linking components:

   ```bash
   bash scripts/batch_train_retriever.sh
   bash scripts/batch_train_reranker.sh
   bash scripts/batch_entity_linker.sh
   ```

5. Run the ATLOP training and inference sequence under “Relation extraction,” followed by any selected relation ensemble. Prepare the baseline environment separately as required by that workflow.
6. Evaluate the final predictions using the instructions in Section 5.

Batch scripts run the active models in their arrays; inspect those arrays and their configurations before starting a complete grid. For end-to-end evaluation, use predicted upstream mentions rather than gold mentions, and disclose any component-only evaluation using gold inputs.

## 5. Performance

- **Evaluation data.** GutBrainIE development and held-out test collections selected through the experiment configuration. Report the dataset edition, split, training collections, and whether gold or predicted entities are used for linking and relation extraction. These are literature-based benchmark experiments, not clinical validation.
- **Metrics and results.** The evaluator reports precision, recall, and F1 with macro- and micro-averaging for supported tasks. The four official 2026 tasks are NER (6.1.1), NERD (6.1.2), mention-level RE (6.2.1), and concept-level RE (6.2.2). Term extraction is additionally available as a component evaluation. Numerical results must identify the selected run or ensemble; they are not interchangeable across configurations.

[TO COMPLETE: insert the verified numerical TWIX results from the Working Notes paper, identifying the data split, model or ensemble, and configuration. No numerical TWIX results are stated in the supplied README.]

- **Results by subgroup.** No evaluation by patient age, sex, country, or clinical site is documented in the supplied README. Report any available per-entity or per-relation analysis separately from demographic subgroup evaluation.

### Run evaluation

Most task runners evaluate automatically when their configuration supplies `eval_data_path`. The bulk evaluator is `eval/evaluate.py`:

```bash
python eval/evaluate.py \
  --TE_folder runs/termExtractor \
  --NER_folder runs/entityRecognizer \
  --NERD_folder runs/entityLinker \
  --M_RE_folder runs/relationExtractor \
  --C_RE_folder runs/relationExtractor
```

Pass only the folders for the tasks you want to score. Before running from the repository root, set `GROUND_TRUTH_PATH` and `GROUND_TRUTH_PATH_TE` near the top of the evaluator to root-relative paths such as `data/Annotations/Dev/json_format/dev.json` and `data/Annotations/Dev/json_format/dev_term_only.json`, or use absolute paths. Replace these with the corresponding test references for test evaluation. The checked-in `../data/...` paths must be adjusted when running from the root.

The script writes CSV summaries to the current working directory, including `TE_evaluation_results.csv`, `NER_evaluation_results.csv`, `NERD_evaluation_results.csv`, `MENTION_LEVEL_RE_evaluation_results.csv`, and `CONCEPT_LEVEL_RE_evaluation_results.csv` for the evaluated tasks.

Keep the evaluated split out of model fitting and lookup-resource construction. Select models, thresholds, and ensemble combinations using development data; report final test results without selecting configurations on test scores. Record model identifiers, random seeds, configurations, prompts, provider/deployment identifiers, and resource versions when reproducing an experiment.

## 6. Known limitations and risks

- **Generalizability.** The baseline was evaluated on English PubMed titles and abstracts selected for the gut–brain domain. Performance on other languages, domains, full-text articles, clinical notes, or individual patient records has not been established.
- **Known biases and confounders.** The collection reflects its retrieval queries, publication coverage, and annotation schema. Annotation quality varies across the Platinum, Gold, Silver, and Bronze tiers; Bronze annotations are automatically generated and were not manually revised. Uneven entity and relation frequencies may affect performance on less frequent categories.
- **Failure modes.** NER can miss entities, assign incorrect types, or predict inaccurate boundaries. These errors propagate to RE, which can omit relations or assign incorrect predicates and arguments. Extracted associations do not establish causation or clinical efficacy, and outputs require verification against the original text.

## 7. Responsible use and reporting

- **Terms of use.** Use this release for research in accordance with the repository license, the terms of the pretrained and fine-tuned models, and the applicable source-data licenses. The research-only notice describes the intended use and clinical limitations; it does not replace the applicable license terms.
- **Contact for misuse or vulnerability reports.** Email [martinell2@dei.unipd.it](mailto:martinell2@dei.unipd.it). General questions, reproducibility issues, and proposed improvements may also be submitted through [GitHub Issues](https://github.com/MMartinelli-hub/GutBrainIE_2025_Baseline/issues). Do not post sensitive data or credentials in public issues.

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.
