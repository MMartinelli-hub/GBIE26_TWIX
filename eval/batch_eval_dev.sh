#!/bin/bash
set -e

python evaluate.py --NER_folder ../runs/termMerged/dev 
python evaluate.py --NER_folder ../runs/entityRecognizer/dev 
python evaluate.py --NERD_folder ../runs/entityLinker/dev
python evaluate.py --M_RE_folder ../runs/relationExtractor/dev
python evaluate.py --M_RE_folder ../runs/relationExtractor/ensembling_dev
python evaluate.py --C_RE_folder ../runs/relationExtractor/dev
python evaluate.py --C_RE_folder ../runs/relationExtractor/ensembling_dev

mkdir -p results/dev
mv *.csv results/dev/