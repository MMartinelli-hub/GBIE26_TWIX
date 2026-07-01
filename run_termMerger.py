import argparse
import os

from src.utils.PredictionMerger import TermPredictionMerger

def main():
    term_extractor_path = "runs/termExtractor/dev"
    term_classifier_path = "runs/termClassifier/dev"
    output_path = "runs/termMerged/dev"
    for extractor_file in os.listdir(term_extractor_path):
        if extractor_file.endswith(".json"):
            for classifier_file in os.listdir(term_classifier_path):
                if classifier_file.endswith(".json"):
                    extractor_path = os.path.join(term_extractor_path, extractor_file)
                    classifier_path = os.path.join(term_classifier_path, classifier_file)
                    merger = TermPredictionMerger(
                        term_extractor_path=extractor_path,
                        term_classifier_path=classifier_path,
                        output_path=output_path,
                        missing_mode="ignore",
                        include_na_entities=False,
                        entity_key="entities",
                    )
                    args = argparse.Namespace(
                        term_extractor_path=extractor_path,
                        term_classifier_path=classifier_path,
                        output_path=output_path,
                        missing_mode="ignore",
                        include_na_entities=False,
                        entity_key="entities",
                    )
                    merger.merge_predictions(args=args) 
    print("Merging completed.")

if __name__ == "__main__":
    main()
