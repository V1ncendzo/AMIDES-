
import sys
import os
import argparse

# Add parent dir to path
sys.path.append(os.getcwd() + "/amides")

from amides.persist import Dumper
from amides.data import TrainingResult

def inspect(path):
    dumper = Dumper()
    try:
        print(f"Loading {path}...")
        result = dumper.load_object(path)
        print(f"Type: {type(result)}")
        if isinstance(result, TrainingResult):
            print(f"Name: {result.name}")
            print(f"Timestamp: {result.timestamp}")
            print(f"Estimator: {result.estimator}")
            print(f"Feature Extractors: {result.feature_extractors}")
            if result.feature_extractors:
                print(f"Extractor count: {len(result.feature_extractors)}")
                print(f"Extractor 0 type: {type(result.feature_extractors[0])}")
            else:
                print("Feature Extractors is empty/None!")
        else:
            print("Not a TrainingResult object.")
            
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Usage: python3 inspect_misuse_model.py <path_to_zip>")
