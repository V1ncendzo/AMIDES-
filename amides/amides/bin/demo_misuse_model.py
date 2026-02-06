#!/usr/bin/env python3
"""
Demo script to interactively classify commands using a trained AMIDES misuse model.
"""

import argparse
import sys
import os
import numpy as np

# Ensure amides package is in path
base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(base_dir)

from amides.persist import Dumper
from amides.features.normalize import normalize
from amides.utils import get_logger, set_log_level

set_log_level("error") # Suppress noisy logs
_logger = get_logger("demo")

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        dumper = Dumper(os.path.dirname(model_path))
        result = dumper.load_object(model_path)
        return result
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict(command, estimator, vectorizer, scaler=None):
    # 1. Normalize
    # normalize() expects a list of strings and returns a list of normalized strings
    normalized = normalize([command])[0]
    
    # 2. Vectorize
    # transform() expects an iterable of strings
    vector = vectorizer.transform([normalized])
    
    # 3. Predict
    # decision_function returns the distance to the hyperplane (confidence score)
    score = estimator.decision_function(vector)[0]
    
    # predict() returns the class label (0=Benign, 1=Malicious)
    label = estimator.predict(vector)[0]
    
    return label, score, normalized

def select_model():
    # Look for models in predefined locations
    model_dirs = ["./models_linux", "./models", ".", "amides/models_linux", "amides/amides/models_linux"]
    potential_models = []
    
    for d in model_dirs:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".zip") and "train_rslt" in f]
            potential_models.extend(files)
            
    if not potential_models:
        print("No models found in ./models_linux or current directory.")
        sys.exit(1)
        
    print("\nAvailable Models:")
    for i, m in enumerate(potential_models):
        print(f"[{i+1}] {m}")
        
    while True:
        try:
            choice = input("\nSelect a model (number): ")
            idx = int(choice) - 1
            if 0 <= idx < len(potential_models):
                return potential_models[idx]
            print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="AMIDES Misuse Model Demo")
    parser.add_argument("--model-path", help="Path to the trained model .zip file")
    parser.add_argument("--input", help="Single command to classify (optional)")
    args = parser.parse_args()

    model_path = args.model_path
    if not model_path:
        model_path = select_model()

    result = load_model(model_path)
    
    estimator = result.estimator
    # vectorizer is stored in feature_extractors list
    if not result.feature_extractors:
        print("Error: Model does not contain a feature extractor.")
        sys.exit(1)
    vectorizer = result.feature_extractors[0]
    scaler = result.scaler

    print("\nModel Loaded Successfully!")
    print("-------------------------")

    if args.input:
        # Single shot mode
        label, score, normalized = predict(args.input, estimator, vectorizer, scaler)
        print(f"Input:      {args.input}")
        print(f"Normalized: {normalized}")
        print(f"Prediction: {'MALICIOUS' if label == 1 else 'BENIGN'}")
        print(f"Score:      {score:.4f}")
    else:
        # Interactive mode
        print("\n=== Interactive Misuse Detection Model Demo ===")
        print("Type a shell command to see if it's classified as BENIGN or MALICIOUS.")
        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                command = input("Enter command > ")
                if command.strip().lower() in ["exit", "quit"]:
                    break
                if not command.strip():
                    continue
                
                label, score, normalized = predict(command, estimator, vectorizer, scaler)
                
                result_str = "MALICIOUS" if label == 1 else "BENIGN"
                # Add color if supported (simple ANSI codes)
                if label == 1:
                    result_str = f"\033[91m{result_str}\033[0m" # Red
                else:
                    result_str = f"\033[92m{result_str}\033[0m" # Green
                
                print(f"Result:     {result_str} (Score: {score:.4f})")
                print(f"Normalized: {normalized}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}\n")
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
