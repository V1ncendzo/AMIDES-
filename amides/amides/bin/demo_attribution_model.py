#!/usr/bin/env python3
"""
Demo script to interactively attribute commands to Sigma rules using a trained AMIDES attribution model.
"""

import argparse
import sys
import os
import numpy as np

# Ensure amides package is in path
# Assumes script is in amides/amides/bin/ relative to project root amides/
base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(base_dir)

from amides.persist import Dumper
from amides.features.normalize import normalize
from amides.utils import get_logger, set_log_level

set_log_level("error") # Suppress noisy logs

def select_model():
    # Look for models in predefined locations
    model_dirs = ["./models_linux", "./models", ".", "amides/models_linux", "amides/amides/models_linux"]
    potential_models = []
    
    for d in model_dirs:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".zip") and "multi_train_rslt" in f]
            potential_models.extend(files)
            
    if not potential_models:
        print("No attribution models ('multi_train_rslt*.zip') found in ./models_linux or current directory.")
        sys.exit(1)
        
    print("\nAvailable Attribution Models:")
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

def load_model(model_path):
    print(f"Loading model bundle from {model_path}...")
    try:
        dumper = Dumper(os.path.dirname(model_path))
        result = dumper.load_object(model_path)
        return result
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def unpack_ensemble(multi_result):
    """
    Unpacks the MultiTrainingResult into a usable dictionary of classifiers.
    """
    ensemble = {}
    print(f"Unpacking {len(multi_result.results)} classifiers...")
    for rule_name, train_result in multi_result.results.items():
        if not train_result.feature_extractors:
            continue
        ensemble[rule_name] = {
            "estimator": train_result.estimator,
            "vectorizer": train_result.feature_extractors[0],
            "scaler": train_result.scaler
        }
    return ensemble

# ANSI Definitions
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def attribute(command, ensemble, top_k=5):
    normalized = normalize([command])[0]
    scores = []
    
    for rule_name, components in ensemble.items():
        estimator = components["estimator"]
        vectorizer = components["vectorizer"]
        
        vector = vectorizer.transform([normalized])
        try:
            raw_score = estimator.decision_function(vector)[0]
            # Convert decision function score to a pseudo-probability (percentage)
            # SVM scores are distances, usually in range [-1, 1] or larger.
            # We scale it slightly to make the percentage transition smoother around 0.
            prob = sigmoid(raw_score) * 100 
            scores.append((rule_name, prob, raw_score))
        except Exception:
            pass
            
    # Sort by probability (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k], normalized

def draw_bar(percentage, width=20):
    # Create a visual progress bar
    filled = int(width * percentage / 100)
    bar = "█" * filled + "░" * (width - filled)
    return bar

def main():
    parser = argparse.ArgumentParser(description="AMIDES Rule Attribution Demo")
    parser.add_argument("--model-path", help="Path to the trained multi-model .zip file")
    parser.add_argument("--input", help="Single command to classify (optional)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top matching rules to show")
    args = parser.parse_args()

    model_path = args.model_path
    if not model_path:
        model_path = select_model()

    multi_result = load_model(model_path)
    ensemble = unpack_ensemble(multi_result)

    print(f"\n{Colors.GREEN}{Colors.BOLD}Model Ensemble Loaded Successfully!{Colors.RESET}")
    print(f"{Colors.BLUE}-----------------------------------{Colors.RESET}")

    if args.input:
        top_hits, normalized = attribute(args.input, ensemble, args.top_k)
        print(f"\n{Colors.BOLD}Input:{Colors.RESET}      {args.input}")
        print(f"{Colors.BOLD}Normalized:{Colors.RESET} {normalized}\n")
        print(f"{Colors.HEADER}=== Top {args.top_k} Likely Rules ==={Colors.RESET}")
        for rank, (rule, prob, raw) in enumerate(top_hits, 1):
            color = Colors.RED if prob > 50 else Colors.YELLOW
            bar = draw_bar(prob)
            print(f"{rank}. {rule:<40} {color}{bar} {prob:5.1f}%{Colors.RESET} (Raw: {raw:+.2f})")
    else:
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Interactive Rule Attribution Demo ==={Colors.RESET}")
        print("Type a shell command to see which Sigma rules it triggers.")
        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                command = input(f"{Colors.CYAN}{Colors.BOLD}Enter command > {Colors.RESET}")
                if command.strip().lower() in ["exit", "quit"]:
                    break
                if not command.strip():
                    continue
                
                top_hits, normalized = attribute(command, ensemble, args.top_k)
                
                print(f"{Colors.BLUE}Normalized:{Colors.RESET} {normalized}\n")
                
                # Check if top hit is strong
                top_prob = top_hits[0][1]
                if top_prob > 50:
                    print(f"{Colors.RED}{Colors.BOLD}🚨 ALERT: SUSPICIOUS ACTIVITY DETECTED 🚨{Colors.RESET}")
                    print(f"Most Likely Rule: {Colors.RED}{Colors.BOLD}{top_hits[0][0]}{Colors.RESET}\n")
                
                print(f"{Colors.UNDERLINE}Top {args.top_k} Attribution Matches:{Colors.RESET}")
                for rank, (rule, prob, raw) in enumerate(top_hits, 1):
                    # Color coding based on confidence
                    if prob >= 80: color = Colors.RED + Colors.BOLD
                    elif prob >= 50: color = Colors.RED
                    elif prob >= 20: color = Colors.YELLOW
                    else: color = Colors.GREEN
                    
                    bar = draw_bar(prob)
                    print(f"  {rank}. {rule:<45} {color}{bar} {prob:6.2f}%{Colors.RESET}")
                print("\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.RESET}\n")
    print(f"\n{Colors.GREEN}Goodbye!{Colors.RESET}")

if __name__ == "__main__":
    main()
