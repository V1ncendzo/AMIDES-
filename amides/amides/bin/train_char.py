#!/usr/bin/env python3
"""
AMIDES Character-Level Training Script
---------------------------------------
This script trains a misuse classification model using CHARACTER-LEVEL TF-IDF features
instead of word-level features. This makes the model more robust to obfuscation techniques
like string concatenation, random casing, and inserted noise characters.

Key Difference from train.py:
- TfidfVectorizer(analyzer='char', ngram_range=(3, 6)) instead of analyzer='word'

Usage:
    python3 train_char.py \
        --benign-samples "path/to/benign.txt" \
        --events-dir "path/to/sigma/events" \
        --rules-dir "path/to/sigma/rules" \
        --malicious-samples-type matches \
        --result-name "misuse_model_char" \
        --out-dir "path/to/output"
"""

import os
import sys
import argparse
import numpy as np
print("DEBUG: Script started", file=sys.stderr)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer

# Add parent dir to path for imports
base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(base_dir, "amides"))

from amides.persist import Dumper
from amides.sigma import RuleSetDataset
from amides.data import DataBunch, TrainingResult, TrainTestValidSplit
from amides.features.extraction import CommandlineExtractor
from amides.features.normalize import normalize
from amides.scale import create_symmetric_mcc_min_max_scaler
from amides.utils import get_logger, set_log_level, execution_time

set_log_level("info")
_logger = get_logger("train_char")

# Default paths (Linux)
DEFAULT_BENIGN = os.path.join(base_dir, "data/socbed/linux/process_creation/train")
DEFAULT_EVENTS = os.path.join(base_dir, "data/sigma/events/linux/process_creation")
DEFAULT_RULES = os.path.join(base_dir, "data/sigma/rules/linux/process_creation")
DEFAULT_OUTPUT = os.path.join(base_dir, "models/linux/process_creation")


def load_benign_samples(path):
    """Load benign samples from a text file (one per line)."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(line.strip())
    return samples


def load_malicious_samples(events_dir, rules_dir, sample_type="matches"):
    """Load malicious samples from Sigma events."""
    rule_set = RuleSetDataset()
    rule_set.load_rule_set_data(events_path=events_dir, rules_path=rules_dir)
    
    if sample_type == "matches":
        events = rule_set.matches.data
        samples = CommandlineExtractor.extract_commandline(events)
    else:
        samples = rule_set.extract_field_values_from_filter(
            search_fields=["process.command_line"]
        )
    
    return normalize(samples)


def create_char_vectorizer():
    """
    Create a CHARACTER-LEVEL TF-IDF vectorizer.
    
    Key parameters:
    - analyzer='char': Use character n-grams instead of word tokens
    - ngram_range=(3, 6): Extract 3 to 6 character sequences
    
    This captures patterns like:
    - 'who' from 'whoami'
    - 'b64' from 'base64'
    - '/sh' from '/bin/sh'
    """
    return TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 6),
        max_features=50000,  # Limit to top 50k features to prevent memory issues
        lowercase=True,
    )


def create_labels(num_benign, num_malicious):
    """Create labels array: 0 for benign, 1 for malicious."""
    labels = np.zeros(num_benign + num_malicious, dtype=int)
    labels[num_benign:] = 1
    return labels


@execution_time
def train_model(X_train, y_train, search_params=True):
    """Train SVM model with optional grid search."""
    
    if search_params:
        _logger.info("Running Grid Search for best parameters...")
        param_grid = {
            'C': np.logspace(-2, 1, num=20),
            'kernel': ['linear'],
            'class_weight': ['balanced'],
        }
        
        grid_search = GridSearchCV(
            SVC(cache_size=2000),
            param_grid,
            cv=5,
            scoring=make_scorer(f1_score),
            n_jobs=4,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)
        
        _logger.info(f"Best parameters: {grid_search.best_params_}")
        _logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        # Use reasonable defaults
        estimator = SVC(
            C=0.34,
            kernel='linear',
            class_weight='balanced',
            cache_size=2000,
        )
        estimator.fit(X_train, y_train)
        return estimator


def main():
    print("DEBUG: Entering main", file=sys.stderr)
    parser = argparse.ArgumentParser(description="Train Character-Level Misuse Model")
    parser.add_argument("--benign-samples", type=str, default=DEFAULT_BENIGN,
                        help="Path to benign samples file")
    parser.add_argument("--events-dir", type=str, default=DEFAULT_EVENTS,
                        help="Path to Sigma events directory")
    parser.add_argument("--rules-dir", type=str, default=DEFAULT_RULES,
                        help="Path to Sigma rules directory")
    parser.add_argument("--malicious-samples-type", type=str, default="matches",
                        choices=["matches", "rule_filters"],
                        help="Type of malicious samples")
    parser.add_argument("--result-name", type=str, default="misuse_model_char",
                        help="Name for the result files")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUTPUT,
                        help="Output directory")
    parser.add_argument("--no-search", action="store_true",
                        help="Skip grid search, use default params")
    parser.add_argument("--mcc-threshold", type=float, default=0.5,
                        help="MCC threshold for scaling")
    
    args = parser.parse_args()
    
    # Initialize dumper
    dumper = Dumper(args.out_dir)
    
    # Load samples
    _logger.info("Loading benign samples...")
    benign_samples = load_benign_samples(args.benign_samples)
    _logger.info(f"Loaded {len(benign_samples)} benign samples")
    
    _logger.info("Loading malicious samples...")
    malicious_samples = load_malicious_samples(
        args.events_dir, args.rules_dir, args.malicious_samples_type
    )
    _logger.info(f"Loaded {len(malicious_samples)} malicious samples")
    
    # Combine samples
    all_samples = benign_samples + malicious_samples
    labels = create_labels(len(benign_samples), len(malicious_samples))
    
    # Create CHARACTER-LEVEL vectorizer
    _logger.info("Creating character-level TF-IDF vectorizer...")
    vectorizer = create_char_vectorizer()
    
    # Transform samples
    _logger.info("Transforming samples to feature vectors...")
    X = vectorizer.fit_transform(all_samples)
    _logger.info(f"Feature matrix shape: {X.shape}")
    
    # Train model
    estimator = train_model(X, labels, search_params=not args.no_search)
    
    # Create scaler
    _logger.info("Creating MCC scaler...")
    df_values = estimator.decision_function(X)
    scaler = create_symmetric_mcc_min_max_scaler(
        df_values, labels,
        num_mcc_samples=50,
        mcc_threshold=args.mcc_threshold,
    )
    
    # Prepare training data
    train_data = DataBunch(
        samples=X,
        labels=labels,
        label_names=["benign", "malicious"],
    )
    
    # Create data split
    data_split = TrainTestValidSplit(
        train_data=train_data,
        name=args.result_name,
    )
    
    # Create result
    result = TrainingResult(
        estimator=estimator,
        feature_extractors=[vectorizer],
        scaler=scaler,
        data=data_split,
        tainted_share=0.0,
        tainted_seed=42,
        name=f"{args.result_name}_0",
        timestamp="",
    )
    
    # Save result
    _logger.info(f"Saving result as {args.result_name}...")
    dumper.save_object(result)
    
    _logger.info("Training complete!")
    _logger.info(f"Output: {args.out_dir}/train_rslt_{args.result_name}_0.zip")


if __name__ == "__main__":
    main()
