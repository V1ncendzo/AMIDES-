#!/bin/bash
set -e

# Ensure we can find the amides package.
# Assuming this script is run from the 'amides' directory (where this script is located) or project root.
# We will use absolute paths based on PWD to be safe.

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Assuming the script is in 'amides/', the project root is 'amides/' (relative to download) or just PWD.
# We need to point PYTHONPATH to the directory CONTAINING the 'amides' package directory.
# Based on project structure: AMIDES/amides/amides/bin...
# We need PYTHONPATH to point to AMIDES/amides

echo ">>> Setting up Environment..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/amides

echo "----------------------------------------------------------------"
echo "PHASE 1: Training Misuse Detection Model (Matches)"
echo "----------------------------------------------------------------"

python3 amides/bin/train.py \
    --model-type misuse \
    --result-name linux_misuse_model_final \
    --malicious-samples-type matches \
    --benign-samples data/socbed/linux/process_creation/train \
    --events-dir data/sigma/events/linux/process_creation \
    --rules-dir data/sigma/rules/linux/process_creation \
    --normalize \
    --deduplicate \
    --search-params \
    --scoring f1 \
    --out-dir models_linux

echo ""
echo "----------------------------------------------------------------"
echo "PHASE 2: Training Rule Attribution Model"
echo "----------------------------------------------------------------"

python3 amides/bin/train.py \
    --model-type attribution \
    --result-name linux_attribution_model_final \
    --benign-samples data/socbed/linux/process_creation/train \
    --events-dir data/sigma/events/linux/process_creation \
    --rules-dir data/sigma/rules/linux/process_creation \
    --malicious-samples-type matches \
    --normalize \
    --deduplicate \
    --search-params \
    --num-subprocesses 4 \
    --out-dir models_linux

echo ""
echo ">>> SUCCESS! Both models have been trained and saved to 'models_linux/'."
