# AMIDES: Adaptive Misuse Detection System (Linux)

This repository contains the implementation of the AMIDES misuse and attribution models for Linux command line data.

## 1. Installation

Ensure you have Python 3 installed. It is recommended to use a virtual environment.

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r amides/amides/requirements.txt
```

## 2. Data Overview

### Socbed Data (Training Data)
The primary dataset used for training and validation is located in the `socbed` directory:
- **Location**: `amides/data/socbed/linux/process_creation/`
- **Structure**:
  - `train`: Training dataset
  - `validation`: Validation dataset
  - `all`: Combined dataset

These files contain the processed command lines used to train the models.

### Evasion and Matches Data
The project uses specific rule sets for evaluation, particularly for checking model robustness against evasions and attributing commands to Sigma rules.

- **Evasion Rules**: Listed in `evasions_rules.txt`. These represent known evasion techniques (e.g., `base64_decode`, `clear_logs`) that the model is tested against.
- **Matches Rules**: Listed in `matches_rules.txt`. These are the Sigma rules that the attribution model aims to identify.

## 3. Running Demos

Two demo scripts are provided to interact with the trained models interactively.

### Misuse Model Demo
This script loads a trained misuse detection model (Benign vs. Malicious) and classifies distinct commands.

**Command:**
```bash
python3 amides/amides/bin/demo_misuse_model.py
# Or optionally specify a model path directly:
# python3 amides/amides/bin/demo_misuse_model.py --model-path <path_to_model.zip>
```

**Usage:**
1. Run the script.
2. Select a model from the list if not provided.
3. Enter a shell command (e.g., `cat /etc/passwd` or `base64 -d payload`) to see the prediction and confidence score.

### Attribution Model Demo
This script loads a trained attribution model (ensemble) and identifies which Sigma rules a command likely triggers.

**Command:**
```bash
PYTHONPATH=$(pwd)/amides/amides python3 amides/amides/bin/demo_attribution_model.py
# Or optionally specify a model path:
# PYTHONPATH=$(pwd)/amides/amides python3 amides/amides/bin/demo_attribution_model.py --model-path <path_to_model.zip>
```

**Usage:**
1. Run the script.
2. Select an attribution model (`multi_train_rslt...zip`) from the list.
3. Enter a shell command.
4. The script will output the top matching Sigma rules and their probabilities.
