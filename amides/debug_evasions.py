
import sys
import os
import json
import glob
import numpy as np

# Add parent dir to path
sys.path.append(os.getcwd() + "/amides")

from amides.sigma import RuleSetDataset
from amides.features.extraction import CommandlineExtractor

events_dir = "amides/data/sigma/events/linux/process_creation"
rules_dir = "amides/data/sigma/rules/linux/process_creation"

print(f"Loading rules from {rules_dir}...")
rs = RuleSetDataset()
rs.load_rule_set_data(events_dir, rules_dir)

print("\n--- Detailed Evasion Counts ---")
print(f"{'Rule Name':<50} | {'Files':<5} | {'Loaded':<6} | {'Valid CMD':<9}")
print("-" * 90)

total_files = 0
total_loaded = 0
total_valid = 0
all_cmds = []

# Get list of rule directories
rule_dirs = [d for d in os.listdir(events_dir) if os.path.isdir(os.path.join(events_dir, d))]

for rule_name in sorted(rs.rule_datasets.keys()):
    rule_data = rs.rule_datasets[rule_name]
    
    # Count evasions loaded
    num_loaded = rule_data.evasions.size
    
    # Count files on disk (approximated by rule name matching directory?)
    # Rule names in datasets match directory names usually, but normalized.
    # Let's try to match exactly.
    
    # Find matching directory
    matching_dir = None
    for d in rule_dirs:
        # Simple heuristic: exact match
        if d == rule_name:
            matching_dir = d
            break
            
    num_files = 0
    if matching_dir:
        path = os.path.join(events_dir, matching_dir)
        num_files = len(glob.glob(os.path.join(path, "*Evasion*.json")))
    
    # Extract
    if num_loaded > 0:
        raw_cmds = CommandlineExtractor.extract_commandline(rule_data.evasions.data)
        num_valid = sum(1 for c in raw_cmds if c is not None)
        # Collect all commands for global deduplication check
        all_cmds.extend([c for c in raw_cmds if c is not None])
    else:
        num_valid = 0
        
    if num_files > 0 or num_loaded > 0:
        print(f"{rule_name:<50} | {num_files:<5} | {num_loaded:<6} | {num_valid:<9}")
        total_files += num_files
        total_loaded += num_loaded
        total_valid += num_valid

print("-" * 90)
print(f"{'TOTAL':<50} | {total_files:<5} | {total_loaded:<6} | {total_valid:<9}")
print("-" * 90)
unique_cmds = len(set(all_cmds))
print(f"Total Unique Command Lines: {unique_cmds}")
print(f"Deduplication Loss: {total_valid - unique_cmds}")
