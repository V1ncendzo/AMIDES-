
import sys
import os
# Add parent dir to path
sys.path.append(os.getcwd() + "/amides")

from amides.sigma import RuleSetDataset

events_dir = "amides/data/sigma/events/linux/process_creation"
rules_dir = "amides/data/sigma/rules/linux/process_creation"

rs = RuleSetDataset()
rs.load_rule_set_data(events_dir, rules_dir)

if "base64_decode" in rs.rule_datasets:
    rd = rs.rule_datasets["base64_decode"]
    print(f"Loaded Evasions: {rd.evasions.size}")
else:
    print("Rule base64_decode not found in dataset")
