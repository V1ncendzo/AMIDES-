
import random
import os

base_dir = "amides/data/socbed/linux/process_creation"
all_file = os.path.join(base_dir, "all")
train_file = os.path.join(base_dir, "train")
valid_file = os.path.join(base_dir, "validation")

print(f"Reading from {all_file}...")
with open(all_file, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

total = len(lines)
print(f"Total lines: {total}")

random.shuffle(lines)

split_idx = int(total * 0.8)
train_data = lines[:split_idx]
valid_data = lines[split_idx:]

print(f"Writing {len(train_data)} lines to {train_file}...")
with open(train_file, "w") as f:
    f.write("\n".join(train_data) + "\n")

print(f"Writing {len(valid_data)} lines to {valid_file}...")
with open(valid_file, "w") as f:
    f.write("\n".join(valid_data) + "\n")

print("Done.")
