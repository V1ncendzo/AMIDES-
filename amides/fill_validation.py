
import os

base_dir = "amides/data/socbed/linux/process_creation"
all_path = os.path.join(base_dir, "all")
train_path = os.path.join(base_dir, "train")
valid_path = os.path.join(base_dir, "validation")

print("Reading files...")
with open(all_path, 'r') as f:
    all_lines = [l.strip() for l in f if l.strip()]

try:
    with open(train_path, 'r') as f:
        train_lines = set(l.strip() for l in f if l.strip())
except FileNotFoundError:
    train_lines = set()

print(f"Total lines in all: {len(all_lines)}")
print(f"Total unique lines in train: {len(train_lines)}")

validation_lines = []
for line in all_lines:
    if line not in train_lines:
        validation_lines.append(line)

print(f"Found {len(validation_lines)} samples missing from train.")
print(f"Writing to {valid_path}...")

with open(valid_path, 'w') as f:
    f.write("\n".join(validation_lines) + "\n")

print("Done.")
