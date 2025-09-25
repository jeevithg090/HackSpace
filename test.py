import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
base_dir = "../archive/1920x1080/1920x1080"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# New split folders
new_base = os.path.join(base_dir, "split_70_10_20")
train_new = os.path.join(new_base, "train")
val_new = os.path.join(new_base, "val")
test_new = os.path.join(new_base, "test")

# Create new directories
for d in [train_new, val_new, test_new]:
    os.makedirs(d, exist_ok=True)

# --- Copy over training data directly (already 70%) ---
for f in os.listdir(train_dir):
    src_path = os.path.join(train_dir, f)
    if os.path.isfile(src_path): # only copy files
        shutil.copy(src_path, train_new)

# --- Split the existing test set into val (1/3) and test (2/3) ---
all_test_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]

val_files, test_files = train_test_split(all_test_files, test_size=2/3, random_state=42)

# Copy validation files
for f in val_files:
    shutil.copy(os.path.join(test_dir, f), val_new)

# Copy test files
for f in test_files:
    shutil.copy(os.path.join(test_dir, f), test_new)

print("✅ New split created at:", new_base)