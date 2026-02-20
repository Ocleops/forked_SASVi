"""Check raw mask values without going through the dataset __getitem__"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# Get the dataset paths
data_root = Path('/home/data/tumai/splatgraph/data/cholecseg8k/')

# Find mask directories
mask_dirs = list(data_root.glob('**/masks')) + list(data_root.glob('**/mask'))
print(f"Found mask directories: {mask_dirs}")

all_unique_labels = set()

# Check a few masks to see what values they contain
for mask_dir in mask_dirs[:1]:  # Just check first directory
    mask_files = sorted(list(mask_dir.glob('*.png')))[:100]  # Sample 100 masks

    print(f"\nChecking {len(mask_files)} masks in {mask_dir}")

    for mask_file in tqdm(mask_files, desc="Reading masks"):
        # Read mask as numpy array
        mask = np.array(Image.open(mask_file))
        unique = np.unique(mask)
        all_unique_labels.update(unique.tolist())

print(f"\n{'='*60}")
print("RESULTS:")
print(f"{'='*60}")
print(f"All unique label values found in raw masks: {sorted(all_unique_labels)}")
print(f"Number of unique labels: {len(all_unique_labels)}")
print(f"Min label: {min(all_unique_labels)}")
print(f"Max label: {max(all_unique_labels)}")
print(f"\nExpected by dataset class: 0 to 12 (13 classes)")
print(f"Actual range in data: 0 to {max(all_unique_labels)} ({len(all_unique_labels)} unique values)")
