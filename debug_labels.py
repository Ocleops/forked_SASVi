"""Debug script to find invalid labels in CholecSeg8k dataset"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/src')
from utils import load_maskrcnn_overseer

def check_dataset_labels():
    chckpt = Path('/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/checkpoints/cholecseg8k/maskrcnn/best_val_dice.pth')
    data = Path('/home/data/tumai/splatgraph/data/cholecseg8k/')
    device = 'cpu'

    print("Loading dataset...")
    m, ds, hidden_ft, backbone, ignore_indices, shift_by_1, keep_ignore = load_maskrcnn_overseer(chckpt, data, device)

    print(f"Dataset size: {len(ds)}")
    print(f"Expected num_classes: {ds.num_classes}")

    print("\nChecking all masks for invalid labels...")
    all_unique_labels = set()
    invalid_samples = []

    for idx in tqdm(range(len(ds)), desc="Checking samples"):
        try:
            # Try to get the item
            _, mask, _, _ = ds[idx]

            # Get unique labels in this mask
            unique_labels = torch.unique(mask)
            all_unique_labels.update(unique_labels.tolist())

            # Check for out-of-bounds labels
            max_label = unique_labels.max().item()
            if max_label >= ds.num_classes:
                invalid_samples.append({
                    'idx': idx,
                    'max_label': max_label,
                    'unique_labels': unique_labels.tolist()
                })

        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            invalid_samples.append({
                'idx': idx,
                'error': str(e)
            })

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"All unique labels found in dataset: {sorted(all_unique_labels)}")
    print(f"Expected range: 0 to {ds.num_classes - 1}")
    print(f"Number of samples with invalid labels: {len(invalid_samples)}")

    if invalid_samples:
        print(f"\nFirst 10 invalid samples:")
        for sample in invalid_samples[:10]:
            print(f"  Sample {sample['idx']}: {sample}")

    return all_unique_labels, invalid_samples

if __name__ == "__main__":
    all_labels, invalid = check_dataset_labels()
