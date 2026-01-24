#%%
import os
import re
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
import argparse
# %%

def extract_frame_number_from_folder(folder_name):
    """Extract frame number from folder name like 'video01_00080' -> 80"""
    match = re.search(r'_(\d+)$', folder_name)
    if match:
        return int(match.group(1))
    return None

def extract_frame_number_from_file(filename):
    """Extract frame number from filename like 'frame_80_endo.png' -> 80"""
    match = re.search(r'frame_(\d+)_endo\.png$', filename)
    if match:
        return int(match.group(1))
    return None

def preprocess_cholecseg8k(
    source_dir: str,
    target_dir: str,
    convert_to_jpg: bool = True,
    copy_masks: bool = False,
    video_filter: str = None,
    dry_run: bool = False
):
    """
    Reorganize CholecSeg8k data for SASVi inference.
    
    Args:
        source_dir: Path to original cholecseg8k directory
        target_dir: Path to output directory for reorganized data
        convert_to_jpg: Convert PNG to JPEG (required for SASVi)
        copy_masks: Also copy ground truth masks to a separate directory
        video_filter: Only process specific video (e.g., 'video01')
        dry_run: Print actions without executing
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_path}")
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video directories
    video_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    
    if video_filter:
        video_dirs = [d for d in video_dirs if video_filter in d.name]
    
    print(f"Found {len(video_dirs)} video directories")
    
    total_frames = 0
    
    for video_dir in video_dirs:
        video_name = video_dir.name
        print(f"\nProcessing {video_name}...")
        
        # Create target video directory
        target_video_dir = target_path / video_name
        if not dry_run:
            target_video_dir.mkdir(parents=True, exist_ok=True)
        
        if copy_masks:
            target_mask_dir = target_path / f"{video_name}_masks"
            if not dry_run:
                target_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all frame subdirectories (e.g., video01_00080)
        frame_subdirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        
        frame_count = 0
        
        for frame_subdir in tqdm(frame_subdirs, desc=f"  {video_name}"):
            # Find ALL frame_*_endo.png files in this subdirectory
            endo_files = list(frame_subdir.glob("frame_*_endo.png"))
            
            if not endo_files:
                print(f"  Warning: No frame_*_endo.png found in {frame_subdir}")
                continue
            
            for endo_file in endo_files:
                frame_num = extract_frame_number_from_file(endo_file.name)
                
                if frame_num is None:
                    print(f"  Warning: Could not extract frame number from {endo_file.name}")
                    continue
                
                # Target filename (padded to 5 digits for proper sorting)
                if convert_to_jpg:
                    target_filename = f"{frame_num:05d}.jpg"
                else:
                    target_filename = f"{frame_num:05d}.png"
                
                target_file = target_video_dir / target_filename
                
                if dry_run:
                    print(f"  Would copy: {endo_file} -> {target_file}")
                else:
                    if convert_to_jpg:
                        # Convert PNG to JPEG
                        img = Image.open(endo_file).convert('RGB')
                        img.save(target_file, 'JPEG', quality=95)
                    else:
                        shutil.copy2(endo_file, target_file)
                
                # Optionally copy masks
                if copy_masks:
                    mask_file = frame_subdir / f"frame_{frame_num}_endo_mask.png"
                    if mask_file.exists():
                        target_mask_file = target_mask_dir / f"{frame_num:05d}_mask.png"
                        if not dry_run:
                            shutil.copy2(mask_file, target_mask_file)
                
                frame_count += 1
        
        print(f"  Processed {frame_count} frames")
        total_frames += frame_count
    
    print(f"\n{'='*50}")
    print(f"Done! Total frames processed: {total_frames}")
    print(f"Reorganized data saved to: {target_path}")
    print(f"\nYou can now run SASVi with:")
    print(f"  --base_video_dir {target_path}")

def process_SASVi_test(
    file: str = 'test_split.txt',
    convert_to_jpg: bool = True, 
    save_folder: str = './test_set_masks/video01/'
):
    with open(file) as f: 
        video_list = [line.strip() for line in f.readlines()]

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    for fname in video_list:
        frame_num = extract_frame_number_from_file(fname)
        if frame_num is None:
            print(f"Warning: Could not extract frame number from {fname}")
            continue

        if convert_to_jpg:
            target_filename = f"{frame_num:05d}.jpg"
        else:
            target_filename = f"{frame_num:05d}.png"
        
        if convert_to_jpg:
            # Convert PNG to JPEG
            img = Image.open(fname).convert('RGB')

            img.save(save_folder+target_filename, 'JPEG', quality=95)

#%%
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Chunked video segmentation inference")
    # parser.add_argument("--source_dir", type=str, default="/home/data/tumai/splatgraph/data/cholecseg8k")
    # parser.add_argument("--target_dir", type=str, default=None)
    # parser.add_argument("--convert_to_jpg", type=bool, default=True)
    # parser.add_argument("--copy_masks", type=bool, default=False)
    # parser.add_argument("--video_filter", type=bool, default=None)
    # parser.add_argument("--dry_run", type=bool, default=False)
    # args = parser.parse_args()

    # preprocess_cholecseg8k(
    #     source_dir= args.source_dir,
    #     target_dir=args.target_dir,
    #     convert_to_jpg = True,
    #     copy_masks = False,
    #     video_filter = None,
    #     dry_run = False
    # )

    process_SASVi_test()

# %%
