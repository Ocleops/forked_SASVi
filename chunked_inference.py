#!/usr/bin/env python3
"""
Chunked inference script for SAM2 video segmentation.

This script processes videos in chunks to avoid memory errors when dealing with
long videos. It creates temporary directories with frame subsets, runs the
original eval_sasvi.py on each chunk, and combines the results.
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path


def get_frame_names(video_dir):
    """Get sorted list of frame names from a video directory."""
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(p))
    return frame_names


def create_chunk_directory(source_dir, chunk_dir, frame_names):
    """Create a temporary directory with symlinks to the specified frames."""
    os.makedirs(chunk_dir, exist_ok=True)

    for frame_name in frame_names:
        # Find the actual file (could be .jpg or .jpeg)
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
            source_path = os.path.join(source_dir, f"{frame_name}{ext}")
            if os.path.exists(source_path):
                dest_path = os.path.join(chunk_dir, f"{frame_name}{ext}")
                # Use symlink to save disk space
                if not os.path.exists(dest_path):
                    os.symlink(source_path, dest_path)
                break


def run_inference_on_chunk(args, chunk_base_dir, chunk_output_dir, video_name):
    """Run the original eval_sasvi.py script on a chunk."""
    cmd = [
        "python3", "src/sam2/eval_sasvi.py",
        "--device", args.device,
        "--sam2_cfg", args.sam2_cfg,
        "--sam2_checkpoint", args.sam2_checkpoint,
        "--overseer_checkpoint", args.overseer_checkpoint,
        "--overseer_type", args.overseer_type,
        "--dataset_type", args.dataset_type,
        "--base_video_dir", chunk_base_dir,
        "--output_mask_dir", chunk_output_dir,
    ]

    if args.save_binary_mask:
        cmd.append("--save_binary_mask")

    if args.score_thresh != 0.0:
        cmd.extend(["--score_thresh", str(args.score_thresh)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=args.working_dir)
    return result.returncode == 0


def copy_chunk_outputs(chunk_output_dir, final_output_dir, video_name):
    """Copy output masks from chunk output to final output directory."""
    chunk_video_output = os.path.join(chunk_output_dir, video_name)
    final_video_output = os.path.join(final_output_dir, video_name)

    os.makedirs(final_video_output, exist_ok=True)

    if os.path.exists(chunk_video_output):
        for mask_file in os.listdir(chunk_video_output):
            src = os.path.join(chunk_video_output, mask_file)
            dst = os.path.join(final_video_output, mask_file)
            shutil.copy2(src, dst)


def process_video_in_chunks(args, video_name, frame_names, temp_base_dir):
    """Process a single video in chunks."""
    num_frames = len(frame_names)
    chunk_size = args.chunk_size
    overlap = args.overlap  # Overlap between chunks for continuity

    print(f"\nProcessing {video_name}: {num_frames} frames in chunks of {chunk_size}")

    chunk_idx = 0
    start_idx = 0

    while start_idx < num_frames:
        end_idx = min(start_idx + chunk_size, num_frames)
        chunk_frames = frame_names[start_idx:end_idx]

        print(f"\n--- Chunk {chunk_idx + 1}: frames {start_idx} to {end_idx - 1} ({len(chunk_frames)} frames) ---")

        # Create temporary directories for this chunk
        chunk_base_dir = os.path.join(temp_base_dir, f"chunk_{chunk_idx}", "input")
        chunk_output_dir = os.path.join(temp_base_dir, f"chunk_{chunk_idx}", "output")
        chunk_video_dir = os.path.join(chunk_base_dir, video_name)

        # Create symlinks for this chunk's frames
        source_video_dir = os.path.join(args.base_video_dir, video_name)
        create_chunk_directory(source_video_dir, chunk_video_dir, chunk_frames)

        # Run inference on this chunk
        success = run_inference_on_chunk(args, chunk_base_dir, chunk_output_dir, video_name)

        if not success:
            print(f"Warning: Chunk {chunk_idx + 1} failed for {video_name}")

        # Copy outputs to final directory
        copy_chunk_outputs(chunk_output_dir, args.output_mask_dir, video_name)

        # Clean up chunk directories to save space
        shutil.rmtree(os.path.join(temp_base_dir, f"chunk_{chunk_idx}"), ignore_errors=True)

        # Move to next chunk (with overlap for continuity)
        start_idx = end_idx - overlap if end_idx < num_frames else num_frames
        chunk_idx += 1

    print(f"\nCompleted {video_name}: processed {chunk_idx} chunks")


def main():
    parser = argparse.ArgumentParser(description="Chunked video segmentation inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_checkpoint", type=str, default="src/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--overseer_checkpoint", type=str, required=True)
    parser.add_argument("--overseer_type", type=str, required=True, choices=['MaskRCNN', 'DETR', 'Mask2Former'])
    parser.add_argument("--dataset_type", type=str, required=True, choices=['CADIS', 'CHOLECSEG8K', 'CATARACT1K'])
    parser.add_argument("--base_video_dir", type=str, required=True)
    parser.add_argument("--output_mask_dir", type=str, required=True)
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--save_binary_mask", action="store_true")

    # Chunking parameters
    parser.add_argument("--chunk_size", type=int, default=150,
                        help="Number of frames per chunk (default: 150)")
    parser.add_argument("--overlap", type=int, default=10,
                        help="Number of overlapping frames between chunks for continuity (default: 10)")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Temporary directory for chunk processing (default: ./temp_chunks)")
    parser.add_argument("--working_dir", type=str, default=None,
                        help="Working directory for running eval_sasvi.py")

    args = parser.parse_args()

    # Set defaults
    if args.temp_dir is None:
        args.temp_dir = os.path.join(os.path.dirname(args.base_video_dir), "temp_chunks")
    if args.working_dir is None:
        args.working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get list of videos to process
    video_names = [
        p for p in os.listdir(args.base_video_dir)
        if os.path.isdir(os.path.join(args.base_video_dir, p))
    ]
    video_names = sorted(video_names)

    print(f"Found {len(video_names)} videos to process: {video_names}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")

    # Create output directory
    os.makedirs(args.output_mask_dir, exist_ok=True)

    # Process each video
    for video_idx, video_name in enumerate(video_names):
        print(f"\n{'='*60}")
        print(f"Video {video_idx + 1}/{len(video_names)}: {video_name}")
        print(f"{'='*60}")

        video_dir = os.path.join(args.base_video_dir, video_name)
        frame_names = get_frame_names(video_dir)

        if len(frame_names) == 0:
            print(f"No frames found in {video_dir}, skipping...")
            continue

        # Create temp directory for this video
        temp_video_dir = os.path.join(args.temp_dir, video_name)
        os.makedirs(temp_video_dir, exist_ok=True)

        try:
            process_video_in_chunks(args, video_name, frame_names, temp_video_dir)
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_video_dir, ignore_errors=True)

    # Final cleanup
    shutil.rmtree(args.temp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Completed processing all videos!")
    print(f"Output masks saved to: {args.output_mask_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
