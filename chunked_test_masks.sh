#!/bin/bash
# Chunked video segmentation - processes videos in chunks to avoid memory errors

pixi run python chunked_inference.py \
--device cuda \
--sam2_cfg configs/sam2.1_hiera_l.yaml \
--sam2_checkpoint src/sam2/checkpoints/sam2.1_hiera_large.pt \
--overseer_checkpoint checkpoints/cholecseg8k/maskrcnn/best_val_dice.pth \
--overseer_type MaskRCNN \
--dataset_type CHOLECSEG8K \
--base_video_dir /home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_set_masks \
--output_mask_dir ./test_segmentations \
--save_binary_mask \
--chunk_size 100 \
--overlap 1 \
--working_dir /home/guests/myron_theocharakis/longform-surgery/forked_SASVi
