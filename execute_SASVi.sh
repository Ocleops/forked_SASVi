python3 src/sam2/eval_sasvi.py \
--device cuda \
--sam2_cfg configs/sam2.1_hiera_l.yaml \
--sam2_checkpoint src/sam2/checkpoints/sam2.1_hiera_large.pt \
--overseer_checkpoint checkpoints/cholecseg8k/maskrcnn/best_val_dice.pth \
--overseer_type MaskRCNN \
--dataset_type CHOLECSEG8K \
--base_video_dir /home/guests/myron_theocharakis/process_cholec \
--output_mask_dir ./output_masks_cholecseg8k \
--save_binary_mask
