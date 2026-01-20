import torch

# 1. Test SAM2 import
print("Testing SAM2...")
try:
    from sam2.build_sam import build_sam2_video_predictor
    print("✓ SAM2 imported successfully")
except ImportError as e:
    print(f"✗ SAM2 import failed: {e}")

# 2. Test SDS_Playground (for overseer models)
print("\nTesting SDS_Playground...")
try:
    import sds_playground
    print("✓ SDS_Playground imported successfully")
except ImportError as e:
    print(f"✗ SDS_Playground import failed: {e}")

# 3. Test loading SAM2 model
print("\nTesting SAM2 model loading...")
try:
    predictor = build_sam2_video_predictor(
        config_file="configs/sam2.1_hiera_l.yaml",
        ckpt_path="./checkpoints/sam2.1_hiera_large.pt"  # adjust to your checkpoint
    )
    print("✓ SAM2 model loaded successfully")
except Exception as e:
    print(f"✗ SAM2 model loading failed: {e}")

# 4. Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")