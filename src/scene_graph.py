"""
Mask R-CNN Inference with Visualization and Scene Graph Generation

This script:
1. Loads an image
2. Runs Mask R-CNN to detect objects
3. Visualizes the detections (bounding boxes and masks)
4. Generates a scene graph from the detections
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import json
import os

# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

# CholecSeg8k classes
# Note: In Mask R-CNN, labels are shifted by 1 (0 is background)
# So label 1 in model output = class index 0 in original dataset
CHOLECSEG8K_CLASSES = {
    1: "Abdominal_Wall",
    2: "Liver",
    3: "Gastrointestinal_Tract",
    4: "Fat",
    5: "Grasper",
    6: "Connective_Tissue",
    7: "Blood",
    8: "Cystic_Duct",
    9: "L-hook_Electrocautery",
    10: "Gallbladder",
    11: "Hepatic_Vein",
    12: "Liver_Ligament",
}

# Tool classes (for detecting "retracting" motion)
TOOL_CLASSES = {5, 9}  # Grasper, L-hook

# Classes to skip in scene graph (background, abdominal wall)
SKIP_CLASSES = {1}  # Abdominal wall is not useful for scene understanding


def get_color_for_class(class_id, num_classes=13):
    """Generate a distinct color for each class using HSV color space."""
    hue = class_id / num_classes
    return hsv_to_rgb([hue, 0.8, 0.9])


# =============================================================================
# MODEL LOADING AND INFERENCE
# =============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load the Mask R-CNN model."""
    from model import get_model_instance_segmentation
    
    model = get_model_instance_segmentation(
        num_classes=13,
        trainable_backbone_layers=0,
        hidden_ft=64,
        custom_in_ft_box=None,
        custom_in_ft_mask=None,
        backbone='ResNet50',
        img_size=(299, 299)
    )
    
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path: str, target_size: tuple = (299, 299)):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return img_tensor, img_array  # Return both tensor and numpy array for visualization

def run_inference(model, image_tensor, device: str = 'cuda', score_threshold: float = 0.5):
    """Run inference and filter by score threshold."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model([image_tensor])
    
    # Filter detections by score
    output = outputs[0]
    keep = output['scores'] > score_threshold
    
    filtered_output = {
        'boxes': output['boxes'][keep].cpu(),
        'labels': output['labels'][keep].cpu(),
        'scores': output['scores'][keep].cpu(),
        'masks': output['masks'][keep].cpu(),
    }
    
    return filtered_output


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_detections(image_array, detections, save_path=None, show=True):
    """
    Visualize bounding boxes and masks on the image.
    
    Creates a figure with 3 subplots:
    1. Original image
    2. Image with bounding boxes
    3. Image with masks overlaid
    """
    boxes = detections['boxes'].numpy()
    labels = detections['labels'].numpy()
    scores = detections['scores'].numpy()
    masks = detections['masks'].numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original image
    axes[0].imshow(image_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 2. Image with bounding boxes
    axes[1].imshow(image_array)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        label = labels[i]
        score = scores[i]
        
        color = get_color_for_class(label)
        class_name = CHOLECSEG8K_CLASSES.get(label, f"Class_{label}")
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Add label text
        axes[1].text(
            x1, y1 - 5,
            f'{class_name}: {score:.2f}',
            color='white', fontsize=8,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )
    
    axes[1].set_title('Bounding Boxes')
    axes[1].axis('off')
    
    # 3. Image with masks
    axes[2].imshow(image_array)
    
    # Create a combined mask overlay
    overlay = np.zeros((*image_array.shape[:2], 4))  # RGBA
    
    for i in range(len(masks)):
        mask = masks[i, 0] > 0.5  # Binary mask
        label = labels[i]
        color = get_color_for_class(label)
        
        # Add this mask to the overlay with transparency
        overlay[mask, :3] = color
        overlay[mask, 3] = 0.5  # Alpha
    
    axes[2].imshow(overlay)
    axes[2].set_title('Segmentation Masks')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_scene_graph(image_array, scene_graph, save_path=None, show=True):
    """
    Visualize the scene graph overlaid on the image.
    
    Shows objects as labeled boxes and relationships as arrows.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_array)
    
    objects = scene_graph['objects']
    relationships = scene_graph['relationships']
    
    # Get image dimensions for denormalizing coordinates
    H, W = image_array.shape[:2]
    
    # Draw objects
    obj_centers = {}  # Store centers for drawing relationship arrows
    
    for obj in objects:
        # Denormalize bounding box
        x1 = obj['bbox'][0] * W
        y1 = obj['bbox'][1] * H
        x2 = obj['bbox'][2] * W
        y2 = obj['bbox'][3] * H
        
        color = get_color_for_class(obj['class_id'])
        
        # Draw box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            x1, y1 - 5,
            f"[{obj['id']}] {obj['class']}",
            color='white', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.9)
        )
        
        # Store center
        cx = obj['center'][0] * W
        cy = obj['center'][1] * H
        obj_centers[obj['id']] = (cx, cy)
    
    # Draw relationships (only "close_to" to avoid clutter)
    for rel in relationships:
        if rel['pred'] == 'close_to':
            if rel['sub'] in obj_centers and rel['obj'] in obj_centers:
                x1, y1 = obj_centers[rel['sub']]
                x2, y2 = obj_centers[rel['obj']]
                
                ax.annotate(
                    '', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5, alpha=0.7)
                )
    
    ax.set_title('Scene Graph Visualization', fontsize=14)
    ax.axis('off')
    
    # Add legend for relationships
    legend_text = "Arrows show 'close_to' relationships"
    ax.text(
        0.02, 0.02, legend_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scene graph visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# SCENE GRAPH GENERATION
# =============================================================================

def build_scene_graph(detections, frame_id="frame_001", image_size=(299, 299), 
                      close_threshold=0.15, score_threshold=0.5):
    """
    Build a scene graph from Mask R-CNN detections.
    
    A scene graph contains:
    - objects: List of detected objects with their properties
    - relationships: List of spatial relationships between objects
    
    Args:
        detections: Output from run_inference()
        frame_id: Identifier for this frame
        image_size: (height, width) of the image
        close_threshold: Distance threshold for "close_to" relationship (fraction of image diagonal)
        score_threshold: Minimum confidence to include a detection
    
    Returns:
        Dictionary with 'frame_id', 'objects', and 'relationships'
    """
    H, W = image_size
    
    boxes = detections['boxes'].numpy()
    labels = detections['labels'].numpy()
    scores = detections['scores'].numpy()
    masks = detections['masks'].numpy()
    
    # Build list of objects
    objects = []
    obj_id = 1
    
    for i in range(len(boxes)):
        label = int(labels[i])
        score = float(scores[i])
        
        # Skip low confidence or unwanted classes
        if score < score_threshold:
            continue
        if label in SKIP_CLASSES:
            continue
        
        x1, y1, x2, y2 = boxes[i]
        
        # Normalize coordinates to [0, 1]
        bbox_normalized = [
            round(x1 / W, 4),
            round(y1 / H, 4),
            round(x2 / W, 4),
            round(y2 / H, 4),
        ]
        
        # Compute center (normalized)
        center = (
            (bbox_normalized[0] + bbox_normalized[2]) / 2,
            (bbox_normalized[1] + bbox_normalized[3]) / 2,
        )
        
        # Compute area from mask
        mask = masks[i, 0] > 0.5
        pixel_area = np.sum(mask)
        area_normalized = round(pixel_area / (H * W), 6)
        
        obj = {
            'id': obj_id,
            'class_id': label,
            'class': CHOLECSEG8K_CLASSES.get(label, f"Unknown_{label}"),
            'bbox': bbox_normalized,
            'center': list(center),
            'area': area_normalized,
            'score': round(score, 4),
        }
        objects.append(obj)
        obj_id += 1
    
    # Build relationships
    relationships = []
    n = len(objects)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            subj = objects[i]
            obj = objects[j]
            
            # Check if subject is inside object
            if is_inside(subj['bbox'], obj['bbox']):
                relationships.append({
                    'sub': subj['id'],
                    'pred': 'inside',
                    'obj': obj['id']
                })
                continue
            
            # Compute distance between centers
            dist = compute_distance(subj['center'], obj['center'])
            
            if dist < close_threshold:
                # Objects are close to each other
                relationships.append({
                    'sub': subj['id'],
                    'pred': 'close_to',
                    'obj': obj['id']
                })
            else:
                # Compute directional relationships
                # Horizontal
                if subj['center'][0] < obj['center'][0]:
                    relationships.append({
                        'sub': subj['id'],
                        'pred': 'left_of',
                        'obj': obj['id']
                    })
                else:
                    relationships.append({
                        'sub': subj['id'],
                        'pred': 'right_of',
                        'obj': obj['id']
                    })
                
                # Vertical
                if subj['center'][1] < obj['center'][1]:
                    relationships.append({
                        'sub': subj['id'],
                        'pred': 'above',
                        'obj': obj['id']
                    })
                else:
                    relationships.append({
                        'sub': subj['id'],
                        'pred': 'below',
                        'obj': obj['id']
                    })
    
    return {
        'frame_id': frame_id,
        'objects': objects,
        'relationships': relationships,
    }


def is_inside(bbox_a, bbox_b):
    """Check if bbox_a is fully contained within bbox_b."""
    return (
        bbox_a[0] >= bbox_b[0] and
        bbox_a[1] >= bbox_b[1] and
        bbox_a[2] <= bbox_b[2] and
        bbox_a[3] <= bbox_b[3]
    )


def compute_distance(center_a, center_b):
    """Compute Euclidean distance between two normalized centers."""
    return np.sqrt(
        (center_a[0] - center_b[0]) ** 2 +
        (center_a[1] - center_b[1]) ** 2
    )


def print_scene_graph(scene_graph):
    """Print the scene graph in a readable format."""
    print("\n" + "=" * 60)
    print("SCENE GRAPH")
    print("=" * 60)
    print(f"Frame: {scene_graph['frame_id']}")
    print(f"Number of objects: {len(scene_graph['objects'])}")
    print(f"Number of relationships: {len(scene_graph['relationships'])}")
    
    print("\n--- OBJECTS ---")
    for obj in scene_graph['objects']:
        print(f"  [{obj['id']}] {obj['class']}")
        print(f"      Score: {obj['score']:.3f}")
        print(f"      Center: ({obj['center'][0]:.2f}, {obj['center'][1]:.2f})")
        print(f"      Area: {obj['area']*100:.2f}% of image")
    
    print("\n--- RELATIONSHIPS ---")
    # Group by predicate for cleaner output
    by_predicate = {}
    for rel in scene_graph['relationships']:
        pred = rel['pred']
        if pred not in by_predicate:
            by_predicate[pred] = []
        by_predicate[pred].append(rel)
    
    for pred, rels in by_predicate.items():
        print(f"\n  {pred}:")
        for rel in rels[:5]:  # Show first 5 of each type
            # Find object names
            subj_name = next((o['class'] for o in scene_graph['objects'] if o['id'] == rel['sub']), '?')
            obj_name = next((o['class'] for o in scene_graph['objects'] if o['id'] == rel['obj']), '?')
            print(f"    [{rel['sub']}] {subj_name} --{pred}--> [{rel['obj']}] {obj_name}")
        if len(rels) > 5:
            print(f"    ... and {len(rels) - 5} more")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    
    CHECKPOINT_PATH = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/checkpoints/cholecseg8k/maskrcnn/best_val_dice.pth'
    # IMAGE_PATH = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_set_masks/video01/00200.jpg'  # <-- CHANGE THIS
    IMAGE_PATH = Path('/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_set_masks/video01')

    OUTPUT_DIR = Path('/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/src/outputs')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SCORE_THRESHOLD = 0.5  # Minimum confidence for detections
    
    # -------------------------------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------------------------------
    
    print("=" * 60)
    print("MASK R-CNN + SCENE GRAPH PIPELINE")
    print("=" * 60)
    
    # Step 1: Load model
    print("\n[1/5] Loading model...")
    model = load_model(CHECKPOINT_PATH, DEVICE)
    print("      Model loaded!")
    
    for img_name in os.listdir(IMAGE_PATH):
        # Step 2: Load and preprocess image
        print(f"\n[2/5] Loading image: {IMAGE_PATH}")
        image_tensor, image_array = preprocess_image(IMAGE_PATH / img_name)
        print(f"      Image shape: {image_array.shape}")
        
        # Step 3: Run inference
        print(f"\n[3/5] Running inference (threshold={SCORE_THRESHOLD})...")
        detections = run_inference(model, image_tensor, DEVICE, score_threshold=SCORE_THRESHOLD)
        print(f"      Found {len(detections['boxes'])} detections")
        
        # Step 4: Visualize detections
        print("\n[4/5] Visualizing detections...")
        vis_path = OUTPUT_DIR / img_name
        visualize_detections(image_array, detections, save_path=str(vis_path), show=True)
        
        # # Step 5: Build scene graph
        # print("\n[5/5] Building scene graph...")
        # frame_id = Path(IMAGE_PATH).stem
        # scene_graph = build_scene_graph(
        #     detections, 
        #     frame_id=frame_id,
        #     image_size=image_array.shape[:2],
        #     close_threshold=0.15,
        #     score_threshold=SCORE_THRESHOLD
        # )
        
        # # Print scene graph
        # print_scene_graph(scene_graph)
        
        # # Visualize scene graph
        # sg_vis_path = OUTPUT_DIR / 'scene_graph.png'
        # visualize_scene_graph(image_array, scene_graph, save_path=str(sg_vis_path), show=False)
        
        # # Save scene graph as JSON
        # json_path = OUTPUT_DIR / 'scene_graph.json'
        # with open(json_path, 'w') as f:
        #     json.dump(scene_graph, f, indent=2)
        # print(f"\nSaved scene graph JSON to: {json_path}")
        
        # print("\n" + "=" * 60)
        # print("DONE!")
        # print("=" * 60)
        # print(f"\nOutput files saved to: {OUTPUT_DIR}/")
        # print("  - detections.png    : Visualization of bounding boxes and masks")
        # print("  - scene_graph.png   : Visualization of scene graph")
        # print("  - scene_graph.json  : Scene graph in JSON format")