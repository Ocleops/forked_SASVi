#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import torch
#%%
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


# Classes to skip in scene graph (background, abdominal wall)
SKIP_CLASSES = 1  # Abdominal wall is not useful for scene understanding


def get_color_for_class(class_id, num_classes=13):
    """Generate a distinct color for each class using HSV color space."""
    hue = class_id / num_classes
    return hsv_to_rgb([hue, 0.8, 0.9])

def preprocess_image(image_path: str, target_size: tuple = (299, 299)):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return img_tensor, img_array  # Return both tensor and numpy array for visualization

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
        img_size=(299, 299) # This is needed because this is the size that the model was trained on.
    )
    
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    return model

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

def bounding_boxes(image_array, detections):
    boxes = detections['boxes'] # the coordinates of the bounding boxes are returned as (x1, y1, x2, y2) according to the mask_rcnn.py file.
    labels = detections['labels']
    masks = detections['masks']
    scores = detections['scores']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(image_array)

    bounding_boxes =[]

    for i in range(len(boxes)):
        if labels[i] == SKIP_CLASSES:
            continue

        class_name = CHOLECSEG8K_CLASSES.get(labels[i].item())
        color = get_color_for_class(class_id=labels[i], num_classes=12)

        x1, y1, x2, y2 = boxes[i]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )

        bounding_boxes.append(rect)

        axes[1].add_patch(rect)
        axes[1].text(
            x1, y1 - 5,
            f'{class_name}: {scores[i]:.2f}',
            color='white', fontsize=8,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )
    
    axes[1].set_title('Bounding Boxes')
    axes[1].axis('off')


    axes[2].imshow(image_array)

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


# %%
CHECKPOINT_PATH = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/checkpoints/cholecseg8k/maskrcnn/best_val_dice.pth'
device = 'cuda'
# img_path = '/home/data/tumai/splatgraph/data/cholecseg8k/video01/video01_00160/frame_174_endo.png'
img_path = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_set_masks/video01/00359.jpg'
model = load_model(checkpoint_path=CHECKPOINT_PATH, device=device)
img_tensor, img_array = preprocess_image(image_path=img_path)
detections = run_inference(model=model, image_tensor=img_tensor)
bounding_boxes(image_array=img_array, detections=detections)

# %%

masks = detections['masks']
height, width = masks.shape[2:]
class_ids = np.zeros((height, width))
for i in range(len(masks)):
    mask = masks[i, 0]
    mask = mask > 0.5
    label = detections['labels'][i].item()
    class_ids[mask] = label



#%%
# Generate instance masks from semantic masks using connected components
# 2d arrays, 0 is background, > 0 are instance ids

NEW_CHOLECSEG8K_CLASSES = {}
new_detections = {
    'boxes': [],
    'labels': [],
    'masks': [],
    'scores': []
}

instance_ids = np.zeros_like(class_ids, dtype=np.int32)
instance_counter = 1
unique_classes = np.unique(class_ids)
unique_classes = unique_classes[unique_classes != 0].astype(int)

for class_id in unique_classes:
    label_name = CHOLECSEG8K_CLASSES[class_id]
    class_binary = (class_ids == class_id).astype(np.uint8)
    num_components, labeled_components = cv2.connectedComponents(
        class_binary, connectivity=8
    )
    
    for component_id in range(1, num_components):
        component_mask = labeled_components == component_id
        component_area = component_mask.sum()
        
        if component_area < 200:
            continue

        instance_ids[component_mask] = instance_counter
        NEW_CHOLECSEG8K_CLASSES[instance_counter] = label_name
        
        # Compute bounding box from the component mask
        # np.where returns (row_indices, col_indices), i.e., (y, x)
        ys, xs = np.where(component_mask)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        
        new_detections['boxes'].append([x1, y1, x2, y2])
        new_detections['labels'].append(class_id)
        new_detections['masks'].append(component_mask.astype(np.float32))
        # We lose the original per-instance score since we merged masks first;
        # you could propagate scores from the original detections if needed
        new_detections['scores'].append(1.0)
        
        instance_counter += 1

# Convert to tensors matching the original format
new_detections['boxes'] = torch.tensor(new_detections['boxes'], dtype=torch.float32)
new_detections['labels'] = torch.tensor(new_detections['labels'], dtype=torch.int64)
new_detections['masks'] = torch.tensor(np.stack(new_detections['masks'])[:, None, :, :], dtype=torch.float32)
new_detections['scores'] = torch.tensor(new_detections['scores'], dtype=torch.float32)



# %%
bounding_boxes(image_array=img_array, detections=new_detections)
# %%
