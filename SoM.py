#%%
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
#%%

def find_label_position(mask, existing_label_regions=None):
    """
    Find the best position for a label within a mask region.
    Uses distance transform to find the point farthest from boundaries.
    """
    if existing_label_regions is not None:
        # Exclude regions already covered by other labels
        mask = mask & ~existing_label_regions
    
    if not mask.any():
        # Fallback to center of original mask if fully occluded
        return None
    
    # Distance transform finds pixels farthest from edges
    distance = ndimage.distance_transform_edt(mask)
    
    # Find the maximum distance point
    max_pos = np.unravel_index(np.argmax(distance), distance.shape)
    
    return (max_pos[1], max_pos[0])  # (x, y) format

def allocate_marks(masks):
    """
    Allocate mark positions for all masks, processing smaller regions first.
    """
    # Sort masks by area (ascending) - smaller regions get priority
    mask_areas = [(i, mask.sum()) for i, mask in enumerate(masks)] #calculate the area of each mask

    sorted_indices = [i for i, _ in sorted(mask_areas, key=lambda x: x[1])] 
    
    positions = {}
    label_radius = 15  # Adjust based on your label size
    occupied = np.zeros_like(masks[0], dtype=bool)
    
    for idx in sorted_indices:
        mask = masks[idx]
        # Exclude already-occupied regions from this mask
        available_mask = mask & ~occupied
        
        pos = find_label_position(available_mask)
        if pos is None:
            pos = find_label_position(mask)  # Fallback
        

        positions[idx] = pos
        # Mark a region around this label as occupied
        y, x = pos[1], pos[0]
        y_min, y_max = max(0, y - label_radius), min(mask.shape[0], y + label_radius)
        x_min, x_max = max(0, x - label_radius), min(mask.shape[1], x + label_radius)
        occupied[y_min:y_max, x_min:x_max] = True
    
    return positions

def create_som_image(original_image, masks, positions, 
                     alpha=0.4, mark_type='number'):
    """
    Overlay masks and labels onto the original image.
    """
    img = original_image.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0)) # A(last channel) controls the opacity of the mask.
    draw = ImageDraw.Draw(overlay)

    # Generate distinct colors for each mask
    colors = generate_colors(len(masks))
    
    # Draw semi-transparent masks
    for i, mask in enumerate(masks):
        color_with_alpha = (*colors[i], int(255 * alpha))
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        colored_mask = Image.new('RGBA', img.size, color_with_alpha)
        overlay.paste(colored_mask, mask=mask_img)
    
    # Composite the overlay
    img = Image.alpha_composite(img, overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for idx, pos in positions.items():
        label = str(idx + 1) if mark_type == 'number' else chr(65 + idx)
        x, y = pos
        
        # Draw background rectangle for readability
        bbox = draw.textbbox((x, y), label, font=font)
        padding = 3
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, 
             bbox[2] + padding, bbox[3] + padding],
            fill=colors[idx]
        )
        
        # Draw text in contrasting color
        text_color = get_contrasting_color(colors[idx])
        draw.text((x, y), label, fill=text_color, font=font)
    
    return img
# %%

# data = np.load('/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_segmentations/video01/00002_binary_mask.npz')

# masks = data['arr']
# masks_list = []
# for mask in masks:
#     masks_list.append(mask)

# x,y = find_label_position(mask=mask)


# %%

img_path = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_set_masks/video01/'
seg_path = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_segmentations/video01/'

data = np.load('/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_segmentations/video01/00002_binary_mask.npz')
masks = data['arr']

for fname in os.listdir(img_path):
    img = Image.open(img_path + fname)

    break
#%%

overlay, draw = create_som_image(original_image=img, masks=masks, positions=1)
plt.imshow(overlay)

# %%
