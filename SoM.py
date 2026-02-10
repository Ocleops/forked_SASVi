#%%
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
#%%

def find_label_position(mask):
    """
    Find the best position for a label within a mask region.
    Uses distance transform to find the point farthest from boundaries.
    """
    
    if not mask.any():# certain masks contain only false values meaning that there is actually
                      # nothing existing in that part of the masks channel. We skip those by 
                      # returning none.
        return None
    
    # Distance transform finds pixels farthest from edges
    distance = ndimage.distance_transform_edt(mask)
    
    max_pos = np.unravel_index(np.argmax(distance), distance.shape) # argmax treats the distance array as an 1D array.
                                                                    # we need to unravel to get the x,y coordinates. 
    
    return (max_pos[1], max_pos[0])  # (x, y) format

def skip_dark_background(original_image, t=15):
    arr = np.array(original_image.convert("RGB"))
    dark = (arr[..., 0] < t) & (arr[..., 1] < t) & (arr[..., 2] < t)
    valid_region = ~dark
    return valid_region

def allocate_marks(masks, original_image=None, label_radius=70, margin=50,
                   darkness_threshold=30):
    """
    Allocate mark positions for all masks, processing smaller regions first.
    Labels are constrained to the valid (non-dark) region of the image.
    """
    img_height, img_width = masks[0].shape
    
    valid_region = skip_dark_background(
        original_image=original_image
    )

    # Pre-occupy invalid regions (dark areas)
    occupied = ~valid_region
    
    # Sort masks by area (ascending) - smaller regions get priority
    mask_areas = [(i, mask.sum()) for i, mask in enumerate(masks)]
    sorted_indices = [i for i, _ in sorted(mask_areas, key=lambda x: x[1])]
    
    positions = {}
    
    i = 0
    for idx in sorted_indices[::-1]:
        mask = masks[idx]
        
        if not mask.any():
            continue
        
        # Find available region: inside the mask AND not occupied
        available_mask = mask & ~occupied
        
        pos = find_label_position(available_mask)
        
        x, y = pos
        # x = max(margin, min(x, img_width - margin))
        # y = max(margin, min(y, img_height - margin))
        pos = (x, y)

        positions[i] = pos
        
        # Mark a region around this label as occupied
        y_min, y_max = max(0, y - label_radius), min(img_height, y + label_radius)
        x_min, x_max = max(0, x - label_radius), min(img_width, x + label_radius)
        occupied[y_min:y_max, x_min:x_max] = True
        i += 1

    
    return positions, ~valid_region

def create_som_image(original_image, masks, positions, occupied, alpha=0.35):
    """
    Overlay masks and labels onto the original image.
    No clipping - masks are drawn as-is.
    """
    img = original_image.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))

    # Generate distinct colors for each mask
    colors = generate_colors(len(masks))
    
    # Draw semi-transparent masks (NO clipping - draw full masks)
    for i, mask in enumerate(masks[1:]):
        color_with_alpha = (*colors[i], int(255 * alpha))
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        
        colored_mask = Image.new('RGBA', img.size, color_with_alpha)
        overlay.paste(colored_mask, mask=mask_img)

    occupied = Image.fromarray((occupied*255).astype(np.uint8), mode='L')
    colored_mask = Image.new('RGBA', img.size, (0, 0, 0, 1))
    overlay.paste(colored_mask, occupied)
    
    # Composite the overlay
    img = Image.alpha_composite(img, overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for idx, pos in positions.items():
        label = str(idx + 1) 
        x, y = pos
        
        bbox = draw.textbbox((x, y), label, font=font)
        padding = 3
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, 
             bbox[2] + padding, bbox[3] + padding],
            fill=colors[idx]
        )
        
        text_color = get_contrasting_color(colors[idx])
        draw.text((x, y), label, fill=text_color, font=font)
    
    return img

def generate_colors(n):
    """Generate n visually distinct colors."""
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9) # arguments: hue is the clor itself, 0.7 is the saturation, 0.9 is the brightness.
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def get_contrasting_color(rgb):
    """Return black or white depending on background brightness."""
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

# %%
img_path = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_set_masks/video01/'
data_path = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_segmentations/video01/'

img_names = []
data_names = []
for img in sorted(os.listdir(img_path)):
    img_names.append(img)

for data in sorted(os.listdir(data_path)):
    if data.endswith(".png"):
        continue
    data_names.append(data)

# %%
save_folder_path = "/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/som_imgs/"
for img_name, data_name in zip(img_names, data_names):
    data_pil = np.load(data_path + data_name)
    image = Image.open(img_path + img_name)

    masks = data_pil["arr"]
    positions, occupied = allocate_marks(masks=masks, original_image=image)
    som_img = create_som_image(original_image=image, masks=masks, positions=positions, occupied=occupied)

    som_img = som_img.convert("RGB")
    # plt.imshow(som_img)
    filename = img_name.removesuffix(".jpg") + ".png"
    som_img.save(save_folder_path + filename)
    print(img_name)

# %%

img = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_set_masks/video01/00232.jpg'
data = '/home/guests/myron_theocharakis/longform-surgery/forked_SASVi/test_segmentations/video01/00232_binary_mask.npz'
data = np.load(data)
masks = data["arr"]
image = Image.open(img)
positions, occupied = allocate_marks(masks=masks, original_image=image)
som_img = create_som_image(original_image=image, masks=masks, positions=positions, occupied=occupied)
plt.imshow(som_img)

# %%
