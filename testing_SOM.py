#%%
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
# %%
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
    
    return (max_pos[1], max_pos[0])  

