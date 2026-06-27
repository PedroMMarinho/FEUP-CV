import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
 
mask_steps = {}


def get_table_reference_points(image_rgb):
    img_h, img_w = image_rgb.shape[:2]
    
    blurred = cv2.GaussianBlur(image_rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    
    cx, cy = img_w // 2, img_h // 2

    # 1. DOMINANT COLOR SAMPLING (Histogram Approach)
    # Grab a large central chunk of the image (30% width and height)
    patch_h = int(img_h * 0.15) 
    patch_w = int(img_w * 0.15)
    center_patch = hsv[cy - patch_h : cy + patch_h, cx - patch_w : cx + patch_w]
    
    # Filter out dark shadows or bright glares (Saturation > 30, Value > 30)
    # This ensures we are looking at actual colors, not white balls or black shadows.
    mask_valid_colors = (center_patch[:, :, 1] > 30) & (center_patch[:, :, 2] > 30)
    valid_pixels = center_patch[mask_valid_colors]
    
    # Fallback just in case the center is completely greyscale
    if len(valid_pixels) < 100:
        valid_pixels = center_patch.reshape(-1, 3)
        
    # Calculate a histogram of the HUE channel (0-179 in OpenCV)
    hue_hist, _ = np.histogram(valid_pixels[:, 0], bins=180, range=(0, 180))
    
    # The absolute peak of the histogram is our dominant cloth color!
    peak_hue = int(np.argmax(hue_hist))

    # Get the median saturation and value for pixels that match this dominant hue
    # (Handling wraparound isn't strictly necessary for green/blue tables, but good practice)
    cloth_pixels = valid_pixels[np.abs(valid_pixels[:, 0].astype(int) - peak_hue) <= 12]
    if len(cloth_pixels) > 0:
        s_med = int(np.median(cloth_pixels[:, 1]))
        v_med = int(np.median(cloth_pixels[:, 2]))
    else:
        s_med = int(np.median(valid_pixels[:, 1]))
        v_med = int(np.median(valid_pixels[:, 2]))

    # Create a tight boundary around the dominant cloth color
    lower_bound = (max(0, peak_hue - 12), max(30, s_med - 60), max(30, v_med - 60))
    upper_bound = (min(180, peak_hue + 12), 255, 255)
    
    raw_cloth_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # 2. ISOLATE THE TRUE TABLE BLOB (With Center Verification)
    contours, _ = cv2.findContours(raw_cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Check the top 3 largest blobs. 
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    
    table_contour = None
    for contour in largest_contours:
        # Get the bounding box center of the contour
        x, y, w, h = cv2.boundingRect(contour)
        cnt_cx, cnt_cy = x + w//2, y + h//2
        
        # If the center of this blob is within the middle 50% of the image, it's our table
        if (img_w * 0.25 < cnt_cx < img_w * 0.75) and (img_h * 0.25 < cnt_cy < img_h * 0.75):
            table_contour = contour
            break
            
    # Fallback to the absolute largest if none meet the center criteria
    if table_contour is None:
        table_contour = largest_contours[0]

    # 3. Dilation (Fill Holes)
    working_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(working_mask, [table_contour], -1, 255, cv2.FILLED)
    
    kernel_size = max(25, img_w // 50)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2+1, kernel_size*2+1))
    working_mask = cv2.dilate(working_mask, dilation_kernel, iterations=2)
    
    # 4. Convex Hull (Straighten Edges)
    contours_dilated, _ = cv2.findContours(working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_dilated:
        return None
        
    largest_dilated = max(contours_dilated, key=cv2.contourArea)
    hull = cv2.convexHull(largest_dilated)
    
    # Draw the hull onto a blank image canvas so OpenCV has pixels to manipulate
    hull_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
    
    # 5a. ERODE (Trim and sharpen the edges)
    # Using your original 2% erosion logic to clean the boundary
    erosion_padding = max(4, int(min(img_h, img_w) * 0.020))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_padding*2+1, erosion_padding*2+1))
    eroded_mask = cv2.erode(hull_mask, erosion_kernel, iterations=2)
    
    # 5b. EXPAND 5% TO ALL SIDES (Dilate)
    # Calculate a 5% margin
    padding_pixels = int(min(img_h, img_w) * 0.01)
    padding_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (padding_pixels*2+1, padding_pixels*2+1))
    
    # Dilate the *eroded* mask to push it outward
    expanded_mask = cv2.dilate(eroded_mask, padding_kernel, iterations=1)
    
    # Dilation rounds the corners. We run one final Convex Hull to 
    # snap those expanded edges back into sharp, straight lines.
    expanded_contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_padded_hull = cv2.convexHull(max(expanded_contours, key=cv2.contourArea))
    
    # 6. GET REFERENCE POINTS
    return final_padded_hull.squeeze()

images = Path("../datasets/main_dataset/data/train").glob("*.jpg")
i = 1
for test_img in images:

    test_img = cv2.imread(str(test_img))

    # 1. Get the array of coordinates
    final_points = get_table_reference_points(test_img)

    if final_points is not None:
        print(f"Success! Found {len(final_points)} reference points.")

        # ---------------------------------------------------------
        # OPTION A: Save a picture so you can visually verify it
        # ---------------------------------------------------------
        # Create a copy of the original image to draw on
        visual_img = test_img.copy()
        
        # OpenCV's drawContours expects points in a specific shape: (N, 1, 2)
        points_to_draw = final_points.reshape((-1, 1, 2))
        
        # Draw a bright green line (0, 255, 0) with thickness 3 connecting the points
        cv2.drawContours(visual_img, [points_to_draw], -1, (0, 255, 0), 3)
        
        filename = f"preprocessing/preprocessing_final_mask_{i}.png"
        cv2.imwrite(filename, visual_img)
        # NOW you can use imwrite, because visual_img is an actual image
        print("Saved visual verification image.")

        i += 1
        # ---------------------------------------------------------
        # OPTION B: Save the actual coordinates for later use
        # ---------------------------------------------------------
        # If you need to load these exact [x, y] coordinates in another script, 
        # save them as a NumPy binary file (.npy) or a text file.
        #np.save(f"preprocessing/table_coordinates_{i}.npy", final_points)
        # To load them later in a different script: points = np.load(f"preprocessing/table_coordinates_{i}.npy")
        #print("Saved raw coordinate data.")

    else:
        print("Could not find the table contour.")