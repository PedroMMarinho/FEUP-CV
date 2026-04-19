import json
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt



json_file = "input.json" 
loaded_images = {}

with open(json_file, 'r') as f:
    data = json.load(f)

image_list = data.get("image_path", []) if isinstance(data, dict) else data

for full_path in image_list:
    img_bgr = cv2.imread(full_path)
    
    if img_bgr is not None:
        clean_name = os.path.basename(full_path)
        
        loaded_images[clean_name] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print(f"Loaded {len(loaded_images)} images.")


def get_table_mask(image_rgb, ball_detection=False):

    img_h, img_w = image_rgb.shape[:2]
    
    # Blur to reduce noise and convert to HSV for color isolation
    blurred = cv2.GaussianBlur(image_rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    
    cx, cy = img_w // 2, img_h // 2

    # 1. Dynamic Color Sampling
    # Define rectangular patch size from image dimensions 
    patch_h = int(img_h * 0.1) 
    patch_w = int(img_w * 0.2)
    
    # Grab the dynamic rectangular patch around the center pixel
    center_patch = hsv[cy - patch_h : cy + patch_h, cx - patch_w : cx + patch_w]
    
    # Calculate the median Hue, Saturation, and Value of the center cloth
    h_med = int(np.median(center_patch[:, :, 0]))
    s_med = int(np.median(center_patch[:, :, 1]))
    v_med = int(np.median(center_patch[:, :, 2]))

    # Create a tight boundary around that specific median color
    lower_bound = (max(0, h_med - 12), max(60, s_med - 50), max(100, v_med - 60))
    upper_bound = (min(180, h_med + 12), 255, 255)
    
    raw_cloth_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    
    # 2. Isolate the True Table Blob
    contours, _ = cv2.findContours(raw_cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    table_contour = None
    
    # Check the 5 largest blobs. Pick the one that covers the center pixel.
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for contour in largest_contours:
        if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
            table_contour = contour
            break
            
    # Fallback: If center is somehow missed (e.g., covered by a ball), pick the largest blob
    if table_contour is None:
        table_contour = max(contours, key=cv2.contourArea)

    isolated_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(isolated_mask, [table_contour], -1, 255, cv2.FILLED)
    

    # 3. Dilation (Fill Holes)
    # Draw the raw contour onto a blank canvas
    working_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(working_mask, [table_contour], -1, 255, cv2.FILLED)
    

    # Grow the white pixels to swallow balls and shadows
    kernel_size = max(25, img_w // 50)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2+1, kernel_size*2+1))
    working_mask = cv2.dilate(working_mask, dilation_kernel, iterations=2)

    
    # 4. Convex Hull (Straighten Edges)
    contours_dilated, _ = cv2.findContours(working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_dilated = max(contours_dilated, key=cv2.contourArea)
    
    
    hull = cv2.convexHull(largest_dilated)
    
    solid_polygon_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(solid_polygon_mask, [hull], 255)

    
    # 5. Erosion (Make edges Sharper by removing a small border)
    if ball_detection:
        erosion_padding = max(4, int(min(img_h, img_w) * 0.025))
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_padding*2+1, erosion_padding*2+1))
        final_table_mask = cv2.erode(solid_polygon_mask, erosion_kernel, iterations=3)
    else:
        erosion_padding = max(4, int(min(img_h, img_w) * 0.020))
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_padding*2+1, erosion_padding*2+1))
        final_table_mask = cv2.erode(solid_polygon_mask, erosion_kernel, iterations=2)
        
    
    return final_table_mask



# Sorts corners using the center point as reference 
def order_points_clockwise(pts):
    cx, cy = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    sorted_pts = pts[np.argsort(angles)]
    
    s = sorted_pts.sum(axis=1)
    tl_index = np.argmin(s)
    return np.roll(sorted_pts, shift=-tl_index, axis=0)

# Line intersection function (returns a point if possible, None if parallel)
def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    
    A1, B1 = vy1, -vx1
    C1 = vy1 * x1 - vx1 * y1
    
    A2, B2 = vy2, -vx2
    C2 = vy2 * x2 - vx2 * y2
    
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return None 
        
    x = (C1 * B2 - C2 * B1) / det
    y = (A1 * C2 - A2 * C1) / det
    return [x, y]

def draw_infinite_line(img, line, color, thickness=2):
    vx, vy, x, y = line
    mult = 3000
    p1 = (int(x - mult * vx), int(y - mult * vy))
    p2 = (int(x + mult * vx), int(y + mult * vy))
    cv2.line(img, p1, p2, color, thickness)


def get_corners_from_mask(table_mask, ball_detection=False):
    # For visualization purposes
    mask_rgb = cv2.cvtColor(table_mask, cv2.COLOR_GRAY2RGB)

    cnts, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
        
    largest_contour = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    
    # Get corner points using approxPolyDP to find 4 corners
    guide_corners = None
    for eps_multiplier in np.linspace(0.005, 0.2, 100):
        epsilon = eps_multiplier * peri
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            guide_corners = approx.reshape(4, 2).astype("float32")
            break
            
    if guide_corners is None: return None

    if ball_detection: return guide_corners
    
    # Order the guideposts: 0:TL, 1:TR, 2:BR, 3:BL
    ordered_guides = order_points_clockwise(guide_corners)
    
    # Image 1: Show guideposts on the mask
    step1_img = mask_rgb.copy()
    cv2.polylines(step1_img, [np.int32(ordered_guides)], True, (255, 255, 0), 2)
    for pt in ordered_guides:
        cv2.circle(step1_img, (int(pt[0]), int(pt[1])), 15, (255, 0, 0), -1)


    pts = largest_contour.reshape(-1, 2)
    edge_points = [[], [], [], []]
    
    # Assign every pixel to one of the 4 edges
    for pt in pts:
        min_dist = float('inf')
        best_edge = -1
        for i in range(4):
            p1 = ordered_guides[i]
            p2 = ordered_guides[(i + 1) % 4]
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0: continue
            t = max(0, min(1, np.dot(pt - p1, p2 - p1) / l2))
            proj = p1 + t * (p2 - p1)
            dist = np.linalg.norm(pt - proj)
            if dist < min_dist:
                min_dist = dist
                best_edge = i
        edge_points[best_edge].append(pt)
    
    # Image 2: Show edge pixels colored by assigned edge
    step2_img = np.zeros_like(mask_rgb)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] 
    for i in range(4):
        for pt in edge_points[i]:
            cv2.circle(step2_img, tuple(pt), 5, colors[i], -1)


    # Fit a line to each edge
    fitted_lines = []
    step3_img = mask_rgb.copy()
    
    for i in range(4):
        ep = np.array(edge_points[i])
        if len(ep) < 10: return None
        
        p1 = ordered_guides[i]
        dists = np.linalg.norm(ep - p1, axis=1)
        ep_sorted = ep[np.argsort(dists)]
        
        trim = int(len(ep_sorted) * 0.15)
        if trim > 0:
            ep_trimmed = ep_sorted[trim:-trim]
        else:
            ep_trimmed = ep_sorted
        for pt in ep_trimmed:
            cv2.circle(step3_img, tuple(pt), 5, (255, 0, 255), -1)
            
        [vx, vy, x, y] = cv2.fitLine(ep_trimmed, cv2.DIST_L2, 0, 0.01, 0.01)
        line = (vx[0], vy[0], x[0], y[0])
        fitted_lines.append(line)
        
        draw_infinite_line(step3_img, line, colors[i], 2)
        
    # Image 3: Show fitted lines on the mask

    # Intersect the 4 lines to get the new corners
    tl = intersect_lines(fitted_lines[0], fitted_lines[3]) # Top intersects Left
    tr = intersect_lines(fitted_lines[0], fitted_lines[1]) # Top intersects Right
    br = intersect_lines(fitted_lines[1], fitted_lines[2]) # Right intersects Bottom
    bl = intersect_lines(fitted_lines[2], fitted_lines[3]) # Bottom intersects Left
    
    if None in (tl, tr, br, bl): return None
    
    step4_img = mask_rgb.copy()

    final_corners = np.array([tl, tr, br, bl], dtype="float32")
    
    for pt in final_corners:
        cv2.circle(step4_img, (int(pt[0]), int(pt[1])), 15, (255, 0, 255), -1)

    return final_corners





# Warps the 4 corners into an horizontal top-down view.
# Returns the warped image AND the transformation matrix (M) needed for backtracking.
def warp_to_top_down(image_rgb, ordered_corners, width=1200, height=800):
    top_edge = np.linalg.norm(ordered_corners[0] - ordered_corners[1])
    left_edge = np.linalg.norm(ordered_corners[0] - ordered_corners[3])
    
    ratio = top_edge / left_edge
    
    if ratio < 1.8:
        # END-VIEW mapping
        dst_pts = np.array([
            [0, 0],                  
            [0, height - 1],         
            [width - 1, height - 1], 
            [width - 1, 0]           
        ], dtype="float32")
    else:
        # SIDE-VIEW mapping
        dst_pts = np.array([
            [0, 0],                  
            [width - 1, 0],          
            [width - 1, height - 1], 
            [0, height - 1]          
        ], dtype="float32")

    # M is the mathematical map from Original -> Flat
    M = cv2.getPerspectiveTransform(ordered_corners, dst_pts)

    warped = cv2.warpPerspective(image_rgb, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    
    return warped, M


# ### 2.4 Save Table Warping Images

# Save warped images
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

print(f"Starting batch process. Saving top-down images to '{output_folder}'...")

success_count = 0
fail_count = 0

homography_matrices = {}

for filename, img_rgb in loaded_images.items():
    test_mask = get_table_mask(img_rgb)
    raw_corners = get_corners_from_mask(test_mask)
    
    if raw_corners is not None:
        ordered_corners = order_points_clockwise(raw_corners)
        
        warped_img, M = warp_to_top_down(img_rgb, ordered_corners)
        
        homography_matrices[filename] = M
        
        warped_bgr = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, warped_bgr)
        
        success_count += 1
    else:
        print(f"Warning: Could not find corners for {filename}")
        fail_count += 1

print(f"Batch complete! Warped {success_count} images. ({fail_count} failed)")
print(f"Successfully saved {len(homography_matrices)} matrices for backtracking.")


BALL_PROFILES = [
    # (num, name, hue_low, hue_high, sat_min, val_min, val_max)    
    (1,  "yellow",       15,  30,  110,  140, 255), 
    (2,  "blue",         98, 112,  160,   90, 255), 
    (3,  "red_low",       0,   8,  120,  110, 220),  
    (3,  "red_high",    165, 180,  120,  110, 220),    
    (4,  "purple",      113, 130,   70,   80, 150),    
    (5,  "orange",        9,  20,  130,  140, 255),    
    (6,  "green",        70,  95,  100,   60, 240),    
    (7,  "maroon",       10,  25,  110,   80, 145),   
    (8,  "black",         0, 180,    0,    0,  75), 
]

STRIPE_OFFSET = 8
WHITE_BALL_ID = 0

def _extract_ball_roi(img_bgr: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """Extract ROI around ball center."""
    x, y, rad = int(round(cx)), int(round(cy)), int(round(r))
    h, w = img_bgr.shape[:2]
    x1, y1 = max(x - rad, 0), max(y - rad, 0)
    x2, y2 = min(x + rad, w),  min(y + rad, h)
    return img_bgr[y1:y2, x1:x2].copy()

def _circular_mask(roi: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """Boolean mask for circular region."""
    rh, rw = roi.shape[:2]
    Y, X = np.ogrid[:rh, :rw]
    rad = int(round(r))
    return (X - rad) ** 2 + (Y - rad) ** 2 <= rad ** 2

def _adaptive_white_threshold(roi_hsv, mask):
    """Compute adaptive white thresholds."""
    ball_pixels = roi_hsv[mask]
    if len(ball_pixels) == 0:
        return 60, 180
    vals = ball_pixels[:, 2]
    ref_val = np.percentile(vals, 70)
    val_thresh = max(120, ref_val * 0.7)
    sat_thresh = 65
    return sat_thresh, val_thresh

def _pixel_stats(roi_bgr: np.ndarray, mask: np.ndarray, debug=False):
    """Analyze ball ROI: color, white ratio, stripe detection."""
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]
    rh, rw = roi_hsv.shape[:2]

    Y, X = np.ogrid[:rh, :rw]
    dist_from_core = np.sqrt((X - rw * 0.5)**2 + (Y - rh * 0.4)**2)
    core_mask = dist_from_core < (rw * 0.25)

    glare_mask = (v_chan > 240) & (s_chan < 30)
    shadow_mask = (v_chan < 50)
    valid_mask = mask & (~glare_mask) & (~shadow_mask)
    
    valid_pixels = roi_hsv[valid_mask]
    if valid_pixels.size == 0:
        if debug:
            return (0.0, 0.0, None, 0.0, False), (None, None)
        return (0.0, 0.0, None, 0.0, False)

    sat_thresh, val_thresh = _adaptive_white_threshold(roi_hsv, valid_mask)
    is_white_1d = (valid_pixels[:, 1] < (sat_thresh + 10)) & (valid_pixels[:, 2] > (val_thresh - 20))
    white_ratio = float(is_white_1d.sum()) / len(valid_pixels)

    stripe_by_spread = False
    temp_white_mask = np.zeros(roi_hsv.shape[:2], dtype=bool)
    temp_white_mask[valid_mask] = is_white_1d
    
    white_coords = np.column_stack(np.where(temp_white_mask))
    if len(white_coords) > 15: 
        y_min, x_min = white_coords.min(axis=0)
        y_max, x_max = white_coords.max(axis=0)
        avg_y = np.mean(white_coords[:, 0])
        is_glare_top = avg_y < (rh * 0.35)
        max_span = max(y_max - y_min, x_max - x_min)
        if max_span > (rh * 0.75) and white_ratio > 0.15 and not is_glare_top:
            stripe_by_spread = True

    core_valid_mask = valid_mask & core_mask
    core_pixels = roi_hsv[core_valid_mask]
    is_white_core = (core_pixels[:, 1] < sat_thresh) & (core_pixels[:, 2] > val_thresh)
    colored_core = core_pixels[~is_white_core]

    if colored_core.size > 10:
        median_hsv = np.median(colored_core, axis=0)
    else:
        colored_general = valid_pixels[~is_white_1d]
        median_hsv = np.median(colored_general, axis=0) if colored_general.size > 0 else None

    colored_ratio = 1.0 - white_ratio
    sat_spread = float(np.percentile(valid_pixels[:, 1], 75) - 
                       np.percentile(valid_pixels[:, 1], 25))
    
    white_mask_2d = (temp_white_mask.astype(np.uint8)) * 255
    stats = (white_ratio, colored_ratio, median_hsv, sat_spread, stripe_by_spread)
    
    if debug:
        return stats, (valid_mask, white_mask_2d)
    return stats

def _score_profile(h, s, v, profile):
    """Score profile for HSV color matching."""
    num, name, hL, hH, sMin, vMin, vMax = profile
    
    if not (s >= sMin and vMin <= v <= vMax):
        return float("inf")

    h_centre = (hL + hH) / 2
    hue_diff = abs(h - h_centre)
    circular_hue_diff = min(hue_diff, 180 - hue_diff)

    if circular_hue_diff > 20:
        return float("inf")

    score = circular_hue_diff + (255 - s) * 0.1
    return score

def _is_stripe(white_ratio, sat_spread, colored_ratio, stripe_by_spread):
    """Detect if ball is striped."""
    return stripe_by_spread

def _best_profile_all(median_hsv, white_ratio, sat_spread, colored_ratio, stripe_by_spread):
    """Rank all ball candidates by score."""
    if median_hsv is None:
        return [(8, float("inf"))]

    h, s, v = map(float, median_hsv)
    results = []

    quality = (0.5 * (s / 255.0) + 0.3 * abs(white_ratio - 0.4) / 0.6 + 
               0.2 * min(sat_spread / 100.0, 1.0))

    detected_stripe = _is_stripe(white_ratio, sat_spread, colored_ratio, stripe_by_spread)

    for profile in BALL_PROFILES:
        base_num = profile[0]

        if base_num == 8:
            score = 0.0 if v < 95 and s < 110 and white_ratio < 0.35 else float("inf")
        else:
            score = _score_profile(h, s, v, profile)

        if score == float("inf"):
            score = 1e6

        if base_num in [0, 8]:
            cand_ids = [base_num]
        else:
            cand_ids = [base_num, base_num + STRIPE_OFFSET]

        for cid in cand_ids:
            is_stripe_candidate = cid > 8
            adjusted_score = score

            if base_num not in [0, 8]:
                if is_stripe_candidate != detected_stripe:
                    adjusted_score += 200

            adjusted_score *= (1.0 - 0.3 * quality)
            results.append((cid, adjusted_score))

    if white_ratio > 0.6 or (v > 180 and s < 50):
        results.append((0, 0.0))

    results.sort(key=lambda x: x[1])
    return results


def identify_balls(all_balls, ball_r, img, ball_mask, store=False, filename="annotated.png"):
    """Identify all detected balls by color."""
    effective_r = ball_r * 0.7
    candidates = []

    for i, ball_pos in enumerate(all_balls):
        cx, cy = ball_pos
        roi = _extract_ball_roi(img, cx, cy, effective_r)
        mask = _circular_mask(roi, cx, cy, effective_r)
        mask_roi = _extract_ball_roi(ball_mask, cx, cy, effective_r)
        valid_mask = mask & (mask_roi > 0)

        stats_tuple, debug_masks = _pixel_stats(roi, valid_mask, debug=True)
        w_rat, c_rat, med_hsv, s_spr, s_spread_bool = stats_tuple        

        ranked = _best_profile_all(med_hsv, w_rat, s_spr, c_rat, s_spread_bool)

        candidates.append({
            "idx": i,
            "pos": ball_pos,
            "ranked": ranked,
            "hsv": med_hsv,
            "white": w_rat,
            "colored": c_rat,
            "sat_spread": s_spr,
            "stripe_spread": s_spread_bool
        })
    
    balls = [None] * 16
    assigned = {}
    used_ids = set()
    all_pairs = []

    for ball in candidates:
        idx = ball["idx"]
        for cid, score in ball["ranked"]:
            all_pairs.append({"idx": idx, "cid": cid, "score": score})

    all_pairs.sort(key=lambda x: x["score"])

    for pair in all_pairs:
        idx = pair["idx"]
        cid = pair["cid"]
        if idx in assigned or cid in used_ids:
            continue
        assigned[idx] = cid
        used_ids.add(cid)
        if len(assigned) == len(candidates):
            break

    id_to_candidate = {cid: idx for idx, cid in assigned.items()}

    for ball in candidates:
        idx = ball["idx"]
        if idx in assigned:
            cid = assigned[idx]
            balls[cid] = ball["pos"]

    balls_positions = balls


    return balls_positions



def remove_pockets_from_mask(balls_raw, ordered_corners, ball_r):
    # Projects ideal pocket positions via homography and removes them from the mask

    top_edge = np.linalg.norm(ordered_corners[0] - ordered_corners[1])
    left_edge = np.linalg.norm(ordered_corners[0] - ordered_corners[3])
    ratio = top_edge / left_edge

    long_side = 2000
    short_side = 1000

    balls_clean = balls_raw.copy()
    pocket_radii = []

    # CASE A: END-VIEW (Deep Perspective) -> ratio < 1.8
    if ratio < 1.8:

        w, h = short_side, long_side

        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

        perfect_pockets = np.array([
            [0, 0], [w, 0], [w, h], [0, h],
            [0, h/2], [w, h/2]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(dst_pts, ordered_corners)

        image_pockets = cv2.perspectiveTransform(
            perfect_pockets.reshape(-1, 1, 2), M
        ).reshape(-1, 2)

        min_y = np.min(image_pockets[:, 1])
        max_y = np.max(image_pockets[:, 1])
        y_range = max(max_y - min_y, 1.0)

        min_mult = 1.3
        max_mult = 2.2

        for px, py in image_pockets:

            t = (py - min_y) / y_range
            t_curved = t ** 2.5

            mult = min_mult + t_curved * (max_mult - min_mult)

            r = int(ball_r * mult)

            pocket_radii.append(r)

            cv2.circle(balls_clean, (int(px), int(py)), r, 0, -1)
            
    # CASE B: SIDE-VIEW / TOP-DOWN (Flat) -> ratio >= 1.8
    else:

        w, h = long_side, short_side

        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

        perfect_pockets = np.array([
            [0, 0], [w, 0], [w, h], [0, h],
            [w/2, 0], [w/2, h]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(dst_pts, ordered_corners)

        image_pockets = cv2.perspectiveTransform(
            perfect_pockets.reshape(-1, 1, 2), M
        ).reshape(-1, 2)

        for px, py in image_pockets:

            r = int(ball_r * 3.2)

            pocket_radii.append(r)

            cv2.circle(balls_clean, (int(px), int(py)), r, 0, -1)

    return balls_clean, image_pockets, pocket_radii


def local_peaks(comp_mask, ball_r, min_frac=0.28):
    # Finds ball centers in merged blobs using distance transform peaks

    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)

    dist_s = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), ball_r * 0.15)

    neigh = max(5, int(ball_r * 0.80))
    if neigh % 2 == 0:
        neigh += 1

    kernel = np.ones((neigh, neigh), np.uint8)
    local_max = (cv2.dilate(dist_s, kernel) == dist_s) & (dist_s > ball_r * min_frac)

    lm = local_max.astype(np.uint8) * 255

    nlm, lm_lab = cv2.connectedComponents(lm)

    centers = []

    for pid in range(1, nlm):

        ys, xs = np.where(lm_lab == pid)

        if len(xs) == 0:
            continue

        peak_val = dist_s[ys, xs].max()

        if peak_val < ball_r * 0.45 or peak_val > ball_r * 1.4:
            continue

        centers.append((float(xs.mean()), float(ys.mean())))

    return centers


def calibrate_ball_radius(balls_raw):
    # Estimates ball radius from circular connected components

    n, labels, stats, _ = cv2.connectedComponentsWithStats(balls_raw)

    radii = []

    for i in range(1, n):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < 500:
            continue

        comp = (labels == i).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            continue

        cnt = cnts[0]

        perim = cv2.arcLength(cnt, True)

        if perim == 0:
            continue

        circ = 4 * np.pi * area / (perim * perim)

        if circ < 0.65:
            continue

        r = np.sqrt(area / np.pi)

        radii.append(r)

    if len(radii) == 0:
        return 38.0

    return float(np.median(radii))

def detect_balls(image_rgb, filename):
    # Pipeline: cloth mask -> pocket removal ->  ball detection 

    img_h, img_w = image_rgb.shape[:2]

    blurred = cv2.GaussianBlur(image_rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    cy, cx = img_h // 2, img_w // 2
    patch = hsv[cy - img_h // 10 : cy + img_h // 10, cx - img_w // 5 : cx + img_w // 5]

    h_med = int(np.median(patch[:, :, 0]))
    s_med = int(np.median(patch[:, :, 1]))
    v_med = int(np.median(patch[:, :, 2]))

    cloth_mask = cv2.inRange(
        hsv,
        (max(0, h_med - 2), max(40, s_med - 60), max(80, v_med - 70)),
        (min(170, h_med + 2), 255, 255)
    )

    cnts, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    table_mask = np.zeros_like(cloth_mask)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        cv2.drawContours(table_mask, [hull], -1, 255, -1)

    balls_raw = cv2.bitwise_and(cv2.bitwise_not(cloth_mask), table_mask)

    ball_r = calibrate_ball_radius(balls_raw)

    table_mask_to_be_used = get_table_mask(image_rgb, True)
    raw_corners = get_corners_from_mask(table_mask_to_be_used, True)

    if raw_corners is not None:
        ordered_corners = order_points_clockwise(raw_corners)
        balls_clean, image_pockets, pocket_radius = remove_pockets_from_mask(
            balls_raw, ordered_corners, ball_r
        )
    else:
        ordered_corners = None
        balls_clean = balls_raw.copy()

    n0, lab0, stats0, cent0 = cv2.connectedComponentsWithStats(balls_clean)

    SINGLE_MAX = np.pi * ball_r ** 2 * 1.8
    all_balls = []

    for i in range(1, n0):
        area = stats0[i, cv2.CC_STAT_AREA]
        w = stats0[i, cv2.CC_STAT_WIDTH]
        h = stats0[i, cv2.CC_STAT_HEIGHT]

        if area < SINGLE_MAX * 0.15:
            continue

        comp = (lab0 == i).astype(np.uint8) * 255

        dist_c = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        dist_max = dist_c.max()
        aspect = max(w, h) / (min(w, h) + 1)

        if aspect > 5 and dist_max < ball_r * 0.35:
            continue  # rail artifact

        if area <= SINGLE_MAX and dist_max > ball_r * 0.25:
            cnts_c, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnts_c:
                perim = cv2.arcLength(cnts_c[0], True)
                circ = 4 * np.pi * area / perim ** 2 if perim > 0 else 0

                if circ > 0.25:
                    all_balls.append((cent0[i][0], cent0[i][1]))
                    continue

        if dist_max > ball_r * 0.28:
            all_balls.extend(local_peaks(comp, ball_r, min_frac=0.28))

    unique = []
    for (x, y) in all_balls:
        if not any(np.hypot(x - ux, y - uy) < ball_r * 0.85 for ux, uy in unique):
            unique.append((x, y))

    all_balls = unique



    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    identified_balls = identify_balls(all_balls, ball_r, image_bgr, balls_clean, True, filename)

    # Annotate and save results

    output = {}
    output["num_balls"] = len(all_balls)
    output["balls"] = []


    for i, ball in enumerate(identified_balls):
        if ball is not None:
            ball_data = {}
            ball_data["number"] = i
            ball_data["x_min"] = ball[0] - ball_r
            ball_data["x_max"] = ball[0] + ball_r
            ball_data["y_min"] = ball[1] - ball_r
            ball_data["y_max"] = ball[1] + ball_r
            output["balls"].append(ball_data)



    return output

data = []

for filename, img_rgb in loaded_images.items():    
    img_data = {}
    img_data["image_path"] = f"development_set/{filename}"
    extra_info = detect_balls(img_rgb, filename)
    img_data.update(extra_info)
    data.append(img_data)

with open("output.json", "w") as f:
    json.dump(data, f, indent=2)




