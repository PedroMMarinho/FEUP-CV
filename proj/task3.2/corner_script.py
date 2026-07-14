# %% [markdown]
# # Imports

# %%
from roboflow import Roboflow
import os
from pathlib import Path
from ultralytics import YOLO
import shutil
import cv2
import numpy as np
import torch

# %%
dataset_path = Path(".")
current_dir = Path.cwd()

# %% [markdown]
# # Dataset Loading

# %%
load_data = False
train = False
test = True

# %%
if load_data:
    os.chdir(dataset_path)

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

    workspace_name = "arena-qye1f"
    project_name = "billiard-7sno1"
    project = rf.workspace(workspace_name).project(project_name)
    versions = project.versions()

    latest_version = max(versions, key=lambda v: int(v.version))

    print(f"  -> Downloading version {latest_version.version}...")
    latest_version.download("yolov8")  # Download in YOLOv8 format

    os.chdir(current_dir)

# %% [markdown]
# # Train

# %%
# Load segmentation model (NOT pose)
if train:
    model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data="billiard-9/data.yaml",  # your dataset yaml
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        workers=0,
        name="pool_table_seg2_model",
        project="runs/seg",
        exist_ok=True,
    )

    best_model_path = results.save_dir / "weights" / "best.pt"
    final_path = "pool_table_seg2_best.pt"

    shutil.copy(best_model_path, final_path)

    print(f"Best model saved at: {best_model_path}")
    print(f"Copied final model to: {final_path}")

    metrics = YOLO(best_model_path).val(
        data="data/billiard-9/data.yaml",
        device="0",
        batch=8,
        workers=0,
    )
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50 (seg): {metrics.seg.map50:.4f}")
    print(f"mAP50-95 (seg): {metrics.seg.map:.4f}")

# %% [markdown]
# # Test on main_dataset

# %%
model = YOLO("pool_table_seg2_best.pt")

# %% [markdown]
# ## Perform homography

# %%
def order_corners_long_side_horizontal(points):
    pts = np.array(points, dtype=np.float32)

    # Get convex hull in correct circular order
    hull = cv2.convexHull(pts).reshape(-1, 2)

    if len(hull) != 4:
        rect = cv2.minAreaRect(pts)
        hull = cv2.boxPoints(rect).astype(np.float32)

    # Sort corners around center
    center = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
    ordered = hull[np.argsort(angles)]

    # Compute edge lengths
    edges = []
    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        edges.append(length)

    # Find a long edge
    long_edge_idx = int(np.argmax(edges))

    # Rotate points so the first edge is a long edge
    ordered = np.roll(ordered, -long_edge_idx, axis=0)

    # Now ordered[0] -> ordered[1] is a long side
    # Decide which long side should be top
    if ordered[0][1] > ordered[2][1]:
        ordered = np.roll(ordered, 2, axis=0)

    # Ensure order is TL, TR, BR, BL
    left_two = ordered[np.argsort(ordered[:, 0])[:2]]
    right_two = ordered[np.argsort(ordered[:, 0])[2:]]

    tl = left_two[np.argmin(left_two[:, 1])]
    bl = left_two[np.argmax(left_two[:, 1])]
    tr = right_two[np.argmin(right_two[:, 1])]
    br = right_two[np.argmax(right_two[:, 1])]

    return np.array([tl, tr, br, bl], dtype=np.float32)

def homography(img, kpts, out_w=800, out_h=400):
    src_pts = order_corners_long_side_horizontal(kpts)

    dst_pts = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, H, (out_w, out_h))

    return warped_img, H


# %% [markdown]
# ## Apply model to images

# %%
def get_corners_from_polygon(polygon):
    # polygon: (N,2)

    hull = cv2.convexHull(polygon.astype(np.float32))

    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # If not exactly 4 points, force fallback
    if len(approx) != 4:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)

    return approx.reshape(4, 2).astype(np.float32)

# %%
def get_corners_fast(polygon):
    rect = cv2.minAreaRect(polygon.astype(np.float32))
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)

# %%
target_images = "../datasets/main_dataset/data/train"
output_dir = "./results2/run1"
mid_dir = "./results2"

os.makedirs(output_dir, exist_ok=True)

count = 0

def draw_keypoints(img, kpts):
    vis = img.copy()

    for i, (x, y) in enumerate(kpts):
        x, y = int(x), int(y)

        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            vis,
            str(i),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return vis

if test:
    for image_file in sorted(os.listdir(target_images)):

        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(target_images, image_file)
        img = cv2.imread(img_path)

        results = model.predict(
            img_path,
            conf=0.10,
            verbose=False,
            save=True,
            project=os.path.abspath(mid_dir),
            name="mid",
            exist_ok=True
        )

        result = results[0]

        if result.masks is None:
            print(result)
            print(f"No table detected: {image_file}")
            continue

        polygon = result.masks.xy[0]
        kpts = get_corners_from_polygon(polygon)

        seg_img = result.plot()
        seg_keypoints_img = draw_keypoints(seg_img, kpts)

        os.makedirs(os.path.join(mid_dir, "keypoints"), exist_ok=True)

        cv2.imwrite(
            os.path.join(mid_dir, "keypoints", f"seg_keypoints_{image_file}"),
            seg_keypoints_img
        )

        save_img, H = homography(img, kpts)

        cv2.imwrite(
            os.path.join(output_dir, f"homography_{image_file}"),
            save_img
        )

        count += 1

    print(f"Processed {count} images.")

# %%
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


