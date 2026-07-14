import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════
#  1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
INPUT_JSON_PATH  = "input.json"
OUTPUT_JSON_PATH = "output.json"

YOLO_MODEL_PATH  = "pool_table_detector_best.pt"
CLF_MODEL_PATH   = "ConvNeXt_Tiny_best.pth"

NUM_CLASSES = 17
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = (300, 300)  

# ══════════════════════════════════════════════════════════════════════
#  2. MODEL ARCHITECTURE & PREPROCESSING TRANSFORMS
# ══════════════════════════════════════════════════════════════════════
class SquarePad:
    """Pad the shorter side with black so the image becomes square."""
    def __call__(self, img):
        w, h    = img.size
        max_dim = max(w, h)
        pad_l   = (max_dim - w) // 2
        pad_t   = (max_dim - h) // 2
        pad_r   = max_dim - w - pad_l
        pad_b   = max_dim - h - pad_t
        return ImageOps.expand(img, (pad_l, pad_t, pad_r, pad_b), fill=0)

infer_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def _head(in_features):
    """Custom classification head matching your training configuration."""
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )

def build_convnext_tiny():
    """Builds the ConvNeXt-Tiny structure to match the weights file."""
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = _head(in_features)
    return model

# ══════════════════════════════════════════════════════════════════════
#  3. PHASE 1: YOLO PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
def run_yolo_preprocessing(image_paths):
    """Runs YOLO on all images and returns a list of cropped PIL images."""
    print(f"--- PHASE 1: Preprocessing {len(image_paths)} images with YOLO ---")
    
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"CRITICAL: Could not find YOLO weights at {YOLO_MODEL_PATH}")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    processed_data = []

    for img_path in image_paths:
        try:
            result = yolo_model.predict(img_path, verbose=False)
            img_pil = Image.open(img_path).convert('RGB')
            
            # Crop to the best bounding box if a table is found
            if len(result[0].boxes) > 0:
                best_box = max(result[0].boxes, key=lambda box: float(box.conf[0]))
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                img_pil = img_pil.crop((x1, y1, x2, y2))
            
            processed_data.append({
                "image_path": img_path,
                "cropped_img": img_pil 
            })
            
        except Exception as e:
            print(f"[ERROR] YOLO failed to process {img_path}: {e}")
            processed_data.append({
                "image_path": img_path,
                "cropped_img": None
            })

    del yolo_model
    torch.cuda.empty_cache()
    
    return processed_data

# ══════════════════════════════════════════════════════════════════════
#  4. PHASE 2: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════
def run_classification(processed_data):
    """Runs ConvNeXt classification on the preprocessed/cropped images."""
    print(f"\n--- PHASE 2: Classifying {len(processed_data)} preprocessed images ---")
    
    if not os.path.exists(CLF_MODEL_PATH):
        raise FileNotFoundError(f"CRITICAL: Could not find Classifier weights at {CLF_MODEL_PATH}")
    
    clf_model = build_convnext_tiny().to(DEVICE)
    clf_model.load_state_dict(torch.load(CLF_MODEL_PATH, map_location=DEVICE))
    clf_model.eval()

    output_data = []

    for item in processed_data:
        img_path = item["image_path"]
        cropped_img = item["cropped_img"]

        if cropped_img is None:
            output_data.append({"image_path": img_path, "num_balls": 0})
            continue

        try:
            tensor = infer_transform(cropped_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = clf_model(tensor)
                pred_count = torch.argmax(outputs, dim=1).item()
                
            output_data.append({
                "image_path": img_path,
                "num_balls": pred_count
            })
            
        except Exception as e:
            print(f"[ERROR] Classifier failed on {img_path}: {e}")
            output_data.append({"image_path": img_path, "num_balls": 0})

    return output_data

# ══════════════════════════════════════════════════════════════════════
#  5. MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"Reading images from {INPUT_JSON_PATH}...")
    with open(INPUT_JSON_PATH, 'r') as f:
        input_data = json.load(f)
        
    image_paths = input_data.get("image_path", [])
    if not image_paths:
        print("Warning: No images found in the input JSON under the key 'image_path'.")
        return

    # Execute the two phases sequentially
    preprocessed_data = run_yolo_preprocessing(image_paths)
    final_predictions = run_classification(preprocessed_data)

    # Save Output
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_predictions, f, indent=4)
        
    print(f"\nSuccess! Predictions for {len(final_predictions)} images saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()