import os


from ultralytics import YOLO

# Load your custom trained model
model = YOLO("pool_table_detector_best-s.pt")

# Run inference on a new image and save the visual output
count = 0
for image_file in sorted(os.listdir("../datasets/main_dataset/data/train")):
    # Check if the file is an image (you can add more extensions if needed)
    if image_file.endswith((".jpg", ".jpeg", ".png")):
        result = model.predict(f"../datasets/main_dataset/data/train/{image_file}", save=True, project=os.path.abspath("./results"), name="run2", exist_ok=True)
        count += 1
