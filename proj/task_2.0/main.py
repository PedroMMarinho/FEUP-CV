import shutil
from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="../datasets/pool-table-cyrrm/data/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        workers=0,
        name="pool_table_model",
        project="runs/detect",
        exist_ok=True,
    )

    best_model_path = results.save_dir / "weights" / "best.pt"
    final_path = "pool_table_detector_best-s.pt"

    shutil.copy(best_model_path, final_path)

    print(f"Best model saved at: {best_model_path}")
    print(f"Copied final model to: {final_path}")

    metrics = YOLO(best_model_path).val(
        data="../datasets/pool-table-cyrrm/data/data.yaml",
        workers=0,
        batch=8,
        device="0",
    )

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()