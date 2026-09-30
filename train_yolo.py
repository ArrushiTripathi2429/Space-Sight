from ultralytics import YOLO


def main():
    # Load pretrained YOLOv8 Nano model
    model = YOLO("models/yolov8n.pt")

    # Train on custom dataset
    results = model.train(
        data="data.yaml",

        # Training
        epochs=80,
        imgsz=640,
        batch=16,

        # GPU
        device=0,

        # Windows stability
        workers=0,

        # Early stopping
        patience=15,

        # Output
        project="runs",
        name="red_yellow_boxes",
        save=True,

        verbose=True
    )

    print("\nTraining complete.")
    print("Best model:")
    print("runs/red_yellow_boxes/weights/best.pt")


if __name__ == "__main__":
    main()