from ultralytics import YOLO


def main():

    model = YOLO(
        "models/best.pt"
    )

    results = model.predict(
        source="test_images",
        conf=0.25,
        imgsz=640,
        device=0,
        save=True,
        show=True
    )

    print("\nTesting complete.")


if __name__ == "__main__":
    main()