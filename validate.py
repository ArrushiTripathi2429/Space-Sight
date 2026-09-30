from ultralytics import YOLO


def main():

    model_path = "runs/detect/runs/red_yellow_boxes/weights/best.pt"

    print(f"Loading model: {model_path}")

    model = YOLO(model_path)

    metrics = model.val(
        data="data.yaml",
        imgsz=640,
        device=0
    )

    print("\n========== VALIDATION RESULTS ==========")
    print(f"mAP50:       {metrics.box.map50:.4f}")
    print(f"mAP50-95:    {metrics.box.map:.4f}")

    print("\nPer-class mAP50-95:")

    for class_id, value in enumerate(metrics.box.maps):
        class_name = model.names[class_id]
        print(f"{class_id} ({class_name}): {value:.4f}")


if __name__ == "__main__":
    main()