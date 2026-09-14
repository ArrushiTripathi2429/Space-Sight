import cv2
import json
import os

from vision.camera import Camera
from vision.detector import ObjectDetector
from vision.tracker import ObjectTracker


def main():

    camera = Camera(camera_id=0)

    detector = ObjectDetector(
        model_path="yolov8n.pt",
        confidence=0.2  
    )

    tracker = ObjectTracker()

    os.makedirs("data/processed", exist_ok=True)

    output_file = open(
        "data/processed/detections.jsonl",
        "w"
    )

    try:

        while True:

            # -------------------------
            # 1. CAMERA
            # -------------------------

            data = camera.read()

            if data is None:
                break

            frame = data["frame"]
            frame_id = data["frame_id"]
            timestamp = data["timestamp"]

            # -------------------------
            # 2. YOLO DETECTION
            # -------------------------

            detections = detector.detect(frame)

            # -------------------------
            # 3. BYTE TRACK
            # -------------------------

            tracked_objects = tracker.update(
                detections
            )

            # -------------------------
            # 4. STANDARD OUTPUT
            # -------------------------

            detection_data = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "objects": []
            }

            for obj in tracked_objects:

                class_id = obj["class_id"]

                # Find original detection class
                class_name = "unknown"

                for detection in detections:
                    if detection.get("class_id") == class_id:
                        class_name = detection["class"]
                        break

                detection_data["objects"].append({
                    "class": class_name,
                    "bbox": obj["bbox"],
                    "confidence": obj["confidence"],
                    "track_id": obj["track_id"]
                })

            # -------------------------
            # 5. SAVE
            # -------------------------

            output_file.write(
                json.dumps(detection_data) + "\n"
            )

            # -------------------------
            # 6. VISUALIZATION
            # -------------------------

            for obj in detection_data["objects"]:

                x1, y1, x2, y2 = obj["bbox"]

                label = (
                    f'{obj["class"]} '
                    f'ID:{obj["track_id"]} '
                    f'{obj["confidence"]:.2f}'
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # Frame information
            cv2.putText(
                frame,
                f"Frame: {frame_id}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "BAS-HAR Vision",
                frame
            )

            # ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:

        output_file.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()