import cv2
from dataclasses import asdict

from common.schemas import FrameData
from common.recording import FrameRecorder
from vision.camera import Camera
from vision.detector import ObjectDetector
from vision.tracker import ObjectTracker
from human.hand_pipeline import HandLandmarker
from human.interaction import process_frame
from human.pose_pipeline import PoseLandmarker


def main():

    camera = Camera(camera_id=0)

    detector = ObjectDetector(
        model_path="yolov8n.pt",
        confidence=0.2  
    )

    tracker = ObjectTracker()
    hand_landmarker = HandLandmarker()
    pose_landmarker = PoseLandmarker()

    recorder = FrameRecorder()

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
            timestamp_ms = data["timestamp_ms"]

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

            class_names = {
                detection["class_id"]: detection["class"]
                for detection in detections
            }
            tracked_objects = [
                {
                    **tracked_object,
                    "class": class_names.get(
                        tracked_object["class_id"],
                        "unknown",
                    ),
                }
                for tracked_object in tracked_objects
            ]

            height, width = frame.shape[:2]
            hands = hand_landmarker.detect(frame)
            pose = pose_landmarker.detect(frame)
            interactions = process_frame(
                hands=hands,
                objects=tracked_objects,
                image_width=width,
                image_height=height,
                select_best=False,
            )

            # -------------------------
            # 4. STANDARD OUTPUT
            # -------------------------

            frame_data = FrameData(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                objects=[{
                    "class": obj["class"],
                    "class_id": obj["class_id"],
                    "bbox": obj["bbox"],
                    "confidence": obj["confidence"],
                    "track_id": obj["track_id"]
                } for obj in tracked_objects],
                pose=pose,
                hands=hands,
                interactions=interactions,
            )
            detection_data = asdict(frame_data)

            # -------------------------
            # 5. SAVE
            # -------------------------

            recorder.write(detection_data)

            # -------------------------
            # 6. VISUALIZATION
            # -------------------------

            for obj in frame_data.objects:

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

            for interaction in interactions:
                if interaction["state"] == "NONE":
                    continue

                x, y = interaction["hand_reference_point"].values()
                label = (
                    f'{interaction["hand"]} -> '
                    f'{interaction["object_class"]}: '
                    f'{interaction["state"]}'
                )
                cv2.putText(
                    frame,
                    label,
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
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

        recorder.close()
        hand_landmarker.close()
        pose_landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()