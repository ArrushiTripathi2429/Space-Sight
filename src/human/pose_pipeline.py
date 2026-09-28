from __future__ import annotations

from typing import Any, Dict

import cv2
import mediapipe as mp


class PoseLandmarker:
    def __init__(self, model_path: str = "models/pose_landmarker.task"):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        )

    def detect(self, frame) -> Dict[str, Any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(image)
        landmarks = []

        if result.pose_landmarks:
            landmarks = [
                {
                    "id": index,
                    "x": round(float(landmark.x), 4),
                    "y": round(float(landmark.y), 4),
                    "z": round(float(landmark.z), 4),
                    "visibility": round(float(landmark.visibility), 4),
                }
                for index, landmark in enumerate(result.pose_landmarks[0])
            ]

        return {
            "pose_detected": bool(landmarks),
            "landmarks": landmarks,
        }

    def close(self) -> None:
        self._landmarker.close()