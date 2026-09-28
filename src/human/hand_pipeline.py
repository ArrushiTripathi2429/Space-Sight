from __future__ import annotations

from typing import Any, Dict

import cv2
import mediapipe as mp


class HandLandmarker:
    def __init__(self, model_path: str = "models/hand_landmarker.task"):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )

    def detect(self, frame) -> Dict[str, Any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(image)
        hands = {
            "hands_detected": False,
            "left_hand": self._empty_hand(),
            "right_hand": self._empty_hand(),
        }

        for hand_index, landmarks in enumerate(result.hand_landmarks):
            category = result.handedness[hand_index][0]
            handedness = category.category_name or "Unknown"
            hand_data = {
                "detected": True,
                "handedness": handedness,
                "handedness_confidence": round(float(category.score), 4),
                "landmarks": [
                    {
                        "id": index,
                        "x": round(float(landmark.x), 4),
                        "y": round(float(landmark.y), 4),
                        "z": round(float(landmark.z), 4),
                    }
                    for index, landmark in enumerate(landmarks)
                ],
            }
            key = "left_hand" if handedness.lower() == "left" else "right_hand"
            hands[key] = hand_data
            hands["hands_detected"] = True

        return hands

    @staticmethod
    def _empty_hand() -> Dict[str, Any]:
        return {
            "detected": False,
            "handedness": None,
            "handedness_confidence": None,
            "landmarks": [],
        }

    def close(self) -> None:
        self._landmarker.close()