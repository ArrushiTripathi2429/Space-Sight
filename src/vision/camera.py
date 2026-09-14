import cv2
import time


class Camera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.frame_id = 0

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        timestamp = time.time()

        self.frame_id += 1

        return {
            "frame": frame,
            "frame_id": self.frame_id,
            "timestamp": timestamp
        }

    def release(self):
        self.cap.release()