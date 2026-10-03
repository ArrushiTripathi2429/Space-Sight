from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from common.recording import FrameRecorder
from common.schemas import FrameData
from human.hand_pipeline import HandLandmarker
from human.interaction import process_frame
from human.pose_pipeline import PoseLandmarker
from vision.detector import ObjectDetector
from vision.tracker import ObjectTracker


def _draw_landmarks(frame: Any, landmarks: list[dict[str, Any]], color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    for landmark in landmarks:
        cv2.circle(
            frame,
            (int(landmark["x"] * width), int(landmark["y"] * height)),
            3,
            color,
            -1,
        )


class VisionVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.detector = ObjectDetector(model_path=str(ROOT / "models" / "best.pt"), confidence=0.3)
        self.tracker = ObjectTracker()
        self.hand_landmarker = HandLandmarker(model_path=str(ROOT / "models" / "hand_landmarker.task"))
        self.pose_landmarker = PoseLandmarker(model_path=str(ROOT / "models" / "pose_landmarker.task"))
        self.recorder = FrameRecorder(root=str(ROOT / "data" / "processed"))
        self.frame_id = 0
        self.stats: dict[str, Any] = {
            "frame_id": 0,
            "objects": [],
            "hands_detected": False,
            "pose_detected": False,
            "interactions": [],
            "session_dir": str(self.recorder.session_dir),
        }
        self._stats_lock = Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_id += 1

        detections = self.detector.detect(image)
        tracked_objects = self.tracker.update(detections)
        class_names = {item["class_id"]: item["class_name"] for item in detections}
        tracked_objects = [
            {**item, "class_name": class_names.get(item["class_id"], "unknown")}
            for item in tracked_objects
        ]

        height, width = image.shape[:2]
        hands = self.hand_landmarker.detect(image)
        pose = self.pose_landmarker.detect(image)
        interactions = process_frame(
            hands=hands,
            objects=tracked_objects,
            image_width=width,
            image_height=height,
            select_best=False,
        )
        frame_data = FrameData(
            frame_id=self.frame_id,
            timestamp_ms=time.time_ns() // 1_000_000,
            objects=tracked_objects,
            pose=pose,
            hands=hands,
            interactions=interactions,
        )
        self.recorder.write(asdict(frame_data))
        self._annotate(image, frame_data)

        with self._stats_lock:
            self.stats = {
                "frame_id": self.frame_id,
                "objects": tracked_objects,
                "hands_detected": hands["hands_detected"],
                "pose_detected": pose["pose_detected"],
                "interactions": interactions,
                "session_dir": str(self.recorder.session_dir),
            }
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    @staticmethod
    def _annotate(image: Any, frame_data: FrameData) -> None:
        for obj in frame_data.objects:
            x1, y1, x2, y2 = obj["bbox"]
            label = f'{obj["class_name"]} ID:{obj["track_id"]} {obj["confidence"]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 90), 2)
            cv2.putText(image, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 90), 2)

        for hand_name in ("left_hand", "right_hand"):
            hand = frame_data.hands[hand_name]
            if hand["detected"]:
                _draw_landmarks(image, hand["landmarks"], (255, 180, 0))
        _draw_landmarks(image, frame_data.pose["landmarks"], (220, 80, 255))

        for interaction in frame_data.interactions:
            if interaction["state"] == "NONE":
                continue
            point = interaction["hand_reference_point"]
            text = f'{interaction["hand"]} -> {interaction["object_class"]}: {interaction["state"]}'
            cv2.putText(image, text, (int(point["x"]), int(point["y"])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 2)

        cv2.putText(image, f"Frame: {frame_data.frame_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def get_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return dict(self.stats)

    def __del__(self) -> None:
        for resource in (getattr(self, "hand_landmarker", None), getattr(self, "pose_landmarker", None)):
            if resource is not None:
                resource.close()
        recorder = getattr(self, "recorder", None)
        if recorder is not None:
            recorder.close()


st.set_page_config(page_title="Space Sight", page_icon=":camera:", layout="wide")
st.title("Space Sight")
st.caption("Live perception dashboard")

with st.sidebar:
    st.subheader("Pipeline")
    st.write("YOLO + ByteTrack")
    st.write("MediaPipe hands + pose")
    st.write("Hand-object interactions")
    st.info("Allow camera access when prompted, then click START.")

context = webrtc_streamer(
    key="space-sight-live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VisionVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

@st.fragment(run_every=1)
def render_stats() -> None:
    if not context.video_processor:
        st.info("The processed live stream and detection overlays will appear here after the camera starts.")
        return

    stats = context.video_processor.get_stats()
    columns = st.columns(4)
    columns[0].metric("Frame", stats["frame_id"])
    columns[1].metric("Objects", len(stats["objects"]))
    columns[2].metric("Hands", "Detected" if stats["hands_detected"] else "Not detected")
    columns[3].metric("Pose", "Detected" if stats["pose_detected"] else "Not detected")
    st.caption(f"Recording: {stats['session_dir']}")
    active_interactions = [item for item in stats["interactions"] if item["state"] != "NONE"]
    if active_interactions:
        st.subheader("Active interactions")
        st.json(active_interactions)


render_stats()