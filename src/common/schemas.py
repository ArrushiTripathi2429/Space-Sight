from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ObjectDetection:

    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    track_id: int


@dataclass
class DetectionData:

    frame_id: int
    timestamp: float
    objects: List[ObjectDetection]


@dataclass
class FrameData:
    frame_id: int
    timestamp_ms: int
    objects: List[Dict[str, Any]]
    pose: Dict[str, Any]
    hands: Dict[str, Any]
    interactions: List[Dict[str, Any]]