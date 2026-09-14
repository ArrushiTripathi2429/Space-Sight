from dataclasses import dataclass
from typing import List, Tuple


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