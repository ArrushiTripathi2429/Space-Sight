"""
BAS-HAR hand-object interaction module.

Input:
  MediaPipe Hands output + YOLO/ByteTrack tracked objects.

Output:
  NONE / NEAR / GRASPING interaction records.

This module does not run the camera, MediaPipe, YOLO, ByteTrack,
temporal HAR, or the experiment FSM.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


# we need to calibrate these thresholds for each camera and object detector
DEFAULT_NEAR_DISTANCE_PX = 50.0
DEFAULT_GRASP_DISTANCE_PX = 20.0
DEFAULT_GRASP_IOU = 0.10

DEFAULT_MIN_HAND_CONFIDENCE = 0.50
DEFAULT_MIN_OBJECT_CONFIDENCE = 0.30

DISTANCE_WEIGHT = 0.45
IOU_WEIGHT = 0.35
DETECTION_CONFIDENCE_WEIGHT = 0.20

EPSILON = 1e-8


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def normalized_to_pixel(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float]:
    """Convert MediaPipe normalized coordinates to pixels."""
    return float(x) * image_width, float(y) * image_height


def get_object_class_name(object_data: Dict[str, Any]) -> str:
    """Support both 'class_name' and the current detector/tracker 'class' key."""
    return str(object_data.get("class_name", object_data.get("class", "unknown")))


def get_hand_reference_point(
    hand: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    """Return MediaPipe wrist landmark (landmark id 0), normalized."""
    if not hand or not hand.get("detected", False):
        return None

    for landmark in hand.get("landmarks", []):
        if int(landmark.get("id", -1)) == 0:
            return float(landmark["x"]), float(landmark["y"])

    return None


def get_hand_bbox(
    hand: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> Optional[List[float]]:
    """Build a pixel-space bbox around all detected hand landmarks."""
    if not hand or not hand.get("detected", False):
        return None

    landmarks = hand.get("landmarks", [])
    if not landmarks:
        return None

    points = [
        normalized_to_pixel(
            float(lm["x"]),
            float(lm["y"]),
            image_width,
            image_height,
        )
        for lm in landmarks
    ]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return [
        clamp(min(xs), 0.0, float(image_width)),
        clamp(min(ys), 0.0, float(image_height)),
        clamp(max(xs), 0.0, float(image_width)),
        clamp(max(ys), 0.0, float(image_height)),
    ]


def validate_bbox(
    bbox: Sequence[float],
) -> Optional[Tuple[float, float, float, float]]:
    """Validate and normalize [x1, y1, x2, y2]."""
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def point_to_box_distance(
    point: Tuple[float, float],
    bbox: Sequence[float],
) -> float:
    """Shortest Euclidean distance from a point to a rectangle."""
    box = validate_bbox(bbox)
    if box is None:
        return float("inf")

    px, py = map(float, point)
    x1, y1, x2, y2 = box

    closest_x = clamp(px, x1, x2)
    closest_y = clamp(py, y1, y2)

    return math.hypot(px - closest_x, py - closest_y)


def calculate_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """Calculate intersection-over-union for two bounding boxes."""
    a = validate_bbox(box_a)
    b = validate_bbox(box_b)

    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0.0, ix2 - ix1)
        * max(0.0, iy2 - iy1)
    )

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    if union <= EPSILON:
        return 0.0

    return intersection / union


def calculate_distance_score(
    distance: float,
    near_threshold: float,
) -> float:
    """Convert proximity into a [0, 1] score."""
    if not math.isfinite(distance) or near_threshold <= 0:
        return 0.0

    return clamp(1.0 - distance / near_threshold, 0.0, 1.0)


def calculate_interaction_confidence(
    distance: float,
    iou: float,
    hand_confidence: Optional[float],
    object_confidence: Optional[float],
    near_threshold: float,
) -> float:
    """
    Interpretable geometric confidence score.

    This is a heuristic score, NOT a calibrated probability.
    """
    distance_score = calculate_distance_score(distance, near_threshold)
    iou_score = clamp(float(iou), 0.0, 1.0)

    hand_score = (
        clamp(float(hand_confidence), 0.0, 1.0)
        if hand_confidence is not None
        else 0.0
    )
    object_score = clamp(float(object_confidence), 0.0, 1.0)

    detection_score = hand_score * object_score

    score = (
        DISTANCE_WEIGHT * distance_score
        + IOU_WEIGHT * iou_score
        + DETECTION_CONFIDENCE_WEIGHT * detection_score
    )

    return round(clamp(score, 0.0, 1.0), 4)


def determine_interaction_state(
    distance: float,
    iou: float,
    near_distance: float = DEFAULT_NEAR_DISTANCE_PX,
    grasp_distance: float = DEFAULT_GRASP_DISTANCE_PX,
    grasp_iou: float = DEFAULT_GRASP_IOU,
) -> str:
    """
    First-version geometric interaction classifier.

    NONE:
        hand is sufficiently far from object.

    NEAR:
        hand is close, but grasp evidence is insufficient.

    GRASPING:
        hand is very close and its landmark bbox overlaps the object bbox.

    Note: a single frame cannot prove a physical grasp. Temporal
    persistence will be handled later by the temporal HAR layer.
    """
    if not math.isfinite(distance):
        return "NONE"

    if distance > near_distance:
        return "NONE"

    if distance <= grasp_distance and iou >= grasp_iou:
        return "GRASPING"

    return "NEAR"


def calculate_interaction(
    hand: Dict[str, Any],
    object_data: Dict[str, Any],
    image_width: int,
    image_height: int,
    near_distance: float = DEFAULT_NEAR_DISTANCE_PX,
    grasp_distance: float = DEFAULT_GRASP_DISTANCE_PX,
    grasp_iou: float = DEFAULT_GRASP_IOU,
    min_hand_confidence: float = DEFAULT_MIN_HAND_CONFIDENCE,
    min_object_confidence: float = DEFAULT_MIN_OBJECT_CONFIDENCE,
) -> Optional[Dict[str, Any]]:
    """Calculate the interaction between one hand and one tracked object."""
    if not hand or not hand.get("detected", False):
        return None

    object_bbox = object_data.get("bbox")
    if validate_bbox(object_bbox) is None:
        return None

    hand_confidence = hand.get("handedness_confidence")
    object_confidence = float(object_data.get("confidence", 0.0))

    if hand_confidence is not None:
        hand_confidence = float(hand_confidence)
        if hand_confidence < min_hand_confidence:
            return None

    if object_confidence < min_object_confidence:
        return None

    wrist_normalized = get_hand_reference_point(hand)
    if wrist_normalized is None:
        return None

    wrist_px = normalized_to_pixel(
        wrist_normalized[0],
        wrist_normalized[1],
        image_width,
        image_height,
    )

    hand_bbox = get_hand_bbox(
        hand,
        image_width,
        image_height,
    )
    if hand_bbox is None:
        return None

    distance = point_to_box_distance(wrist_px, object_bbox)
    iou = calculate_iou(hand_bbox, object_bbox)

    state = determine_interaction_state(
        distance=distance,
        iou=iou,
        near_distance=near_distance,
        grasp_distance=grasp_distance,
        grasp_iou=grasp_iou,
    )

    interaction_confidence = calculate_interaction_confidence(
        distance=distance,
        iou=iou,
        hand_confidence=hand_confidence,
        object_confidence=object_confidence,
        near_threshold=near_distance,
    )

    return {
        "hand": hand.get("handedness"),
        "object_class": get_object_class_name(object_data),
        "class_id": object_data.get("class_id"),
        "track_id": object_data.get("track_id"),
        "hand_reference_point": {
            "x": round(wrist_px[0], 2),
            "y": round(wrist_px[1], 2),
        },
        "hand_bbox": [round(v, 2) for v in hand_bbox],
        "object_bbox": [round(float(v), 2) for v in object_bbox],
        "distance_px": round(distance, 2),
        "iou": round(iou, 4),
        "hand_confidence": (
            round(hand_confidence, 4)
            if hand_confidence is not None
            else None
        ),
        "object_confidence": round(object_confidence, 4),
        "state": state,
        "interaction_confidence": interaction_confidence,
    }


def process_interactions(
    hands: Dict[str, Any],
    objects: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    near_distance: float = DEFAULT_NEAR_DISTANCE_PX,
    grasp_distance: float = DEFAULT_GRASP_DISTANCE_PX,
    grasp_iou: float = DEFAULT_GRASP_IOU,
) -> List[Dict[str, Any]]:
    """
    Evaluate every detected hand against every tracked object.

    Returns all valid hand-object pairs. Use select_best_interactions()
    if you want only the strongest object per hand.
    """
    interactions: List[Dict[str, Any]] = []

    if not hands or not objects:
        return interactions

    for hand_name in ("left_hand", "right_hand"):
        hand = hands.get(hand_name)

        if not hand or not hand.get("detected", False):
            continue

        # Preserve the canonical name if handedness is missing.
        hand_for_processing = dict(hand)
        if not hand_for_processing.get("handedness"):
            hand_for_processing["handedness"] = (
                "Left" if hand_name == "left_hand" else "Right"
            )

        for object_data in objects:
            interaction = calculate_interaction(
                hand=hand_for_processing,
                object_data=object_data,
                image_width=image_width,
                image_height=image_height,
                near_distance=near_distance,
                grasp_distance=grasp_distance,
                grasp_iou=grasp_iou,
            )

            if interaction is not None:
                interactions.append(interaction)

    return interactions


def select_best_interactions(
    interactions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep the strongest interaction for each hand."""
    best_by_hand: Dict[str, Dict[str, Any]] = {}

    for interaction in interactions:
        hand = interaction.get("hand")
        if hand is None:
            continue

        current = best_by_hand.get(hand)

        if (
            current is None
            or interaction["interaction_confidence"]
            > current["interaction_confidence"]
        ):
            best_by_hand[hand] = interaction

    return list(best_by_hand.values())


def process_frame(
    hands: Dict[str, Any],
    objects: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    select_best: bool = False,
    near_distance: float = DEFAULT_NEAR_DISTANCE_PX,
    grasp_distance: float = DEFAULT_GRASP_DISTANCE_PX,
    grasp_iou: float = DEFAULT_GRASP_IOU,
) -> List[Dict[str, Any]]:
    """Convenience entry point for main.py."""
    interactions = process_interactions(
        hands=hands,
        objects=objects,
        image_width=image_width,
        image_height=image_height,
        near_distance=near_distance,
        grasp_distance=grasp_distance,
        grasp_iou=grasp_iou,
    )

    if select_best:
        return select_best_interactions(interactions)

    return interactions

