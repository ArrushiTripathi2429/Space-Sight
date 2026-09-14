import numpy as np
import supervision as sv


class ObjectTracker:

    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections):

        if not detections:
            self.tracker.update_with_detections(
                sv.Detections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    confidence=np.empty(0, dtype=np.float32),
                    class_id=np.empty(0, dtype=np.int32)
                )
            )
            return []

        xyxy = np.array(
            [d["bbox"] for d in detections],
            dtype=np.float32
        )

        confidence = np.array(
            [d["confidence"] for d in detections],
            dtype=np.float32
        )

        class_ids = np.array(
            [d["class_id"] for d in detections],
            dtype=np.int32
        )

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids
        )

        tracked = self.tracker.update_with_detections(
            sv_detections
        )

        output = []

        for i in range(len(tracked)):

            output.append({
                "class_id": int(tracked.class_id[i]),
                "bbox": [
                    int(tracked.xyxy[i][0]),
                    int(tracked.xyxy[i][1]),
                    int(tracked.xyxy[i][2]),
                    int(tracked.xyxy[i][3])
                ],
                "confidence": float(tracked.confidence[i]),
                "track_id": int(tracked.tracker_id[i])
            })

        return output