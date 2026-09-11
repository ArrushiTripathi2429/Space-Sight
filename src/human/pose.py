import cv2
import mediapipe as mp
import time
import json
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


MODEL_PATH = r"models\pose_landmarker.task"
OUTPUT_DIR = "outputs"

CAMERA_ID = 0


session_history = []

# Maps MediaPipe timestamp -> our internal frame ID.
#
# Example:
# timestamp 1757600000123 -> frame 1
# timestamp 1757600000156 -> frame 2
#
# This is necessary because LIVE_STREAM processing is asynchronous.
timestamp_to_frame_id = {}



BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Stores the latest MediaPipe result for visualization.
latest_result = None


def print_result(
    result: PoseLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    """
    Callback executed asynchronously by MediaPipe.

    Important:
    Do NOT use the current global frame_id here because the
    callback may execute after several newer frames have already
    been submitted.

    Instead, timestamp_ms is used to recover the correct frame ID.
    """

    global latest_result

    latest_result = result


    current_frame_id = timestamp_to_frame_id.get(timestamp_ms)

    # This should normally never happen, but we handle it safely.
    if current_frame_id is None:
        print(
            f"Warning: No frame ID found for timestamp "
            f"{timestamp_ms}"
        )
        return


    frame_data = {
        "frame_id": current_frame_id,
        "timestamp_ms": timestamp_ms,
        "system_time": time.time(),
        "pose_detected": False,
        "landmarks": []
    }


    if result.pose_landmarks:

        frame_data["pose_detected"] = True

        # MediaPipe Pose provides 33 landmarks.
        for idx, landmark in enumerate(result.pose_landmarks[0]):

            frame_data["landmarks"].append({
                "id": idx,
                "x": round(landmark.x, 4),
                "y": round(landmark.y, 4),
                "z": round(landmark.z, 4),
                "visibility": round(landmark.visibility, 4)
            })


    session_history.append(frame_data)


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.LIVE_STREAM,

    result_callback=print_result
)


cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


frame_id = 0


with PoseLandmarker.create_from_options(options) as landmarker:

    print(
        "Pose Landmarker is initialized and running."
    )

    print(
        "Press 'q' to stop recording."
    )

    while cap.isOpened():


        success, image = cap.read()

        # IMPORTANT:
        # Check capture success BEFORE incrementing frame ID.
        if not success:
            print("ERROR: Failed to read frame from camera.")
            break


        frame_id += 1


        image = cv2.flip(image, 1)


        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image
        )

        frame_timestamp_ms = int(
            time.time_ns() / 1_000_000
        )

        timestamp_to_frame_id[
            frame_timestamp_ms
        ] = frame_id

        image.flags.writeable = False


        landmarker.detect_async(
            mp_image,
            frame_timestamp_ms
        )

        image.flags.writeable = True

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        if latest_result:

            try:

                annotated_image = drawing_utils.draw_landmarks(
                    image=image,
                    landmarks=latest_result.pose_landmarks,
                    connections=mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=
                        drawing_styles.get_default_pose_landmarks_style()
                )

                cv2.imshow(
                    "BAS-HAR - Pose Landmarker",
                    annotated_image
                )

            except AttributeError:

                # Fallback if drawing API differs between
                # MediaPipe versions.

                cv2.imshow(
                    "BAS-HAR - Pose Landmarker",
                    image
                )

        else:

            cv2.imshow(
                "BAS-HAR - Pose Landmarker",
                image
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

if session_history:

    # Current date/time
    timestamp_str = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"{OUTPUT_DIR}/"
        f"pose_session_{timestamp_str}.json"
    )

    print(
        f"\nSaving {len(session_history)} frames "
        f"of movement data..."
    )

    # Create output directory if necessary
    import os

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Save JSON
    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            session_history,
            f,
            indent=4
        )

    print(
        f"Session saved successfully: {filename}"
    )

else:

    print(
        "No pose data was recorded during this session."
    )