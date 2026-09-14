import cv2
import mediapipe as mp
import time
import json
import os

from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


MODEL_PATH = r"models\\hand_landmarker.task"

OUTPUT_DIR = "outputs"

CAMERA_ID = 0

session_history = []


# Maps MediaPipe timestamp -> our internal frame ID.
#
# Example:
# timestamp 1757600000123 -> frame 1
# timestamp 1757600000156 -> frame 2
#
# This is necessary because LIVE_STREAM processing
# is asynchronous.

timestamp_to_frame_id = {}


# Stores the latest MediaPipe result for visualization.

latest_result = None


BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

HandLandmarkerResult = (
    mp.tasks.vision.HandLandmarkerResult
)

VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(
    result: HandLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    """
    Callback executed asynchronously by MediaPipe.

    IMPORTANT:
    Do NOT use the current global frame_id here.

    The callback may execute after several newer frames
    have already been submitted.

    Instead, timestamp_ms is used to recover the correct
    frame ID.
    """

    global latest_result

    latest_result = result

    current_frame_id = timestamp_to_frame_id.get(
        timestamp_ms
    )

    # This should normally never happen.
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

        "hands_detected": False,

        "left_hand": {
            "detected": False,
            "handedness": None,
            "handedness_confidence": None,
            "landmarks": []
        },

        "right_hand": {
            "detected": False,
            "handedness": None,
            "handedness_confidence": None,
            "landmarks": []
        }
    }


    if result.hand_landmarks:

        frame_data["hands_detected"] = True

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):


            handedness_category = (
                result.handedness[hand_index][0]
            )

            handedness = (
                handedness_category.category_name
            )

            handedness_confidence = (
                handedness_category.score
            )


            hand_data = {
                "detected": True,
                "handedness": handedness,
                "handedness_confidence": round(
                    handedness_confidence,
                    4
                ),
                "landmarks": []
            }



            for idx, landmark in enumerate(
                landmarks
            ):

                hand_data["landmarks"].append({
                    "id": idx,
                    "x": round(landmark.x, 4),
                    "y": round(landmark.y, 4),
                    "z": round(landmark.z, 4)
                })


            if handedness.lower() == "left":

                frame_data["left_hand"] = hand_data

            elif handedness.lower() == "right":

                frame_data["right_hand"] = hand_data



    session_history.append(frame_data)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.LIVE_STREAM,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    result_callback=print_result
)


cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)


frame_id = 0


with HandLandmarker.create_from_options(
    options
) as landmarker:

    print(
        "Hand Landmarker is initialized and running."
    )

    print(
        "Press 'q' to stop recording."
    )

    while cap.isOpened():


        success, image = cap.read()

        # Check capture success BEFORE incrementing
        # frame ID.

        if not success:

            print(
                "ERROR: Failed to read frame from camera."
            )

            break


        frame_id += 1



        image = cv2.flip(
            image,
            1
        )


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


        landmarker.detect_async(
            mp_image,
            frame_timestamp_ms
        )


        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )


        if latest_result:

            try:

                annotated_image = image.copy()

                for hand_landmarks in (
                    latest_result.hand_landmarks
                ):

                    annotated_image = (
                        drawing_utils.draw_landmarks(
                            image=annotated_image,
                            landmarks=hand_landmarks,
                            connections=(
                                mp.solutions.hands
                                .HAND_CONNECTIONS
                            ),
                            landmark_drawing_spec=(
                                drawing_styles
                                .get_default_hand_landmarks_style()
                            )
                        )
                    )


                cv2.imshow(
                    "BAS-HAR - Hand Landmarker",
                    annotated_image
                )


            except AttributeError:

                # Fallback if the drawing API differs
                # between MediaPipe versions.

                cv2.imshow(
                    "BAS-HAR - Hand Landmarker",
                    image
                )


        else:

            cv2.imshow(
                "BAS-HAR - Hand Landmarker",
                image
            )


        if cv2.waitKey(1) & 0xFF == ord("q"):

            break



cap.release()

cv2.destroyAllWindows()



if session_history:

    timestamp_str = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    filename = (
        f"{OUTPUT_DIR}/"
        f"hand_session_{timestamp_str}.json"
    )


    print(
        f"\nSaving {len(session_history)} frames "
        f"of hand movement data..."
    )


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
        "No hand data was recorded during this session."
    )