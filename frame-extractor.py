import cv2
import os


def extract_frames(video_path, output_dir, frame_interval=10, resize=None):
    """
    Extract frames from a video.

    Parameters:
        video_path      : Path to input video
        output_dir      : Directory where frames will be saved
        frame_interval  : Save every Nth frame
        resize          : Tuple (width, height), or None
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save every Nth frame
        if frame_count % frame_interval == 0:

            # Resize if requested
            if resize is not None:
                frame = cv2.resize(frame, resize)

            filename = os.path.join(
                output_dir,
                f"frame_{saved_count:06d}.jpg"
            )

            cv2.imwrite(filename, frame)

            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Video: {video_path}")
    print(f"Total frames: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print("-" * 50)


# -----------------------------
# VIDEO PATHS
# -----------------------------

video1 = "red.mp4"


# -----------------------------
# EXTRACT FRAMES
# -----------------------------

extract_frames(
    video1,
    "dataset/red",
    frame_interval=1,
)