from dataclasses import dataclass
from typing import Optional, Tuple

# Number of consecutive GRASPING frames required before
# considering the interaction stable.
GRASPING_MIN_FRAMES = 3

# Number of consecutive HOLDING frames required before
# considering the object genuinely held.
HOLD_MIN_FRAMES = 5

# Number of consecutive NONE frames required to confirm
# that the object has been released.
RELEASE_MIN_FRAMES = 2

# Minimum object-center movement in pixels required to
# consider the object to have moved.
MOVEMENT_THRESHOLD_PX = 10.0


@dataclass
class InteractionFrame:
    """
    One frame of interaction information.

    This is the input given to the ActionRecognizer.
    """

    frame_id: int

    # Expected values:
    # NONE / NEAR / GRASPING
    interaction_state: str

    object_class: Optional[str] = None

    track_id: Optional[int] = None

    # (x1, y1, x2, y2)
    object_bbox: Optional[
        Tuple[float, float, float, float]
    ] = None

    # (x, y)
    hand_position: Optional[
        Tuple[float, float]
    ] = None

    interaction_confidence: float = 0.0


@dataclass
class ActionEvent:
    """
    Represents a completed physical action.
    """

    action: str

    object_class: Optional[str]

    track_id: Optional[int]

    start_frame: int

    end_frame: int

    confidence: float

class ActionRecognizer:

    def __init__(self):

        self.state = "IDLE"

        self.object_class = None

        self.track_id = None

        self.start_frame = None

        self.grasping_frames = 0

        self.holding_frames = 0

        self.release_frames = 0

        self.previous_object_center = None

        self.object_moved = False

        self.max_confidence = 0.0


    def reset(self):

        self.state = "IDLE"

        self.object_class = None

        self.track_id = None

        self.start_frame = None

        self.grasping_frames = 0

        self.holding_frames = 0

        self.release_frames = 0

        self.previous_object_center = None

        self.object_moved = False

        self.max_confidence = 0.0



    def update(
        self,
        frame: InteractionFrame,
    ) -> Optional[ActionEvent]:


        self._update_object_information(frame)


        self._update_object_movement(
            frame.object_bbox
        )

        self.max_confidence = max(
            self.max_confidence,
            frame.interaction_confidence
        )

        if frame.interaction_state == "NEAR":

            return self._handle_near(frame)

        elif frame.interaction_state == "GRASPING":

            return self._handle_grasping(frame)

        elif frame.interaction_state == "NONE":

            return self._handle_none(frame)

        # Unknown interaction state
        return None


    def _update_object_information(
        self,
        frame: InteractionFrame,
    ):

        # If no object is currently being tracked,
        # initialize the current interaction.

        if self.object_class is None:

            self.object_class = frame.object_class

        if self.track_id is None:

            self.track_id = frame.track_id

        if self.start_frame is None:

            self.start_frame = frame.frame_id


    @staticmethod
    def _get_object_center(
        bbox
    ):

        if bbox is None:

            return None

        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2.0

        center_y = (y1 + y2) / 2.0

        return (
            center_x,
            center_y
        )


    def _update_object_movement(
        self,
        bbox
    ):

        current_center = self._get_object_center(
            bbox
        )

        # No valid bounding box
        if current_center is None:

            return

        # First frame containing the object
        if self.previous_object_center is None:

            self.previous_object_center = current_center

            return

        previous_x, previous_y = (
            self.previous_object_center
        )

        current_x, current_y = current_center

        dx = current_x - previous_x

        dy = current_y - previous_y

        distance = (
            dx ** 2 + dy ** 2
        ) ** 0.5

        # Check whether movement is significant
        if distance >= MOVEMENT_THRESHOLD_PX:

            self.object_moved = True

        self.previous_object_center = current_center


    def _handle_near(
        self,
        frame: InteractionFrame,
    ):

        if self.state == "IDLE":

            self.state = "APPROACHING"

            self.object_class = (
                frame.object_class
            )

            self.track_id = (
                frame.track_id
            )

            self.start_frame = (
                frame.frame_id
            )

        return None



    def _handle_grasping(
        self,
        frame: InteractionFrame,
    ):


        if self.state == "IDLE":

            self.state = "GRASPING"

            self.object_class = (
                frame.object_class
            )

            self.track_id = (
                frame.track_id
            )

            self.start_frame = (
                frame.frame_id
            )

            self.grasping_frames = 1

            return None

        if self.state == "APPROACHING":

            self.state = "GRASPING"

            self.grasping_frames = 1

            return None

        if self.state == "GRASPING":

            self.grasping_frames += 1

            # Stable grasp achieved
            if (
                self.grasping_frames
                >= GRASPING_MIN_FRAMES
            ):

                self.state = "HOLDING"

                self.holding_frames = 1

            return None

        if self.state == "HOLDING":

            self.holding_frames += 1

            if (
                self.holding_frames
                >= HOLD_MIN_FRAMES
                and self.object_moved
            ):

                self.state = "PICKED"

                return ActionEvent(

                    action="PICK",

                    object_class=self.object_class,

                    track_id=self.track_id,

                    start_frame=self.start_frame,

                    end_frame=frame.frame_id,

                    confidence=self.max_confidence,
                )

            return None

        if self.state == "PICKED":

            return None

        return None


    def _handle_none(
        self,
        frame: InteractionFrame,
    ):


        if self.state in (
            "HOLDING",
            "PICKED",
        ):

            self.release_frames += 1

            # Confirm release only after consecutive NONE
            # frames.
            if (
                self.release_frames
                >= RELEASE_MIN_FRAMES
            ):

                event = ActionEvent(

                    action="RELEASE",

                    object_class=self.object_class,

                    track_id=self.track_id,

                    start_frame=self.start_frame,

                    end_frame=frame.frame_id,

                    confidence=self.max_confidence,
                )

                self.reset()

                return event

            return None


        if self.state in (
            "APPROACHING",
            "GRASPING",
        ):

            self.reset()

            return None

        return None