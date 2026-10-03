from dataclasses import dataclass
from typing import Optional
from action.action_recognizer import ActionEvent


@dataclass
class SemanticActionEvent:
    """
    Experiment-level action produced from a physical ActionEvent.
    """

    action: str

    object_class: Optional[str]

    track_id: Optional[int]

    start_frame: int

    end_frame: int

    confidence: float

class SemanticActionMapper:

    def __init__(self):

        # Mapping between physical actions and experiment
        # objects.
        #
        # Example:
        #
        # PICK + red_box
        #       ↓
        # PICK_RED

        self.action_map = {

            ("PICK", "red_box"):
                "PICK_RED",

            ("PICK", "yellow_box"):
                "PICK_YELLOW",

            # These can be enabled once placement logic
            # has been implemented.
            #
            # ("PLACE", "red_box"):
            #     "PLACE_RED",
            #
            # ("PLACE", "yellow_box"):
            #     "PLACE_YELLOW",
        }

    def map_action(
        self,
        action_event: ActionEvent,
    ) -> Optional[SemanticActionEvent]:

        key = (
            action_event.action,
            action_event.object_class,
        )


        semantic_action = self.action_map.get(
            key
        )

        if semantic_action is None:

            return None


        return SemanticActionEvent(

            action=semantic_action,

            object_class=(
                action_event.object_class
            ),

            track_id=(
                action_event.track_id
            ),

            start_frame=(
                action_event.start_frame
            ),

            end_frame=(
                action_event.end_frame
            ),

            confidence=(
                action_event.confidence
            ),
        )

