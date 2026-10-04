import json
from dataclasses import dataclass
from typing import Optional

from action.semantic_action import SemanticActionEvent


@dataclass
class FSMResult:
    """
    Result produced after the FSM receives a semantic action.
    """

    accepted: bool

    action: str

    expected_action: Optional[str]

    previous_state: str

    current_state: str

    message: str

class ExperimentFSM:

    def __init__(self, config_path: str):

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(file)

        self.sequence = config["sequence"]

        self.start_state = config.get(
            "start_state",
            "START"
        )

        self.current_state = self.start_state

        self.current_index = 0

        self.completed_actions = []

    @property
    def expected_action(self) -> Optional[str]:

        if self.current_index >= len(self.sequence):

            return None

        return self.sequence[
            self.current_index
        ]

    def process(
        self,
        event: SemanticActionEvent,
    ) -> FSMResult:

        observed_action = event.action

        previous_state = self.current_state

        expected_action = self.expected_action

        if expected_action is None:

            return FSMResult(

                accepted=False,

                action=observed_action,

                expected_action=None,

                previous_state=previous_state,

                current_state=self.current_state,

                message=(
                    "Experiment already completed."
                ),
            )

        if observed_action == expected_action:

            self.completed_actions.append(
                observed_action
            )

            self.current_index += 1

            self.current_state = observed_action

            if self.expected_action is None:

                message = (
                    "Correct action. "
                    "Experiment completed."
                )

            else:

                message = (
                    "Correct action."
                )

            return FSMResult(

                accepted=True,

                action=observed_action,

                expected_action=expected_action,

                previous_state=previous_state,

                current_state=self.current_state,

                message=message,
            )


        return FSMResult(

            accepted=False,

            action=observed_action,

            expected_action=expected_action,

            previous_state=previous_state,

            current_state=self.current_state,

            message=(
                f"Protocol violation. "
                f"Expected '{expected_action}', "
                f"received '{observed_action}'."
            ),
        )

    def reset(self):

        self.current_state = self.start_state

        self.current_index = 0

        self.completed_actions = []

    def get_status(self):

        return {

            "current_state":
                self.current_state,

            "expected_action":
                self.expected_action,

            "current_index":
                self.current_index,

            "completed_actions":
                self.completed_actions.copy(),

        }