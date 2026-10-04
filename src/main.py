import cv2
from dataclasses import asdict

from common.schemas import FrameData

from common.recording import FrameRecorder

from vision.camera import Camera

from vision.detector import ObjectDetector

from vision.tracker import ObjectTracker

from human.hand_pipeline import HandLandmarker

from human.pose_pipeline import PoseLandmarker

from human.interaction import process_frame



from action.action_recognizer import (

    ActionRecognizer,

    InteractionFrame,

)



from action.semantic_action import (

    SemanticActionMapper,

)

from protocol.fsm import ExperimentFSM













def main():



    camera = Camera(

        camera_id=0

    )



    detector = ObjectDetector(

        model_path="models/best.pt",

        confidence=0.3

    )



    tracker = ObjectTracker()



    hand_landmarker = HandLandmarker()

    pose_landmarker = PoseLandmarker()



    action_recognizer = ActionRecognizer()



    semantic_mapper = SemanticActionMapper()



    fsm = ExperimentFSM(

        config_path="config/experiment.json"

    )


    recorder = FrameRecorder()



    try:



        while True:



            data = camera.read()



            if data is None:

                break



            frame = data["frame"]



            frame_id = data["frame_id"]



            timestamp_ms = data["timestamp_ms"]





            detections = detector.detect(

                frame

            )



            tracked_objects = tracker.update(

                detections

            )





            # RESTORE CLASS NAMES

            # ByteTrack currently gives us class_id.

            # Map that ID back to the detector's class name.



            class_names = {

                detection["class_id"]:

                detection["class_name"]

                for detection in detections

            }



            tracked_objects = [

                {

                    **tracked_object,



                    "class_name":

                    class_names.get(

                        tracked_object["class_id"],

                        "unknown"

                    ),

                }



                for tracked_object

                in tracked_objects

            ]



            height, width = frame.shape[:2]



            hands = hand_landmarker.detect(

                frame

            )



            pose = pose_landmarker.detect(

                frame

            )



            interactions = process_frame(



                hands=hands,



                objects=tracked_objects,



                image_width=width,



                image_height=height,



                select_best=False,

            )



            action_events = []



            semantic_events = []



            for interaction in interactions:



                if interaction["state"] == "NONE":

                    continue



                object_class = interaction.get(

                    "object_class"

                )



                track_id = interaction.get(

                    "track_id"

                )



                object_bbox = None



                for obj in tracked_objects:



                    if obj["track_id"] == track_id:



                        object_bbox = obj["bbox"]



                        break



                hand_point = interaction.get(

                    "hand_reference_point"

                )



                hand_position = None



                if hand_point is not None:



                    hand_position = (

                        hand_point["x"],

                        hand_point["y"],

                    )



                confidence = interaction.get(

                    "confidence",

                    0.0

                )



                interaction_frame = InteractionFrame(



                    frame_id=frame_id,



                    interaction_state=(

                        interaction["state"]

                    ),



                    object_class=(

                        object_class

                    ),



                    track_id=(

                        track_id

                    ),



                    object_bbox=(

                        object_bbox

                    ),



                    hand_position=(

                        hand_position

                    ),



                    interaction_confidence=(

                        confidence

                    ),

                )



                action_event = (

                    action_recognizer.update(

                        interaction_frame

                    )

                )



                if action_event is not None:



                    action_events.append(

                        action_event

                    )



                    print(

                        "\n[ACTION] "

                        f"{action_event.action} | "

                        f"Object: "

                        f"{action_event.object_class} | "

                        f"Track: "

                        f"{action_event.track_id} | "

                        f"Frames: "

                        f"{action_event.start_frame}-"

                        f"{action_event.end_frame} | "

                        f"Confidence: "

                        f"{action_event.confidence:.2f}"

                    )



                    semantic_event = (

                        semantic_mapper.map_action(

                            action_event

                        )

                    )



                    if semantic_event is not None:



                        semantic_events.append(

                            semantic_event

                        )



                        print(

                            "[SEMANTIC ACTION] "

                            f"{semantic_event.action} | "

                            f"Object: "

                            f"{semantic_event.object_class} | "

                            f"Track: "

                            f"{semantic_event.track_id} | "

                            f"Frames: "

                            f"{semantic_event.start_frame}-"

                            f"{semantic_event.end_frame} | "

                            f"Confidence: "

                            f"{semantic_event.confidence:.2f}"

                        )



            # Process semantic actions through the experiment FSM.
            # The FSM decides whether the observed action is valid
            # for the current protocol state.
            if semantic_events:

                for semantic_event in semantic_events:

                    fsm_result = fsm.process(
                        semantic_event
                    )

                    print(
                        "[FSM] "
                        f"{fsm_result.message} | "
                        f"State: {fsm_result.current_state} | "
                        f"Expected: {fsm_result.expected_action}"
                    )

            frame_data = FrameData(



                frame_id=frame_id,



                timestamp_ms=timestamp_ms,



                objects=[

                    {

                        "class_name":

                        obj["class_name"],



                        "class_id":

                        obj["class_id"],



                        "bbox":

                        obj["bbox"],



                        "confidence":

                        obj["confidence"],



                        "track_id":

                        obj["track_id"]

                    }



                    for obj in tracked_objects

                ],



                pose=pose,



                hands=hands,



                interactions=interactions,

            )



            detection_data = asdict(

                frame_data

            )



            recorder.write(

                detection_data

            )



            for obj in frame_data.objects:



                x1, y1, x2, y2 = obj["bbox"]



                label = (

                    f'{obj["class_name"]} '

                    f'ID:{obj["track_id"]} '

                    f'{obj["confidence"]:.2f}'

                )



                cv2.rectangle(

                    frame,



                    (x1, y1),



                    (x2, y2),



                    (0, 255, 0),



                    2

                )



                cv2.putText(

                    frame,



                    label,



                    (x1, y1 - 10),



                    cv2.FONT_HERSHEY_SIMPLEX,



                    0.5,



                    (0, 255, 0),



                    2

                )



            for interaction in interactions:



                if interaction["state"] == "NONE":

                    continue



                point = interaction[

                    "hand_reference_point"

                ]



                x = point["x"]



                y = point["y"]



                label = (

                    f'{interaction["hand"]} -> '

                    f'{interaction["object_class"]}: '

                    f'{interaction["state"]}'

                )



                cv2.putText(

                    frame,



                    label,



                    (int(x), int(y)),



                    cv2.FONT_HERSHEY_SIMPLEX,



                    0.5,



                    (0, 0, 255),



                    2

                )



            if action_events:



                latest_action = action_events[-1]



                action_text = (

                    f"ACTION: "

                    f"{latest_action.action} "

                    f"{latest_action.object_class}"

                )



                cv2.putText(

                    frame,



                    action_text,



                    (20, 65),



                    cv2.FONT_HERSHEY_SIMPLEX,



                    0.7,



                    (255, 0, 255),



                    2

                )



            if semantic_events:



                latest_semantic = (

                    semantic_events[-1]

                )



                semantic_text = (

                    f"SEMANTIC: "

                    f"{latest_semantic.action}"

                )



                cv2.putText(

                    frame,



                    semantic_text,



                    (20, 95),



                    cv2.FONT_HERSHEY_SIMPLEX,



                    0.7,



                    (255, 255, 0),



                    2

                )



            fsm_status = fsm.get_status()

            cv2.putText(
                frame,
                f"FSM: {fsm_status['current_state']}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Expected: {fsm_status['expected_action']}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Frame: {frame_id}",



                (20, 30),



                cv2.FONT_HERSHEY_SIMPLEX,



                0.7,



                (255, 255, 255),



                2

            )



            cv2.imshow(

                "BAS-HAR Vision",

                frame

            )



            if cv2.waitKey(1) & 0xFF == 27:

                break



    finally:



        recorder.close()



        hand_landmarker.close()



        pose_landmarker.close()



        camera.release()



        cv2.destroyAllWindows()





if __name__ == "__main__":

    main()