 # BAS HAR: AI Human Activity Recognition for Onboard Experiments

 Final architecture, end-to-end pipeline, and prototype plan for an offline, experiment-aware Human Activity Recognition (HAR) co-pilot.

 ## 1. Overview

 The system observes an astronaut performing a predefined experiment, recognizes actions and human-object interactions, reasons over their temporal order, detects protocol violations, provides corrective guidance, and maintains a lightweight digital record.

 The final sensing definition is:

 - Fixed RGB camera for semantic visual perception.
 - Wi-Fi Channel State Information (CSI) for complementary motion and context evidence.
 - 3D Human Mesh Recovery (3D HMR) as part of the vision representation.

 No separate radar, mmWave, point, or other sensing subsystem is used.

 ## 2. Canonical Workflow

 ```text
 Fixed Camera
	 -> OpenCV
	 -> YOLOv8-nano + ByteTrack
	 -> MediaPipe Pose + Hands
	 -> Hand-Object Interaction
	 -> 3D HMR (ROMP / PARE)
	 -> Vision/HMR Features

 Wi-Fi CSI
	 -> Capture
	 -> Processing and Filtering
	 -> 1D-CNN
	 -> 32-D CSI Embedding

 Vision/HMR Features + CSI Embedding
	 -> Confidence-Aware Feature Fusion
	 -> 30-Frame Temporal Window
	 -> 2-Layer LSTM
	 -> Debounce
	 -> Experiment FSM
	 -> Async Event Bus
	 -> Voice / Structured Log / Video + RTSP / Dashboard
 ```

 Camera/HMR and CSI remain independent until feature fusion. Object identity and fine-grained object semantics remain primarily visual responsibilities.

 ## 3. Problem and Representative Protocol

 ### Core challenges

 - Microgravity removes reliable floor, up, and down assumptions.
 - A single frame is ambiguous; activity recognition requires temporal context.
 - ML perception and deterministic protocol validation must remain separate.
 - CSI can provide useful motion and presence evidence when vision is uncertain, but does not inherently identify the manipulated object.
 - The real-time loop must operate locally without continuous Earth communication.

 ### Example experiment

 ```text
 OPEN_CONTAINER
	 -> PICK_RED
	 -> PLACE_RED
	 -> PICK_YELLOW
	 -> PLACE_YELLOW
	 -> CLOSE_CONTAINER
	 -> DONE
 ```

 If `PICK_YELLOW` occurs while `PLACE_RED` is expected, the FSM rejects the transition and the system provides corrective guidance.

 ## 4. Architecture Components

 | Stage | Technology | Primary output |
 | --- | --- | --- |
 | Camera acquisition | Fixed RGB camera, OpenCV | Frames |
 | Object perception | YOLOv8-nano, ByteTrack | Classes, boxes, confidence, track IDs |
 | Human perception | MediaPipe Pose and Hands | Body, wrist, and hand landmarks |
 | Interaction | IoU, keypoint-to-box distance, proximity geometry | `NONE`, `NEAR`, `GRASPING` |
 | 3D human representation | ROMP or PARE | 3D pose, joints, and mesh parameters |
 | CSI acquisition | Nexmon CSI or equivalent | CSI stream |
 | CSI processing | Filtering, normalization, PyTorch 1D-CNN | 32-D CSI embedding |
 | Fusion | Confidence-aware multimodal fusion | Fused representation |
 | Temporal HAR | PyTorch 2.x, 2-layer LSTM | Activity step and confidence |
 | Stabilization | Confidence threshold and debounce | Confirmed step |
 | Protocol validation | JSON-defined FSM | Valid transition or violation |
 | Event distribution | Async `asyncio` event bus | Non-blocking events |
 | Outputs | `pyttsx3`, JSON/CSV/TXT, FFmpeg/RTSP, FastAPI/WebSocket | Guidance, logs, video, dashboard |

 ## 5. End-to-End Processing

 1. **Capture:** A fixed camera continuously writes frames to a bounded queue.
 2. **Vision perception:** YOLOv8-nano detects experiment objects, ByteTrack maintains identities, and MediaPipe estimates body and hand landmarks.
 3. **Hand-object interaction:** Hand-to-object geometry converts detections and landmarks into `NONE`, `NEAR`, or `GRASPING` states.
 4. **3D HMR:** ROMP or PARE processes RGB frames into 3D pose and mesh features. These become part of the vision representation.
 5. **CSI capture:** A CSI-capable Wi-Fi node acquires CSI independently at approximately 100 Hz.
 6. **CSI processing:** CSI is timestamped, buffered, normalized, filtered, and encoded into a 32-D embedding.
 7. **Feature fusion:** Vision/HMR features and CSI embeddings are fused with confidence awareness.
 8. **Temporal HAR:** A 30-frame sliding window is passed to the LSTM to infer a candidate experiment step and confidence.
 9. **Debounce:** A step is confirmed only after stable evidence. Transient predictions do not advance the protocol.
 10. **FSM validation:** The confirmed action is checked against the expected action in the JSON-defined experiment state machine.
 11. **Event dispatch:** An asynchronous event bus distributes results without blocking inference.
 12. **Response:** Valid progression triggers next-step guidance; violations trigger corrective voice alerts, while logs, dashboard, and video are updated.

 ## 6. Vision Perception

 ### Object detection and tracking

 - **Detector:** YOLOv8-nano using the Ultralytics framework.
 - **Training:** Fine-tune on images of the actual experiment tools and objects.
 - **Tracking:** ByteTrack.
 - **Output:** Class, bounding box, confidence, track ID, presence, and position.
 - **Optimization:** Validate functionally first, then consider ONNX and INT8 deployment.

 ### Pose and hands

 - **Body:** MediaPipe Pose.
 - **Hands:** MediaPipe Hands, or MediaPipe Holistic as an alternative.
 - **Output:** Body landmarks, wrist keypoints, and 21 landmarks per hand.
 - **Base hand representation:** 126-D coordinate representation.

 ### Hand-object interaction

 The interaction layer connects human pose with object semantics:

 | Signal | Use |
 | --- | --- |
 | Bounding-box IoU | Object overlap and association |
 | Keypoint-to-box distance | Hand-object association |
 | Hand-object distance | Proximity reasoning |
 | Combined geometry | `NONE`, `NEAR`, or `GRASPING` state |

 ## 7. 3D Human Mesh Recovery

 3D HMR is included in the final vision architecture.

 | Property | Design |
 | --- | --- |
 | Candidate frameworks | ROMP or PARE |
 | Input | RGB camera frames |
 | Output | 3D body pose, 3D joints, and human mesh parameters |
 | Reference coordinates | Payload-rack-relative coordinates |
 | Position in pipeline | After hand-object interaction, before feature fusion |
 | Purpose | Orientation-aware 3D human reasoning |
 | Deployment | GPU-oriented, Jetson-class target |
 | Reference datasets | AGORA and 3DPW |

 The supplied design specifies a 147-D base vision representation and a 32-D CSI embedding. Since it does not specify the HMR projection size, the final fused feature dimension is:

 ```text
 147 + 32 + D_HMR = 179 + D_HMR
 ```

 `D_HMR` must be selected and documented during implementation. The LSTM input size must use the same final dimension. The earlier 179-D value is valid only for the base camera/interaction plus CSI representation without HMR features.

 ## 8. CSI Complementary Sensing

 CSI means Wi-Fi Channel State Information and is the only complementary sensing modality in this design.

 | Stage | Specification |
 | --- | --- |
 | Acquisition | Nexmon CSI or another CSI-capable Wi-Fi setup |
 | Node | Raspberry Pi 5 or equivalent CSI-capable node |
 | Target sampling | Approximately 100 Hz |
 | Transport | UDP |
 | Buffer | Rolling CSI buffer |
 | Synchronization | Camera and CSI timestamp alignment |
 | Processing | Feature extraction, normalization, and filtering |
 | Reference filter | 0.5-10 Hz |
 | Encoder | PyTorch 1D-CNN |
 | Embedding | 32-D CSI embedding |

 CSI supplies movement, presence, and motion-pattern evidence. It improves robustness when visual information is unreliable, but fine-grained object semantics remain primarily visual.

 ## 9. Feature Representation and Fusion

 ### Base representation

 | Feature | Dimension |
 | --- | ---: |
 | Object presence | 4 |
 | Object positions | 6 |
 | Wrist keypoints | 6 |
 | Hand landmarks | 126 |
 | Interaction state | 5 |
 | Base vision representation | 147 |
 | CSI embedding | 32 |
 | Base camera + CSI representation | 179 |
 | HMR projection | `D_HMR` |
 | Final fused representation | `179 + D_HMR` |

 ### Confidence behavior

 | Condition | System behavior |
 | --- | --- |
 | Vision, HMR, and CSI reliable | Use normal fused inference |
 | Visual uncertainty | Use CSI as complementary motion/context evidence |
 | CSI unavailable | Use camera + HMR degraded mode |
 | Both uncertain | Hold or abstain; do not advance the FSM |
 | Object semantics required | Prefer vision-derived evidence |

 Fusion occurs only after independent camera/HMR and CSI processing.

 ## 10. Temporal HAR and Stabilization

 | Parameter | Reference design |
 | --- | --- |
 | Framework | PyTorch 2.x |
 | Model | 2-layer LSTM |
 | Hidden size | 256 |
 | Sequence length | 30 frames |
 | Input size | `179 + D_HMR` |
 | Temporal context | Approximately 1 second at 30 FPS |
 | Classification head | 256 -> 128 -> step/confidence |
 | Activation | ReLU |
 | Dropout | 0.3 |

 The temporal model distinguishes similar instantaneous poses by their trajectories over time.

 Stabilization uses a confidence threshold above `0.75`, enabled temporal debounce, and a 15-frame stable-confirmation reference. Low-confidence predictions are held or abstained from, and transition gating is required.

 ## 11. Experiment FSM

 The ML model proposes an observed action. The deterministic JSON-defined FSM decides whether that action is legal in the current protocol state.

 | Current state | Expected action | Next state |
 | --- | --- | --- |
 | `START` | `OPEN_CONTAINER` | `OPEN_CONTAINER` |
 | `OPEN_CONTAINER` | `PICK_RED` | `PICK_RED` |
 | `PICK_RED` | `PLACE_RED` | `PLACE_RED` |
 | `PLACE_RED` | `PICK_YELLOW` | `PICK_YELLOW` |
 | `PICK_YELLOW` | `PLACE_YELLOW` | `PLACE_YELLOW` |
 | `PLACE_YELLOW` | `CLOSE_CONTAINER` | `CLOSE_CONTAINER` |
 | `CLOSE_CONTAINER` | `DONE` | `DONE` |

 | Event | Response |
 | --- | --- |
 | `STEP_OK` | Advance and announce the next step |
 | `STEP_SKIPPED` | Violation and corrective voice guidance |
 | `OUT_OF_ORDER` | Violation and corrective voice guidance |
 | `INVALID_TRANSITION` | Reject transition |
 | `LOW_CONFIDENCE` | Hold current state |
 | `EXPERIMENT_DONE` | Close session and write final log |

 ## 12. Event Bus and Outputs

 The event bus uses asynchronous `asyncio` dispatch so output consumers do not block the inference loop.

 | Consumer | Technology or responsibility |
 | --- | --- |
 | Voice guidance | `pyttsx3`, offline TTS |
 | Structured logging | JSON, CSV, or TXT |
 | Local video | OpenCV `VideoWriter`, MP4 |
 | Streaming | FFmpeg/RTSP; GStreamer as an alternative |
 | Optional RTSP server | MediaMTX |
 | Dashboard backend | FastAPI |
 | Live transport | WebSocket |
 | Prototype GUI | Streamlit or PyQt5 |

 The dashboard should expose the current experiment, FSM state, current step, confidence, progress, alerts, protocol status, camera status, CSI status, degraded sensing mode, and optional live video.

 ## 13. Data and Training Strategy

 Collect synchronized fixed-camera and CSI sequences covering:

 - Complete correct runs.
 - Normal execution at different speeds.
 - Step skips and wrong-object actions.
 - Pauses and temporal variations.
 - Occlusions and visual degradation.
 - Lighting changes.
 - Held-out sessions for generalization evaluation.

 | Dataset or source | Use |
 | --- | --- |
 | Custom synchronized camera + CSI dataset | Primary multimodal training and evaluation |
 | COCO | Object detection reference and pretraining |
 | Open Images | Object detection reference |
 | Roboflow | Dataset management and annotation |
 | COCO-WholeBody | Whole-body and hand reference |
 | AGORA | 3D HMR |
 | 3DPW | 3D human pose and mesh |
 | EPIC-KITCHENS | Human-object interaction reference |
 | HOI4D | Human-object interaction reference |

 ## 14. Project Setup and Model Files

### Python environment

Create and activate a project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` needs to be generated or updated from the active virtual environment:

```powershell
pip freeze > requirements.txt
```

### MediaPipe Pose Landmarker model

The MediaPipe Python package and the Pose Landmarker model file are separate.

The Python package is installed through `requirements.txt` (or directly with
`pip install mediapipe`). The `.task` model file is downloaded separately
and is not a Python package dependency.

Create the model directory and download the Pose Landmarker Heavy model:

```powershell
mkdir models -ErrorAction SilentlyContinue

Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" -OutFile "models\pose_landmarker.task"
```

The expected local path is:

```text
models/
└── pose_landmarker.task
```

The application should reference the model using:

```python
model_path = "models\\pose_landmarker.task"
```

### Git handling of model files

Model weight files are kept locally and are not committed to Git. Add the
following rule to `.gitignore`:

```gitignore
models/*.task
```

Each developer should download the required model files locally after
cloning the repository.

Do not ignore the entire `models/` directory because future project
artifacts may need to be tracked there.

### Current setup distinction

- `requirements.txt` → Python dependencies such as `mediapipe`, `opencv-python`, `numpy`, and other packages used by the project.
- `models/*.task` → downloaded model weight files, kept outside Git.
- `.venv/` → local Python virtual environment, kept outside Git.
- `README.md` → setup and model-download instructions.

## 15. Deployment and Runtime

 ### Deployment targets

 - **Primary edge target:** NVIDIA Jetson Orin/Xavier-class hardware.
 - **Prototype fallback:** Laptop or Intel NUC.
 - **CSI node:** Raspberry Pi-class or equivalent CSI-capable device.
 - **Runtime:** Decoupled camera, CSI, inference, audio, logging, and dashboard/streaming processes.
 - **Optimization path:** ONNX, ONNX Runtime, INT8, and optionally TensorRT.

 3D HMR is GPU-oriented. Final latency must be measured on the actual demonstration hardware; design estimates are not measured guarantees.

 ### Runtime responsibilities

 | Thread or process | Responsibility |
 | --- | --- |
 | Camera thread | Camera acquisition and bounded frame queue |
 | CSI thread | CSI packet reception and embedding buffer |
 | Main inference | Vision, HMR, interaction, fusion, LSTM, and FSM |
 | Audio thread | Voice playback |
 | I/O thread | Session logging |
 | Stream thread | Video and RTSP |
 | Async server | FastAPI and WebSocket |

 ## 16. Failure Handling

 | Failure or condition | Response |
 | --- | --- |
 | Camera uncertainty | Use CSI as supporting evidence |
 | CSI unavailable | Continue with camera + HMR degraded mode |
 | Wrong object | Reject through interaction layer and FSM |
 | Skipped action | FSM violation and corrective voice guidance |
 | Out-of-order action | FSM violation and corrective voice guidance |
 | Both modalities uncertain | Hold or abstain |
 | Low confidence | Do not transition state |
 | Experiment complete | Close session and write final record |

 ## 17. Prototype Build Plan

 1. Freeze a 5-7 step experiment protocol and its JSON FSM.
 2. Build a fixed-camera mock testbed and collect complete sequences.
 3. Train and validate YOLOv8-nano on the actual experiment tools.
 4. Add MediaPipe Pose, MediaPipe Hands, and hand-object interaction.
 5. Integrate ROMP or PARE and define the selected `D_HMR` projection.
 6. Implement CSI capture and the 32-D CSI embedding.
 7. Build the fused representation with input size `179 + D_HMR`.
 8. Train and integrate the 30-frame temporal LSTM.
 9. Implement debounce and deterministic FSM validation.
 10. Add voice, structured logs, dashboard, and video/RTSP output.
 11. Demonstrate CSI-assisted robustness under visual uncertainty.
 12. Optimize and measure the complete pipeline on target edge hardware.

 ## 18. Judge Demonstration

 The demonstration should show:

 1. Several correct experiment steps with next-step guidance.
 2. A deliberate `PLACE_RED` skip followed by `PICK_YELLOW`, producing an FSM violation and corrective alert.
 3. Visual occlusion or confidence degradation while CSI continues to provide motion/context evidence.
 4. The 3D HMR pose/mesh representation in the perception layer.
 5. Recovery of visual confidence and successful experiment completion.
 6. The resulting structured log and dashboard state.

 ## 19. Core Innovation

 - Experiment-aware temporal HAR.
 - Probabilistic perception separated from deterministic protocol validation.
 - Confidence-aware fusion of camera/HMR and Wi-Fi CSI evidence.
 - 3D human pose and mesh representation for rack-relative, orientation-aware reasoning.
 - Offline, edge-first operation.
 - Proactive next-step experiment guidance.

 ## 20. Final Architecture Statement

 The final proposed system is an offline, edge-deployed, experiment-aware HAR co-pilot. A fixed RGB camera feeds object detection and tracking, body and hand estimation, hand-object interaction, and 3D human mesh recovery. In parallel, Wi-Fi CSI is captured and processed into a compact 32-D motion embedding. Vision/HMR and CSI representations are fused using confidence-aware multimodal reasoning, accumulated over a 30-frame temporal window, and classified by an LSTM into an experiment step and confidence. Debounce confirms stable actions, after which a JSON-defined finite-state machine validates the action against the expected protocol state. Valid transitions advance the experiment and trigger next-step guidance; violations generate corrective offline voice alerts. An asynchronous event bus distributes events to structured logging, local video/RTSP, and a FastAPI/WebSocket dashboard.

 ## 21. Design Boundaries

 **Included:** Fixed RGB camera, OpenCV, YOLOv8-nano, ByteTrack, MediaPipe Pose/Hands, hand-object interaction, 3D HMR (ROMP/PARE), Wi-Fi CSI, CSI processing, 1D-CNN CSI embedding, confidence-aware fusion, LSTM, debounce, FSM, event bus, voice, logging, video, RTSP, dashboard, and documented local model-file setup.

 **Not included:** Any separate radar, mmWave, point-cloud, or other non-CSI sensing subsystem.

 **Consistency rule:** One workflow is authoritative everywhere. Camera/HMR and CSI remain separate until feature fusion, and the final HMR-inclusive dimension is `179 + D_HMR` unless a concrete HMR projection is selected and documented.
