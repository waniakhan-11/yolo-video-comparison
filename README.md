# YOLO Video Comparison and Person Tracking

This project contains two experiments using YOLO11 for person detection and tracking.

1. **YOLO Video Comparison:** Compares YOLO11n and YOLO11x using original and darkened video frames at different confidence thresholds.
2. **Person Detection and Tracking:** Uses YOLO11 and ByteTrack to detect people, assign tracking IDs, and count unique tracked people.

## Part 1: YOLO Video Comparison

### Experiment Setup

- Sampled frames: 10, approximately one frame per second
- Models: yolo11n and yolo11x
- Conditions: original and darkened to 40% brightness
- Confidence thresholds: 0.2, 0.4, 0.6, 0.8
- Detection class: person
- Device: CPU
- YOLO image size: 640

The same sampled frames and settings were used for both models.

### Detection Results

Values show average person detections per sampled frame.

| **Confidence** | **YOLO11n Original** | **YOLO11n Dark** | **YOLO11x Original** | **YOLO11x Dark** |
| -------------- | -------------------- | ---------------- | -------------------- | ---------------- |
| 0.2            | 10.4                 | 9.7              | 12.5                 | 11.9             |
| 0.4            | 8.2                  | 7.9              | 9.2                  | 8.8              |
| 0.6            | 6.6                  | 6.5              | 7.8                  | 7.6              |
| 0.8            | 3.5                  | 3.5              | 6.1                  | 5.8              |

### Speed Comparison

Original frames at confidence 0.4:

| **Model** | **Average Inference Time** |
| --------- | -------------------------- |
| YOLO11n   | 41.70 ms/frame             |
| YOLO11x   | 696.87 ms/frame            |

YOLO11x took approximately 16.7 times as long as YOLO11n for model inference in this experiment. These timings exclude video loading and saving.

### Findings

- Higher confidence thresholds reduced the number of detections.
- Darkening slightly reduced detections in most experiments.
- YOLO11x produced more detections at every tested threshold.
- YOLO11n had much lower inference time on this computer.

Detection counts are not accuracy scores or unique person counts. The same person may appear in multiple frames. Without manually labelled ground truth, this experiment cannot determine which model is more accurate.

### Charts

**Confidence Comparison**

[Confidence comparison](results/confidence_comparison.png)

**Original vs Dark**

[Darkness comparison](results/darkness_comparison.png)

**Model Speed**

[Model speed](results/model_speed.png)

---

## Part 2: Person Detection and Tracking Using ByteTrack

### Objective

The objective of this experiment is to detect and track people throughout a video using YOLO11 and ByteTrack. The system assigns a unique tracking ID to each detected person and maintains these IDs across video frames.

### Experiment Setup

- Detection model: YOLO11n
- Tracking algorithm: ByteTrack
- Detection class: Person (class 0)
- Confidence threshold: 0.4
- Image size: 640
- Device: CPU
- Total video frames: 279
- Frame rate: 29.97 FPS
- Video duration: Approximately 9.3 seconds

### Implementation

YOLO11 was used to detect people in each video frame. ByteTrack was integrated using the Ultralytics tracking function.

The system performs the following operations:

1. Reads the input video frame by frame.
2. Detects people using YOLO11.
3. Assigns tracking IDs using ByteTrack.
4. Draws bounding boxes around detected people.
5. Displays each person's tracking ID on the video.
6. Records tracking information in CSV files.
7. Counts the total number of unique tracking IDs.

### Tracking Results

The system processed all 279 frames and generated the following results:

| **Measurement**        | **Result** |
| ---------------------- | ---------- |
| Total video frames     | 279        |
| Estimated manual count | 25         |
| Uni                    |            |

| que tracking IDs | 29 |
| ---------------- | -- |
| Difference       | 4  |

The tracking system generated 29 unique IDs, while the estimated manual count was 25. This difference may be caused by temporary detection loss, occlusion, or assigning new IDs to previously detected people.

**Note:** The manual count is an estimate and has not been independently verified across all frames.

### ID Switch Observation

A possible ID switch was suspected around frame 183.

At this point, several pedestrians were walking close to each other, and some tracking IDs temporarily disappeared from the detection results.

This may have happened because pedestrians were partially overlapping or blocking each other, making it difficult for the tracking algorithm to maintain their identities.

**Note:** Frame 183 is a suspected ID-switch location, not a visually confirmed ID switch.

### Output Files

The tracking experiment generates:

- `tracked_video.mp4`: Video with bounding boxes and person IDs.
- `tracking_frames.csv`: Frame-wise tracking information.
- `unique_people.csv`: Unique tracking IDs, first frame, last frame, and number of frames tracked.

[View Tracked Output Video](results/tracked_video.mp4)

---

## How to Run

Python 3.13 was used for this project.

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run YOLO Comparison

Place a video containing people in the project folder and name it `input.mp4`.

Run:

```bash
python compare.py
```

To regenerate charts from the saved summary:

```bash
python plot_results.py
```

### 3. Run Person Tracking

Run:

```bash
python tracking.py
```

The tracked video and CSV results will be saved in the `results/` folder.

Model weights download automatically on first use. The input video and model weights are excluded from this repository.

---

## Project Files

| **File / Folder**  | **Description**                                                   |
| ------------------ | ----------------------------------------------------------------- |
| `compare.py`       | Runs the YOLO comparison experiments                              |
| `plot_results.py`  | Generates comparison charts                                       |
| `tracking.py`      | Performs person detection and tracking using ByteTrack            |
| `requirements.txt` | Python dependencies                                               |
| `results/`         | Contains charts, CSV results, annotated images, and tracked video |

## Technologies Used

- Python
- YOLO11
- Ultralytics
- ByteTrack
- OpenCV
- Matplotlib
- CSV
- Git and GitHub
- Git LFS
