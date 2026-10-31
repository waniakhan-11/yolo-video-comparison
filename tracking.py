
from pathlib import Path
import cv2
import csv
from collections import defaultdict
from ultralytics import YOLO

# Project paths
BASE = Path(__file__).resolve().parent
VIDEO = BASE / "input.mp4"
OUTPUT = BASE / "results"
OUTPUT.mkdir(exist_ok=True)

MODEL_PATH = BASE / "yolo11n.pt"
CONFIDENCE = 0.4
IMAGE_SIZE = 640


def main():

    # Load YOLO model
    model = YOLO(str(MODEL_PATH))

    # Open input video
    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video
    output_path = OUTPUT / "tracked_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Cannot create output video.")

    frame_number = 0

    # Store unique IDs and their frame appearances
    unique_ids = set()
    id_frames = defaultdict(list)
    frame_results = []

    print("Starting person tracking...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Track people using ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )

        result = results[0]
        current_ids = []

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()

            for box, person_id in zip(boxes, ids):

                x1, y1, x2, y2 = map(int, box)

                unique_ids.add(person_id)
                id_frames[person_id].append(frame_number)
                current_ids.append(person_id)

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display person ID
                label = f"Person ID: {person_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # Display current number of tracked people
        cv2.putText(
            frame,
            f"Unique IDs: {len(unique_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        writer.write(frame)

        frame_results.append({
            "frame": frame_number,
            "tracked_people": len(current_ids),
            "person_ids": ",".join(map(str, current_ids))
        })

        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames.")

    cap.release()
    writer.release()

    # Save frame-by-frame tracking results
    with open(
        OUTPUT / "tracking_frames.csv",
        "w",
        newline=""
    ) as file:

        fieldnames = [
            "frame",
            "tracked_people",
            "person_ids"
        ]

        csv_writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        csv_writer.writeheader()
        csv_writer.writerows(frame_results)

    # Save unique person tracking information
    with open(
        OUTPUT / "unique_people.csv",
        "w",
        newline=""
    ) as file:

        fieldnames = [
            "person_id",
            "first_frame",
            "last_frame",
            "frames_tracked"
        ]

        csv_writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        csv_writer.writeheader()

        for person_id, frames in sorted(id_frames.items()):

            csv_writer.writerow({
                "person_id": person_id,
                "first_frame": min(frames),
                "last_frame": max(frames),
                "frames_tracked": len(frames)
            })

    print("\nTracking completed!")
    print("Output video:", output_path)
    print("Unique tracked people:", len(unique_ids))
    print("Results saved in:", OUTPUT)


if __name__ == "__main__":
    main()
