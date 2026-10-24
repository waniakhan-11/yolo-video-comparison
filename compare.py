from pathlib import Path
import gc

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from statistics import mean
from ultralytics import YOLO


# Paths are relative to this script.
BASE = Path(__file__).resolve().parent
VIDEO = BASE / "input.mp4"
OUTPUT = BASE / "results"
OUTPUT.mkdir(exist_ok=True)

MODELS = ["yolo11n", "yolo11x"]
CONFIDENCES = [0.2, 0.4, 0.6, 0.8]
CONDITIONS = ["original", "dark"]

# Sample one frame per second to keep the experiment manageable.
# Both models use exactly the same sampled frames.
SAMPLE_SECONDS = 1
IMAGE_SIZE = 640


def load_frames():
    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Cannot read the video's frame rate.")

    step = max(1, round(fps * SAMPLE_SECONDS))
    frames = []
    frame_number = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_number % step == 0:
            # Limit stored frame size to reduce memory use.
            height, width = frame.shape[:2]
            scale = min(1.0, IMAGE_SIZE / max(height, width))
            if scale < 1:
                frame = cv2.resize(
                    frame,
                    (round(width * scale), round(height * scale))
                )

            frames.append((frame_number, frame))

        frame_number += 1

    cap.release()

    if not frames:
        raise RuntimeError("No frames could be read.")

    return frames


def main():
    frames = load_frames()
    print(f"Loaded {len(frames)} sampled frames.")

    rows = []

    for model_name in MODELS:
        print(f"\nLoading {model_name}...")
        model = YOLO(str(BASE / f"{model_name}.pt"))

        # Warm up before collecting inference timings.
        model.predict(
            frames[0][1],
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )

        for condition in CONDITIONS:
            for confidence in CONFIDENCES:
                print(
                    f"Running: {model_name}, "
                    f"{condition}, confidence={confidence}"
                )

                for index, (frame_number, original) in enumerate(frames):
                    if condition == "dark":
                        # Keep 40% of the original pixel brightness.
                        frame = (original * 0.4).astype("uint8")
                    else:
                        frame = original.copy()

                    result = model.predict(
                        frame,
                        conf=confidence,
                        classes=[0],  # Person class in these COCO models
                        imgsz=IMAGE_SIZE,
                        device="cpu",
                        verbose=False
                    )[0]

                    rows.append({
                        "model": model_name,
                        "condition": condition,
                        "confidence": confidence,
                        "frame_number": frame_number,
                        "detections": len(result.boxes),
                        "inference_ms": result.speed["inference"]
                    })

                    # Save one annotated example for every experiment.
                    if index == 0:
                        filename = (
                            f"{model_name}_{condition}_"
                            f"conf_{confidence:.1f}.jpg"
                        )
                        saved = cv2.imwrite(
                            str(OUTPUT / filename),
                            result.plot()
                        )
                        if not saved:
                            raise RuntimeError(f"Cannot save {filename}")

        del result
        del model
        gc.collect()

    # Save individual frame results and averages.
       # Save individual frame results.
    with open(OUTPUT / "frame_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Calculate averages for each experiment.
    summary = []

    for model_name in MODELS:
        for condition in CONDITIONS:
            for confidence in CONFIDENCES:
                selected = [
                    row for row in rows
                    if row["model"] == model_name
                    and row["condition"] == condition
                    and row["confidence"] == confidence
                ]

                summary.append({
                    "model": model_name,
                    "condition": condition,
                    "confidence": confidence,
                    "average_detections": mean(
                        row["detections"] for row in selected
                    ),
                    "average_inference_ms": mean(
                        row["inference_ms"] for row in selected
                    ),
                    "sampled_frames": len(selected)
                })

    with open(OUTPUT / "summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # Chart 1: confidence sweep.
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name in MODELS:
        for condition in CONDITIONS:
            selected = sorted(
                [
                    row for row in summary
                    if row["model"] == model_name
                    and row["condition"] == condition
                ],
                key=lambda row: row["confidence"]
            )

            ax.plot(
                [row["confidence"] for row in selected],
                [row["average_detections"] for row in selected],
                marker="o",
                label=f"{model_name} / {condition}"
            )

    ax.set(
        title="Effect of Confidence Threshold on Person Detections",
        xlabel="Confidence threshold",
        ylabel="Average person detections per sampled frame"
    )
    ax.set_xticks(CONFIDENCES)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "confidence_comparison.png", dpi=200)
    plt.close(fig)

    # Chart 2: original vs dark at confidence 0.4.
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(len(MODELS)))
    width = 0.35

    for condition_index, condition in enumerate(CONDITIONS):
        values = [
            next(
                row["average_detections"]
                for row in summary
                if row["model"] == model_name
                and row["condition"] == condition
                and row["confidence"] == 0.4
            )
            for model_name in MODELS
        ]

        offset = (condition_index - 0.5) * width
        ax.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=condition
        )

    ax.set_xticks(positions, MODELS)
    ax.set(
        title="Original vs Dark Video — Confidence 0.4",
        xlabel="Model",
        ylabel="Average person detections per sampled frame"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "darkness_comparison.png", dpi=200)
    plt.close(fig)

    # Chart 3: inference time on original frames at confidence 0.4.
    times = [
        next(
            row["average_inference_ms"]
            for row in summary
            if row["model"] == model_name
            and row["condition"] == "original"
            and row["confidence"] == 0.4
        )
        for model_name in MODELS
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(MODELS, times, color=["steelblue", "orange"])
    ax.set(
        title="Model Inference Time — Original Video, Confidence 0.4",
        xlabel="Model",
        ylabel="Average inference time (ms/frame; lower is faster)"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "model_speed.png", dpi=200)
    plt.close(fig)

    print("\nFinished! Results saved in:", OUTPUT)

    for row in summary:
        print(
            f"{row['model']} | {row['condition']} | "
            f"confidence={row['confidence']:.1f} | "
            f"detections/frame={row['average_detections']:.2f} | "
            f"inference={row['average_inference_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()