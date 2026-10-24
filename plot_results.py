from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT = Path(__file__).resolve().parent / "results"
MODELS = ["yolo11n", "yolo11x"]
CONDITIONS = ["original", "dark"]
CONFIDENCES = [0.2, 0.4, 0.6, 0.8]

# Read the experiment results already saved by compare.py.
with open(OUTPUT / "summary.csv", newline="") as file:
    summary = list(csv.DictReader(file))

for row in summary:
    for key in [
        "confidence",
        "average_detections",
        "average_inference_ms"
    ]:
        row[key] = float(row[key])


def get_value(model, condition, confidence, metric):
    return next(
        row[metric]
        for row in summary
        if row["model"] == model
        and row["condition"] == condition
        and row["confidence"] == confidence
    )


# Chart 1: confidence comparison.
fig, ax = plt.subplots(figsize=(9, 5))

for model in MODELS:
    for condition in CONDITIONS:
        values = [
            get_value(model, condition, conf, "average_detections")
            for conf in CONFIDENCES
        ]
        ax.plot(
            CONFIDENCES,
            values,
            marker="o",
            label=f"{model} / {condition}"
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

for index, condition in enumerate(CONDITIONS):
    values = [
        get_value(model, condition, 0.4, "average_detections")
        for model in MODELS
    ]
    offset = (index - 0.5) * width
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

# Chart 3: model inference time.
times = [
    get_value(model, "original", 0.4, "average_inference_ms")
    for model in MODELS
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

print("Success! All three charts are saved in the results folder.")