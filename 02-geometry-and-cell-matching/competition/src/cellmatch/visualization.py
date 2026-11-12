"""Render original-size mask contours for a selected training region."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .data import read_rows, read_image, image_path, parse_instances
from .segmentation import normalize


def preview(data_root, predictions, sample_id, output):
    truth = read_rows(Path(data_root) / "training/train_ground_truth.csv")[sample_id]
    prediction = read_rows(predictions)[sample_id]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for column, modality in enumerate(("invivo", "exvivo")):
        image = read_image(image_path(data_root, "training", sample_id, modality))
        for row, (record, name) in enumerate(((truth, "Ground truth"), (prediction, "Prediction"))):
            ax = axes[row, column]
            ax.imshow(normalize(image), cmap="gray", vmin=0, vmax=1)
            instances = parse_instances(record, modality, image.shape)
            for instance in instances.values():
                if not instance.area:
                    continue
                points = instance.coordinates()
                xmin, ymin = points.min(axis=0)
                xmax, ymax = points.max(axis=0)
                patch = np.zeros((ymax-ymin+3, xmax-xmin+3))
                patch[points[:, 1]-ymin+1, points[:, 0]-xmin+1] = 1
                ax.contour(np.arange(xmin-1, xmax+2), np.arange(ymin-1, ymax+2), patch,
                           levels=[0.5], colors=["#56e6ba" if row == 0 else "#ffbd59"], linewidths=0.45)
            ax.set_title(f"{name} | {modality} | {len(instances)} instances")
            ax.set_xlim(0, image.shape[1]-1)
            ax.set_ylim(image.shape[0]-1, 0)
            ax.set_axis_off()
    fig.suptitle(f"{sample_id}\n{Path(predictions).stem} — contours shown at original image coordinates", fontsize=13)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
