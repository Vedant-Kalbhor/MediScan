"""Helpers for logging MediScan training experiments to MLflow.

This module is intentionally lightweight so it can be imported from the
per-organ training notebooks or future train.py scripts without affecting the
inference API.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - training-only dependency
    plt = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - training-only dependency
    np = None

try:
    from sklearn.metrics import confusion_matrix
except ImportError:  # pragma: no cover - training-only dependency
    confusion_matrix = None

try:
    import mlflow
except ImportError:  # pragma: no cover - training-only dependency
    mlflow = None


DEFAULT_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "MediScan")
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


def mlflow_available() -> bool:
    return mlflow is not None and os.getenv("DISABLE_MLFLOW", "").lower() not in {"1", "true", "yes"}


def configure_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
) -> bool:
    """Configure the MLflow client and create the experiment if needed."""
    if not mlflow_available():
        return False

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    return True


@contextmanager
def start_run(
    run_name: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """Context manager for a single tracked training run."""
    if not configure_mlflow(experiment_name=experiment_name, tracking_uri=tracking_uri):
        yield None
        return

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        yield mlflow


def log_params(params: Dict[str, Any]) -> None:
    if not mlflow_available():
        return
    mlflow.log_params({key: str(value) for key, value in params.items()})


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if not mlflow_available():
        return
    numeric_metrics = {key: float(value) for key, value in metrics.items()}
    mlflow.log_metrics(numeric_metrics, step=step)


def log_confusion_matrix(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    class_names: Iterable[str],
    artifact_name: str = "confusion_matrix.png",
    normalize: bool = False,
) -> None:
    """Render and log a confusion matrix image as an MLflow artifact."""
    if not mlflow_available() or plt is None or np is None or confusion_matrix is None:
        return

    labels = list(class_names)
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    numeric_labels = all(
        isinstance(value, (int, np.integer))
        for value in (*y_true_list, *y_pred_list)
    )
    if numeric_labels:
        matrix = confusion_matrix(y_true_list, y_pred_list, labels=range(len(labels)))
    else:
        matrix = confusion_matrix(y_true_list, y_pred_list, labels=labels)
    matrix_to_plot = matrix.astype(float)

    if normalize and matrix_to_plot.sum() > 0:
        row_sums = np.clip(matrix_to_plot.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
        matrix_to_plot = matrix_to_plot / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix_to_plot, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (Normalized)" if normalize else ""),
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix_to_plot.max() / 2.0 if matrix_to_plot.size else 0.0
    for i in range(matrix_to_plot.shape[0]):
        for j in range(matrix_to_plot.shape[1]):
            value = matrix_to_plot[i, j]
            if normalize:
                text = f"{value:.2f}"
            else:
                text = f"{int(value)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / artifact_name
        fig.savefig(artifact_path, dpi=200, bbox_inches="tight")
        mlflow.log_artifact(str(artifact_path))

    plt.close(fig)


def log_classification_report(
    accuracy: float,
    f1_score: float,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    extra_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Log the core interview-friendly metrics and hyperparameters."""
    if not mlflow_available():
        return

    params = {
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    if extra_params:
        params.update(extra_params)

    log_params(params)
    log_metrics(
        {
            "accuracy": accuracy,
            "f1_score": f1_score,
        }
    )
