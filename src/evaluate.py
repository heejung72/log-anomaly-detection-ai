"""평가 지표 계산 및 저장."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate(y_true, y_pred, labels=None) -> dict:
    """per-class precision/recall/F1 + macro/weighted F1 + confusion matrix."""
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(report["accuracy"]),
        "per_class": {
            k: v for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": cm.tolist(),
        "labels": list(labels) if labels is not None else sorted(set(map(int, y_true))),
    }


def save_metrics(metrics: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def pretty_print(metrics: dict) -> None:
    print(f"accuracy    : {metrics['accuracy']:.4f}")
    print(f"macro-F1    : {metrics['macro_f1']:.4f}")
    print(f"weighted-F1 : {metrics['weighted_f1']:.4f}")
    print("\nper-class (precision / recall / f1 / support):")
    for cls, m in sorted(metrics["per_class"].items(), key=lambda x: str(x[0])):
        print(f"  level {cls}: {m['precision']:.3f} / {m['recall']:.3f} / "
              f"{m['f1-score']:.3f} / {int(m['support'])}")
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(np.array(metrics["confusion_matrix"]))
