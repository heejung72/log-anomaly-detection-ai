#!/usr/bin/env python3
"""Baseline(TF-IDF + LogisticRegression) 학습 · 평가 CLI.

사용법:
    python scripts/run_baseline.py --data data/train.csv
    python scripts/run_baseline.py --nrows 50000      # 빠른 실험
결과는 results/baseline_metrics.json 에 저장된다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_train, make_splits, class_distribution, ALL_LEVELS
from src.evaluate import evaluate, save_metrics, pretty_print
from src.models.baseline import build_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--max-features", type=int, default=50_000)
    ap.add_argument("--out", default="results/baseline_metrics.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] loading {args.data} (nrows={args.nrows}) ...")
    df = load_train(args.data, nrows=args.nrows)
    print(f"      rows={len(df)}  dist={class_distribution(df['level'])}")

    print("[2/4] splitting (train/val/test) ...")
    s = make_splits(df)
    print(f"      train={len(s.X_train)} val={len(s.X_val)} test={len(s.X_test)}")

    print("[3/4] training TF-IDF + LogisticRegression(balanced) ...")
    model = build_baseline(max_features=args.max_features)
    model.fit(s.X_train, s.y_train)

    print("[4/4] evaluating on held-out test ...")
    y_pred = model.predict(s.X_test)
    metrics = evaluate(s.y_test, y_pred, labels=ALL_LEVELS)
    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    metrics["n_rows"] = len(df)
    metrics["model"] = "tfidf+logreg(balanced)"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_metrics(metrics, args.out)
    pretty_print(metrics)
    print(f"\nsaved -> {args.out}  ({metrics['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
