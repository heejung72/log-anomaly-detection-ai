#!/usr/bin/env python3
"""DistilBERT + Focal Loss 학습 · 평가 CLI (고급, GPU 권장).

transformers/torch 설치 필요:  pip install -r requirements-nlp.txt
CPU에서는 매우 느리므로 소량(--nrows)으로만 검증을 권장한다.

사용법:
    python scripts/run_distilbert.py --nrows 20000 --epochs 3
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_train, make_splits, ALL_LEVELS
from src.evaluate import evaluate, save_metrics, pretty_print


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--nrows", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default="results/distilbert_metrics.json")
    args = ap.parse_args()

    import torch
    from src.models.distilbert import DistilBertConfig, LogDataset, train_distilbert

    cfg = DistilBertConfig(
        epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len
    )

    df = load_train(args.data, nrows=args.nrows)
    s = make_splits(df)
    model, tok = train_distilbert(
        s.X_train, s.y_train, s.X_val, s.y_val, cfg
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    test_ds = LogDataset(s.X_test, s.y_test, tok, cfg.max_len)
    dl = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size)
    preds, trues = [], []
    with torch.no_grad():
        for batch in dl:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            trues.extend(labels.tolist())

    metrics = evaluate(trues, preds, labels=ALL_LEVELS)
    metrics["model"] = "distilbert+focalloss"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_metrics(metrics, args.out)
    pretty_print(metrics)


if __name__ == "__main__":
    main()
