"""고급 모델: DistilBERT fine-tuning + Focal Loss (reference 프로젝트 재현).

reference 프로젝트의 핵심 아이디어를 정리해 올바른 코드로 재구성한 것이다.
 - DistilBERT: BERT를 지식증류(knowledge distillation)로 경량화한 모델. 로그를 텍스트로
   보고 위험도(level)를 분류한다.
 - Focal Loss: 극심한 클래스 불균형(2/4/6)에 대응하는 손실함수(gamma로 easy sample 억제).
 - Oversampling: 소수 클래스(2/4/6)를 복제해 배치에 더 자주 등장시킴.
 - max_len 512: 로그가 길어 512 토큰으로 자른다.

주의: 이 모듈은 transformers/torch가 설치되고 (가급적) GPU가 있을 때 실행하도록 만든 것이다.
CPU만 있는 환경에서는 매우 느리므로, 저장소의 "실행된 결과"는 baseline이며 이 코드는
재현 가능한 형태로 제공한다. scripts/run_distilbert.py 참고.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # torch 미설치 환경에서도 import 가능하도록
    _TORCH = False
    nn = object  # type: ignore


@dataclass
class DistilBertConfig:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 7          # level 0~6
    max_len: int = 512
    batch_size: int = 16
    lr: float = 1e-5
    epochs: int = 12
    focal_gamma: float = 2.0
    oversampling_scale: int = 10  # 소수 클래스 복제 배수
    minority_levels: tuple = (2, 4, 6)
    seed: int = 20210425


if _TORCH:

    class FocalLoss(nn.Module):
        """Multi-class focal loss. alpha(클래스 가중) 옵션 포함."""

        def __init__(self, gamma: float = 2.0, alpha: "torch.Tensor | None" = None):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** self.gamma) * ce
            return loss.mean()

    class LogDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_len: int):
            self.texts = list(texts)
            self.labels = list(labels)
            self.tok = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = self.tok(
                str(self.texts[i]),
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
            return item


def oversample(texts, labels, cfg: "DistilBertConfig"):
    """소수 클래스(2/4/6)를 oversampling_scale 배로 복제."""
    import pandas as pd
    df = pd.DataFrame({"t": list(texts), "y": list(labels)})
    extra = df[df["y"].isin(cfg.minority_levels)]
    dup = pd.concat([extra] * (cfg.oversampling_scale - 1), ignore_index=True)
    out = pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=cfg.seed)
    return out["t"].tolist(), out["y"].tolist()


def train_distilbert(train_texts, train_labels, val_texts, val_labels,
                     cfg: "DistilBertConfig | None" = None):
    """DistilBERT fine-tuning. transformers/torch 필요. (GPU 권장)

    반환: (model, tokenizer). 상세 실행은 scripts/run_distilbert.py 참고.
    """
    if not _TORCH:
        raise RuntimeError("torch/transformers가 필요합니다. requirements-nlp.txt 참고.")

    import numpy as np
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    cfg = cfg or DistilBertConfig()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    ).to(device)

    tr_texts, tr_labels = oversample(train_texts, train_labels, cfg)
    train_ds = LogDataset(tr_texts, tr_labels, tok, cfg.max_len)
    val_ds = LogDataset(val_texts, val_labels, tok, cfg.max_len)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)

    # 클래스 가중(alpha) = 역빈도
    counts = np.bincount(list(train_labels), minlength=cfg.num_labels).astype(float)
    counts[counts == 0] = 1
    alpha = torch.tensor((counts.sum() / counts), dtype=torch.float).to(device)
    alpha = alpha / alpha.sum() * cfg.num_labels

    criterion = FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_dl:
            optim.zero_grad()
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
        # (간결화를 위해 검증 루프는 run_distilbert.py에서 수행)
    return model, tok
