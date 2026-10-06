"""Baseline 모델: TF-IDF + 선형 분류기.

포맷이 섞인 로그에 견고하고 CPU에서 빠르게 학습된다.
극심한 클래스 불균형(2/4/6이 각 8~12개)에 대응해 class_weight='balanced'를 사용.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..features import build_vectorizer


def build_baseline(max_features: int = 50_000, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", build_vectorizer(max_features=max_features)),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # 소수 클래스 가중
            n_jobs=-1,
            C=C,
        )),
    ])
