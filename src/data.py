"""데이터 로딩 및 분할.

DACON 시스템 로그 분석 경진대회(235717) 데이터를 다룬다.
train.csv: id, level(위험도 0~6), full_log(원본 로그 문자열)
test.csv : id, full_log  (라벨 없음 — 대회 제출용이라 자체 평가에는 쓰지 않음)

라벨이 없는 test.csv로는 정량 평가를 할 수 없으므로,
정직한 성능 측정을 위해 train.csv를 stratified 분할해 train/val/test로 쓴다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

ENCODING = "ISO-8859-1"
LABEL_COL = "level"
TEXT_COL = "full_log"
# 전체 클래스. 2/4/6은 극소수 클래스(각 8~12개)로 reference 프로젝트가 oversampling 대상으로 삼았다.
ALL_LEVELS = [0, 1, 2, 3, 4, 5, 6]
MINORITY_LEVELS = [2, 4, 6]


@dataclass
class Splits:
    X_train: pd.Series
    y_train: pd.Series
    X_val: pd.Series
    y_val: pd.Series
    X_test: pd.Series
    y_test: pd.Series


def load_train(path: str = "data/train.csv", nrows: int | None = None) -> pd.DataFrame:
    """train.csv 로드. nrows로 샘플링 가능(빠른 실험용)."""
    df = pd.read_csv(path, encoding=ENCODING, nrows=nrows)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def _safe_stratify(y: pd.Series):
    """클래스별 표본이 2개 미만이면 stratify 불가 → None 반환."""
    return y if y.value_counts().min() >= 2 else None


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> Splits:
    """train/val/test = (1-val)*(1-test) / (1-test)*val / test 비율로 분할.

    극소수 클래스(2/4/6)가 있어 가능한 한 stratify 하되, 불가하면 랜덤 분할로 폴백한다.
    """
    X, y = df[TEXT_COL], df[LABEL_COL]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=_safe_stratify(y)
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, random_state=seed, stratify=_safe_stratify(y_tmp)
    )
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)


def class_distribution(y: pd.Series) -> dict[int, int]:
    return {int(k): int(v) for k, v in y.value_counts().sort_index().items()}
