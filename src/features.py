"""로그 파싱 및 피처 추출.

핵심 교훈(원래 프로젝트가 멈췄던 이유): train 데이터에는 kibana/logstash/wazuh 등
여러 유형의 로그가 섞여 있고 포맷이 제각각이다. 따라서 "하나의 정규식으로 모든 필드를
뽑는" 방식은 깨지기 쉽다. 여기서는 두 갈래로 접근한다.

1) 구조적 파싱(parse_log): 가능한 범위에서 timestamp/host/application을 '관대하게' 추출.
   추출 실패해도 None으로 두고 죽지 않는다(여러 포맷 허용).
2) 텍스트 피처(TF-IDF): 포맷에 의존하지 않고 로그 전체를 문자/단어 n-gram으로 벡터화.
   포맷이 섞여 있어도 견고하게 동작하므로 baseline 모델의 주 입력으로 쓴다.
"""
from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# "Sep 24 10:02:22" 형태의 syslog 타임스탬프
_TS = re.compile(r"[A-Z][a-z]{2}\s+\d+\s+\d{1,2}:\d{2}:\d{2}")
# "localhost kibana:" 처럼 host + application 토큰
_APP = re.compile(r"\b(?:localhost|[\w.-]+)\s+([\w./-]+)\s*[:\[]")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_LEVEL_KW = re.compile(r"\b(error|warn(?:ing)?|info|debug|critical|alert|fail(?:ed|ure)?)\b", re.I)


def parse_log(log: str) -> dict:
    """여러 포맷을 관대하게 파싱. 실패한 필드는 None."""
    log = str(log)
    ts = _TS.search(log)
    app = _APP.search(log)
    lvl = _LEVEL_KW.search(log)
    return {
        "timestamp": ts.group(0) if ts else None,
        "application": app.group(1) if app else None,
        "log_level_kw": lvl.group(1).lower() if lvl else None,
        "n_ip": len(_IP.findall(log)),
        "length": len(log),
    }


def parse_frame(logs: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([parse_log(x) for x in logs], index=logs.index)


def normalize_log(log: str) -> str:
    """노이즈(숫자/IP/타임스탬프)를 토큰으로 치환해 일반화. TF-IDF 입력 전처리."""
    log = str(log)
    log = _TS.sub(" <TS> ", log)
    log = _IP.sub(" <IP> ", log)
    log = re.sub(r"\b\d+\b", " <NUM> ", log)
    return log.lower()


def build_vectorizer(max_features: int = 50_000) -> TfidfVectorizer:
    """단어 + 문자 n-gram 혼합 대신, 로그에 강한 word(1,2)-gram TF-IDF를 사용."""
    return TfidfVectorizer(
        preprocessor=normalize_log,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )
