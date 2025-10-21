# risk_utils.py  — Polars 1.34.0 호환 / H3 단일 해상도 + 타깃인코딩 / Student 시간=삼각함수
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import polars as pl
import numpy as np
import holidays
import h3  # pip install h3

# =============================================================================
# 공통 상수/피처 리스트
# =============================================================================
TEACHER_TARGET = "label"

# Teacher: 시간/달력 + 교통 피처 + H3 인코딩 (hour_sin/cos은 선택적이지만 포함해둠)
TEACHER_FEATURES_H3 = [
    "LINK_ID", "ts", "date", "hour",
    "sped", "tfvl", "ocpy_rate", "trvl_hh", "lane_num",
    # 시간/달력
    "is_holiday_or_weekend", "is_night", "season_idx",
    # 시간 삼각함수(Teacher에서는 선택 사항이지만 포함)
    "hour_sin", "hour_cos",
    # H3 인코딩
    "h3_id", "h3_enc", "h3_cnt_log",
    TEACHER_TARGET,
]

# Student: 시간=삼각함수만, raw hour 제외 (경량)
STUDENT_FEATURES_H3 = [
    "ts",
    "hour_sin", "hour_cos",
    "is_holiday_or_weekend", "is_night", "season_idx",
    "h3_enc", "h3_cnt_log",
]

# 한국 공휴일 (2015~2030)
KR_HOLIDAYS = holidays.KR(years=range(2015, 2031))
HOLI_DATES = set(KR_HOLIDAYS.keys())  # Python date 객체

# =============================================================================
# 시간/달력 파생
# =============================================================================
def add_calendar_flags(ldf: pl.LazyFrame) -> pl.LazyFrame:
    ldf = ldf.with_columns(pl.col("ts").cast(pl.Datetime).alias("ts"))

    # ✅ hour/date는 무조건 ts에서 생성 (존재해도 덮어쓰기)
    ldf = ldf.with_columns([
        pl.col("ts").dt.hour().cast(pl.Int16).alias("hour"),
        pl.col("ts").dt.date().alias("date"),
    ])

    # is_night
    ldf = ldf.with_columns(((pl.col("hour") >= 22) | (pl.col("hour") <= 5)).cast(pl.Int8).alias("is_night"))

    # 주말/공휴일 + 시즌
    ldf = ldf.with_columns([
        pl.col("date").dt.weekday().alias("weekday_tmp"),
        pl.col("date").is_in(list(HOLI_DATES)).alias("is_holiday_tmp"),
        pl.col("date").dt.month().alias("month_tmp"),
    ]).with_columns([
        ((pl.col("weekday_tmp").is_in([5, 6])) | pl.col("is_holiday_tmp")).cast(pl.Int8).alias("is_holiday_or_weekend"),
        pl.when(pl.col("month_tmp").is_in([3, 4, 5])).then(0)
         .when(pl.col("month_tmp").is_in([6, 7, 8])).then(1)
         .when(pl.col("month_tmp").is_in([9,10,11])).then(2)
         .otherwise(3)
         .cast(pl.Int8).alias("season_idx"),
    ]).drop(["weekday_tmp", "is_holiday_tmp", "month_tmp"])

    # hour_sin/cos (Student용)
    radians = pl.col("hour").cast(pl.Float64) * (2 * np.pi / 24.0)
    ldf = ldf.with_columns([
        radians.sin().alias("hour_sin"),
        radians.cos().alias("hour_cos"),
    ])
    return ldf


# =============================================================================
# H3 단일 해상도 — 계산/인코딩
# =============================================================================
@dataclass
class H3Spec:
    """H3 단일 해상도 설정"""
    res: int = 7
    m_smooth: float = 200.0  # m-스무딩: 희소 셀 안정화

def _build_linkid_to_h3(
    ldf: pl.LazyFrame,
    spec: H3Spec,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pl.DataFrame:
    """
    LINK_ID 고유 좌표만 수집 → Python에서 h3 계산 (소형 DF) → 매핑 반환.
    대량 map_elements 없이 빠름.
    """
    uniq = (
        ldf.select("LINK_ID", lat_col, lon_col)
           .unique(maintain_order=False)
           .collect()
    )
    h3_ids = [
        h3.latlng_to_cell(float(lat), float(lon), spec.res)
        for lat, lon in zip(uniq[lat_col].to_list(), uniq[lon_col].to_list())
    ]
    return pl.DataFrame({"LINK_ID": uniq["LINK_ID"], "h3_id": h3_ids})

def attach_h3_id(
    ldf: pl.LazyFrame,
    spec: H3Spec,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pl.LazyFrame:
    """
    LINK_ID→H3 매핑을 조인하여 h3_id 추가.
    """
    link2h3 = _build_linkid_to_h3(ldf, spec, lat_col, lon_col)  # small DF
    return ldf.join(link2h3.lazy(), on="LINK_ID", how="left")

def build_h3_target_encoding(
    train_ldf: pl.LazyFrame,
    spec: H3Spec,
    label_col: str = TEACHER_TARGET,
) -> Tuple[pl.DataFrame, float]:
    """
    (train 전용) h3_id별 sum_y/cnt로 m-스무딩 타깃인코딩과 카운트 로그 생성.
    반환: (enc_map[h3_id, h3_enc, h3_cnt_log], global_mean)
    """
    mu = train_ldf.select(pl.col(label_col).mean().alias("mu")).collect()["mu"][0]
    enc_map = (
        train_ldf.group_by("h3_id")
                 .agg([
                     pl.col(label_col).sum().alias("sum_y"),
                     pl.len().alias("cnt"),
                 ])
                 .with_columns([
                     ((pl.col("sum_y") + spec.m_smooth * mu) / (pl.col("cnt") + spec.m_smooth)).alias("h3_enc"),
                     (pl.col("cnt").cast(pl.Float64).log1p()).alias("h3_cnt_log"),
                 ])
                 .select(["h3_id", "h3_enc", "h3_cnt_log"])
                 .collect()
    )
    return enc_map, float(mu)

def attach_h3_encoding(
    ldf: pl.LazyFrame,
    enc_map: pl.DataFrame,
    mu: float,
) -> pl.LazyFrame:
    """
    h3 인코딩을 조인하고, 미관측 셀은 전역 평균/0으로 채움.
    """
    return (
        ldf.join(enc_map.lazy(), on="h3_id", how="left")
           .with_columns([
               pl.col("h3_enc").fill_null(mu).alias("h3_enc"),
               pl.col("h3_cnt_log").fill_null(0.0).alias("h3_cnt_log"),
           ])
    )

# =============================================================================
# 음성 다운샘플 (빠른 모듈러 방식; 파이썬 UDF 없음)
# =============================================================================
def downsample_neg_by_mod(
    ldf: pl.LazyFrame,
    label_col: str = TEACHER_TARGET,
    neg_frac: float = 0.002,
) -> pl.LazyFrame:
    """
    라벨 0(음성)만 neg_frac 비율로 보존, 라벨 1(양성)은 전량 보존.
    Lazy에서 row_index() + 모듈러로 처리 (빠르고 안정적).
    """
    if not (0.0 < neg_frac <= 1.0):
        return ldf
    mod_base = 1_000_000
    thresh = int(mod_base * neg_frac)
    return (
        ldf.with_row_index(name="_rid")
           .filter(
               (pl.col(label_col) == 1) |
               ((pl.col(label_col) == 0) & ((pl.col("_rid") % mod_base) < thresh))
           )
           .drop("_rid")
    )

# =============================================================================
# 편의: 최종 피처 선택
# =============================================================================
def select_teacher_features(ldf: pl.LazyFrame) -> pl.LazyFrame:
    return ldf.select([pl.col(c) for c in TEACHER_FEATURES_H3])

def select_student_features(ldf: pl.LazyFrame) -> pl.LazyFrame:
    return ldf.select([pl.col(c) for c in STUDENT_FEATURES_H3])

# =============================================================================
# __all__ (선택)
# =============================================================================
__all__ = [
    "TEACHER_TARGET",
    "TEACHER_FEATURES_H3",
    "STUDENT_FEATURES_H3",
    "H3Spec",
    "add_calendar_flags",
    "attach_h3_id",
    "build_h3_target_encoding",
    "attach_h3_encoding",
    "downsample_neg_by_mod",
    "select_teacher_features",
    "select_student_features",
]
