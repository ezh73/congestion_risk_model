# train_student.py — risk_utils를 참조하는 버전 (H3 Student)
import os
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor

# === risk_utils 참조 ===
from risk_utils import (
    STUDENT_FEATURES_H3,          # ["ts","hour_sin","hour_cos","is_holiday_or_weekend","is_night","season_idx","h3_enc","h3_cnt_log"]
    select_student_features,      # LazyFrame -> 위 피처만 선택
    add_calendar_flags,           # ts 기반으로 hour/date/holiday/season/hour_sin/cos 생성
)

# ===== 경로 (기존 유지) =====
MODEL_DIR = Path(os.getenv("MODEL_DIR", "run/teacher_model_h3")).resolve()
DATA_DIR  = Path(os.getenv("DATA_DIR",  "run/teacher_prep_h3")).resolve()
IN_PQ     = MODEL_DIR / "student_train_from_test.parquet"      # p_teacher 포함
BASE_TEST = DATA_DIR / "student_base_test.parquet"             # 피처만 (추론 예)

OUT_DIR   = Path(os.getenv("STUDENT_OUT", "run/student_model")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "lgbm_student.pkl"
IMP_PATH   = OUT_DIR / "feature_importance.csv"
PRED_PATH  = OUT_DIR / "student_pred_test.parquet"

TARGET = "p_teacher"

# ts(datetime) → 숫자형(일 단위 float)로 치환
def _with_numeric_ts(lf: pl.LazyFrame, col: str = "ts") -> pl.LazyFrame:
    return lf.with_columns(
        (pl.col(col).dt.epoch(time_unit="us") / pl.lit(86_400_000_000)).alias(col)
    )

# NA 제거(입력/타깃)
def _drop_na(df: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    cols = feats + ([TARGET] if TARGET in df.columns else [])
    return df.drop_nulls(subset=cols).drop_nans(subset=cols)

def _load_xy(parquet_path: Path):
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} 없음. 먼저 teacher 예측 병합을 생성하세요.")

    # 1) 캘린더 플래그(holiday/season/hour_sin/cos) 보장
    lf = pl.scan_parquet(str(parquet_path))
    lf = add_calendar_flags(lf)                 # ts 기준 파생 생성(이미 있어도 갱신됨)
    lf = select_student_features(lf)            # STUDENT_FEATURES_H3만 선택
    lf = _with_numeric_ts(lf, col="ts")         # ts → float days

    # 2) 타깃 포함 수집
    lf_y = pl.scan_parquet(str(parquet_path)).select(pl.col(TARGET))
    df  = lf.collect().with_columns(lf_y.collect()[TARGET])    # 같은 row 순서 가정
    df  = _drop_na(df, STUDENT_FEATURES_H3)

    X = df.select(STUDENT_FEATURES_H3).to_pandas()
    y = df[TARGET].to_numpy()
    return X, y

def _predict_on_base(reg, base_parquet: Path) -> pl.DataFrame:
    lf = pl.scan_parquet(str(base_parquet))
    lf = add_calendar_flags(lf)
    lf = select_student_features(lf)
    lf = _with_numeric_ts(lf, col="ts")

    df = lf.collect()
    df = _drop_na(df, STUDENT_FEATURES_H3)

    X = df.select(STUDENT_FEATURES_H3).to_pandas()
    p = reg.predict(X)
    return df.with_columns(pl.Series("p_student", p))

def main():
    # 1) 데이터 로드
    X, y = _load_xy(IN_PQ)
    print(f"[STUDENT] data={X.shape}, target_mean={y.mean():.6f}")

    # 2) 분할 (p_teacher 분포 유지: 간단한 quantile bin 층화 흉내)
    q = np.clip((y * 1000).astype(int), 0, 10)
    X_tr, X_ev, y_tr, y_ev = train_test_split(X, y, test_size=0.1, random_state=42, stratify=q)

    # 3) 경량 LGBM 회귀
    reg = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_bin=255,
        min_data_in_bin=3,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], eval_metric="l2")

    # 4) 간단 검증
    p_ev = reg.predict(X_ev)
    r2  = r2_score(y_ev, p_ev)
    mae = mean_absolute_error(y_ev, p_ev)
    spm = float(spearmanr(y_ev, p_ev).correlation)
    print(f"[STUDENT EVAL] R2={r2:.4f}  MAE={mae:.6f}  Spearman={spm:.4f}")

    # 5) 저장
    joblib.dump(reg, MODEL_PATH)
    fi = getattr(reg, "feature_importances_", None)
    if fi is not None:
        pl.DataFrame({"feature": STUDENT_FEATURES_H3, "importance": fi}).write_csv(str(IMP_PATH))

    # 6) 베이스 추론 저장(히트맵/서빙용)
    out = _predict_on_base(reg, BASE_TEST)
    out.write_parquet(str(PRED_PATH))

    print("[DONE] saved:")
    print(" -", MODEL_PATH)
    print(" -", IMP_PATH)
    print(" -", PRED_PATH)

if __name__ == "__main__":
    main()
