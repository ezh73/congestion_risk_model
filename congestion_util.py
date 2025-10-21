import os
import json
import pickle
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import holidays

# ─────────────────────────────
# 0) 공휴일 집합 (전역 1회)
# ─────────────────────────────
KR_HOLIDAYS = holidays.KR(years=range(2015, 2030))
HOLIDAY_DATES = set(KR_HOLIDAYS.keys())

def load_json_with_fallback(path: str, default: Optional[dict] = None) -> dict:
    """JSON 로드(+결측 키 보강). 실패/누락 시 default."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if default:
                    for k, v in (default.items()):
                        data.setdefault(k, v)
                return data
            return default or {}
        except Exception as e:
            print(f"[WARN] JSON 로드 실패({path}) → 기본값 사용: {e}")
            return default or {}
    else:
        print(f"[INFO] 파일 없음({path}) → 기본값 사용")
        return default or {}

# ─────────────────────────────
# 1) 타입 보정 & 시간/계절 피처
# ─────────────────────────────
def ensure_datetime(df: pd.DataFrame, col: str = "dt") -> pd.DataFrame:
    """dt 컬럼을 tz-naive UTC datetime64로 표준화."""
    if col not in df.columns:
        return df.copy()
    s = df[col]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        dt = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        out = df.copy(); out[col] = dt; return out
    elif pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            out = df.copy(); out[col] = s.dt.tz_convert("UTC").dt.tz_localize(None); return out
        return df.copy()
    elif pd.api.types.is_dtype_equal(s.dtype, "date"):
        out = df.copy(); out[col] = pd.to_datetime(s); return out
    else:
        out = df.copy()
        out[col] = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        return out

def _season_idx(m: int) -> int:
    if m in (3,4,5): return 0
    if m in (6,7,8): return 1
    if m in (9,10,11): return 2
    return 3

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - year, month, is_weekend
    - hour_sin, hour_cos
    - season_idx (봄0/여름1/가을2/겨울3)
    - is_holiday_or_weekend
    - open_mask_t (09~18시)  ※ 모델 피처엔 넣지 않더라도 유틸은 생성 유지
    """
    out = ensure_datetime(df, "dt").copy()

    if "hour" in out.columns and pd.api.types.is_string_dtype(out["hour"]):
        out["hour"] = pd.to_numeric(out["hour"], errors="coerce").astype("Int64")

    if "dt" in out.columns:
        out["year"] = out["dt"].dt.year
        out["month"] = out["dt"].dt.month
        out["is_weekend"] = (out["dt"].dt.weekday >= 5).astype("int8")
    else:
        out["year"] = pd.NA; out["month"] = pd.NA; out["is_weekend"] = pd.NA

    if "hour" in out.columns:
        theta = 2 * np.pi * out["hour"].astype("float64") / 24.0
        out["hour_sin"] = np.sin(theta)
        out["hour_cos"] = np.cos(theta)
    else:
        out["hour_sin"] = np.nan; out["hour_cos"] = np.nan

    out["season_idx"] = out["month"].apply(
        lambda m: _season_idx(int(m)) if pd.notna(m) else pd.NA
    ).astype("Int8")

    if "dt" in out.columns:
        dt_dates = out["dt"].dt.date
        out["is_holiday_or_weekend"] = (
            out["is_weekend"].eq(1).astype(bool) | dt_dates.isin(HOLIDAY_DATES)
        ).astype("int8")
    else:
        out["is_holiday_or_weekend"] = pd.NA

    if "hour" in out.columns:
        out["open_mask_t"] = out["hour"].between(9, 18, inclusive="both").astype("int8")
    else:
        out["open_mask_t"] = pd.NA

    return out

def _add_compat_features(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    """학습시 사용된 피처 호환 보강 (is_night, year_group)."""
    out = df.copy()
    if "is_night" in feats and "is_night" not in out.columns:
        if "hour" in out.columns:
            h = pd.to_numeric(out["hour"], errors="coerce")
            out["is_night"] = ((h >= 20) | (h <= 6)).astype("int8")
        else:
            out["is_night"] = 0
    if "year_group" in feats and "year_group" not in out.columns:
        if "dt" in out.columns:
            y = pd.to_datetime(out["dt"], errors="coerce").dt.year
            out["year_group"] = (y >= 2023).astype("int8")
        else:
            out["year_group"] = 1
    return out

# ─────────────────────────────
# 2) H3 + 인코딩
# ─────────────────────────────
def _latlon_to_h3_str(lat: float, lon: float, res: int = 9):
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    try:
        import h3
        if hasattr(h3, "geo_to_h3"):
            return str(h3.geo_to_h3(lat, lon, res))
        elif hasattr(h3, "latlng_to_cell"):
            return str(h3.latlng_to_cell(lat, lon, res))
    except Exception:
        try:
            import h3.api.basic_str as h3s
            return h3s.latlng_to_cell(lat, lon, res)
        except Exception:
            return None

def add_h3_res9(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not {"lat","lon"}.issubset(out.columns):
        return out
    out["h3_res9"] = out.apply(lambda r: _latlon_to_h3_str(r["lat"], r["lon"], 9), axis=1)
    return out

def apply_h3_encoder(df: pd.DataFrame, enc: dict, mode: str = "compat") -> pd.DataFrame:
    out = df.copy()
    if "h3_res9" not in out.columns:
        return out
    if mode == "compat":
        out["h3_enc"] = out["h3_res9"].map(lambda s: enc.get(s, 0)).astype("Int32")
        return out
    return out

# ─────────────────────────────
# 3) 아티팩트 로딩
# ─────────────────────────────
def load_artifacts(model_path: str = "congestion_model.pkl",
                   scaler_path: str = "score_scaler.json",
                   feature_contract_path: str = "feature_contract.json",
                   h3_encoder_path: str = "h3_encoder.json") -> Dict[str, Any]:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # 스케일러 파일은 레거시 호환용으로만 로드(현재 z-score만 사용)
    scaler = load_json_with_fallback(scaler_path, default={"version": "zscale-1.0"})
    with open(feature_contract_path, "r", encoding="utf-8") as f:
        fc = json.load(f)
    h3_map = None
    if os.path.exists(h3_encoder_path):
        with open(h3_encoder_path, "r", encoding="utf-8") as f:
            h3_map = json.load(f)
    print("[cfg] loaded scaler(meta) =", scaler)
    return {"model": model, "scaler_meta": scaler, "feature_contract": fc, "h3_map": h3_map}

# ─────────────────────────────
# 4) Z-score 스케일러 (공통)
# ─────────────────────────────
def fit_z_params(values, scale: float = 20.0, clip: tuple = (0, 100)) -> dict:
    """원시 예측 배열에서 z-score 파라미터 산출(mean/std/scale/clip)."""
    v = np.asarray(values, dtype=float)
    mean = float(np.nanmean(v))
    std  = float(np.nanstd(v))
    if not np.isfinite(std) or std < 1e-9:
        std = 1.0  # 분산 거의 0일 때 안전장치
    return {"mean": mean, "std": std, "scale": float(scale), "clip": list(clip), "version": "zscale-1.0"}

def zscore_transform(values, params: dict):
    """z-score → 0–100. 배치 전역 mean/std를 동일 적용하면 장소간 비교 가능."""
    v = np.asarray(values, dtype=float)
    mean, std = float(params["mean"]), float(params["std"])
    scale = float(params.get("scale", 20.0))
    lo, hi = params.get("clip", (0.0, 100.0))
    z = (v - mean) / (std + 1e-9)
    return np.clip(50.0 + scale * z, lo, hi)

# ─────────────────────────────
# 5) 예측 유틸
# ─────────────────────────────
def resolve_latlon(place_name: Optional[str] = None,
                   lat: Optional[float] = None,
                   lon: Optional[float] = None,
                   df_context: Optional[pd.DataFrame] = None,
                   place_meta_csv: str = "place_meta.csv") -> Tuple[str, float, float]:
    if lat is not None and lon is not None:
        return (place_name or "custom_place", float(lat), float(lon))
    if place_name and os.path.exists(place_meta_csv):
        meta = pd.read_csv(place_meta_csv)
        cand = meta.loc[meta["place_name"] == place_name, ["lat", "lon"]]
        if len(cand):
            lat_, lon_ = float(cand.iloc[0]["lat"]), float(cand.iloc[0]["lon"])
            return (place_name, lat_, lon_)
    raise ValueError(f"좌표를 찾을 수 없습니다: {place_name}")

def make_24h_grid(target_date: str) -> pd.DataFrame:
    d = pd.to_datetime(target_date).date()
    hours = np.arange(24, dtype=int)
    df = pd.DataFrame({"hour": hours})
    df["dt"] = pd.to_datetime(d) + pd.to_timedelta(df["hour"], unit="h")
    return df

def ensure_features_for_inference(base_df: pd.DataFrame,
                                  feature_contract: Dict[str, Any],
                                  h3_map: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    - 시간/계절/휴일 피처 생성
    - 학습 피처와 호환 피처 보강(is_night, year_group)
    - H3 인코딩 (use_h3 & h3_enc 필요 시)
    """
    df = add_time_features(base_df)
    feats = feature_contract.get("features", [])
    use_h3 = feature_contract.get("use_h3", False)

    df = _add_compat_features(df, feats)

    if use_h3 and "h3_enc" in feats and "h3_enc" not in df.columns:
        df = add_h3_res9(df)
        if h3_map is not None:
            df = apply_h3_encoder(df, h3_map)
    return df

def predict_hourly_scores(df_infer: pd.DataFrame,
                          artifacts: Dict[str, Any],
                          z_params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    예측 + z-score 점수 변환.
    - z_params를 넘기면 그 파라미터로 변환(권장: 배치 전역 비교용)
    - z_params가 None이면 y_hat 분포로 즉석 산출(서브셋 한정 비교)
    """
    model = artifacts["model"]
    feats = artifacts["feature_contract"]["features"]

    X = (df_infer
         .reindex(columns=feats, fill_value=0.0)
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0))

    y_hat = model.predict(X)

    # z-score 파라미터 준비
    zpar = z_params or fit_z_params(y_hat, scale=20.0, clip=(0, 100))
    score = zscore_transform(y_hat, zpar)

    out = df_infer[["dt", "hour", "place_name", "lat", "lon"]].copy()
    out["pred"] = y_hat          # 혼잡도: 유입량 자체(원시값)
    out["score"] = score         # 0–100 스케일
    return out

def summarize_scores(hourly_df: pd.DataFrame) -> Dict[str, Any]:
    s = hourly_df["score"].to_numpy()
    hours = hourly_df["hour"].to_numpy()
    idx = int(np.argmax(s))
    return {"peak_hour": int(hours[idx]),
            "peak_score": float(np.mean(s[[idx]])),
            "mean_score": float(np.mean(s))}

def predict_for_user(target_date: str,
                     place_name: Optional[str] = None,
                     lat: Optional[float] = None,
                     lon: Optional[float] = None,
                     *,
                     place_meta_csv: str = "place_meta.csv",
                     artifacts_paths: Dict[str, str] = None,
                     z_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    artifacts_paths = artifacts_paths or {}
    art = load_artifacts(
        model_path=artifacts_paths.get("model", "congestion_model.pkl"),
        scaler_path=artifacts_paths.get("scaler", "score_scaler.json"),
        feature_contract_path=artifacts_paths.get("feature_contract", "feature_contract.json"),
        h3_encoder_path=artifacts_paths.get("h3_encoder", "h3_encoder.json"),
    )
    pname, plat, plon = resolve_latlon(place_name, lat, lon, place_meta_csv=place_meta_csv)
    grid = make_24h_grid(target_date)
    grid["place_name"], grid["lat"], grid["lon"] = pname, plat, plon
    df_inf = ensure_features_for_inference(grid, art["feature_contract"], art["h3_map"])
    hourly = predict_hourly_scores(df_inf, art, z_params=z_params)
    summ = summarize_scores(hourly)
    return {
        "date": target_date,
        "place_name": pname,
        "hourly": [{"hour": int(h), "score": float(s)} for h, s in zip(hourly["hour"], hourly["score"])],
        "summary": summ,
        "extra": {"lat": plat, "lon": plon},
    }
