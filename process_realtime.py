import os, glob, re
import pandas as pd
import numpy as np

def find_first_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def to_datetime_safe(s, tz=None):
    return pd.to_datetime(s, errors="coerce", utc=False)

def minutes(delta) -> pd.Series:
    # Uniformly convert to Timedelta and then to minutes
    td = pd.to_timedelta(delta, errors="coerce")
    return td.dt.total_seconds() / 60.0

def service_day(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, errors="coerce").dt.date

def _haversine_np(lat1, lon1, lat2, lon2):

    R = 6371000.0

    def _to_np_f64(x):
        return pd.to_numeric(x, errors="coerce").astype("float64").to_numpy(copy=False)

    lat1 = np.radians(_to_np_f64(lat1))
    lon1 = np.radians(_to_np_f64(lon1))
    lat2 = np.radians(_to_np_f64(lat2))
    lon2 = np.radians(_to_np_f64(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def standardize_columns(df: pd.DataFrame, cmap: dict):
    cols = {}
    for key, cands in cmap.items():
        col = find_first_column(df, cands)
        cols[key] = col
    rename = {v: k for k, v in cols.items() if v is not None}
    out = df.rename(columns=rename).copy()
    return out, cols


def read_and_normalize_csvs(realtime_dir: str, config: dict) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(realtime_dir, "*.csv")))
    if not files:
        return pd.DataFrame()

    dfs = []
    chunksize = config["csv_read"]["chunksize"]

    read_kw = dict(
        sep=config["csv_read"]["sep"],
        encoding=config["csv_read"]["encoding"],
        low_memory=config["csv_read"]["low_memory"],
    )
    engine = config["csv_read"].get("engine", None)
    usecols = config["csv_read"].get("usecols", None)

    if chunksize:
        read_kw["engine"] = "c"
        if usecols: read_kw["usecols"] = usecols
    else:
        if engine: read_kw["engine"] = engine
        if usecols: read_kw["usecols"] = usecols

    # File name date mode
    date_pat1 = re.compile(r"(\d{2}-\d{2}-\d{4})")  # dd-mm-yyyy
    date_pat2 = re.compile(r"(\d{4}-\d{2}-\d{2})")  # yyyy-mm-dd

    def _ensure_service_date(df_chunk: pd.DataFrame, fp: str):
        # 1)Time column → Date
        if "service_date" not in df_chunk.columns or df_chunk["service_date"].isna().all():
            for base in ["actual_departure", "sched_departure", "Time", "OriginAimedDepartureTime"]:
                if base in df_chunk.columns:
                    dt = pd.to_datetime(df_chunk[base], errors="coerce", utc=False)
                    if dt.notna().any():
                        df_chunk["service_date"] = dt.dt.date
                        break
        # 2) File name
        if "service_date" not in df_chunk.columns or df_chunk["service_date"].isna().all():
            name = os.path.basename(fp)
            m1 = date_pat1.search(name); m2 = date_pat2.search(name)
            if m1:
                df_chunk["service_date"] = pd.to_datetime(m1.group(1), format="%d-%m-%Y").date()
            elif m2:
                df_chunk["service_date"] = pd.to_datetime(m2.group(1), format="%Y-%m-%d").date()
        return df_chunk

    for fp in files:
        if chunksize:
            for ch in pd.read_csv(fp, chunksize=int(chunksize), **read_kw):
                df, _ = standardize_columns(ch, config["columns_map"])
                df = df.convert_dtypes(dtype_backend="pyarrow")
                for tcol in ["sched_arrival","sched_departure","actual_arrival","actual_departure",
                             "service_date","Time","OriginAimedDepartureTime"]:
                    if tcol in df.columns: df[tcol] = to_datetime_safe(df[tcol])
                df = _ensure_service_date(df, fp)
                dfs.append(df)
        else:
            df0 = pd.read_csv(fp, **read_kw)
            df, _ = standardize_columns(df0, config["columns_map"])
            df = df.convert_dtypes(dtype_backend="pyarrow")
            for tcol in ["sched_arrival","sched_departure","actual_arrival","actual_departure",
                         "service_date","Time","OriginAimedDepartureTime"]:
                if tcol in df.columns: df[tcol] = to_datetime_safe(df[tcol])
            df = _ensure_service_date(df, fp)
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.convert_dtypes(dtype_backend="pyarrow")

    #Vehicle snapshot aperture → Fill column
    if "actual_departure" not in df.columns and "Time" in df.columns:
        df["actual_departure"] = df["Time"]
    if "sched_departure" not in df.columns and "OriginAimedDepartureTime" in df.columns:
        df["sched_departure"] = df["OriginAimedDepartureTime"]

    # Longitude and latitude
    if "latitude" not in df.columns and "Lat" in df.columns:
        df["latitude"] = pd.to_numeric(df["Lat"], errors="coerce")
    if "longitude" not in df.columns and "Long" in df.columns:
        df["longitude"] = pd.to_numeric(df["Long"], errors="coerce")

    if "stop_sequence" in df.columns:
        df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")

    return df

def _load_origin_coords(config: dict) -> pd.DataFrame:
    cache_dir = config["paths"]["cache_dir"]
    cand = os.path.join(cache_dir, "stops.parquet")
    if os.path.exists(cand):
        try:
            st = pd.read_parquet(cand)
            cols = {c.lower(): c for c in st.columns}
            stop_id_col = "stop_id" if "stop_id" in st.columns else cols.get("stoppointref") or cols.get("atcocode")
            lat_col = "lat" if "lat" in st.columns else cols.get("latitude")
            lon_col = "lon" if "lon" in st.columns else cols.get("longitude")
            st2 = st.rename(columns={stop_id_col: "stop_id", lat_col: "lat", lon_col: "lon"})[["stop_id","lat","lon"]].dropna()
            st2["lat"] = pd.to_numeric(st2["lat"], errors="coerce")
            st2["lon"] = pd.to_numeric(st2["lon"], errors="coerce")
            st2 = st2.dropna()
            return st2
        except Exception:
            pass
    return pd.DataFrame(columns=["stop_id","lat","lon"])

def compute_triplevel_metrics_at_origin(df: pd.DataFrame, config: dict) -> pd.DataFrame:

    required = ["service_date","route","direction","trip_id","stop_id","sched_departure","actual_departure"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=["service_date","route","direction","trip_id","stop_id",
                                     "sched_departure","actual_departure","delay_min","on_time","cancelled"])

    keep = list(set(required + ["latitude","longitude"]))
    df = df[[c for c in keep if c in df.columns]].copy()

    tz_off = int(config.get("tz_offset_minutes", 0) or 0)
    if tz_off != 0:
        df["actual_departure"] = pd.to_datetime(df["actual_departure"], errors="coerce") + pd.to_timedelta(tz_off, unit="m")

    # Starting point coordinate
    stops = _load_origin_coords(config)
    df = df.merge(stops.rename(columns={"stop_id":"_origin_id","lat":"_origin_lat","lon":"_origin_lon"}),
                  left_on="stop_id", right_on="_origin_id", how="left")
    df.drop(columns=["_origin_id"], inplace=True)

    # Distance from the starting point (in meters) - Only counts if there are coordinates. If there is none, set NaN
    has_cols = {"latitude","longitude","_origin_lat","_origin_lon"}.issubset(df.columns)
    if has_cols:
        if df[["_origin_lat","_origin_lon"]].isna().all().all():
            df["dist_m"] = np.nan
        else:
            df["dist_m"] = _haversine_np(df["latitude"], df["longitude"], df["_origin_lat"], df["_origin_lon"])
    else:
        df["dist_m"] = np.nan

    radius = float(config.get("origin_match_radius_m", 120))
    hit_space = df[(df["dist_m"].notna()) & (df["dist_m"] <= radius)].copy()

    keys = ["service_date","route","direction","trip_id","stop_id"]
    sched_cols = keys + ["sched_departure"]
    sched = (
        df[sched_cols]
        .dropna(subset=["sched_departure"])
        .sort_values(keys + ["sched_departure"], kind="mergesort")
        .drop_duplicates(subset=keys, keep="first")
        .reset_index(drop=True)
    )
    hit = hit_space.merge(sched, on=keys, how="left", suffixes=("", "_plan"))
    W = int(config.get("match_time_window_min", 30))
    act_dt  = pd.to_datetime(hit["actual_departure"], errors="coerce", utc=True)
    plan_dt = pd.to_datetime(hit["sched_departure"],   errors="coerce", utc=True)
    # Nanosecond difference → minutes
    try:
        time_diff_min = (act_dt.view("int64") - plan_dt.view("int64")) / 1e9 / 60.0
    except Exception:
        time_diff_min = (act_dt.astype("int64") - plan_dt.astype("int64")) / 1e9 / 60.0
    hit = hit[np.abs(time_diff_min) <= W].copy()

    min_gap = int(config.get("min_ping_gap_seconds", 30))
    if not hit.empty and min_gap > 0:
        hit.sort_values(keys + ["actual_departure"], inplace=True, kind="mergesort")
        act_ns = pd.to_datetime(hit["actual_departure"], errors="coerce", utc=True)
        try:
            act_ns = act_ns.view("int64")
        except Exception:
            act_ns = act_ns.astype("int64")
        grp_key = (hit["service_date"].astype(str) + "|" + hit["route"].astype(str) + "|" +
                   hit["direction"].astype(str) + "|" + hit["trip_id"].astype(str) + "|" + hit["stop_id"].astype(str)).to_numpy()
        # Calculate the adjacent difference within the group (in seconds)
        same_grp = grp_key
        prev_same = np.r_[True, same_grp[1:] == same_grp[:-1]]
        prev_ns   = np.r_[act_ns.values[0], act_ns.values[:-1]]
        gap_sec   = (act_ns.values - prev_ns) / 1e9
        keep = (~prev_same) | (gap_sec >= min_gap)
        hit = hit[keep].copy()

    fallback_needed = hit.empty
    if not fallback_needed:
        total_trips = df.drop_duplicates(subset=keys).shape[0]
        matched_trips = hit.drop_duplicates(subset=keys).shape[0]
        if total_trips > 0 and (matched_trips / total_trips) <= 0.01:
            fallback_needed = True
    if fallback_needed:
        hit = (
            df[keys + ["actual_departure"]]
            .dropna(subset=["actual_departure"])
            .sort_values(keys + ["actual_departure"], kind="mergesort")
            .drop_duplicates(subset=keys, keep="first")
            .reset_index(drop=True)
        )

    actual_cols = keys + ["actual_departure"]
    actual = (
        hit[actual_cols]
        .dropna(subset=["actual_departure"])
        .sort_values(keys + ["actual_departure"], kind="mergesort")
        .drop_duplicates(subset=keys, keep="first")
        .reset_index(drop=True)
    )

    # The planned departure has been approved by sched on it

    trips = sched.merge(actual, on=keys, how="left")


    sched_dt = pd.to_datetime(trips["sched_departure"], errors="coerce", utc=True)
    act_dt   = pd.to_datetime(trips["actual_departure"], errors="coerce", utc=True)
    mask = act_dt.notna() & sched_dt.notna()

    delay_min = np.full(len(trips), np.nan, dtype="float64")
    try:
        act_ns   = act_dt.view("int64")
        sched_ns = sched_dt.view("int64")
    except Exception:
        act_ns   = act_dt.astype("int64")
        sched_ns = sched_dt.astype("int64")
    delay_min[mask] = (act_ns[mask] - sched_ns[mask]) / 1e9 / 60.0

    # ---- Abnormal truncation (default: [-30, 120] minutes)----
    lo = float(config.get("outlier_clip_min", -30))
    hi = float(config.get("outlier_clip_max", 120))
    if lo is not None or hi is not None:
        delay_min = np.clip(delay_min, lo, hi)

    trips["delay_min"] = delay_min

    # ---- OTP / cancel ----
    e_ok = config["otp"]["early_ok_minutes"]
    l_ok = config["otp"]["late_ok_minutes"]
    early_bad = config["otp"]["count_early_as_not_on_time"]

    cond_nan = ~mask  # No actual → Cancelled
    if early_bad:
        good = (~cond_nan) & (trips["delay_min"] >= -e_ok) & (trips["delay_min"] <= l_ok)
    else:
        good = (~cond_nan) & (trips["delay_min"] <=  l_ok) & (trips["delay_min"] >= -e_ok)

    trips["on_time"] = np.nan
    trips.loc[~cond_nan, "on_time"] = good.astype(float)
    trips["cancelled"] = cond_nan.astype(int)

    return trips

def aggregate_metrics(trips: pd.DataFrame) -> dict:
    if trips is None or trips.empty:
        return {}

    keys = [k for k in ["service_date","route","direction"] if k in trips.columns]
    g = trips.groupby(keys, dropna=False)
    daily = g.agg(
        trips_planned=("sched_departure","count"),
        trips_actual=("actual_departure", lambda x: x.notna().sum()),
        cancel_rate=("cancelled","mean"),        # The denominator = planned train number
        otp=("on_time","mean"),
        delay_med=("delay_min","median"),
        delay_p90=("delay_min", lambda x: x.quantile(0.90)),
        delay_avg=("delay_min","mean"),
    ).reset_index()
    daily["events"] = daily["trips_actual"]

    # Ranking at the OriginRef level
    stop_rank = pd.DataFrame()
    keys_stop = [k for k in ["service_date","route","direction","stop_id"] if k in trips.columns]
    if keys_stop:
        s = trips[trips["actual_departure"].notna()].copy()
        gs = s.groupby(keys_stop, dropna=False)
        stop_rank = gs.agg(
            events=("on_time","count"),
            otp=("on_time","mean"),
            delay_med=("delay_min","median")
        ).reset_index()

    return {"daily_route": daily, "stop_rank": stop_rank}
