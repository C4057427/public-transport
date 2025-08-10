# run_all.py — robust
import os, sys, yaml
from pathlib import Path

DEFAULT_CFG = {
    "paths": {
        "realtime_dir": "./data/realtime",
        "timetable_zip": "./data/TimetableData.zip",
        "timetable_dir": "./data/timetable_extracted",
        "output_dir": "./output",
        "cache_dir": "./cache",
    },
    "otp": {
        "early_ok_minutes": 1,
        "late_ok_minutes": 5,
        "count_early_as_not_on_time": False,
    },
    "csv_read": {
        "encoding": "utf-8",
        "sep": ",",
        "infer_datetime": True,
        "low_memory": False,
        "chunksize": 200000,   # Reducing it saves more memory
        "usecols": None,
        "engine": "pyarrow",
        "dtype_backend": "pyarrow",
    },
    "origin_match_radius_m": 120,
    "tz_offset_minutes": 0,
    "columns_map": {
        "route": ["LineRef","route_id","RouteId"],
        "direction": ["DirectionRef","direction"],
        "trip_id": ["DatedVehicleJourneyRef","trip_id"],
        "vehicle_id": ["VehicleRef","vehicle_id"],
        "stop_id": ["OriginRef","stop_id"],
        "sched_departure": ["OriginAimedDepartureTime","sched_dep","ScheduledDepartureTime"],
        "actual_departure": ["Time","act_dep","ActualDepartureTime"],
        "latitude": ["Lat","Latitude"],
        "longitude": ["Long","Longitude"],
        "service_date": [],
    },
}

def load_cfg():
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        print(f"[WARN]  {cfg_path}not found，Use the default configuration.")
        return DEFAULT_CFG.copy()
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] Failed to read config.yaml ：{e}，use the default configuration.")
        return DEFAULT_CFG.copy()
    def merge(d, default):
        for k, v in default.items():
            if k not in d:
                d[k] = v
            elif isinstance(v, dict) and isinstance(d[k], dict):
                merge(d[k], v)
        return d
    return merge(cfg, DEFAULT_CFG.copy())

def main():
    cfg = load_cfg()
    print("[INFO] paths =", cfg.get("paths", {}))

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)

    try:
        from src.parse_txc import build_timetable
        build_timetable(cfg["paths"].get("timetable_zip"), cfg["paths"].get("timetable_dir"), cfg["paths"]["cache_dir"])
    except Exception as e:
        print(f"[WARN] timetable analysis Skip：{e}")

    from src.process_realtime import (
        read_and_normalize_csvs,
        compute_triplevel_metrics_at_origin,
        aggregate_metrics,
    )

    df = read_and_normalize_csvs(cfg["paths"]["realtime_dir"], cfg)
    if df.empty:
        print(f"[ERROR] CSV was not found in {cfg['paths']['realtime_dir']}.")
        sys.exit(1)

    trips = compute_triplevel_metrics_at_origin(df, cfg)
    if trips.empty:
        print("[ERROR] The train number table is empty. Please check the column mapping/date deduction/time zone/coordinate matching.")
        sys.exit(1)

    agg = aggregate_metrics(trips)

    # Export
    if "daily_route" in agg and len(agg["daily_route"]):
        agg["daily_route"].to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig")
    if "stop_rank" in agg and len(agg["stop_rank"]):
        agg["stop_rank"].to_csv(os.path.join(out_dir, "stop_ranking.csv"), index=False, encoding="utf-8-sig")

    from src.dashboard import build_dashboard
    html_path = os.path.join(out_dir, "dashboard.html")
    build_dashboard(agg, html_path)
    print("[OK] Dashboard:", html_path)

if __name__ == "__main__":
    main()
