# Stage D.0: download race telemetry for every race in the corpus via FastF1

# Imports
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import fastf1
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Main
cfg = load_config()
raw = path(cfg, "raw")
cache = raw / "fastf1_cache"
cache.mkdir(exist_ok=True)
out_root = raw / "telemetry"
out_root.mkdir(exist_ok=True)
log_dir = path(cfg, "results") / "stage_d"
log_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache))
logging.getLogger("fastf1").setLevel(logging.WARNING)

clips = pd.read_parquet(path(cfg, "interim") / "clips.parquet")
races = clips.groupby("race_id")["session_date"].first().sort_index()

schedules = {}
log = []

# Settings
RATE_LIMIT_WAIT_S = 600


# Functions
def fetch(race_id, date):
    out = out_root / race_id
    t0 = time.time()
    try:
        year = int(race_id[:4])
        if year not in schedules:
            schedules[year] = fastf1.get_event_schedule(year, include_testing=False)
        sch = schedules[year]
        hit = sch[sch["EventDate"].dt.strftime("%Y-%m-%d") == date]
        if len(hit) != 1:
            raise ValueError(f"{len(hit)} events on {date}")
        ev = hit.iloc[0]
        s = fastf1.get_session(year, int(ev["RoundNumber"]), "R")
        s.load(laps=True, telemetry=True, weather=True, messages=True)
        out.mkdir(exist_ok=True)
        pd.DataFrame(s.laps).to_parquet(out / "laps.parquet")
        car_ok = True
        try:
            cd = []
            for drv, df in s.car_data.items():
                df = pd.DataFrame(df).copy()
                df["DriverNumber"] = str(drv)
                cd.append(df)
            pd.concat(cd, ignore_index=True).to_parquet(out / "car_data.parquet")
        except Exception as e:
            car_ok = False
            print(f"{race_id}: no car data ({type(e).__name__})", flush=True)
        if s.weather_data is not None:
            pd.DataFrame(s.weather_data).to_parquet(out / "weather.parquet")
        if s.race_control_messages is not None:
            pd.DataFrame(s.race_control_messages).to_parquet(out / "race_control.parquet")
        pd.DataFrame(s.results).to_parquet(out / "results.parquet")
        (out / "meta.json").write_text(json.dumps({
            "race_id": race_id, "event": ev["EventName"], "round": int(ev["RoundNumber"]),
            "session_date": date, "t0_date": str(s.t0_date), "car_data": car_ok,
        }, indent=1))
        return {"race_id": race_id, "status": "ok" if car_ok else "ok_no_car_data",
                "event": ev["EventName"], "n_laps": len(s.laps), "secs": round(time.time() - t0)}
    except Exception as e:
        if "RateLimit" in type(e).__name__:
            return None
        traceback.print_exc()
        return {"race_id": race_id, "status": "fail", "error": f"{type(e).__name__}: {e}"[:300]}


for race_id, date in races.items():
    if (out_root / race_id / "meta.json").exists():
        log.append({"race_id": race_id, "status": "ok_cached"})
        continue
    while (res := fetch(race_id, date)) is None:
        print(f"{time.strftime('%H:%M')} rate limit hit at {race_id}; waiting {RATE_LIMIT_WAIT_S}s", flush=True)
        time.sleep(RATE_LIMIT_WAIT_S)
    log.append(res)
    print(res, flush=True)
    pd.DataFrame(log).to_csv(log_dir / "fetch_log.csv", mode="w", index=False)
