# Stage D.3: T2 corroboration rules evaluated for EVERY (race, car, lap)

# Imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Settings
GRIP_T = [0.10, 0.20, 0.30, 0.50, 0.80]
TRAFFIC_G = [1.0, 1.5, 2.0, 3.0]
RELPACE_P = [0.15, 0.30, 0.50]
INCIDENT_W = {"L-1..L+2": (-1, 2), "L..L+1": (0, 1), "L-2..L+3": (-2, 3)}
POWER_V = [4, 8, 15]
POWER_R = [300, 500, 1000]

# Main
cfg = load_config()
tel = path(cfg, "raw") / "telemetry"
R = pd.read_parquet(path(cfg, "processed") / "lap_residuals.parquet")
R["DriverNumber"] = R["DriverNumber"].astype(str)


# Functions
def roll_lagged(s: pd.Series, lo: int, hi: int, fn: str) -> pd.Series:
    parts = [s.shift(-k) for k in range(lo, hi + 1)]
    m = pd.concat(parts, axis=1)
    return getattr(m, fn)(axis=1, skipna=True).where(m.notna().any(axis=1))


out = []
for race_id, RL in R.groupby("race_id"):
    rdir = tel / race_id
    L = pd.read_parquet(rdir / "laps.parquet", columns=["DriverNumber", "LapNumber", "Time", "LapStartDate", "LapTime"])
    L["DriverNumber"] = L["DriverNumber"].astype(str)
    L = L.dropna(subset=["Time"])
    L["end_s"] = L["Time"].dt.total_seconds()
    L = L.sort_values(["LapNumber", "end_s"])
    L["gap_ahead_s"] = L.groupby("LapNumber")["end_s"].diff()
    L["gap_behind_s"] = -L.groupby("LapNumber")["end_s"].diff(-1)

    lapmax = None
    cdp = rdir / "car_data.parquet"
    if cdp.exists():
        cd = pd.read_parquet(cdp, columns=["Date", "Speed", "RPM", "DriverNumber"])
        cd["DriverNumber"] = cd["DriverNumber"].astype(str)
        parts = []
        for drv, c in cd.groupby("DriverNumber"):
            ll = L[(L["DriverNumber"] == drv)].dropna(subset=["LapStartDate"]).sort_values("LapStartDate")
            if ll.empty:
                continue
            c = pd.merge_asof(c.sort_values("Date"), ll[["LapStartDate", "LapNumber"]],
                              left_on="Date", right_on="LapStartDate", direction="backward")
            parts.append(c.groupby("LapNumber").agg(max_speed=("Speed", "max"), max_rpm=("RPM", "max"))
                         .assign(DriverNumber=drv).reset_index())
        lapmax = pd.concat(parts) if parts else None

    rc = None
    rcp = rdir / "race_control.parquet"
    if rcp.exists():
        rc = pd.read_parquet(rcp)

    for drv, g in RL.groupby("DriverNumber"):
        g = g.sort_values("LapNumber").set_index("LapNumber")
        laps = pd.RangeIndex(int(g.index.min()), int(g.index.max()) + 1)
        g = g.reindex(laps)
        f = pd.DataFrame(index=laps)
        f.index.name = "LapNumber"
        lg = L[L["DriverNumber"] == drv].set_index("LapNumber").reindex(laps)

        clean_res = g["residual_s"].where(g["clean"] == True)
        grip_mean = roll_lagged(clean_res, -2, 0, "mean")
        for t in GRIP_T:
            f[f"GRIP_DEG@{t}"] = (grip_mean >= t).astype(float).where(grip_mean.notna())

        gap_min = roll_lagged(lg["gap_ahead_s"], -1, 0, "min")
        blue = pd.Series(0.0, index=laps)
        if rc is not None and "Flag" in rc and "RacingNumber" in rc and "Lap" in rc:
            b = rc[(rc["Flag"] == "BLUE") & (rc["RacingNumber"].astype(str) == drv)]["Lap"].dropna().astype(int)
            hit = pd.Series(0.0, index=laps)
            hit.loc[hit.index.intersection(b.values)] = 1.0
            blue = roll_lagged(hit, -1, 1, "max").fillna(0)
        for gg in TRAFFIC_G:
            v = ((gap_min < gg) | (blue > 0)).astype(float)
            f[f"TRAFFIC@{gg}"] = v.where(gap_min.notna() | (blue > 0))

        d_ahead = lg["gap_ahead_s"].diff().abs()
        d_behind = lg["gap_behind_s"].diff().abs()
        dmax = pd.concat([d_ahead, d_behind], axis=1).max(axis=1, skipna=True)
        pace = roll_lagged(dmax, -2, 0, "mean")
        for p in RELPACE_P:
            f[f"REL_PACE@{p}"] = (pace >= p).astype(float).where(pace.notna())

        if rc is not None and "Lap" in rc:
            num = rc["RacingNumber"].astype(str) == drv if "RacingNumber" in rc else False
            txt = rc["Message"].fillna("").str.contains(rf"\bCAR {drv}\b", regex=True)
            m = rc[(num | txt) & rc["Lap"].notna()]
            m = m[~m["Message"].fillna("").str.contains("BLUE FLAG", case=False)]
            hit = pd.Series(0.0, index=laps)
            hit.loc[hit.index.intersection(m["Lap"].astype(int).values)] = 1.0
            for name, (lo, hi) in INCIDENT_W.items():
                f[f"INCIDENT@{name}"] = roll_lagged(hit, lo, hi, "max").fillna(0)
        else:
            for name in INCIDENT_W:
                f[f"INCIDENT@{name}"] = np.nan

        if lapmax is not None and (lapmax["DriverNumber"] == drv).any():
            lm = lapmax[lapmax["DriverNumber"] == drv].set_index("LapNumber").reindex(laps)
            cl = (g["clean"] == True)
            base_s = lm["max_speed"].where(cl).shift(1).rolling(5, min_periods=3).median()
            base_r = lm["max_rpm"].where(cl).shift(1).rolling(5, min_periods=3).median()
            drop_s = roll_lagged(base_s - lm["max_speed"], 0, 1, "max")
            drop_r = roll_lagged(base_r - lm["max_rpm"], 0, 1, "max")
            for v in POWER_V:
                for r in POWER_R:
                    val = ((drop_s >= v) | (drop_r >= r)).astype(float)
                    f[f"POWER@{v}kmh_{r}rpm"] = val.where(drop_s.notna() | drop_r.notna())
        else:
            for v in POWER_V:
                for r in POWER_R:
                    f[f"POWER@{v}kmh_{r}rpm"] = np.nan

        f["neutralised"] = g["neutralised"].astype(float)
        f["pit_lap"] = g["pit_lap"].astype(float)
        out.append(f.reset_index().assign(race_id=race_id, DriverNumber=drv))

T = pd.concat(out, ignore_index=True)
T.to_parquet(path(cfg, "processed") / "lap_rules.parquet", index=False)
prim = ["GRIP_DEG@0.3", "TRAFFIC@1.5", "REL_PACE@0.3", "INCIDENT@L-1..L+2", "POWER@8kmh_500rpm"]
print(len(T), "car-laps |", T["race_id"].nunique(), "races")
print("base rates (all laps, primary thresholds):")
print(T[prim].mean().round(3).to_string())
print("gradeable share:")
print(T[prim].notna().mean().round(3).to_string())
