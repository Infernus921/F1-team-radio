# Stage D.5: telemetry windows for every in-race driver utterance (input to neural T2)

# Imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path
from f1radio.utterances import driver_utterances

# Main
PRE_S, POST_S, SHIFT_LAPS, LAP_LO, LAP_HI = 120, 60, 3, -4, 2

# Settings
CH = ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]
cfg = load_config()
proc, tel = path(cfg, "processed"), path(cfg, "raw") / "telemetry"

U = driver_utterances(pd.read_parquet(proc / "turns_attributed.parquet"))
J = pd.read_parquet(proc / "clip_laps.parquet")[["id", "race_id", "racing_number", "phase", "lap", "ts"]]
U = U.merge(J, left_on="clip_id", right_on="id").drop(columns="id")
U = U[(U["phase"] == "race") & U["lap"].notna()].reset_index(drop=True)
U["lap"] = U["lap"].astype(int)
U["racing_number"] = U["racing_number"].astype(str)
rules = pd.read_parquet(proc / "lap_rules.parquet")
rules["DriverNumber"] = rules["DriverNumber"].astype(str)
R = pd.read_parquet(proc / "lap_residuals.parquet")
R["DriverNumber"] = R["DriverNumber"].astype(str)
print("in-race driver utterances:", len(U), flush=True)

N = len(U)
car = np.zeros((N, PRE_S + POST_S, len(CH) + 1), np.float32)
car_pb = np.zeros_like(car)
laps = np.zeros((N, LAP_HI - LAP_LO + 1, 12), np.float32)
laps_pb = np.zeros_like(laps)

for race_id, g in U.groupby("race_id"):
    rdir = tel / race_id
    Lr = pd.read_parquet(rdir / "laps.parquet", columns=["DriverNumber", "LapNumber", "LapStartDate", "Time"])
    Lr["DriverNumber"] = Lr["DriverNumber"].astype(str)
    Lr = Lr.sort_values(["LapNumber", "Time"])
    Lr["gap_ahead"] = Lr.groupby("LapNumber")["Time"].diff().dt.total_seconds()
    Lr["gap_behind"] = -Lr.groupby("LapNumber")["Time"].diff(-1).dt.total_seconds()
    rc = pd.read_parquet(rdir / "race_control.parquet") if (rdir / "race_control.parquet").exists() else None
    cd = pd.read_parquet(rdir / "car_data.parquet") if (rdir / "car_data.parquet").exists() else None
    if cd is not None:
        cd["DriverNumber"] = cd["DriverNumber"].astype(str)
    lapmax = None
    if cd is not None:
        lapmax = {}
    for drv, gd in g.groupby("racing_number"):
        lr = Lr[Lr["DriverNumber"] == drv].set_index("LapNumber")
        rr = R[(R["race_id"] == race_id) & (R["DriverNumber"] == drv)].set_index("LapNumber")
        ru = rules[(rules["race_id"] == race_id) & (rules["DriverNumber"] == drv)].set_index("LapNumber")
        c = None
        if cd is not None:
            c = cd[cd["DriverNumber"] == drv].sort_values("Date")
            if len(c):
                c = c.set_index("Date")[CH].astype(float)
                mu, sd = c.mean(), c.std().replace(0, 1)
                c1 = ((c - mu) / sd).resample("1s").mean()
                lm = pd.merge_asof(c.reset_index().sort_values("Date"),
                                   lr["LapStartDate"].dropna().reset_index().sort_values("LapStartDate"),
                                   left_on="Date", right_on="LapStartDate").groupby("LapNumber").agg(
                    ms=("Speed", "max"), mr=("RPM", "max"))
            else:
                c = None
        rcl = set()
        bl = set()
        if rc is not None and "Lap" in rc:
            m = rc["RacingNumber"].astype(str) == drv
            rcl = set(rc.loc[m & (rc["Flag"] != "BLUE"), "Lap"].dropna().astype(int))
            bl = set(rc.loc[m & (rc["Flag"] == "BLUE"), "Lap"].dropna().astype(int))

        def lap_tok(L0):
            out = np.zeros((LAP_HI - LAP_LO + 1, 12), np.float32)
            for j, l in enumerate(range(L0 + LAP_LO, L0 + LAP_HI + 1)):
                if l not in rr.index:
                    continue
                row = rr.loc[l]
                res = row["residual_s"] if bool(row["clean"]) else np.nan
                vals = [res, lr["gap_ahead"].get(l, np.nan), lr["gap_behind"].get(l, np.nan), row["Position"],
                        row["TyreLife"], (lm["ms"].get(l, np.nan) if c is not None else np.nan),
                        (lm["mr"].get(l, np.nan) if c is not None else np.nan), float(row["pit_lap"]),
                        float(row["neutralised"]), float(l in rcl), float(l in bl)]
                vals = np.array(vals, np.float32)
                out[j, :11] = np.nan_to_num(vals, nan=0.0)
                out[j, 11] = 1.0
            return out

        for i, u in gd.iterrows():
            laps[i] = lap_tok(u["lap"])
            laps_pb[i] = lap_tok(u["lap"] - SHIFT_LAPS)
            if c is None:
                continue
            ts = pd.Timestamp(u["ts"]).floor("s")
            grid = pd.date_range(start=ts - pd.Timedelta(seconds=PRE_S), periods=PRE_S + POST_S, freq="s")
            w = c1.reindex(grid)
            car[i, :, :len(CH)] = np.nan_to_num(w.to_numpy(np.float32))
            car[i, :, -1] = w.notna().all(axis=1).to_numpy(np.float32)
            s0, s1 = lr["LapStartDate"].get(u["lap"] - SHIFT_LAPS), lr["LapStartDate"].get(u["lap"])
            if pd.notna(s0) and pd.notna(s1):
                wpb = c1.reindex(grid - pd.Timedelta(s1 - s0).round("s"))
                car_pb[i, :, :len(CH)] = np.nan_to_num(wpb.to_numpy(np.float32))
                car_pb[i, :, -1] = wpb.notna().all(axis=1).to_numpy(np.float32)
    print(race_id, flush=True) if race_id.endswith("Abu_Dhabi_Grand_Prix") else None

mu = laps[..., :11].reshape(-1, 11).mean(0)
sd = laps[..., :11].reshape(-1, 11).std(0) + 1e-6
laps[..., :11] = (laps[..., :11] - mu) / sd
laps_pb[..., :11] = (laps_pb[..., :11] - mu) / sd
np.savez_compressed(proc / "t2_windows.npz", utt_id=U["utt_id"].to_numpy(), race_id=U["race_id"].to_numpy(),
                    lap=U["lap"].to_numpy(), car=car, car_pb=car_pb, laps=laps, laps_pb=laps_pb)
print("saved", N, "| car coverage:", round(float(car[:, :, -1].mean()), 3),
      "| placebo car coverage:", round(float(car_pb[:, :, -1].mean()), 3))
