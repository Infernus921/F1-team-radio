# Stage D.1: expected-lap-time model and lap-time residuals

# Imports
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Settings
TRAFFIC_GAP_S = 1.5
CLEAN_MAX_RATIO = 1.07
N_FOLDS = 5

# Main
cfg = load_config()
rng = np.random.default_rng(cfg["seed"])
tel = path(cfg, "raw") / "telemetry"
out_dir = path(cfg, "results") / "stage_d"
out_dir.mkdir(exist_ok=True)


# Functions
def design(d: pd.DataFrame, drivers, compounds):
    X = [np.ones(len(d))]
    for drv in drivers[1:]:
        X.append((d["DriverNumber"] == drv).to_numpy(float))
    X.append(d["LapNumber"].to_numpy(float))
    for c in compounds:
        X.append(((d["Compound"] == c).to_numpy(float)) * d["TyreLife"].fillna(0).to_numpy(float))
    for c in compounds[1:]:
        X.append((d["Compound"] == c).to_numpy(float))
    X.append(d["traffic"].to_numpy(float))
    return np.column_stack(X)


def fit_predict(train, test, drivers, compounds):
    Xtr, Xte = design(train, drivers, compounds), design(test, drivers, compounds)
    beta, *_ = np.linalg.lstsq(Xtr, train["lap_s"].to_numpy(), rcond=None)
    return Xte @ beta, beta


all_laps, diag_rows = [], []
for rdir in sorted(tel.iterdir()):
    f = rdir / "laps.parquet"
    if not f.exists():
        continue
    L = pd.read_parquet(f)
    L = L[L["LapTime"].notna()].copy()
    if L.empty:
        continue
    L["race_id"] = rdir.name
    L["lap_s"] = L["LapTime"].dt.total_seconds()
    L["DriverNumber"] = L["DriverNumber"].astype(str)
    L["Compound"] = L["Compound"].fillna("UNKNOWN").astype(str)
    ts = L["TrackStatus"].fillna("").astype(str)
    L["neutralised"] = ts.str.contains("[4567]", regex=True)
    L["pit_lap"] = L["PitInTime"].notna() | L["PitOutTime"].notna()
    L["end_s"] = L["Time"].dt.total_seconds()
    L = L.sort_values(["LapNumber", "end_s"])
    L["gap_ahead_s"] = L.groupby("LapNumber")["end_s"].diff()
    L = L.sort_values(["DriverNumber", "LapNumber"])
    L["gap_ahead_prev_s"] = L.groupby("DriverNumber")["gap_ahead_s"].shift(1)
    L["traffic"] = (L["gap_ahead_prev_s"] < TRAFFIC_GAP_S).fillna(False)
    med = L.groupby("DriverNumber")["lap_s"].transform("median")
    L["clean"] = ((L["LapNumber"] > 1) & ~L["pit_lap"] & ~L["neutralised"]
                  & (L["lap_s"] < CLEAN_MAX_RATIO * med))
    C = L[L["clean"]]
    drivers = sorted(C["DriverNumber"].unique())
    compounds = sorted(C["Compound"].unique())
    if len(C) < 100 or len(drivers) < 2:
        diag_rows.append({"race_id": rdir.name, "status": "too_few_clean_laps", "n_clean": int(len(C))})
        continue

    fold = rng.integers(0, N_FOLDS, len(C))
    oos = np.full(len(C), np.nan)
    for k in range(N_FOLDS):
        tr, te = C[fold != k], C[fold == k]
        if te.empty:
            continue
        oos[fold == k] = te["lap_s"].to_numpy() - fit_predict(tr, te, drivers, compounds)[0]
    naive = C["lap_s"] - C.groupby("DriverNumber")["lap_s"].transform("mean")

    known = L[L["DriverNumber"].isin(drivers) & L["Compound"].isin(compounds)].copy()
    pred, beta = fit_predict(C, known, drivers, compounds)
    known["expected_s"] = pred
    known["residual_s"] = known["lap_s"] - known["expected_s"]
    all_laps.append(known[["race_id", "DriverNumber", "Driver", "LapNumber", "lap_s", "expected_s",
                           "residual_s", "Compound", "TyreLife", "Stint", "Position", "traffic",
                           "gap_ahead_prev_s", "clean", "neutralised", "pit_lap", "LapStartDate"]])
    ss_tot = ((C["lap_s"] - C["lap_s"].mean()) ** 2).sum()
    diag_rows.append({
        "race_id": rdir.name, "status": "ok", "n_clean": int(len(C)), "n_all": int(len(known)),
        "oos_rmse_s": float(np.sqrt(np.nanmean(oos ** 2))), "oos_mae_s": float(np.nanmean(np.abs(oos))),
        "naive_driver_mean_rmse_s": float(np.sqrt((naive ** 2).mean())),
        "oos_r2": float(1 - np.nansum(oos ** 2) / ss_tot),
        "fuel_slope_s_per_lap": float(beta[len(drivers)]),
    })

laps = pd.concat(all_laps, ignore_index=True)
laps.to_parquet(path(cfg, "processed") / "lap_residuals.parquet", index=False)
D = pd.DataFrame(diag_rows)
ok = D[D["status"] == "ok"]
summary = {
    "n_races_modelled": int(len(ok)), "n_races_skipped": int((D["status"] != "ok").sum()),
    "clean_laps": int(ok["n_clean"].sum()),
    "oos_rmse_s_median_race": round(float(ok["oos_rmse_s"].median()), 3),
    "oos_rmse_s_iqr": [round(float(ok["oos_rmse_s"].quantile(q)), 3) for q in (0.25, 0.75)],
    "oos_mae_s_median_race": round(float(ok["oos_mae_s"].median()), 3),
    "naive_driver_mean_rmse_s_median_race": round(float(ok["naive_driver_mean_rmse_s"].median()), 3),
    "oos_r2_median_race": round(float(ok["oos_r2"].median()), 3),
    "fuel_slope_median_s_per_lap": round(float(ok["fuel_slope_s_per_lap"].median()), 4),
    "note": "Fuel slope should be negative (cars lighten); ~-0.03 to -0.08 s/lap is typical.",
}
(out_dir / "residual_model_diagnostics.json").write_text(json.dumps({"summary": summary, "per_race": diag_rows}, indent=1, default=str))
print(json.dumps(summary, indent=1))
