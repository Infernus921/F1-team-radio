# Stage G: T4: does lagged arousal predict subsequent lap performance?

# Imports
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--arousal", default=None, help="override input (plumbing tests on fake labels only)")
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()

warnings.filterwarnings("ignore")
cfg = load_config()
proc = path(cfg, "processed")
out_dir = Path(args.out_dir) if args.out_dir else path(cfg, "results") / "stage_g"
out_dir.mkdir(parents=True, exist_ok=True)

arousal_source = "override"
if args.arousal:
    A = pd.read_parquet(args.arousal)
else:
    rese = path(cfg, "results") / "stage_e"
    cands = {}
    m2 = rese / "arousal_metrics.json"
    if m2.exists():
        j = json.loads(m2.read_text())
        cands["e2_" + j["t4_model"]] = (j["cv5"][j["t4_model"]]["macro"]["auc_pr"], proc / "arousal_all_utterances.parquet")
    m5 = rese / "wavlm_ft_metrics.json"
    if m5.exists() and (proc / "arousal_wavlm_ft.parquet").exists():
        j = json.loads(m5.read_text())
        cands["e5_wavlm_partial_ft"] = (j["protocols"]["cv5"]["metrics"]["macro"]["auc_pr"], proc / "arousal_wavlm_ft.parquet")
    arousal_source = max(cands, key=lambda k: cands[k][0])
    print("arousal candidates (cv5 macro AUC-PR):", {k: v[0] for k, v in cands.items()}, "-> using", arousal_source)
    A = pd.read_parquet(cands[arousal_source][1])
J = pd.read_parquet(proc / "clip_laps.parquet")[["id", "phase", "lap", "racing_number"]]
A = A.merge(J, left_on="clip_id", right_on="id").drop(columns="id")
A = A[(A["phase"] == "race") & A["lap"].notna()].copy()
A["lap"] = A["lap"].astype(int)
A["racing_number"] = A["racing_number"].astype(str)
X = A.groupby(["race_id", "driver_id", "racing_number", "lap"]).agg(
    x=("p_high", "mean"), n_utt=("p_high", "size"),
    x_label=("arousal_label", lambda s: (s.dropna().astype(int) >= 2).mean() if s.notna().any() else np.nan)).reset_index()

R = pd.read_parquet(proc / "lap_residuals.parquet")
R["DriverNumber"] = R["DriverNumber"].astype(str)
R = R.sort_values(["race_id", "DriverNumber", "LapNumber"])
Rk = R.set_index(["race_id", "DriverNumber", "LapNumber"])
clean_res = R["residual_s"].where(R["clean"])
R["clean_res"] = clean_res
g = R.groupby(["race_id", "DriverNumber"])["clean_res"]


def fwd(s, lo, hi, fn, min_n):
    m = pd.concat([s.shift(-k) for k in range(lo, hi + 1)], axis=1)
    v = getattr(m, fn)(axis=1, skipna=True)
    return v.where(m.notna().sum(axis=1) >= min_n)


R["Y1"] = g.transform(lambda s: fwd(s, 1, 3, "mean", 2))
R["Y2"] = g.transform(lambda s: fwd(s, 1, 5, "std", 3))
maxlap = R.groupby("race_id")["LapNumber"].transform("max")
R["progress"] = R["LapNumber"] / maxlap
D = X.merge(R[["race_id", "DriverNumber", "LapNumber", "residual_s", "TyreLife", "Position", "progress", "Y1", "Y2"]],
            left_on=["race_id", "racing_number", "lap"], right_on=["race_id", "DriverNumber", "LapNumber"], how="inner")
D = D.rename(columns={"residual_s": "res_L"})
D = D.dropna(subset=["res_L", "TyreLife", "Position", "progress"])
for c in ["res_L", "Y1", "Y2"]:
    lo, hi = D[c].quantile([0.01, 0.99])
    D[c] = D[c].clip(lo, hi)
CTRL = "res_L + TyreLife + progress + Position"

# STEP 1: MDE (no arousal term)
mde = {}
for y in ["Y1", "Y2"]:
    d = D.dropna(subset=[y])
    m0 = smf.ols(f"{y} ~ {CTRL}", d).fit()
    sx = smf.ols(f"x ~ {CTRL}", d).fit().resid.std()
    n = len(d)
    se = m0.resid.std() / (sx * np.sqrt(n))
    m_bar = d.groupby("driver_id").size().pow(2).sum() / n
    v_drv = float(smf.mixedlm(f"{y} ~ {CTRL}", d, groups=d["driver_id"]).fit(reml=True).cov_re.iloc[0, 0])
    icc = max(0.0, v_drv / (v_drv + m0.scale))
    deff = 1 + (m_bar - 1) * icc
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    mde[y] = {"n_laps": int(n), "n_drivers": int(d["driver_id"].nunique()), "sd_x_after_controls": round(float(sx), 4),
              "resid_sd_s": round(float(m0.resid.std()), 4), "design_effect": round(float(deff), 3),
              "mde_s_per_unit_x": round(float(z * se * np.sqrt(deff)), 4),
              "mde_s_for_0_to_1_x": "same as above: x is a probability, so a unit is calm (0) -> aroused (1)",
              "power": 0.8, "alpha": 0.05}
(out_dir / "t4_mde.json").write_text(json.dumps(mde, indent=1))
print("STEP 1 (written before fitting arousal):", json.dumps(mde, indent=1), flush=True)

# STEP 2: arousal models
res = {"mde": mde, "arousal_source": arousal_source, "outcomes": {}}
for y in ["Y1", "Y2"]:
    d = D.dropna(subset=[y]).copy()
    f = f"{y} ~ x + {CTRL}"
    fe = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["driver_id"])[0]})
    m_slope = smf.mixedlm(f, d, groups=d["driver_id"], re_formula="~x").fit(reml=True)
    ml_slope = smf.mixedlm(f, d, groups=d["driver_id"], re_formula="~x").fit(reml=False)
    ml_int = smf.mixedlm(f, d, groups=d["driver_id"]).fit(reml=False)
    lr = max(0.0, 2 * (ml_slope.llf - ml_int.llf))
    p_lrt = 0.5 * stats.chi2.sf(lr, 1) + 0.5 * stats.chi2.sf(lr, 2)
    ci = m_slope.conf_int().loc["x"].tolist()
    res["outcomes"][y] = {
        "n_laps": int(len(d)), "n_drivers": int(d["driver_id"].nunique()),
        "beta_x_mixed": round(float(m_slope.params["x"]), 4), "ci95_mixed": [round(float(v), 4) for v in ci],
        "p_mixed": round(float(m_slope.pvalues["x"]), 4),
        "beta_x_ols_cluster": round(float(fe.params["x"]), 4),
        "ci95_ols_cluster": [round(float(v), 4) for v in fe.conf_int().loc["x"]],
        "beta_res_L": round(float(m_slope.params["res_L"]), 4),
        "slope_variance_component": round(float(m_slope.cov_re.loc["x", "x"]), 6),
        "slope_sd": round(float(np.sqrt(max(0, m_slope.cov_re.loc["x", "x"]))), 4),
        "lrt_slopes_vs_intercepts": {"stat": round(float(lr), 3), "p_mixture": round(float(p_lrt), 4)},
        "abs_beta_below_mde": bool(abs(m_slope.params["x"]) < mde[y]["mde_s_per_unit_x"]),
    }
    dl = d.dropna(subset=["x_label"])
    if len(dl) >= 30 and dl["driver_id"].nunique() >= 3:
        fl = smf.ols(f"{y} ~ x_label + {CTRL}", dl).fit(cov_type="cluster",
                                                         cov_kwds={"groups": pd.factorize(dl["driver_id"])[0]})
        fp = smf.ols(f"{y} ~ x + {CTRL}", dl).fit(cov_type="cluster",
                                                   cov_kwds={"groups": pd.factorize(dl["driver_id"])[0]})
        res["outcomes"][y]["attenuation_check_labelled_subset"] = {
            "n": int(len(dl)), "beta_hand_label": round(float(fl.params["x_label"]), 4),
            "ci95_hand_label": [round(float(v), 4) for v in fl.conf_int().loc["x_label"]],
            "beta_prob_same_rows": round(float(fp.params["x"]), 4)}
    if y == "Y1":
        re = pd.DataFrame({k: {"slope_dev": v.get("x", np.nan)} for k, v in m_slope.random_effects.items()}).T
        re["slope_total"] = m_slope.params["x"] + re["slope_dev"]
        re["n_laps"] = d.groupby("driver_id").size()
        re.sort_values("n_laps", ascending=False).to_csv(out_dir / "t4_per_driver_slopes_UNDERPOWERED.csv")

(out_dir / "t4_results.json").write_text(json.dumps(res, indent=1))
print("STEP 2:", json.dumps(res["outcomes"], indent=1))
