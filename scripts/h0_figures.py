# Stage H: figures and a single summary table from the results JSONs (every number traceable)

# Imports
import json
import sys
from pathlib import Path

import matplotlib

# Main
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, MUTED, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200})
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default=None, help="override (plumbing tests only)")
ap.add_argument("--fig-dir", default=None)
args = ap.parse_args()
cfg = load_config()
res = Path(args.results_dir) if args.results_dir else path(cfg, "results")
fig_dir = Path(args.fig_dir) if args.fig_dir else path(cfg, "figures")
fig_dir.mkdir(parents=True, exist_ok=True)
summary = []


# Functions
def load(rel):
    p = res / rel
    return json.loads(p.read_text()) if p.exists() else None


def dot_ci(ax, labels, est, lo, hi, color, chance=None, ref=None, ref_label=None):
    ys = range(len(labels))[::-1]
    for y, e, a, b in zip(ys, est, lo, hi):
        ax.plot([a, b], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(e, y, "o", color=color, ms=8, mec="#fcfcfb", mew=2)
        ax.annotate(f"{e:.2f}", (e, y), xytext=(0, 9), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_yticks(list(ys), labels)
    if chance is not None:
        ax.axvspan(chance[0], chance[1], color=GREY, alpha=0.15, lw=0)
        ax.text(chance[1], len(labels) - 0.45, " chance (95% of random)", color=MUTED, fontsize=8, va="center")
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=1, ls="--")
        ax.text(ref, -0.75, f" {ref_label}", color=MUTED, fontsize=8)
    ax.set_ylim(-1, len(labels))


a = load("stage_a/attribution_eval.json")
if a:
    names = list(a["systems"])
    s = a["systems"]
    short = {"ensemble_silk": "Primary: voice + text", "ensemble_silk_raw": "Original naming rule",
             "ensemble_k2": "Brief's 2 clusters", "voice_only_silk": "Voice only",
             "voice_only_k2": "Voice only, 2 clusters", "text_only": "Text cues only"}
    labels = [next((v for k, v in short.items() if n.split(" ")[0] == k), n) for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    dot_ci(ax, labels, [s[n]["accuracy"] for n in names], [s[n]["ci95_race_bootstrap"][0] for n in names],
           [s[n]["ci95_race_bootstrap"][1] for n in names], BLUE, ref=a["gate"], ref_label=f"gate {a['gate']}")
    ax.set_xlabel("Accuracy on hand-labelled driver/engineer turns\n(95% CI, race bootstrap)")
    ax.set_title(f"Speaker attribution (n = {a['n_evaluable']})", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_attribution.png")
    for n in names:
        summary.append({"finding": "attribution", "item": n, "value": s[n]["accuracy"],
                        "ci_lo": s[n]["ci95_race_bootstrap"][0], "ci_hi": s[n]["ci95_race_bootstrap"][1]})


# Settings
NICE = {"majority": "Majority", "context": "Context only (no text)", "tfidf": "TF-IDF", "deberta": "DeBERTa (fine-tuned)",
        "audio": "Audio: frozen WavLM", "text": "Text: TF-IDF", "fused": "Audio + text (frozen)",
        "telemetry_lr": "Telemetry, linear", "neural_concat": "Neural, naive concat", "neural_gated": "Neural, gated attention",
        "placebo_neural_gated": "Placebo (3 laps earlier)", "wavlm_ft": "Audio: WavLM partial fine-tune",
        "deberta_text": "Text: DeBERTa multi-task"}


def model_panel(js, key, title, fname, models, tag):
    proto = js
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(models) + 1.4))
    m0 = proto[models[0]][key]["macro"] if key else proto[models[0]]["macro"]
    get = (lambda m: proto[m][key]["macro"]) if key else (lambda m: proto[m]["macro"])
    dot_ci(ax, [NICE.get(m, m) for m in models], [get(m)["auc_pr"] for m in models], [get(m)["ci95"][0] for m in models],
           [get(m)["ci95"][1] for m in models], BLUE,
           chance=(m0.get("chance_auc_pr", m0["no_skill"]), m0.get("chance_95pct", m0["no_skill"])))
    ax.set_xlabel("Macro AUC-PR (95% CI, race bootstrap)")
    ax.set_title(title, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / fname)
    for m in models:
        summary.append({"finding": tag, "item": m, "value": get(m)["auc_pr"], "ci_lo": get(m)["ci95"][0],
                        "ci_hi": get(m)["ci95"][1], "chance": m0.get("chance_auc_pr", m0["no_skill"])})


t = load("stage_e/text_metrics.json")
if t and "cv5" in t["protocols"]:
    ms = [m for m in ["majority", "context", "tfidf", "deberta"] if m in t["protocols"]["cv5"]]
    model_panel(t["protocols"]["cv5"], "claim", "T1 claim type, 5-fold race-grouped CV", "fig2_t1_claims.png", ms, "T1_cv5")
    model_panel(t["protocols"]["cv5"], "arousal", "T3 arousal from text, 5-fold CV", "fig3_t3_text_arousal.png", ms, "T3text_cv5")

ar = load("stage_e/arousal_metrics.json")
if ar:
    panel = dict(ar["cv5"])
    ms = ["majority", "audio", "text", "fused"]
    ftj = load("stage_e/wavlm_ft_metrics.json")
    if ftj:
        panel["wavlm_ft"] = ftj["protocols"]["cv5"]["metrics"]
        ms.append("wavlm_ft")
    if t and "cv5" in t["protocols"]:
        panel["deberta_text"] = t["protocols"]["cv5"]["deberta"]["arousal"]
        ms.append("deberta_text")
    model_panel(panel, None, "T3 arousal, 5-fold race-grouped CV", "fig4_t3_arousal.png", ms, "T3_cv5")

r = load("stage_e/t2_rules.json")
if r:
    rows = []
    for c, v in r["claims"].items():
        p = v.get("thresholds", {}).get(v["primary_rule"])
        if p and p["consistency"] is not None:
            rows.append((f"{c} (n={v['n_in_race']})", p))
    if rows:
        fig, ax = plt.subplots(figsize=(7, 0.7 * len(rows) + 1.6))
        ys = list(range(len(rows)))[::-1]
        for y, (lab, p) in zip(ys, rows):
            ax.plot([p["base_rate_matched"]], [y], "o", mfc="none", mec=GREY, ms=8, mew=2)
            ax.plot([p["other_claim_rate"]], [y + 0.18], "s", color=ORANGE, ms=7)
            ax.plot([p["placebo_rate"]], [y - 0.18], "D", color=AQUA, ms=7)
            ax.plot([p["consistency"]], [y], "o", color=BLUE, ms=9, mec="#fcfcfb", mew=2)
            xr = max(v for v in [p["consistency"], p["base_rate_matched"], p["other_claim_rate"], p["placebo_rate"]] if v == v)
            ax.annotate(f"{p['consistency']:.2f} at claim", (xr, y), xytext=(12, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9, color=INK)
            summary.append({"finding": "T2_rules", "item": lab, "value": p["consistency"],
                            "base": p["base_rate_matched"], "other_claims": p["other_claim_rate"], "placebo": p["placebo_rate"]})
        ax.set_yticks(ys, [lab for lab, _ in rows])
        ax.set_xlabel("Share of laps where the pre-registered rule fires")
        ax.set_xlim(right=min(1.12, ax.get_xlim()[1] + 0.12))
        ax.set_title("T2 telemetry-consistency (primary thresholds)", loc="left", color=INK)
        h = [plt.Line2D([], [], marker="o", color=BLUE, ls="", label="at the claim"),
             plt.Line2D([], [], marker="s", color=ORANGE, ls="", label="other claims, same lap logic"),
             plt.Line2D([], [], marker="D", color=AQUA, ls="", label="3 laps before (placebo)"),
             plt.Line2D([], [], marker="o", mfc="none", mec=GREY, ls="", label="car's base rate")]
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig5_t2_rules.png")

n = load("stage_e/t2_neural.json")
if n:
    ms = [m for m in ["majority", "context", "telemetry_lr", "neural_concat", "neural_gated",
                      "placebo_neural_gated"] if m in n["models"]]
    model_panel(n["models"], None, "T2 neural: which claim does telemetry support?", "fig6_t2_neural.png", ms, "T2_neural")

g = load("stage_g/t4_results.json")
if g:
    fig, ax = plt.subplots(figsize=(7, 2.7))
    labels, ys = [], []
    for i, (y, v) in enumerate(g["outcomes"].items()):
        mde = g["mde"][y]["mde_s_per_unit_x"]
        ax.axvspan(-mde, mde, ymin=(1 - i) / 2 + 0.05, ymax=(2 - i) / 2 - 0.05, color=GREY, alpha=0.15, lw=0)
        ax.plot(v["ci95_mixed"], [1 - i] * 2, color=BLUE, lw=2)
        ax.plot(v["beta_x_mixed"], 1 - i, "o", color=BLUE, ms=8, mec="#fcfcfb", mew=2)
        ax.annotate(f"{v['beta_x_mixed']:+.3f} s  (MDE ±{mde:.2f})", (v["beta_x_mixed"], 1 - i), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9)
        labels.append({"Y1": "mean residual, laps L+1..L+3", "Y2": "SD of residuals, L+1..L+5"}[y])
        summary.append({"finding": "T4", "item": y, "value": v["beta_x_mixed"], "ci_lo": v["ci95_mixed"][0],
                        "ci_hi": v["ci95_mixed"][1], "mde": mde})
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks([1, 0], labels)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Effect of arousal, calm -> aroused (seconds, 95% CI)\ngrey band = smaller than the minimum detectable effect")
    ax.set_title("T4: arousal -> subsequent lap performance", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_t4.png")

lc = load("stage_e/learning_curve.json")
if lc:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    xs = [lc["n_train_mean"][str(f)] for f in lc["fractions"]]
    for m, col, mk in [("deberta", BLUE, "o"), ("tfidf", ORANGE, "s")]:
        ys = [lc[m][str(f)]["macro"] for f in lc["fractions"]]
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=7)
        ax.annotate({"deberta": "DeBERTa (fine-tuned)", "tfidf": "TF-IDF"}[m], (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
        for x_, y_ in zip(xs, ys):
            summary.append({"finding": "learning_curve", "item": f"{m}@{x_:.0f}", "value": y_})
    ax.set_xlabel("Training examples per fold (mean)")
    ax.set_ylabel("Macro AUC-PR")
    ax.set_xlim(right=xs[-1] * 1.35)
    ax.set_title("T1 learning curve (5-fold race-grouped CV)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig9_learning_curve.png")

pd.DataFrame(summary).to_csv(res / "summary.csv", index=False)
print("figures:", sorted(p.name for p in fig_dir.glob("*.png")), "| summary rows:", len(summary))
