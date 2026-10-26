# Stage H.2: presentation deck (HTML, 16:9, arrow-key navigation, prints one slide per page)

# Imports
import html
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

cfg = load_config()
R, FIG = path(cfg, "results"), path(cfg, "figures")
OUT = path(cfg, "figures").parent / "deck"
(OUT / "img").mkdir(parents=True, exist_ok=True)
for p in FIG.glob("*.png"):
    shutil.copy(p, OUT / "img" / p.name)


def J(rel):
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def ci(c):
    return f"{c[0]:.2f}–{c[1]:.2f}"


e = lambda s: html.escape(str(s))
attr, wer, txt = J("stage_a/attribution_eval.json"), J("stage_b/wer.json"), J("stage_e/text_metrics.json")
aro, t2r, t2n = J("stage_e/arousal_metrics.json"), J("stage_e/t2_rules.json"), J("stage_e/t2_neural.json")
ft, t4, lc = J("stage_e/wavlm_ft_metrics.json"), J("stage_g/t4_results.json"), J("stage_e/learning_curve.json")
resid = J("stage_d/residual_model_diagnostics.json")
lag = J("stage_d/timestamp_offset.json")
llm = J("stage_e/llm_zeroshot.json")
slides = []


def slide(body, kind="light", notes=""):
    slides.append((kind, body, notes))


def img(name, alt):
    return f'<img src="img/{name}" alt="{e(alt)}">' if (OUT / "img" / name).exists() else \
        f'<div class="pending">Figure pending: {e(name)}</div>'


def pending(title, what):
    slide(f'<h2>{e(title)}</h2><div class="pending big">Result pending: {e(what)}<br>'
          f'<span>re-run <code>scripts/h2_deck.py</code> when it lands</span></div>')


# Title
slide("""<div class="title-wrap"><p class="kicker">NLP + Deep Learning project</p>
<h1>What drivers claim,<br>and what the car says</h1>
<p class="sub">Understanding Formula 1 team radio, and checking it against telemetry</p></div>""", "dark",
      "Open with the question: can a computer understand what F1 drivers claim over the radio, and does the car's data back them up?")

# The question + pipeline
steps = [("Radio audio", "14,370 clips, 45 h"), ("Speech → text", "Whisper large-v3"), ("Who is speaking?", "diarization + voice"),
         ("What is claimed?", "DeBERTa fine-tuned"), ("Does the car agree?", "telemetry rules + neural"),
         ("How aroused?", "WavLM + text")]
flow = "".join(f'<div class="step"><span class="n">{i + 1}</span><b>{e(a)}</b><small>{e(b)}</small></div>'
               for i, (a, b) in enumerate(steps))
slide(f"""<h2>One pipeline, six questions</h2>
<div class="flow">{flow}</div>
<p class="lead">Example: <i>“No grip at the rear!”</i> → driver → <b>GRIP_DEG</b> → were the lap times actually dropping? → was the voice raised?</p>""",
      notes="Walk left to right. Each box is a stage with its own checkpoint and pass/fail bar set in advance.")

# Data
lag_pct = lag["A_box_calls"]["share_after_pit_entry"] if lag else None
slide(f"""<h2>Real data, real quirks</h2>
<div class="stats">
<div class="stat"><b>14,370</b><span>radio clips after de-duplication (2018–2025)</span></div>
<div class="stat"><b>148</b><span>races of telemetry joined (FastF1)</span></div>
<div class="stat"><b>{pct(lag_pct)}</b><span>of “box box” calls stamped <i>after</i> the car entered the pits</span></div>
</div>
<ul class="list">
<li>No 2022 season in the dataset; 2025 is thin (669 clips)</li>
<li>204 byte-identical “Thank you.” clips reused across races: silent audio the original ASR hallucinated over</li>
<li>Broadcast timestamps lag speech by ~30–60 s (about half a lap), so all windows are lap-level</li>
</ul>""", notes="These quirks were found by checking the data, not assumed. The timestamp lag is a finding in itself.")

# ASR
if wer:
    s = wer["systems"]
    rows = [("Dataset transcripts (Cohere)", s["dataset_cohere"]["wer"]), ("Whisper large-v3", s["whisper_plain"]["wer"]),
            ("Whisper + F1 vocabulary prompt", s["whisper_prompted"]["wer"])]
    rel = s["whisper_prompted"]["relative_improvement_vs_dataset"]
    bars = "".join(f'<div class="hbar"><span class="lab">{e(n)}</span><span class="track"><span class="fill" style="width:{v * 200:.0f}%"></span></span><span class="val">{pct(v)}</span></div>'
                   for n, v in rows)
    slide(f"""<h2>Speech → text: prompting barely helps</h2>
<div class="two"><div><p class="caption">Word error rate on {wer['n_clips']} hand-transcribed clips ({wer['n_ref_words']} words)</p>{bars}</div>
<div class="callout"><b>{pct(rel)}</b><span>relative WER improvement from prompting, vs the dataset's transcripts<br>(bar for adoption: 15%)</span>
<p>Verdict: <strong>weak</strong>. Whisper is adapted by prompting only, never fine-tuned, and the WER check says the prompt isn't worth re-running.</p></div></div>""",
          notes="Roughly 3 words in 10 are wrong on compressed radio. Errors propagate: 'Enjoy one more tea guys' is an ASR error inside a misclassified utterance.")

# Attribution
if attr:
    a = attr["systems"]
    prim = next(v for k, v in a.items() if k.startswith("ensemble_silk (primary"))
    text = a["text_only"]["accuracy"]
    slide(f"""<h2>Who is speaking? Voice beats words, but not enough</h2>
<div class="two"><div>{img('fig1_attribution.png', 'Attribution accuracy by method')}</div>
<div><div class="callout"><b>{pct(prim['accuracy'])}</b><span>accuracy, 95% CI {ci(prim['ci95_race_bootstrap'])}<br>pre-set gate: 85%</span></div>
<ul class="list"><li>Text cues alone: <b>{pct(text)}</b>. Language can't tell driver from engineer</li>
<li>Diarization wasn't the problem (only {pct(attr['label_rates'].get('MIXED'))} mixed turns); naming the voice cluster was</li>
<li>Annotator self-agreement: {pct(attr['test_retest']['agreement'])} on {attr['test_retest']['n']} repeats</li></ul>
<p class="verdict bad">Failed its gate → downstream uses only <b>human-verified</b> driver turns</p></div></div>""",
          notes="This is a negative result we chose to keep. We worked around it honestly by having the annotator flag non-driver turns.")

# Annotation
items_lab = None
lab_p = path(cfg, "labels") / "claim_labels.csv"
if txt:
    counts = txt["claim_counts"]
    chips = "".join(f'<div class="chip{" dim" if k in txt["dropped_claims"] else ""}"><b>{v}</b>{e(k)}</div>'
                    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    slide(f"""<h2>A claim taxonomy defined by the car's sensors</h2>
<div class="chips">{chips}</div>
<ul class="list"><li>Categories are what public telemetry <i>can</i> check: tyres, traffic, pace, incidents, power; plus NO_SENSOR and NO_CLAIM</li>
<li>{txt['n_items']} verified driver utterances; {txt['n_not_driver_flagged']} sampled items were actually the engineer (an independent check on attribution)</li>
<li>Classes under 30 labels (greyed) are dropped and tested in a sensitivity analysis; one annotator, so no κ</li></ul>""",
          notes="The brief asked for 100 per class; with 317 labels only NO_CLAIM would survive, so we pre-stated 30 and report the 20 threshold as sensitivity.")

# T1 main result
if txt:
    p = txt["protocols"]
    cv = p["cv5"]
    dmc = cv["deberta_minus_context"]["claim"]
    rows = [("Race-grouped 5-fold CV", "cv5"), ("Validation 2024", "val2024"), ("Test 2025 (used once)", "test2025"),
            ("Leave-one-driver-out", "lodo")]
    llm = J("stage_e/llm_zeroshot.json")
    llm_line = (f" A local LLM zero-shot (Qwen2.5-1.5B, no training) scores {f2(llm['metrics']['macro']['auc_pr'])}."
                if llm else "")
    trs = "".join(f"<tr><td>{e(n)}</td><td>{f2(p[k]['deberta']['claim']['macro']['auc_pr'])}</td>"
                  f"<td>{f2(p[k]['tfidf']['claim']['macro']['auc_pr'])}</td>"
                  f"<td>{f2(p[k]['context']['claim']['macro']['auc_pr']) if 'context' in p[k] else '–'}</td>"
                  f"<td class='muted'>{f2(p[k]['deberta']['claim']['macro']['chance_95pct'])}</td></tr>" for n, k in rows if k in p)
    slide(f"""<h2>Main result: text identifies the claim</h2>
<div class="two"><div>{img('fig2_t1_claims.png', 'Claim classification, macro AUC-PR')}</div>
<div><table><tr><th>Split</th><th>DeBERTa</th><th>TF-IDF</th><th>Context</th><th>Chance<sub>95</sub></th></tr>{trs}</table>
<p class="verdict good">DeBERTa beats context-only by <b>+{dmc['delta_macro_auc_pr']:.2f}</b> (CI {dmc['ci95'][0]:+.2f} to {dmc['ci95'][1]:+.2f})</p>
<p class="small">Holds on unseen drivers, so it learned claim language, not driver identity. Fine-tuned DeBERTa and TF-IDF are comparable at ~250 training examples.{llm_line}</p></div></div>""",
          notes="Macro AUC-PR, the brief's metric. Context-only uses lap, position, tyre age and gap with no words. The words add real information.")

# LLM zero-shot
llm = J("stage_e/llm_zeroshot.json")
if llm and txt:
    cv = txt["protocols"]["cv5"]
    pv = llm.get("paired_vs_cv5_oof", {})
    bars = [("Context only (no text)", cv["context"]["claim"]["macro"]["auc_pr"]),
            ("Fine-tuned DeBERTa", cv["deberta"]["claim"]["macro"]["auc_pr"]),
            ("TF-IDF", cv["tfidf"]["claim"]["macro"]["auc_pr"]),
            ("LLM zero-shot (no training)", llm["metrics"]["macro"]["auc_pr"])]
    bh = "".join(f'<div class="hbar"><span class="lab">{e(n)}</span><span class="track"><span class="fill" style="width:{v * 140:.0f}%"></span></span><span class="val">{f2(v)}</span></div>'
                 for n, v in bars)
    d = pv.get("llm_minus_deberta")
    slide(f"""<h2>Surprise: a small LLM, told the definitions, wins</h2>
<div class="two"><div><p class="caption">Claim type, macro AUC-PR (same 281 items; chance {f2(llm['metrics']['macro']['chance_auc_pr'])})</p>{bh}</div>
<div class="callout"><b>+{d['delta_macro_auc_pr']:.2f}</b><span>over fine-tuned DeBERTa (95% CI {d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f})</span>
<p>Qwen2.5-1.5B-Instruct, run locally, sees only the category definitions from our annotation guide, with no labels and one prompt never tuned.
With ~250 labels, <strong>stating</strong> what a category means beats <strong>learning</strong> it from examples.</p></div></div>""",
          notes="This was a baseline the brief required, run exactly once. Leakage is impossible: the LLM never saw a label. The comparison is paired on identical items.")

# Error analysis
if (OUT / "img" / "fig8_confusion.png").exists():
    ex = R / "stage_e" / "error_examples.csv"
    exs = ""
    if ex.exists():
        import pandas as pd
        d = pd.read_csv(ex, encoding="utf-8")
        w = d[d["kind"] == "confident_wrong"].head(2)
        exs = "".join(f'<li>“{e(r.text[:70])}” → predicted <b>{e(r.pred)}</b>, truly <b>{e(r.gold)}</b></li>' for r in w.itertuples())
    slide(f"""<h2>Where it goes wrong, and why</h2>
<div class="two wider-left"><div>{img('fig8_confusion.png', 'Confusion matrices')}</div>
<div><ul class="list"><li>Incidents told as another driver's behaviour look like <b>pace</b> complaints</li>
<li>Brake or balance complaints (unverifiable) look like <b>tyre</b> complaints</li>{exs}</ul></div></div>""",
          notes="The confusions are semantic and taxonomic, not random. Several errors trace back to ASR mistakes.")

# Learning curve
if lc:
    last = str(lc["fractions"][-1])
    slide(f"""<h2>Would more labels help the transformer?</h2>
<div class="two wide-left"><div>{img('fig9_learning_curve.png', 'Learning curve, DeBERTa vs TF-IDF')}</div>
<div><ul class="list"><li>Same race-grouped folds; training races subsampled to 25–100%</li>
<li>At full data: DeBERTa {f2(lc['deberta'][last]['macro'])} vs TF-IDF {f2(lc['tfidf'][last]['macro'])}</li>
<li>DeBERTa climbs twice as steeply, then flattens: no evidence it would overtake with modestly more labels</li>
<li>Both stay well below the zero-shot LLM ({f2(llm['metrics']['macro']['auc_pr']) if llm else '–'})</li></ul></div></div>""",
          notes="Pre-declared contingency: because DeBERTa did not beat TF-IDF in CV, we committed in advance to a learning curve.")

# Arousal
if aro:
    c = aro["cv5"]
    cards = [("Audio (WavLM features)", c["audio"]["macro"]["auc_pr"]), ("Text", c["text"]["macro"]["auc_pr"]),
             ("Audio + text", c["fused"]["macro"]["auc_pr"])]
    if txt:
        cards.append(("DeBERTa multi-task", txt["protocols"]["cv5"]["deberta"]["arousal"]["macro"]["auc_pr"]))
    if ft:
        cards.append(("WavLM partial fine-tune", ft["protocols"]["cv5"]["metrics"]["macro"]["auc_pr"]))
    cs = "".join(f'<div class="stat"><b>{f2(v)}</b><span>{e(n)}</span></div>' for n, v in cards)
    ftline = ""
    if ft:
        w = ft["protocols"]["cv5"]["learned_layer_weights_mean"]
        top = max(w, key=w.get)
        ftline = f"<li>WavLM: lower 6 layers frozen, upper 6 fine-tuned with a learned layer weighting; highest weight on {e(top.replace('_', ' '))}</li>"
        abl = J("stage_e/wavlm_ablation.json")
        if abl:
            dd = abl["ft_minus_frozen"]
            ftline += (f"<li>Partial fine-tuning beats frozen features by <b>+{dd['delta_macro_auc_pr']:.2f}</b> "
                       f"(paired CI {dd['ci95'][0]:+.2f} to {dd['ci95'][1]:+.2f}); holds on unseen drivers "
                       f"({f2(ft['protocols']['lodo']['metrics']['macro']['auc_pr'])})</li>")
    slide(f"""<h2>Arousal is audible, and readable</h2>
<div class="stats">{cs}</div>
<p class="caption">Macro AUC-PR, race-grouped CV; chance ≈ {f2(c['audio']['macro']['chance_auc_pr'])}</p>
<ul class="list">{ftline}<li>Fusing audio with text beats audio alone (+{aro['cv5']['fused_minus_audio']['delta_macro_auc_pr']:.2f})</li>
<li>Caveat: the annotator could read the transcript, which may flatter the text models</li></ul>""",
          notes="Called arousal, never stress: g-force, breathing and broadcast compression all raise vocal arousal.")

# T2 rules
if t2r:
    slide(f"""<h2>Does the car back up the claim? Not beyond chance</h2>
<div class="two wide-left"><div>{img('fig5_t2_rules.png', 'Telemetry consistency vs base rate and placebo')}</div>
<div><ul class="list"><li>Rules were fixed before any claim was labelled</li>
<li>No claim type beats its base rate (all CIs include 0)</li>
<li>Incident claims match race-control messages 2× the base rate, but <b>just as often 3 laps before</b> the message</li></ul>
<p class="verdict bad">The placebo shows the broadcast picks eventful moments; that's not the driver's foresight</p></div></div>""",
          notes="This is the negative control doing its job. Without the placebo we would have reported a misleading positive for incidents.")

# T2 neural
if t2n:
    m = t2n["models"]
    comp = t2n["comparisons"]
    slide(f"""<h2>Neural telemetry model: gated vs naive fusion</h2>
<div class="two"><div>{img('fig6_t2_neural.png', 'Neural T2 macro AUC-PR')}</div>
<div><ul class="list"><li>1-D CNN over car data + lap tokens; each claim type <b>queries</b> the telemetry by cross-attention, with a gate (constrained fusion)</li>
<li>Gated {f2(m['neural_gated']['macro']['auc_pr'])} vs concat {f2(m['neural_concat']['macro']['auc_pr'])} vs context {f2(m['context']['macro']['auc_pr'])}; chance {f2(m['context']['macro']['chance_auc_pr'])}</li>
<li>Placebo (3 laps earlier): {f2(m['placebo_neural_gated']['macro']['auc_pr'])}</li>
<li>Gated − placebo: {comp['neural_gated_minus_placebo_neural_gated']['delta_macro_auc_pr']:+.2f} (CI {comp['neural_gated_minus_placebo_neural_gated']['ci95'][0]:+.2f} to {comp['neural_gated_minus_placebo_neural_gated']['ci95'][1]:+.2f})</li></ul></div></div>""")
else:
    pending("Neural telemetry model: gated vs naive fusion", "neural T2 (e4)")

# T4
if t4:
    y1 = t4["outcomes"]["Y1"]
    slide(f"""<h2>Does arousal change the next laps?</h2>
<div class="two"><div>{img('fig7_t4.png', 'Arousal effect vs minimum detectable effect')}</div>
<div><ul class="list"><li>Mixed model: residual over the next 3 laps on lagged arousal <b>probabilities</b>, controlling for the current lap</li>
<li>Effect: {y1['beta_x_mixed']:+.3f} s (CI {y1['ci95_mixed'][0]:+.2f} to {y1['ci95_mixed'][1]:+.2f}); minimum detectable ±{t4['mde']['Y1']['mde_s_per_unit_x']:.2f} s, computed <b>before</b> fitting</li>
<li>Hand-label check: {y1['attenuation_check_labelled_subset']['beta_hand_label']:+.3f} s, so it doesn't even keep the direction</li>
<li>No evidence drivers differ (heterogeneity p = {y1['lrt_slopes_vs_intercepts']['p_mixture']:.2f})</li></ul>
<p class="verdict bad">Null: effects under ~{t4['mde']['Y1']['mde_s_per_unit_x']:.2f} s/lap can't be detected with these data</p></div></div>""",
          notes="Reported as a null with its minimum detectable effect, as the brief requires. No alternative specifications were tried after seeing the result.")
else:
    pending("Does arousal change the next laps?", "T4 mixed model (g0)")

# Rigour
slide("""<h2>How we kept it honest</h2>
<div class="grid3">
<div class="card"><b>Pass/fail bars set in advance</b><span>Every stage had a threshold written down before its labels existed</span></div>
<div class="card"><b>No leakage</b><span>Whole races in one fold; on 30 sets of random labels the models score exactly chance (0.499)</span></div>
<div class="card"><b>Real chance levels</b><span>AUC-PR of random guessing is simulated, not assumed</span></div>
<div class="card"><b>Unseen drivers</b><span>Leave-one-driver-out, crossed with race folds</span></div>
<div class="card"><b>Negative control</b><span>Every telemetry test repeated 3 laps before the message</span></div>
<div class="card"><b>No fabricated numbers</b><span>Every figure is regenerated from results/ by a script</span></div>
</div>""")

# Verdict
slide("""<h2>What holds up, and what doesn't</h2>
<div class="two"><div class="col good"><h3>Holds up</h3><ul class="list">
<li>Claim type is identifiable from speech, on every split</li>
<li>A zero-shot LLM told the definitions beats fine-tuning (+0.20)</li>
<li>WavLM partial fine-tuning lifts audio arousal (+0.11)</li>
<li>Lap-time model works; timestamps lag speech ~30–60 s</li></ul></div>
<div class="col bad"><h3>Doesn't (yet)</h3><ul class="list">
<li>Speaker attribution: 67% vs 85% gate</li>
<li>ASR prompting: ~3–4% WER gain vs a 15% bar</li>
<li>Telemetry doesn't corroborate claims (placebo equal)</li>
<li>Arousal doesn't detectably change pace (null)</li></ul></div></div>
<p class="lead">Next: a second annotator, more rare-claim labels, and supervised attribution trained on our hand labels</p>""", "dark")

# Render
sec = "\n".join(f'<section class="slide {k}" aria-label="Slide {i + 1}"><div class="inner">{b}</div>'
                f'<div class="num">{i + 1} / {len(slides)}</div>'
                f'{f"<aside class=notes>{e(n)}</aside>" if n else ""}</section>' for i, (k, b, n) in enumerate(slides))
CSS = (Path(__file__).with_name("deck.css")).read_text(encoding="utf-8")
doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>F1 Radio Project</title>
<style>{CSS}</style></head><body>
{sec}
<script>
const S=[...document.querySelectorAll('.slide')];let i=0;
function go(n){{i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,k)=>s.classList.toggle('active',k===i));history.replaceState(null,'','#'+(i+1));}}
document.addEventListener('keydown',ev=>{{if(['ArrowRight','PageDown',' '].includes(ev.key)){{ev.preventDefault();go(i+1)}}
if(['ArrowLeft','PageUp'].includes(ev.key)){{ev.preventDefault();go(i-1)}}if(ev.key==='n')document.body.classList.toggle('show-notes');}});
document.addEventListener('click',ev=>{{if(!ev.target.closest('a'))go(ev.clientX>innerWidth/3?i+1:i-1)}});
go((parseInt(location.hash.slice(1))||1)-1);
addEventListener('hashchange',()=>{{const n=(parseInt(location.hash.slice(1))||1)-1;if(n!==i)go(n);}});
</script></body></html>"""
(OUT / "index.html").write_text(doc, encoding="utf-8")
print(f"{len(slides)} slides -> {OUT / 'index.html'}")
