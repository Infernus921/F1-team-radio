# What Drivers Claim, and What the Car Says

Spoken-language understanding on Formula 1 team radio, checked against telemetry. For every
driver radio message: who spoke, what they claimed, how aroused they sounded, and whether the car's
data is consistent with the claim. Full design brief: `BUILD_BRIEF.md`. Findings: `RESULTS.md`.

## Headline results

| Finding | Number | Verdict |
|---|---|---|
| Claim type from text beats a context-only baseline | DeBERTa 0.43 vs 0.32 macro AUC-PR (race-grouped CV); holds on 2025 test and unseen drivers | strong |
| Zero-shot LLM given the category definitions beats fine-tuning | Qwen2.5-1.5B 0.63; +0.20 over DeBERTa (95% CI 0.12–0.27) | strong |
| WavLM partial fine-tuning improves audio arousal | 0.55 vs 0.44 frozen; +0.11 (95% CI 0.03–0.19) | strong |
| Speaker attribution | 67% vs pre-set 85% gate | failed gate |
| Whisper prompt biasing | ~3% relative WER gain vs a 15% bar | weak |
| Telemetry corroborates claims | no claim type beats base rate; placebo window scores the same | null |
| Arousal → subsequent lap pace | +0.15 s, CI −0.02 to +0.31; minimum detectable effect 0.26 s | null |

## What's in this archive

| Path | Contents |
|---|---|
| `scripts/` | Pipeline, run in order: `a*` attribution → `b*` ASR/WER → `c0` annotation sample → `d*` telemetry → `e*` models → `g0` T4 → `h*` figures, deck, demo. GPU steps: `run_all_gpu.ps1` |
| `src/f1radio/` | Shared code: race-grouped splits, metrics (AUC-PR + race bootstrap + simulated chance), utterance definition |
| `tests/` | Leakage assertions: fail if any race appears in two folds (`pytest`) |
| `configs/` | `config.yaml` (single config, fixed seed 1337), F1 vocabulary |
| `labels/` | Hand labels: 380 attribution turns, 42 transcripts, 450 claim/arousal items; annotation guide |
| `results/` | Every reported number as JSON/CSV, plus the pre-registration documents |
| `paper/` | IEEE paper (LaTeX, `IEEEtran`); figures in `reports/figures/` |
| `reports/` | Figures, HTML slide deck, demo page (demo audio not included) |

## Reproducing

1. Python 3.11+, CUDA GPU (8 GB was used). `pip install -r requirements.txt` (PyTorch from the CUDA index; see the file).
2. pyannote's diarization model is gated: accept its terms on Hugging Face and run `hf auth login`.
3. Data are **not** included (the radio audio is broadcast material; coursework use only). `scripts/a0_build_metadata.py` downloads the dataset `MikCil/f1-team-radio`; `scripts/d0_fetch_telemetry.py` fetches telemetry via FastF1.
4. Run the scripts in the order above. Numbers in `results/` should reproduce up to GPU nondeterminism.

## Honesty notes

Analysis decisions were written down before the labels they concern existed (`results/stage_a/PREREGISTRATION.md`,
`results/stage_d/PREREGISTRATION_T2.md`). Deviations from the brief are listed in `RESULTS.md`. The demo's
seven clips are hand-picked for illustration and say so on screen. The T2 target is *telemetry-consistency*,
never truth.
