# Download open-access copies of the cited papers into references/ (official sources only)

# Imports
import urllib.request
from pathlib import Path

# Settings
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "references"

# Main
OUT.mkdir(exist_ok=True)

OPEN = [
    ("01_He2023_DeBERTaV3.pdf", "https://arxiv.org/pdf/2111.09543", "He, Gao, Chen (2023). DeBERTaV3. ICLR."),
    ("02_Chen2022_WavLM.pdf", "https://arxiv.org/pdf/2110.13900", "Chen et al. (2022). WavLM. IEEE JSTSP 16(6). [arXiv version]"),
    ("03_Radford2023_Whisper.pdf", "https://arxiv.org/pdf/2212.04356", "Radford et al. (2023). Robust speech recognition via large-scale weak supervision. ICML. [arXiv version]"),
    ("04_Bredin2023_pyannote21.pdf", "https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.pdf", "Bredin (2023). pyannote.audio 2.1 speaker diarization pipeline. Interspeech."),
    ("05_Plaquet2023_Powerset.pdf", "https://www.isca-archive.org/interspeech_2023/plaquet23_interspeech.pdf", "Plaquet, Bredin (2023). Powerset multi-class cross entropy loss for neural speaker diarization. Interspeech."),
    ("06_Wang2023_WeSpeaker.pdf", "https://arxiv.org/pdf/2210.17016", "Wang et al. (2023). WeSpeaker. ICASSP. [arXiv version]"),
    ("07_Yang2021_SUPERB.pdf", "https://arxiv.org/pdf/2105.01051", "Yang et al. (2021). SUPERB. Interspeech. [arXiv version]"),
    ("08_Lee2026_ConstrainedFusion.pdf", "https://arxiv.org/pdf/2603.22372", "Lee et al. (2026). Rethinking multimodal fusion for time series. arXiv:2603.22372."),
    ("09_Vaswani2017_Attention.pdf", "https://arxiv.org/pdf/1706.03762", "Vaswani et al. (2017). Attention is all you need. NeurIPS. [arXiv version]"),
    ("10_Davis2006_PR_ROC.pdf", "https://www.biostat.wisc.edu/~page/rocpr.pdf", "Davis, Goadrich (2006). The relationship between precision-recall and ROC curves. ICML. [author copy]"),
    ("11_Saito2015_PRplot.pdf", "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0118432&type=printable", "Saito, Rehmsmeier (2015). PLOS ONE 10(3): e0118432."),
    ("12_Efron1979_Bootstrap.pdf", "https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.pdf", "Efron (1979). Bootstrap methods: another look at the jackknife. Annals of Statistics 7(1)."),
    ("13_Nosek2018_Preregistration.pdf", "https://www.pnas.org/doi/pdf/10.1073/pnas.1708274114", "Nosek et al. (2018). The preregistration revolution. PNAS 115(11)."),
    ("14_Pedregosa2011_sklearn.pdf", "https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf", "Pedregosa et al. (2011). Scikit-learn. JMLR 12."),
    ("15_Wolf2020_Transformers.pdf", "https://aclanthology.org/2020.emnlp-demos.6.pdf", "Wolf et al. (2020). Transformers. EMNLP System Demonstrations."),
    ("16_Seabold2010_statsmodels.pdf", "https://proceedings.scipy.org/articles/Majora-92bf1922-011.pdf", "Seabold, Perktold (2010). statsmodels. Proc. 9th Python in Science Conf."),
    ("17_Qwen2024_Qwen25.pdf", "https://arxiv.org/pdf/2412.15115", "Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115."),
    ("18_Thorne2018_FEVER.pdf", "https://aclanthology.org/N18-1074.pdf", "Thorne et al. (2018). FEVER. NAACL-HLT. (presentation)"),
    ("19_Brown2020_GPT3.pdf", "https://arxiv.org/pdf/2005.14165", "Brown et al. (2020). Language models are few-shot learners. NeurIPS. (presentation) [arXiv version]"),
]

ok, failed = [], []
for name, url, cite in OPEN:
    dest = OUT / name
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (academic reference download)"})
        data = urllib.request.urlopen(req, timeout=60).read()
        if not data.startswith(b"%PDF"):
            raise ValueError(f"not a PDF (got {data[:15]!r})")
        dest.write_bytes(data)
        ok.append((name, len(data), cite, url))
        print(f"OK   {name}  {len(data) / 1e6:.1f} MB", flush=True)
    except Exception as e:
        failed.append((name, cite, url, f"{type(e).__name__}: {e}"[:120]))
        print(f"FAIL {name}  {type(e).__name__}: {e}"[:160], flush=True)

print(f"\n{len(ok)} downloaded ({sum(b for _, b, _, _ in ok) / 1e6:.1f} MB), {len(failed)} failed")
for f in failed:
    print("  failed:", f[0], "|", f[3])
