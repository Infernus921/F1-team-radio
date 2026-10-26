# Speaker-attribution helpers: turn/word alignment and linguistic cues

# Imports
from __future__ import annotations

import re

# Settings
FIRST_NAMES = {
    "ALEALB01": ["alex"], "ANDANT01": ["kimi", "andrea"], "ANTGIO01": ["antonio", "gio"],
    "BREHAR01": ["brendon"], "CARSAI01": ["carlos"], "CHALEC01": ["charles"],
    "DANKVY01": ["daniil", "dany", "danny"], "DANRIC01": ["daniel", "dan"], "ESTOCO01": ["esteban", "este"],
    "FERALO01": ["fernando", "nando"], "FRACOL01": ["franco"], "GABBOR01": ["gabriel", "gabi"],
    "GEORUS01": ["george"], "GUAZHO01": ["zhou", "guanyu"], "ISAHAD01": ["isack", "isaac"],
    "JACAIT01": ["jack"], "JACDOO01": ["jack"], "KEVMAG01": ["kevin", "kev"], "KIMRAI01": ["kimi"],
    "LANNOR01": ["lando"], "LANSTR01": ["lance"], "LEWHAM01": ["lewis"], "LIALAW01": ["liam"],
    "LOGSAR01": ["logan"], "MARERI01": ["marcus"], "MAXVER01": ["max"], "MICSCH02": ["mick"],
    "NICHUL01": ["nico"], "NICLAF01": ["nicholas", "nicky"], "NIKMAZ01": ["nikita"], "NYCDEV01": ["nyck"],
    "OLIBEA01": ["ollie", "oliver"], "OSCPIA01": ["oscar"], "PIEFIT01": ["pietro"], "PIEGAS01": ["pierre"],
    "ROBKUB01": ["robert"], "ROMGRO01": ["romain"], "SEBVET01": ["seb", "sebastian"],
    "SERPER01": ["checo", "sergio"], "SERSIR01": ["sergey"], "STOVAN01": ["stoffel"],
    "VALBOT01": ["valtteri", "val"], "YUKTSU01": ["yuki"],
}

ENG_PATTERNS = [
    r"\bbox\b", r"\bbox box\b", r"\bstay out\b", r"\bstaying out\b",
    r"\b(target|delta)\b", r"\b\d:\d\d\b", r"\b\d{1,2}\.\d\b", r"\bgap\b",
    r"\b(he|he's|his|him)\b", r"\bcar (ahead|behind)\b", r"\b(p\d{1,2})\b",
    r"\bstrat\b", r"\b(mode|switch|position) \w+\b", r"\blaps? (to go|remaining)\b",
    r"\bplan [a-z]\b", r"\bwe('re| are| will|'ll) (pit|box|stay|go)\b",
    r"\b(keep|let's|push|manage|look after|bring it)\b", r"\bplease\b",
    r"\byour (tyres?|tires?|car|pace|gap|brakes?)\b", r"\byou('re| are)\b",
    r"\bwell done\b", r"\bgood job\b",
]
DRV_PATTERNS = [
    r"\b(i|i'm|i've|i'll|my|me)\b", r"\b(no|zero) grip\b", r"\bfeels?\b",
    r"\b(tyres?|tires?) (are|is) (gone|dead|finished|done)\b", r"\bcan't\b",
    r"\bwhat (is|was|happened)\b", r"\bwhy\b", r"\?", r"\bf\*+", r"\bs\*+",
    r"\bunderstood\b", r"\bcopy\b", r"\bthanks?( you)?,? guys\b",
]
_ENG = [re.compile(p) for p in ENG_PATTERNS]
_DRV = [re.compile(p) for p in DRV_PATTERNS]


# Functions
def text_cue_score(text: str, driver_id: str) -> float:
    t = text.lower()
    eng = sum(bool(p.search(t)) for p in _ENG)
    drv = sum(bool(p.search(t)) for p in _DRV)
    names = FIRST_NAMES.get(driver_id, [])
    if any(re.search(rf"\b{n}\b", t) for n in names):
        eng += 2
    return (eng - drv) / max(1, eng + drv)


def words_in_span(words, start: float, end: float) -> str:
    return "".join(w[2] for w in words if start <= (w[0] + w[1]) / 2 < end).strip()
