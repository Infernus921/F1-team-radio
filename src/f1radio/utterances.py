# Driver-side utterances: consecutive DRIVER-attributed turns within one clip, merged

# Imports
import pandas as pd

# Settings
MIN_UTT_S = 1.0


# Functions
def driver_utterances(turns: pd.DataFrame, label_col: str = "ens_silk") -> pd.DataFrame:
    T = turns[turns["attributable"]].sort_values(["clip_id", "start"]).copy()
    T["is_drv"] = T[label_col] == "DRIVER"
    T["run"] = ((T["clip_id"] != T["clip_id"].shift()) | (T["is_drv"] != T["is_drv"].shift())).cumsum()
    D = T[T["is_drv"]].groupby("run").agg(
        clip_id=("clip_id", "first"), driver_id=("driver_id", "first"),
        start=("start", "min"), end=("end", "max"),
        text=("text", lambda s: " ".join(x for x in s if x).strip()))
    D = D[(D["end"] - D["start"] >= MIN_UTT_S)].reset_index(drop=True)
    D["utt_id"] = D["clip_id"] + "@" + D["start"].round(2).astype(str)
    return D
