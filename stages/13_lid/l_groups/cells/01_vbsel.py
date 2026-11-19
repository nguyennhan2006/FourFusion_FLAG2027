print("[cell] l_groups: South-Asian VoxCeleb2 speakers (VBSEL-01 vox2_sa_speakers.json)", flush=True)
# New cell (FourFusion reproduction repo, 19/11/2026). Rebuilds Experiment/VBSEL-01/vox2_sa_speakers.json, the frozen list that the
# F-family stages carry in their embedded code (VBSEL-01/run.py line 20), from the FLAG_42 output (VoxLingua107 probabilities of the
# FLAG_37 VoxCeleb2 clips). Rule = Experiment/EDA-VB/eda_vb.py section (4), the computation that produced the file on 10/10/2026:
#   top-1 VoxLingua107 code of each clip; share of a speaker's clips whose code is South-Asian (hi ur pa ta te mr gu ne si ml kn);
#   keep share >= 0.5; remove the Bengali-flagged speakers of the FLAG_42 rule (lid_summary.json speakers_dropped_ids).
# The file is written as the original was (json indent 0, Windows line ends) and compared by sha256 with the embedded copy.
import hashlib, json, os
from pathlib import Path
import numpy as np, pandas as pd

L42 = Path(os.environ.get("FF_VBSEL_L42", "/kaggle/input/notebooks/nguynlcnhtho/flag-2027-nb-42"))
OUT = Path(os.environ.get("FF_VBSEL_OUT", "/kaggle/working/groups"))
OUT.mkdir(parents=True, exist_ok=True)
SA = ["hi", "ur", "pa", "ta", "te", "mr", "gu", "ne", "si", "ml", "kn"]
EMBEDDED_SHA256 = "eccc765365323b2ea52a97822b17eb33ffaa4dc46e0aca00f85f2df1911dd278"   # configs/data_map.json (CRLF form)
DEFINITION = ("VoxCeleb2 speakers (FLAG_37 key list) with >= 50 % of their clips whose VoxLingua107 top-1 language is one of "
              "hi,ur,pa,ta,te,mr,gu,ne,si,ml,kn; Bengali-flagged speakers (FLAG_42 rule) removed. Frozen 10/10 before VBSEL-01 ran "
              "(EDA-VB E3).")

probs = sorted(p for p in L42.rglob("voice_lid_probs.npy") if p.is_file())
assert probs, f"voice_lid_probs.npy not found under {L42} (FLAG_42 output feats_vox2lid)"
LD = probs[0].parent
meta = pd.read_csv(LD / "voice_meta.csv", dtype={"video": str})          # FLAG_37 clip rows, same order as voice_lid_probs.npy
labels = json.loads((LD / "lid_labels.json").read_text())
P = np.load(LD / "voice_lid_probs.npy")
dropped = set(json.loads((LD / "lid_summary.json").read_text())["speakers_dropped_ids"])
assert len(meta) == len(P), (len(meta), P.shape)
codes = np.array([str(l).split(":")[0].strip() for l in labels])
top = codes[P.argmax(1)]
share = pd.Series(np.isin(top, SA)).groupby(meta.spk.values[: len(top)]).mean()
speakers = sorted(set(share.index[share >= .5]) - dropped)
text = json.dumps({"definition": DEFINITION, "languages": SA, "speakers": speakers}, indent=0)
(OUT / "vox2_sa_speakers.json").write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
sha = hashlib.sha256((OUT / "vox2_sa_speakers.json").read_bytes()).hexdigest()
rep = {"clips": int(len(top)), "speakers_scored": int(len(share)), "bengali_flagged_removed": int(len(dropped & set(share.index[share >= .5]))),
       "speakers": len(speakers), "sha256_crlf": sha, "same_as_embedded": sha == EMBEDDED_SHA256}
print(json.dumps(rep, indent=1), flush=True)
if not rep["same_as_embedded"]:
    action, why = guard_action() if "guard_action" in globals() else ("fail", "no guard policy")
    rep["guard"] = {"action": action, "why": why, "FF_MODE": os.environ.get("FF_MODE", "")}
    print(f"!! vox2_sa_speakers.json differs from the embedded copy ({why})", flush=True)
    assert action != "fail", ("vox2_sa_speakers.json differs from the copy the F-family stages embed (FLAG_42 rerun?); use the stored "
                              "FLAG_42 output or set FF_ALLOW_MISMATCH=1 (" + why + ")")
(OUT / "vbsel_check.json").write_text(json.dumps(rep, indent=1))
