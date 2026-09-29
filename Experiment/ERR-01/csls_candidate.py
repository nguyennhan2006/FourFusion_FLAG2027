"""CSLS (a=1, k=5, fixed for all cells) applied to the members we have full matrices for, at fusion level, vs FUSE-03.
English : rank(CSLS s007) + rank(CSLS s010B)               vs submitted rank(007) + rank(010B)
ng/Bn   : rank(003c zip) + rank(CSLS r2mix)                vs submitted rank(003c) + rank(r2_mix)
g/Bn    : duration gate(003c, CSLS r2mix), same a,b        vs submitted gate(003c, r2_mix)"""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "Experiment/EDA-000_raw")); from eda_utils import load_dev, eer_from_scores
LAB = {l: np.load(R / f"Experiment/SEL-01_selector/pseudo_labels_{l}.npz") for l in ["arc", "btc"]}
M = {n: np.load(R / f"Experiment/GRAPH-01/mats/{n}_matrix.npz") for n in ["s007", "s010B", "r2mix"]}
rank = lambda s: rankdata(s) / (len(s) + 1)
FN_ = {"no_gender/English": "no_gender/sub_score_v4_English_heard.txt", "gender/English": "gender/sub_score_v4_English_heard.txt",
       "no_gender/Bangla": "no_gender/sub_score_v4_Bangla_unheard.txt", "gender/Bangla": "gender/sub_score_v4_Bangla_unheard.txt"}
zs = lambda zp, fn: -pd.read_csv(zipfile.ZipFile(R / zp).open(fn), sep=" ", header=None)[1].values
dur = pd.read_csv(R / "Experiment/EDA-001_flag_questions/wav_durations.csv"); dmap = dict(zip(dur.name, dur.dur))


def csls(Mx, a=1.0, k=5):
    Mx = Mx.astype(np.float32); rf = np.sort(Mx, 1)[:, -k:].mean(1); rv = np.sort(Mx, 0)[-k:, :].mean(0)
    return np.diag(Mx) - a / 2 * (rf + rv)


out = {}
for c, fn in FN_.items():
    tk = c.replace("/", "_"); prot, lang = c.split("/")
    cur = zs("Experiment/FUSE-03/out/submission_FUSE03.zip", fn)
    if lang == "English":
        new = (rank(csls(M["s007"][c])) + rank(csls(M["s010B"][c]))) / 2
    else:
        a3, rm = rank(zs("Experiment/EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip", fn)), rank(csls(M["r2mix"][c]))
        if prot == "no_gender":
            new = (a3 + rm) / 2
        else:
            t = load_dev(prot, lang)[2]; d = np.array([dmap[f"dev_set/{prot}/{v}"] for v in t.voice])
            w = 1 / (1 + np.exp(-(0.5 * np.log(np.maximum(d, .5)) - 1.5))); new = w * a3 + (1 - w) * rm
    out[c] = new
    msg = []
    for lb, L in LAB.items():
        kn, y = L[tk + "_known"], L[tk + "_lab"]
        msg.append(f"{lb}: {eer_from_scores(cur[kn], y[kn]):.2f} -> {eer_from_scores(new[kn], y[kn]):.2f}")
    print(f"{c:18s} FUSE-03 -> CSLS candidate | " + " | ".join(msg))
np.savez("csls_candidate_scores.npz", **{k.replace("/", "_"): v for k, v in out.items()})
