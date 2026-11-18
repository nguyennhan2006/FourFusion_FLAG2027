"""Stage dp_score (new code, FourFusion 13.54 reproduction repo): DP (Different Profile) scores of the Evaluation trials from the SAVED
two-tower checkpoint (dp_towers.pt, written by the repo copy of xjf.py in stage dp_train) instead of re-training.

Same arithmetic as the last block of xjf.py ("final models + Evaluation scores", original lines 166-173):
    cos = mean over the 10 seeds of <face_tower(face - MU_F), voice_tower(voice - MU_V)>,  z = (cos - mean) / sd,
    mean / sd = the out-of-fold constants of the FLAG train trials (CONST, stored in the checkpoint; = consts.json of xjf.py).
The helpers l2 / csv_feats / face_cat / voice_cat / Tower / cos and the lambda T are VERBATIM copies of xjf.py (lines 28-45, 94,
100-105, 131-133), so a checkpoint made and scored in the same environment gives the same bits as xjf.py's own xjf_scores.npz.

    python dp_score.py --towers dp_towers.pt --eval-feats <dir of stage dp_extract_eval> --eval-zip eval_set.zip \
                       --out xjf_scores.npz [--compare <xjf_scores.npz of dp_train>]

Inputs: organiser features of the Evaluation files (evaluation_phase/<protocol>/features/<lang>_test_{faces,voices}.csv inside
eval_set.zip) + ArcFace / AdaFace / ECAPA arrays and ids_<t>.txt of stage dp_extract_eval. Output keys as xjf.py:
ids_<t>, cos_<t>, z_<t> for every t = <protocol>_<lang> whose features exist (the submission uses the two English files)."""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F


# ------------------------------------------------------------------------------------------------ verbatim from xjf.py (28-45)
def l2(x):
    x = np.asarray(x, np.float32); return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def csv_feats(z, name, d):
    a = pd.read_csv(io.BytesIO(z.read(name)), header=None).to_numpy(np.float32)
    if a.shape[1] == d + 1 and (np.isnan(a[:, -1]).all() or (a[:, -1] == np.round(a[:, -1])).all()):
        a = a[:, :-1]                                   # trailing comma, or the class label the organisers append to the train csv
    assert a.shape[1] == d and np.isfinite(a).all(), (name, a.shape)
    return a


def face_cat(vgg, ada, arc):
    return (np.hstack([l2(np.sqrt(np.maximum(vgg, 0))), l2(ada), l2(arc)]) / np.sqrt(3)).astype(np.float32)


def voice_cat(btc, e192, e6144):
    return (np.hstack([l2(btc), l2(e192), l2(e6144)]) / np.sqrt(3)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ verbatim from xjf.py (100-105)
class Tower(nn.Module):
    def __init__(self, d):
        super().__init__(); self.n = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.7), nn.Linear(256, 128))

    def forward(self, x):
        return F.normalize(self.n(x), dim=-1)


# ------------------------------------------------------------------------------------------------ verbatim from xjf.py (131-133)
@torch.no_grad()
def cos(models, xf, xv):
    return np.mean([(f(xf) * v(xv)).sum(1).numpy() for f, v, _ in models], axis=0)


# ------------------------------------------------------------------------------------------------ verbatim from xjf.py (94)
T = lambda x, mu: torch.from_numpy(np.ascontiguousarray((x - mu).astype(np.float32)))


def load_checkpoint(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:                                   # torch without the weights_only argument
        return torch.load(p, map_location="cpu")


def compare(new, ref_path):
    """Report how the new scores differ from another xjf_scores.npz (same trial ids required)."""
    R = np.load(ref_path)
    rep = {}
    for k in sorted(x for x in new if x.startswith("z_")):
        t = k[2:]
        if k not in R.files:
            rep[t] = "absent in the reference"; continue
        same_ids = bool(np.array_equal(new[f"ids_{t}"], R[f"ids_{t}"]))
        a, b = np.asarray(new[k], np.float64), np.asarray(R[k], np.float64)
        rho = float(pd.Series(a).corr(pd.Series(b), method="spearman")) if same_ids else float("nan")
        rep[t] = {"same_ids": same_ids, "bit_identical_z": bool(same_ids and np.array_equal(new[k], R[k])),
                  "max_abs_diff_z": float(np.abs(a - b).max()) if same_ids else None, "spearman_z": rho}
        print(f"  compare {t}: {rep[t]}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(description="DP scores of the Evaluation trials from the saved two-tower checkpoint")
    ap.add_argument("--towers", required=True, help="dp_towers.pt of stage dp_train")
    ap.add_argument("--eval-feats", required=True, help="folder with ids_<t>.txt, face_{arcface,adaface}_<t>.npy, voice_ecapa{192,6144}_<t>.npy")
    ap.add_argument("--eval-zip", required=True, help="eval_set.zip (organiser features of the Evaluation files)")
    ap.add_argument("--out", required=True, help="xjf_scores.npz to write")
    ap.add_argument("--compare", default=None, help="optional xjf_scores.npz to compare with (e.g. the one xjf.py wrote in dp_train)")
    a = ap.parse_args()

    ck = load_checkpoint(a.towers)
    assert str(ck.get("format", "")).startswith("FourFusion DP two-tower checkpoint"), f"not a DP checkpoint: {a.towers}"
    dims = [int(x) for x in ck["dims"]]
    models = []
    for tw in ck["towers"]:
        f, v = Tower(dims[0]), Tower(dims[1])
        f.load_state_dict(tw["face"]); v.load_state_dict(tw["voice"]); f.eval(); v.eval()
        models.append((f, v, None))
    MU_F, MU_V = ck["MU_F"].numpy(), ck["MU_V"].numpy()
    CONST = {k: (float(x[0]), float(x[1])) for k, x in ck["CONST"].items()}
    print(f"checkpoint: {len(models)} tower pairs (seeds {list(ck['seeds'])}), dims {dims}, CONST {CONST}", flush=True)

    ez = zipfile.ZipFile(a.eval_zip); FE = Path(a.eval_feats); res = {}
    for prot in ("no_gender", "gender"):
        for lang in ("English", "Bangla"):
            t = f"{prot}_{lang}"
            if not (FE / f"voice_ecapa6144_{t}.npy").exists():
                print("skip (no features):", t, flush=True); continue
            ids = (FE / f"ids_{t}.txt").read_text().split()
            f = face_cat(csv_feats(ez, f"evaluation_phase/{prot}/features/{lang}_test_faces.csv", 4096), np.load(FE / f"face_adaface_{t}.npy"), np.load(FE / f"face_arcface_{t}.npy"))
            v = voice_cat(csv_feats(ez, f"evaluation_phase/{prot}/features/{lang}_test_voices.csv", 192), np.load(FE / f"voice_ecapa192_{t}.npy"), np.load(FE / f"voice_ecapa6144_{t}.npy"))
            assert len(ids) == len(f) == len(v)
            assert f.shape[1] == dims[0] and v.shape[1] == dims[1], (t, f.shape, v.shape, dims)
            c = cos(models, T(f, MU_F), T(v, MU_V)); mu, sd = CONST[t.rsplit("_", 1)[0]]
            res[f"ids_{t}"] = np.array(ids); res[f"cos_{t}"] = c; res[f"z_{t}"] = (c - mu) / sd
            print(f"Evaluation {t}: {len(ids)} trials scored", flush=True)   # no score statistics of the test files are printed
    assert res, "no Evaluation file has DP features (run stage dp_extract_eval first)"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **res)
    print("saved", a.out, flush=True)
    if a.compare:
        rep = compare(res, a.compare)
        Path(a.out).with_name("dp_score_compare.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
