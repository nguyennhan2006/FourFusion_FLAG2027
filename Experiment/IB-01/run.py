"""IB-01 — ImageBind zero-shot face <-> voice on MAV-Celeb v1/v2 (true labels, no training).

The FLAG_10 probe gave 23.73 / 34.02 (ng / g, train-centred) on v4 train, but v4 has no video ids, so a positive trial
may pair a face and a voice of the SAME video: ImageBind embeds scenes, and a shared recording could be matched
instead of the person. v1/v2 carry video ids, so here positives are forced to come from a DIFFERENT video (as in
EXT-02/03), with same-video positives as the comparison that measures the session effect.

Trials per (source, voice language): up to 15 voices per person; each voice gets one positive (same person, other
video) and one negative (other person; same gender for 'g', v1 only). 5 seeds. Scores: raw cosine, and cosine after
subtracting the source-wide mean face / voice (the notebook's "train-centred"). Pre-registered, no fitted parameter.
Output: results.csv."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle"))
from flag_lib import eer_from_scores, l2n

FM = R / "kaggle/output/feats_models"          # FLAG_10 output
FX = R / "kaggle/output/feats_ext_full"        # meta csv of the same rows
SEEDS, PER_PERSON = range(1, 6), 15


def load(src):
    fm, vm = pd.read_csv(FX / f"ext_{src}_face_meta.csv"), pd.read_csv(FX / f"ext_{src}_voice_meta.csv")
    F = np.load(FM / f"ext_{src}_face_ibv.npy").astype(np.float32)
    A = np.load(FM / f"ext_{src}_voice_iba.npy").astype(np.float32)
    assert len(F) == len(fm) and len(A) == len(vm)
    key, gen = {}, {}
    sp = FX / f"ext_{src}_speakers.csv"            # v1 only: merges ids of one person, gives gender
    if sp.exists():
        for r in pd.read_csv(sp).itertuples():
            key[r.ids] = str(r.name).strip().lower()
            gen[key[r.ids]] = "m" if str(r.gender).lower().startswith("m") else "f"
    fm["p"], vm["p"] = fm.spk.map(lambda x: key.get(x, x)), vm.spk.map(lambda x: key.get(x, x))
    return F, A, fm, vm, gen


def trials(fm, vm, gen, lang, seed, same_video, same_gender):
    rng = np.random.RandomState(seed)
    faces_of = {p: g for p, g in fm.groupby("p")}
    persons = np.array(sorted(faces_of))
    vi, fi, lab = [], [], []
    for p, g in vm[vm.lang == lang].groupby("p"):
        if p not in faces_of:
            continue
        for v in rng.choice(g.index, min(len(g), PER_PERSON), replace=False):
            fp = faces_of[p]
            fp = fp[(fp.video == vm.video[v]) if same_video else (fp.video != vm.video[v])]
            others = [q for q in persons if q != p and (not same_gender or gen.get(q) == gen.get(p))]
            if not len(fp) or not others:
                continue
            q = others[rng.randint(len(others))]
            vi += [v, v]; fi += [rng.choice(fp.index), rng.choice(faces_of[q].index)]; lab += [1, 0]
    return np.array(fi), np.array(vi), np.array(lab)


rows = []
for src, langs in [("v1_complete", ["english", "urdu"]), ("v2_complete", ["english", "hindi"])]:
    F, A, fm, vm, gen = load(src)
    score = {"raw": (l2n(F), l2n(A)), "centred": (l2n(F - F.mean(0)), l2n(A - A.mean(0)))}
    protos = ["ng", "g"] if gen else ["ng"]
    for lang in langs:
        for pos in ("other video", "same video"):
            for proto in protos:
                for sc, (Fn, An) in score.items():
                    e, n = [], 0
                    for s in SEEDS:
                        fi, vi, lab = trials(fm, vm, gen, lang, s, pos == "same video", proto == "g")
                        e.append(eer_from_scores(np.sum(Fn[fi] * An[vi], 1), lab)); n = len(lab)
                    rows.append(dict(source=src, voice_lang=lang, positive=pos, protocol=proto, score=sc,
                                     eer=round(float(np.mean(e)), 2), sd=round(float(np.std(e)), 2), trials=n))
                    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).parent / "results.csv", index=False)
print(out.pivot_table(index=["source", "voice_lang", "protocol", "score"], columns="positive", values="eer").to_string())
