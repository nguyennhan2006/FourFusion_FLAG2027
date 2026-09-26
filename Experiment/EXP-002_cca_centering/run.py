"""EXP-002 — Pipeline E on the linear CCA reference: unsupervised per-file feature centering + score normalisation.
Internal simulation: val speakers are centred with the val-set mean (mirrors centring a dev file with its own mean)."""
import sys, json, zipfile
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EDA-000_raw"))
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
Xf, Xv, yf, _, txt, gmap = load_train()
SEEDS = [1, 2, 3]
CFGS = {"k4": dict(k=4, reg=1.0, pca_x=128), "k32": dict(k=32, reg=10.0, pca_x=256)}


def center(X, mode, ref_mean):
    """mode: none | file (subtract the set's own mean, add back train mean so the CCA standardisation stays valid)."""
    if mode == "none":
        return X
    return X - X.mean(0) + ref_mean


def znorm(scores):
    return (scores - scores.mean()) / (scores.std() + 1e-8)


def asnorm(Ef, Ev, fi, vj, top=200):
    """Adaptive S-norm with cohort = all other faces/voices in the same file. Returns normalised cosine (higher = same)."""
    S = Ef @ Ev.T                       # face x voice full similarity within the file
    raw = S[fi, vj]
    # enrol side: for face fi, cohort = top-`top` voices (excluding the trial voice)
    Sf = np.sort(S, axis=1)[:, -top - 1:-1]; mf, sf = Sf.mean(1), Sf.std(1) + 1e-8
    Sv = np.sort(S, axis=0)[-top - 1:-1, :]; mv, sv = Sv.mean(0), Sv.std(0) + 1e-8
    return 0.5 * ((raw - mf[fi]) / sf[fi] + (raw - mv[vj]) / sv[vj])


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        tr, va = speaker_split(yf, seed); idx = np.where(va)[0]
        trials = {sg: build_trials(yf[idx], gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg) for sg in [False, True]}
        for cname, cfg in CFGS.items():
            m = RidgeCCA(**cfg).fit(Xf[tr], Xv[tr])
            for cmode in ["none", "file", "file_voice_only"]:
                Xf_v = center(Xf[idx], "none" if cmode == "file_voice_only" else cmode, Xf[tr].mean(0))
                Xv_v = center(Xv[idx], "none" if cmode == "none" else "file", Xv[tr].mean(0))
                Ef, Ev = m.transform_x(Xf_v), m.transform_y(Xv_v)
                for sg, (fi, vj, lab) in trials.items():
                    cos = np.sum(Ef[fi] * Ev[vj], 1)
                    for snorm, sc in [("raw", cos), ("asnorm", asnorm(Ef, Ev, fi, vj))]:
                        rows.append(dict(seed=seed, cfg=cname, center=cmode, snorm=snorm,
                                         protocol="gender" if sg else "no_gender", eer=eer_from_scores(sc, lab)))
    df = pd.DataFrame(rows); df.to_csv(OUT / "exp002_internal.csv", index=False)
    piv = df.groupby(["cfg", "center", "snorm", "protocol"]).eer.agg(["mean", "std"]).round(2).unstack("protocol")
    print(piv.to_string())
    piv.to_csv(OUT / "exp002_internal_summary.csv")

    # ---------------------------------------------------------------- dev submissions: one zip per (cfg, center, snorm)
    names = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt", ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
             ("gender", "English"): "gender/sub_score_v4_English_heard.txt", ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
    models = {c: RidgeCCA(**cfg).fit(Xf, Xv) for c, cfg in CFGS.items()}
    devs = {k: load_dev(*k) for k in names}
    for cname, m in models.items():
        for cmode in ["file", "file_voice_only"]:
            for snorm in ["raw", "asnorm"]:
                zname = OUT / f"submission_EXP002_{cname}_{cmode}_{snorm}.zip"
                with zipfile.ZipFile(zname, "w") as zf:
                    for key, fn in names.items():
                        a, b, t = devs[key]
                        a2 = center(a, "none" if cmode == "file_voice_only" else "file", Xf.mean(0))
                        b2 = center(b, "file", Xv.mean(0))
                        Ef, Ev = m.transform_x(a2), m.transform_y(b2)
                        ii = np.arange(len(t))
                        cos = np.sum(Ef * Ev, 1) if snorm == "raw" else asnorm(Ef, Ev, ii, ii)
                        d = -cos if snorm == "asnorm" else 2 - 2 * cos     # lower = same
                        zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(t.pair_id, d)) + "\n")
                print("wrote", zname.name)
