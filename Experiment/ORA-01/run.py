"""ORA-01 — which modality's per-sample noise limits us?  Oracle substitution on dev (diagnostic only, never submitted).
Model: EXP-007 recipe, 3 seeds, trained on all 70 train speakers. Person sets from SEL-01b pseudo-identities:
  face person  = ArcFace face cluster of the trial face
  voice person = face cluster of the voice's owner (nearest pseudo-positive voice, own ECAPA-192)
Scores: base cos(u_f, w_v) | FACE-oracle cos(mean u of the face person, w_v) | VOICE-oracle cos(u_f, mean w of the voice
person) | BOTH.  Judged on the INDEPENDENT btc pseudo-labels (VGG + organiser voice clusters) and on arc labels.
Also: the same on TRAIN held-out speakers with TRUE labels (no pseudo-labels at all)."""
import os, sys, numpy as np, torch
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(10)
import flag_v2 as V
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)
LAB = {l: np.load(R / f"Experiment/SEL-01_selector/pseudo_labels_{l}.npz") for l in ["arc", "btc"]}
Xf, Xv = store.Xf, store.Xv
l2n = V.l2n


def oracle_scores(u, w, fperson, vperson):
    mu = {p: l2n(u[fperson == p].mean(0, keepdims=True))[0] for p in np.unique(fperson)}
    mw = {p: l2n(w[vperson == p].mean(0, keepdims=True))[0] for p in np.unique(vperson)}
    U = np.stack([mu[p] for p in fperson]); W = np.stack([mw[p] for p in vperson])
    return dict(base=np.sum(u * w, 1), face_oracle=np.sum(U * w, 1), voice_oracle=np.sum(u * W, 1), both=np.sum(U * W, 1))


# ---- (A) train held-out speakers, TRUE labels
print("(A) v4 train, held-out speakers, TRUE labels (gender protocol trials)")
for seed in (1, 2, 3):
    tr, va = V.speaker_split(store.spk, seed); tri, vai = np.where(tr)[0], np.where(va)[0]
    net, prep = V.train_one(store, REC, tri, seed=seed)
    u, w = V.embed(net, prep, Xf[vai], Xv[vai], prep.f(Xf[tri]).mean(0), prep.v(Xv[tri]).mean(0))
    sp = store.spk[vai]; fi, vj, lab = V.build_trials(sp, store.gmap, seed=seed, same_gender=True)
    # oracle over the held-out person's OTHER samples (exclude the trial's own sample from its mean is negligible here)
    S = oracle_scores(u, w, sp, sp)
    base = V.eer_from_scores(np.sum(u[fi] * w[vj], 1), lab)
    mu = {p: l2n(u[sp == p].mean(0, keepdims=True))[0] for p in np.unique(sp)}
    mw = {p: l2n(w[sp == p].mean(0, keepdims=True))[0] for p in np.unique(sp)}
    Uo = np.stack([mu[p] for p in sp[fi]]); Wo = np.stack([mw[p] for p in sp[vj]])
    print(f"  split {seed}: base {base:.2f} | face-oracle {V.eer_from_scores(np.sum(Uo * w[vj], 1), lab):.2f} | "
          f"voice-oracle {V.eer_from_scores(np.sum(u[fi] * Wo, 1), lab):.2f} | both {V.eer_from_scores(np.sum(Uo * Wo, 1), lab):.2f}", flush=True)

# ---- (B) dev, pseudo-identities
print("\n(B) dev, pseudo-identities (model trained on all 70 speakers, 3 seeds averaged)")
nets = [V.train_one(store, REC, np.arange(len(store.spk)), seed=s) for s in (1, 101, 201)]
FE = R / "kaggle/output/feats_v2"
for k in V.CELLS:
    tk = f"{k[0]}_{k[1]}"; c = "/".join(k)
    a, b = store.face("vgg", c), store.voice("given", c)
    fc, pl, kn = LAB["arc"][tk + "_fc"], LAB["arc"][tk + "_lab"], LAB["arc"][tk + "_known"]
    ev = l2n(np.load(FE / f"voice_ecapa192_{tk}.npy")); ev = l2n(ev - ev.mean(0))
    pos = np.where(kn & (pl == 1))[0]
    vperson = np.array([fc[pos[np.argmax(ev[pos] @ ev[i])]] for i in range(len(ev))])
    acc = {}
    for net, prep in nets:
        u, w = V.embed(net, prep, a, b, prep.f(Xf).mean(0), prep.v(Xv).mean(0))
        for n, s in oracle_scores(u, w, fc, vperson).items():
            acc[n] = acc.get(n, 0) + V.z(s)
    line = []
    for lb in ("btc", "arc"):
        kk, yy = LAB[lb][tk + "_known"], LAB[lb][tk + "_lab"]
        line.append(f"[{lb}] " + " ".join(f"{n} {V.eer_from_scores(s[kk], yy[kk]):.2f}" for n, s in acc.items()))
    print(f"  {c:18s} " + " | ".join(line), flush=True)
