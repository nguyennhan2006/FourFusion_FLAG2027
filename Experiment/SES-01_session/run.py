"""SES-01 — is there SESSION (same-video) information in the organiser features, and are dev positives same-session?

Train rows are (audio clip, a face frame of that clip); several rows share one clip (vgrp). So:
  instance model: InfoNCE where positives = rows of the SAME clip only (other clips of the same speaker = negatives)
  speaker model : InfoNCE where positives = same speaker (our usual recipe)
Test 1 (train, held-out speakers): AUC( same-clip pairs  vs  same-speaker different-clip pairs ).
Test 2 (dev, pseudo-identities SEL-01b): for pseudo-positive trials (f, v), is s(f, v) > s(f, v') where v' is the voice
of ANOTHER pseudo-positive trial of the same person in the same file?  If yes, dev positives share the session.
"""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E.parent / "kaggle"))
torch.set_num_threads(4)
import flag_lib as L
Xf, Xv, spk, gmap, vgrp = L.load_train(E.parent / "kaggle_upload")
CELLS = [("no_gender", "English"), ("no_gender", "Bangla"), ("gender", "English"), ("gender", "Bangla")]
LAB = np.load(E / "SEL-01_selector/pseudo_labels_arc.npz")


def train(tri, pos_key, seed=0, epochs=40):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    prep = L.Prep(Xf[tri], Xv[tri])
    Fa, Va = torch.tensor(prep.f(Xf[tri])), torch.tensor(prep.v(Xv[tri]))
    key = torch.tensor(pd.factorize(pos_key[tri])[0])
    br = lambda d: nn.Sequential(nn.Dropout(.3), nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(.3), nn.Linear(512, 128))
    bf, bv = br(Fa.shape[1]), br(Va.shape[1])
    opt = torch.optim.AdamW(list(bf.parameters()) + list(bv.parameters()), 1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        bf.train(); bv.train(); o = rng.permutation(len(tri))
        for s in range(0, len(o) - 128, 256):
            b = torch.tensor(o[s:s + 256]); u, w = F.normalize(bf(Fa[b]), dim=1), F.normalize(bv(Va[b]), dim=1)
            S = u @ w.T / 0.07; same = (key[b][:, None] == key[b][None]).float()
            loss = 0.5 * ((-(F.log_softmax(S, 1) * same).sum(1) / same.sum(1)).mean() + (-(F.log_softmax(S, 0) * same).sum(0) / same.sum(0)).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    bf.eval(); bv.eval()
    mu_f, mu_v = prep.f(Xf[tri]).mean(0), prep.v(Xv[tri]).mean(0)

    @torch.no_grad()
    def emb(A, B):
        A = L.center_to(prep.f(A), mu_f); B = L.center_to(prep.v(B), mu_v)
        return F.normalize(bf(torch.tensor(A)), dim=1).numpy(), F.normalize(bv(torch.tensor(B)), dim=1).numpy()
    return emb


res = []
for seed in (1, 2, 3):
    tr, va = L.speaker_split(spk, seed); tri, vai = np.where(tr)[0], np.where(va)[0]
    # held-out pairs: same clip (different face frame of that clip) vs same speaker, different clip
    rng = np.random.RandomState(seed); same_clip, other_clip = [], []
    for i in vai:
        sc = [j for j in vai if vgrp[j] == vgrp[i] and j != i]
        oc = [j for j in vai if spk[j] == spk[i] and vgrp[j] != vgrp[i]]
        if sc and oc:
            same_clip.append((i, rng.choice(sc))); other_clip.append((i, rng.choice(oc)))
    P = np.array(same_clip + other_clip); y = np.r_[np.ones(len(same_clip)), np.zeros(len(other_clip))]
    for name, key in [("instance (same clip)", vgrp.astype(str)), ("speaker (usual)", spk)]:
        emb = train(tri, np.asarray(key), seed)
        u, w = emb(Xf[vai], Xv[vai]); pos = {j: k for k, j in enumerate(vai)}
        s = np.array([u[pos[a]] @ w[pos[b]] for a, b in P])     # face of row a with voice of row b
        # also: speaker-level EER on held-out people (does the instance model still know identity?)
        fi, vj, lab = L.build_trials(spk[vai], gmap, seed=seed, same_gender=True)
        e_g = L.eer_from_scores(np.sum(u[fi] * w[vj], 1), lab)
        res.append(dict(seed=seed, model=name, auc_sameclip_vs_otherclip=round(roc_auc_score(y, s), 3), n=len(P) // 2, int_g=round(e_g, 2)))
        print(res[-1], flush=True)
        if seed == 1:
            # Test 2 on dev with the model trained on this split
            for k in CELLS:
                a, b, t = L.load_dev(*k, data=E.parent / "kaggle_upload")
                U, W = emb(a, b); tk = f"{k[0]}_{k[1]}"
                kn, lb, fc = LAB[tk + "_known"], LAB[tk + "_lab"], LAB[tk + "_fc"]
                pos_idx = np.where(kn & (lb == 1))[0]
                by = {}
                for i in pos_idx: by.setdefault(fc[i], []).append(i)
                own, cross = [], []
                for i in pos_idx:
                    others = [j for j in by[fc[i]] if j != i]
                    if others:
                        j = others[np.random.RandomState(i).randint(len(others))]
                        own.append(U[i] @ W[i]); cross.append(U[i] @ W[j])
                own, cross = np.array(own), np.array(cross)
                print(f"   dev {tk:18s} [{name}] P(s(f,v_own) > s(f,v_same-person-other-trial)) = {np.mean(own > cross):.3f}  (n={len(own)})", flush=True)
pd.DataFrame(res).to_csv("ses01.csv", index=False)
print(pd.DataFrame(res).groupby("model")[["auc_sameclip_vs_otherclip", "int_g"]].mean().round(3))
