"""IB-ADAPT (docs/PLAN_EVAL.md E5a): can our paired data IMPROVE ImageBind, or only use it zero-shot? A cheap CPU gate for
the GPU LoRA notebook (FLAG_13_iblora). TRUE labels.

ImageBind embeddings are frozen (FLAG_10: face_ibv / voice_iba, v4 and MAV-Celeb v1 / v2 rows). On top of them, one
linear map per modality, initialised to the identity and pulled back to it (lam * ||W - I||^2), trained with a symmetric
multi-positive InfoNCE on (face, voice) of the same person:
  ZS        zero-shot (identity), the ImageBind term of SET-13
  AD4       adapter trained on the split's 40 v4 train persons
  ADX       adapter trained on the 40 v4 persons + the MAV-Celeb v1 / v2 persons of the 4 other folds (faces and voices of
            DIFFERENT videos when a person has several)
  ZS+ADX    0.5 z(ZS) + 0.5 z(ADX)
Evaluation (5 splits; file-centred cosine, z over the file's matrix, as SET-13):
  v4        30 held-out persons, 16 faces + 16 voices each from different rows, 50 % positives, ng / g
  urdu / hindi / v1_english / v2_english   the held-out fold's v1 / v2 persons, up to 16 voices each in that language,
            the face of a positive from ANOTHER video (the session effect of IB-01 cannot help); ng (+ g for Urdu,
            whose genders are known)
  levels    sample (every trial alone) and person (true identities as blocks, b = 0.99; SET-13 clusters get close)
Gate for FLAG_13 (LoRA): ADX better than ZS by >= 1 EER at person level on urdu AND v4, in >= 4/5 splits.
    python run.py   -> results.csv, summary printed
"""
import os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle"))
torch.set_num_threads(int(os.environ.get("ADAPT_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext_full")
import flag_v2 as V

FM = R / "kaggle/output/feats_models"
OUT = Path(__file__).parent / os.environ.get("ADAPT_OUT", "results.csv")
STEPS, LAM, TAU, LR = int(os.environ.get("ADAPT_STEPS", "1500")), 1e-2, 0.05, 1e-4
SPLITS = [int(x) for x in os.environ.get("ADAPT_SPLITS", "1,2,3,4,5").split(",")]
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
load = lambda n: np.load(FM / n).astype(np.float32)
v4F, v4A = load("face_ibv_train.npy"), load("voice_iba_train.npy")
v4S = store.spk.astype(str)
v4G = np.array([store.gmap.get(s, "u") for s in store.spk])

# MAV-Celeb rows: ids merged per person (v1 lists some people twice), genders where known
EXT = {}
for src in ["v1_complete", "v2_complete"]:
    D = store.ext_source(src)
    key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
    fm, vm = D["fm"].copy(), D["vm"].copy()
    assert len(fm) == len(load(f"ext_{src}_face_ibv.npy")) and len(vm) == len(load(f"ext_{src}_voice_iba.npy")), src
    fm["p"], vm["p"] = fm.spk.map(key), vm.spk.map(key)
    EXT[src] = dict(F=load(f"ext_{src}_face_ibv.npy"), A=load(f"ext_{src}_voice_iba.npy"), fm=fm, vm=vm, gen=D["gen_of"])
persons = sorted(set(EXT["v1_complete"]["fm"].p) | set(EXT["v2_complete"]["fm"].p))
FOLD = dict(zip(persons, np.random.RandomState(0).permutation(len(persons)) % 5))


# ============================================================================ training rows
def ext_pairs(src, keep, rng, per_person=60):
    """(face rows, voice rows, person) of one source, faces and voices from different videos when possible."""
    E = EXT[src]; fm, vm = E["fm"], E["vm"]
    fi, vi, pp = [], [], []
    for p, g in vm[vm.p.isin(keep)].groupby("p"):
        fr = fm[fm.p == p]
        if not len(fr):
            continue
        for i in rng.choice(g.index.values, min(per_person, len(g)), replace=False):
            other = fr[fr.video != vm.video[i]]
            fi.append(rng.choice((other if len(other) else fr).index.values)); vi.append(i); pp.append(p)
    return E["F"][fi], E["A"][vi], np.array(pp)


def train_adapter(Fx, Ax, P, mf, mv, seed):
    """Linear map per modality, init identity, InfoNCE on batches of 32 persons x 2 pairs."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    Fx, Ax = torch.tensor(Fx - mf), torch.tensor(Ax - mv)
    Wf, Wv = (torch.eye(Fx.shape[1], requires_grad=True) for _ in range(2))
    opt = torch.optim.Adam([Wf, Wv], lr=LR)
    by = pd.Series(np.arange(len(P))).groupby(P).apply(np.array).to_dict(); keys = list(by)
    I = torch.eye(Fx.shape[1])
    for step in range(STEPS):
        ps = rng.choice(len(keys), min(32, len(keys)), replace=False)
        idx = np.concatenate([rng.choice(by[keys[k]], 2) for k in ps])
        lab = torch.tensor(np.repeat(np.arange(len(ps)), 2))
        u, w = F.normalize(Fx[idx] @ Wf, dim=1), F.normalize(Ax[idx] @ Wv, dim=1)
        S = u @ w.T / TAU
        pos = (lab[:, None] == lab[None, :]).float()
        loss = -0.5 * ((F.log_softmax(S, 1) * pos).sum(1) / pos.sum(1) + (F.log_softmax(S, 0) * pos).sum(0) / pos.sum(0)).mean()
        loss = loss + LAM * (((Wf - I) ** 2).sum() + ((Wv - I) ** 2).sum())
        opt.zero_grad(); loss.backward(); opt.step()
    return Wf.detach().numpy(), Wv.detach().numpy()


# ============================================================================ evaluation files
def v4_file(rows, sg, rng, m=16):
    Fi, Vi, Y, nf, nv = [], [], [], [], []
    for p in np.unique(v4S[rows]):
        r = rng.permutation(rows[v4S[rows] == p]); k = min(m, len(r) // 2)
        fr, vr = r[:k], r[k:2 * k]; h = k // 2
        Fi += list(fr[:h]); Vi += list(vr[:h]); Y += [1] * h; nf += list(fr[h:]); nv += list(vr[h:])
    nv = list(rng.permutation(nv))
    for f in rng.permutation(nf):
        ok = [j for j, v in enumerate(nv) if v4S[v] != v4S[f] and (not sg or v4G[v] == v4G[f])]
        if ok:
            Fi.append(f); Vi.append(nv.pop(ok[0])); Y.append(0)
    return v4F[Fi], v4A[Vi], np.array(Y), v4S[Fi], v4S[Vi]


def ext_file(src, lang, test_p, sg, rng, m=16):
    E = EXT[src]; fm, vm, gen = E["fm"], E["vm"], E["gen"]
    Fi, Vi, Y, nf, nv = [], [], [], [], []
    for p in test_p:
        vr = vm[(vm.p == p) & (vm.lang == lang)]
        fr = fm[fm.p == p]
        if len(vr) < 2 or not len(fr):
            continue
        vs = rng.choice(vr.index.values, min(m, len(vr)), replace=False)
        for j, v in enumerate(vs):
            other = fr[fr.video != vm.video[v]]
            if not len(other):
                continue
            f = rng.choice(other.index.values)
            if j < len(vs) // 2:
                Fi.append(f); Vi.append(v); Y.append(1)
            else:
                nf.append(f); nv.append(v)
    nv = list(rng.permutation(nv))
    for f in rng.permutation(nf):
        pf = fm.p[f]
        ok = [j for j, v in enumerate(nv) if vm.p[v] != pf and (not sg or gen.get(vm.p[v], "u") == gen.get(pf, "x"))]
        if ok:
            Fi.append(f); Vi.append(nv.pop(ok[0])); Y.append(0)
    return E["F"][Fi], E["A"][Vi], np.array(Y), fm.p.values[Fi], vm.p.values[Vi]


def scores(Fx, Ax, W):
    """File-centred cosine matrix after the adapter W = (Wf, Wv); z over the matrix (as SET-13)."""
    u = V.l2n((Fx - Fx.mean(0)) @ W[0]); w = V.l2n((Ax - Ax.mean(0)) @ W[1])
    M = u @ w.T
    return (M - M.mean()) / (M.std() + 1e-9)


def block_true(M, pf, pv, b=0.99):
    cf, cv = pd.factorize(pf)[0], pd.factorize(pv)[0]
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * np.diag(M) + b * (Af @ M @ Av.T)[cf, cv]


recs = []
I = np.eye(1024, dtype=np.float32)
for s in SPLITS:
    t0 = time.time()
    tr, va = V.speaker_split(store.spk, s, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]
    rng = np.random.RandomState(s)
    train_p = [p for p in persons if FOLD[p] != s - 1]; test_p = [p for p in persons if FOLD[p] == s - 1]
    mf, mv = v4F[tri].mean(0), v4A[tri].mean(0)
    # v4 training pairs: face and voice of different rows of the same person
    P4 = v4S[tri]; fi = tri; vi = np.array([rng.choice(tri[(P4 == v4S[i]) & (tri != i)]) for i in tri])
    W = {"ZS": (I, I)}
    W["AD4"] = train_adapter(v4F[fi], v4A[vi], P4, mf, mv, seed=s)
    xs = [ext_pairs(src, train_p, rng) for src in EXT]
    W["ADX"] = train_adapter(np.concatenate([v4F[fi]] + [x[0] for x in xs]), np.concatenate([v4A[vi]] + [x[1] for x in xs]),
                             np.concatenate([P4] + [x[2] for x in xs]), mf, mv, seed=s)
    files = [("v4", "ng", v4_file(vai, False, rng)), ("v4", "g", v4_file(vai, True, rng)),
             ("urdu", "ng", ext_file("v1_complete", "urdu", test_p, False, rng)),
             ("urdu", "g", ext_file("v1_complete", "urdu", test_p, True, rng)),
             ("hindi", "ng", ext_file("v2_complete", "hindi", test_p, False, rng)),
             ("v1_english", "ng", ext_file("v1_complete", "english", test_p, False, rng)),
             ("v2_english", "ng", ext_file("v2_complete", "english", test_p, False, rng))]
    for data, prot, (Fx, Ax, y, pf, pv) in files:
        if len(y) < 20 or y.min() == y.max():
            continue
        Ms = {k: scores(Fx, Ax, w) for k, w in W.items()}
        Ms["ZS+ADX"] = 0.5 * Ms["ZS"] + 0.5 * Ms["ADX"]
        for arm, M in Ms.items():
            for level, sc in [("sample", np.diag(M)), ("person", block_true(M, pf, pv))]:
                recs.append(dict(split=s, data=data, protocol=prot, arm=arm, level=level, n=len(y),
                                 persons=len(set(pf)), eer=V.eer_from_scores(sc, y)))
    pd.DataFrame(recs).to_csv(OUT, index=False)
    print(f"split {s}: {time.time() - t0:.0f}s, ext train persons {len(train_p)}, test {len(test_p)}", flush=True)

T = pd.DataFrame(recs)
piv = T.pivot_table(index=["data", "protocol", "level"], columns="arm", values="eer").round(2)
for a in ["AD4", "ADX", "ZS+ADX"]:
    piv[f"{a}-ZS"] = (piv[a] - piv["ZS"]).round(2)
print(piv.to_string())
w = T.pivot_table(index=["split", "data", "protocol", "level"], columns="arm", values="eer")
wins = ((w["ZS"] - w["ADX"]) >= 1).groupby(level=["data", "protocol", "level"]).sum()
print("\nsplits where ADX beats ZS by >= 1 EER:\n" + wins.to_string())
print("ALL DONE", flush=True)
