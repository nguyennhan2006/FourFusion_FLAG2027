"""XJF-01 step 2: small in-domain two-tower model on recognition-grade frozen encoders (recipe of the Track 3 report sub99b,
branch XJ-U), trained on OUR feature files: FLAG train (70 persons, 6 485 pairs) + MAV-Celeb v1 / v2 (154 persons).
    python Experiment/XJF-01/xjf.py            -> out/xjf_scores.npz, out/report.txt
Faces  [organiser VGGFace fc7 4096 (sqrt + L2) | AdaFace IR-101 512 | ArcFace buffalo_l 512], each block L2, concat / sqrt(3)
Voices [organiser ECAPA 192 | speechbrain ECAPA 192 | its 6144-d pooled statistics], each block L2, concat / sqrt(3)
Towers Linear -> 256, BatchNorm, ReLU, Dropout 0.7, Linear -> 128, L2; symmetric InfoNCE, learnable temperature;
Adam lr 1e-3, weight decay 1e-4, batch 128, 3 epochs; inputs centred by the FLAG train mean; external : FLAG = min(4, pool) : 1.
Different from the report: same-person pairs inside a batch are masked out of the negatives; no VoxCeleb1 (no ArcFace for it here).
Scores: cosine, mean over the seeds; z = (cos - mean) / sd with mean / sd from the out-of-fold FLAG train trials (5 folds by person).
No test statistic is used; MAV voice files of lid_strict.json `mav_files` are dropped (no Bengali speech in training)."""
import io, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]; KO = Path(os.environ.get("DP_FEATS", REPO / "kaggle" / "output"))   # FF repo: dp_feats folder
OUT = Path(os.environ.get("DP_OUT", HERE / "out")); OUT.mkdir(exist_ok=True)   # FF repo: output folder
EVZ = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/ff_data/evalphase/eval_set.zip")
SEEDS = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]; OOF_SEEDS = SEEDS[:3]
LOG = []


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


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


# ------------------------------------------------------------------------------------------------ data
tz = zipfile.ZipFile(os.environ.get("DP_TRAIN_ZIP", REPO / "Input" / "train_set.zip"))   # FF repo: organisers' train_set.zip
rows = [l.split() for l in tz.read("train_set/train_set/train_English.txt").decode().splitlines() if l.strip()]
t_pid = np.array([r[2].split("/")[2] for r in rows]); assert len(set(t_pid)) == 70
gmap = {l.split(",")[0]: l.split(",")[1].strip() for l in Path(os.environ.get("DP_TRAIN_META", REPO / "Input" / "meta_file_train_set.csv")).read_text().splitlines() if l.strip()}
t_gen = np.array([gmap[p] for p in t_pid])
TF = face_cat(csv_feats(tz, "train_set/features/faces/train_English_faces.csv", 4096),
              np.load(KO / "feats_models" / "face_adaface_train.npy"), np.load(KO / "feats_v2" / "face_arcface_train.npy"))
TV = voice_cat(csv_feats(tz, "train_set/features/voices/train_English_voices.csv", 192),
               np.load(KO / "feats_v2" / "voice_ecapa192_train.npy"), np.load(KO / "feats_v2" / "voice_ecapa6144_train.npy"))
log("FLAG train", TF.shape, TV.shape, "persons", len(set(t_pid)))

lid = json.loads(Path(os.environ.get("DP_LID_STRICT", REPO / "Experiment" / "LIDSTRICT-01" / "lid_strict.json")).read_text()); bad = set(lid.get("mav_files", []))
EF, EV, ef_pid, ev_pid = [], [], [], []
for v in ("v1", "v2"):
    fm = pd.read_csv(KO / "feats_ext_full" / f"ext_{v}_complete_face_meta.csv"); vm = pd.read_csv(KO / "feats_ext_full" / f"ext_{v}_complete_voice_meta.csv")
    arc = np.load(KO / "feats_eval" / f"ext_{v}_complete_face_arcface.npy").astype(np.float32)
    ada = np.load(KO / "feats_models" / f"ext_{v}_complete_face_adaface.npy").astype(np.float32)
    # row alignment check: with the right meta, ArcFace rows of one speaker are far closer than rows of different speakers
    a = l2(arc); sp = fm.spk.to_numpy(); r = np.random.default_rng(0); i = r.integers(0, len(a), 4000); j = r.integers(0, len(a), 4000)
    same = sp[i] == sp[j]; cs = (a[i] * a[j]).sum(1)
    log(f"MAV {v}: ArcFace cos same speaker {cs[same].mean():.3f} / other {cs[~same].mean():.3f} ({len(fm)} faces, {len(vm)} voices)")
    assert cs[same].mean() > cs[~same].mean() + 0.2, "face meta does not line up with the ArcFace rows"
    EF.append(face_cat(np.load(KO / "feats_ext_full" / f"ext_{v}_complete_face.npy").astype(np.float32), ada, arc)); ef_pid += [f"{v}_{s}" for s in fm.spk]
    vpath = vm.path.map(lambda q: q if q.startswith(v + "/") else v + "/" + q)   # the full meta has no "v1/" prefix, lid_strict has it
    keep = ~vpath.isin(bad).to_numpy()
    assert int((~keep).sum()) == sum(b.startswith(v + "/") for b in bad), "every flagged MAV voice file of this version must be found and dropped"
    EV.append(voice_cat(np.load(KO / "feats_ext_full" / f"ext_{v}_complete_voice.npy").astype(np.float32)[keep],
                        np.load(KO / "feats_eval" / f"ext_{v}_complete_voice_ecapa192.npy").astype(np.float32)[keep],
                        np.load(KO / "feats_eval" / f"ext_{v}_complete_voice_ecapa6144.npy").astype(np.float32)[keep])); ev_pid += [f"{v}_{s}" for s in vm.spk[keep]]
    log(f"MAV {v}: dropped {int((~keep).sum())} voice files flagged Bengali")
EF, EV = np.vstack(EF), np.vstack(EV); ef_pid, ev_pid = np.array(ef_pid), np.array(ev_pid)
assert np.isfinite(EF).all() and np.isfinite(EV).all()
ef_by = {p: np.nonzero(ef_pid == p)[0] for p in sorted(set(ef_pid))}
ev_ok = np.array([p in ef_by for p in ev_pid]); log("external", EF.shape, EV.shape, "persons", len(ef_by), "voice rows with a face", int(ev_ok.sum()))

MU_F, MU_V = TF.mean(0), TV.mean(0)                     # centring constants: FLAG train
ez = zipfile.ZipFile(EVZ); FE = Path(os.environ.get("DP_EVAL_FEATS", HERE / "feats_evalphase")); EVAL = {}
for prot in ("no_gender", "gender"):
    for lang in ("English", "Bangla"):
        t = f"{prot}_{lang}"
        if not (FE / f"voice_ecapa6144_{t}.npy").exists(): log("skip (no features):", t); continue
        ids = (FE / f"ids_{t}.txt").read_text().split()
        f = face_cat(csv_feats(ez, f"evaluation_phase/{prot}/features/{lang}_test_faces.csv", 4096), np.load(FE / f"face_adaface_{t}.npy"), np.load(FE / f"face_arcface_{t}.npy"))
        v = voice_cat(csv_feats(ez, f"evaluation_phase/{prot}/features/{lang}_test_voices.csv", 192), np.load(FE / f"voice_ecapa192_{t}.npy"), np.load(FE / f"voice_ecapa6144_{t}.npy"))
        assert len(ids) == len(f) == len(v); EVAL[t] = (ids, f, v)
T = lambda x, mu: torch.from_numpy(np.ascontiguousarray((x - mu).astype(np.float32)))
TFt, TVt, EFt, EVt = T(TF, MU_F), T(TV, MU_V), T(EF, MU_F), T(EV, MU_V)
all_p = sorted(set(t_pid)) + sorted(ef_by); pix = {p: i for i, p in enumerate(all_p)}


# ------------------------------------------------------------------------------------------------ model
class Tower(nn.Module):
    def __init__(self, d):
        super().__init__(); self.n = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.7), nn.Linear(256, 128))

    def forward(self, x):
        return F.normalize(self.n(x), dim=-1)


def train(seed, flag_rows):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    f, v = Tower(TF.shape[1]), Tower(TV.shape[1]); ls = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
    opt = torch.optim.Adam(list(f.parameters()) + list(v.parameters()) + [ls], lr=1e-3, weight_decay=1e-4)
    evr = np.nonzero(ev_ok)[0]; n_ext = min(4 * len(flag_rows), len(evr))
    for ep in range(3):
        ext_v = rng.choice(evr, n_ext, replace=False); ext_f = np.array([rng.choice(ef_by[ev_pid[i]]) for i in ext_v])
        src = np.concatenate([np.zeros(len(flag_rows), int), np.ones(n_ext, int)]); fi = np.concatenate([flag_rows, ext_f]); vi = np.concatenate([flag_rows, ext_v])
        pid = np.array([pix[p] for p in np.concatenate([t_pid[flag_rows], ev_pid[ext_v]])])
        order = rng.permutation(len(src)); f.train(); v.train()
        for lo in range(0, len(order) - 127, 128):
            b = order[lo:lo + 128]; s = src[b]
            xf = torch.where(torch.from_numpy(s == 0)[:, None], TFt[np.where(s == 0, fi[b], 0)], EFt[np.where(s == 1, fi[b], 0)])
            xv = torch.where(torch.from_numpy(s == 0)[:, None], TVt[np.where(s == 0, vi[b], 0)], EVt[np.where(s == 1, vi[b], 0)])
            lg = ls.exp().clamp(max=100) * f(xf) @ v(xv).T
            pb = torch.from_numpy(pid[b]); m = (pb[:, None] == pb[None, :]) & ~torch.eye(len(b), dtype=torch.bool)
            lg = lg.masked_fill(m, float("-inf")); y = torch.arange(len(b))
            loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
    f.eval(); v.eval()
    return f, v, float(loss)


@torch.no_grad()
def cos(models, xf, xv):
    return np.mean([(f(xf) * v(xv)).sum(1).numpy() for f, v, _ in models], axis=0)


def eer(s, y):
    o = np.argsort(-s); y = np.asarray(y)[o]; tp = np.cumsum(y) / y.sum(); fp = np.cumsum(1 - y) / (1 - y).sum()
    i = np.argmin(np.abs((1 - tp) - fp)); return 100 * ((1 - tp[i]) + fp[i]) / 2


# ------------------------------------------------------------------------------------------------ out-of-fold on FLAG train
persons = np.array(sorted(set(t_pid))); pg = np.array([gmap[p] for p in persons])
folds = np.zeros(len(persons), int)
for g in sorted(set(pg)):
    ix = np.nonzero(pg == g)[0]; folds[np.random.default_rng(11).permutation(ix)] = np.arange(len(ix)) % 5
OOF = {"no_gender": [[], []], "gender": [[], []]}
for k in range(5):
    te_p = persons[folds == k]; tr_rows = np.nonzero(~np.isin(t_pid, te_p))[0]
    ms = [train(s, tr_rows) for s in OOF_SEEDS]
    rr = np.random.default_rng(100 + k); by = {p: np.nonzero(t_pid == p)[0] for p in te_p}
    for prot in OOF:
        n = 6000; fi, vi, y = [], [], []
        for _ in range(n // 2):                                        # positives: face and voice rows of one held-out person
            p = rr.choice(te_p); a, b = rr.choice(by[p], 2, replace=False); fi.append(a); vi.append(b); y.append(1)
        while len(y) < n:                                              # negatives: two held-out persons (gender file: same gender)
            p, q = rr.choice(te_p, 2, replace=False)
            if prot == "gender" and gmap[p] != gmap[q]: continue
            fi.append(rr.choice(by[p])); vi.append(rr.choice(by[q])); y.append(0)
        OOF[prot][0].append(cos(ms, TFt[fi], TVt[vi])); OOF[prot][1].append(np.array(y))
    log(f"fold {k}: {len(te_p)} held-out persons, last loss {ms[-1][2]:.3f}")
CONST = {}
for prot, (s, y) in OOF.items():
    s, y = np.concatenate(s), np.concatenate(y); CONST[prot] = (float(s.mean()), float(s.std()))
    log(f"OOF FLAG train {prot}: EER {eer(s, y):.2f} | mean {s.mean():.4f} sd {s.std():.4f} ({len(s)} trials)")

# ------------------------------------------------------------------------------------------------ final models + Evaluation scores
ms = [train(s, np.arange(len(t_pid))) for s in SEEDS]; log("final models:", len(ms), "last losses", [round(m[2], 3) for m in ms[:3]])
# FF repo (save): checkpoint of the 10 final tower pairs + the constants needed to score from it (stage dp_score).
# Nothing here draws a random number or changes a model or an input: every score below is unchanged.
torch.save({"format": "FourFusion DP two-tower checkpoint v1", "seeds": list(SEEDS), "dims": [int(TF.shape[1]), int(TV.shape[1])],
            "towers": [{"face": f.state_dict(), "voice": v.state_dict()} for f, v, _ in ms],
            "MU_F": torch.from_numpy(MU_F.copy()), "MU_V": torch.from_numpy(MU_V.copy()),
            "CONST": {k: [float(a), float(b)] for k, (a, b) in CONST.items()}}, OUT / "dp_towers.pt")
print("saved", OUT / "dp_towers.pt", flush=True)
res = {}
for t, (ids, f, v) in EVAL.items():
    c = cos(ms, T(f, MU_F), T(v, MU_V)); mu, sd = CONST[t.rsplit("_", 1)[0]]
    res[f"ids_{t}"] = np.array(ids); res[f"cos_{t}"] = c; res[f"z_{t}"] = (c - mu) / sd
    log(f"Evaluation {t}: {len(ids)} trials scored")
np.savez(OUT / "xjf_scores.npz", **res); (OUT / "consts.json").write_text(json.dumps(CONST, indent=1))
(OUT / "report.txt").write_text("\n".join(LOG) + "\n"); log("saved", OUT / "xjf_scores.npz")
