"""SAFE-04 step 3: relational bridge sweep on true-label validation folds (needs streams_fold<s>.parquet).

    FOLDS=1 python rel_sweep.py      -> <scratch>/out/safe04/relsweep_fold<s>.parquet (one column per variant)
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["SAFE04_NOMAIN"] = "1"
from lab_streams import *
from rel import Relational

FO = R / "kaggle/output"


def ext_cat(kind, name, folder):
    return np.concatenate([np.load(FO / folder / f"ext_{s_}_{kind}_{name}.npy").astype(np.float32) for s_ in ("v1_complete", "v2_complete")])


FACE = {"farl": (A("v4_face_farl").astype(np.float32), ext_cat("face", "farl", "feats_models")),
        "siglip2": (np.load(FO / "feats_models/face_siglip2_train.npy"), ext_cat("face", "siglip2", "feats_models")),
        "arc": (A("v4_face_arcface").astype(np.float32), ext_cat("face", "arcface", "feats_eval")),
        "vgg": (V4["vgg"], FA["vgg"]),
        "ibf": (V4["ibf"], FA["ibv"]),
        "ada": (np.load(FO / "feats_models/face_adaface_train.npy"), ext_cat("face", "adaface", "feats_models")),
        "vitage": (np.load(FO / "feats_models/face_vitage_train.npy"), ext_cat("face", "vitage", "feats_models"))}
VOICE = {"ecapa": (A("v4_voice_ecapa192").astype(np.float32), ext_cat("voice", "ecapa192", "feats_eval")),
         "btc": (V4["btc"], VA["btc"]),
         "iba": (V4["iba"], VA["iba"]),
         "e6144": (np.load(FO / "feats_eval/voice_ecapa6144_train.npy"), ext_cat("voice", "ecapa6144", "feats_eval")),
         "w2v2ag": (np.load(FO / "feats_models/voice_w2v2ag_train.npy"), ext_cat("voice", "w2v2ag", "feats_models"))}
for k, (a, b) in list(FACE.items()):
    assert len(a) == len(V4S) and len(b) == len(FMETA), k
for k, (a, b) in list(VOICE.items()):
    assert len(a) == len(V4S) and len(b) == len(VMETA), k


def groups(fold, fk, vk, ext=True):
    G4f, G4v = FACE[fk][0], VOICE[vk][0]
    Gef, Gev = FACE[fk][1], VOICE[vk][1]
    tr = fold["v4tr"]
    out = []
    for p in np.unique(V4S[tr]):
        r = tr[V4S[tr] == p]
        out.append((p, G4f[r], G4v[r]))
    if ext:
        fp, vp = FMETA.p.values, VMETA.p.values
        for p in fold["ext_pool"]:
            fr, vr = np.where(fp == p)[0], np.where(vp == p)[0]
            if len(fr) and len(vr):
                out.append((p, Gef[fr], Gev[vr]))
    return out


def run(s, variants):
    fold = make_fold(s)
    D = pd.concat([pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"])) for (data, prot), d in fold["files"].items()],
                  ignore_index=True)
    tr = fold["v4tr"]
    rr = np.random.RandomState(0)
    rf, rv = tr[rr.randint(len(tr), size=N_REF)], tr[rr.randint(len(tr), size=N_REF)]
    for name, (fk, vk, mode, ext) in variants.items():
        t0 = time.time()
        rel = Relational.from_rows(groups(fold, fk, vk, ext), mode=mode)
        z = Std(rel.ref_score(FACE[fk][0][rf], VOICE[vk][0][rv], V4S[rf], V4S[rv]))
        col = []
        for (data, prot), d in fold["files"].items():
            Fs = FACE[fk][0] if data == "v4" else FACE[fk][1]
            Vs = VOICE[vk][0] if data == "v4" else VOICE[vk][1]
            col.append(z(rel.score(Fs[d["fi"]], Vs[d["vi"]])))
        D[name] = np.concatenate(col)
        log(f"fold {s} {name}: {len(rel.ids)} ids [{time.time() - t0:.0f}s]")
    D.to_parquet(SP / f"out/safe04/relonly_{TAG}_fold{s}.parquet")


TAG = os.environ.get("TAG", "main")
if __name__ == "__main__":
    V = {}
    for fk in ("farl", "ibf", "ada", "vgg", "siglip2", "vitage", "arc"):
        for vk in ("ecapa", "btc"):
            V[f"rel_{fk}_{vk}"] = (fk, vk, "z", True)
    for fk in ("farl", "ibf", "ada"):
        V[f"relkr_{fk}_ecapa"] = (fk, "ecapa", "kr", True)
        V[f"relv4_{fk}_ecapa"] = (fk, "ecapa", "z", False)
    for s in [int(x) for x in os.environ.get("FOLDS", "1").split(",")]:
        run(s, V)
