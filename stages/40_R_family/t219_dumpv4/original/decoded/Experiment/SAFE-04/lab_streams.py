"""SAFE-04 step 1: level-0 stream table on true-label validation (CPU).

Re-uses the exact cells of kaggle/build_l0_notebooks.py (ENV / LOAD / LIB / DATA = FLAG_18 ... 29), so folds, eval
files, XALL bridge and the L0 scorer are those of COMBO-01. For every fold it trains the XALL bridge (3 nets, 1200 PK
steps; = COMBO-01 XALL) and the v4-only bridge (= CTRL), scores every validation trial, and adds candidate streams that
need no training on test data:
  acl   AudioCLIP face . voice cosine (train means removed)          [v4 all rows; v1/v2 only rows FLAG_14 covered]
  lb    LanguageBind face . voice cosine                               [same coverage]
  iba2  ImageBind with the face embedding averaged over the two face views (ibv + ibalign)
  gen   gender agreement: P(face male) from a logistic regression on train faces (ImageBind), P(voice male) from w2v2
Every stream is z-scored with 20 000 random pairs of the fold's v4 train rows (as L0). Nothing reads dev or test data.

    python lab_streams.py            -> <scratch>/out/streams_fold<s>.parquet
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[2]
SP = Path(os.environ.get("SAFE04_SCRATCH", r"C:/Users/ASUS/AppData/Local/Temp/claude/d--Sinh-vi-n-CNhan-FLAG-2027-FourFusion/969f2c71-effc-43ec-a610-c253be9d1ea8/scratchpad"))
os.environ.setdefault("FLAG_LOCAL", "1"); os.environ.setdefault("FLAG_IN", str(SP / "in")); os.environ.setdefault("FLAG_OUT", str(SP / "out"))
os.environ.setdefault("L0_NETS", "3")
sys.path.insert(0, str(R / "kaggle"))
import build_l0_notebooks as B

G = {}
exec(B.ENV.replace("__OUTDIR__", "safe04").replace("__ARMS__", "CTRL"), G)
for cell in (B.LOAD, B.LIB, B.DATA):
    exec(cell, G)
globals().update({k: v for k, v in G.items() if not k.startswith("__")})
import torch
torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "10")))

FO = R / "kaggle/output"
l2n_ = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def ext_full(kind, name, folder):
    """v1 then v2 rows (the order of LOAD's FA / VA); rows FLAG_14 did not cover are NaN."""
    parts = []
    for src in ("v1_complete", "v2_complete"):
        n = len(FMETA[FMETA.src == src]) if kind == "face" else len(VMETA[VMETA.src == src])
        a = np.load(FO / folder / f"ext_{src}_{kind}_{name}.npy").astype(np.float32)
        if len(a) == n:
            parts.append(a); continue
        rows = np.load(FO / folder / f"ext_{src}_{kind}_rows.npy")
        full = np.full((n, a.shape[1]), np.nan, np.float32); full[rows] = a
        parts.append(full)
    return np.concatenate(parts)


X4 = dict(aclf=np.load(FO / "feats_av/face_aclip_train.npy"), aclv=np.load(FO / "feats_av/voice_aclip_train.npy"),
          lbf=np.load(FO / "feats_av/face_lbind_train.npy"), lbv=np.load(FO / "feats_av/voice_lbind_train.npy"),
          ibal=np.load(FO / "feats_av/face_ibalign_train.npy"),
          vgen=np.load(FO / "feats_eval/voice_w2v2aggen_train.npy"))
assert np.abs(np.load(FO / "feats_eval/face_ibv_train.npy") - V4["ibf"]).max() < 1e-4
XE = dict(aclf=ext_full("face", "aclip", "feats_av"), aclv=ext_full("voice", "aclip", "feats_av"),
          lbf=ext_full("face", "lbind", "feats_av"), lbv=ext_full("voice", "lbind", "feats_av"),
          ibal=ext_full("face", "ibalign", "feats_av"), vgen=VA["w2v2aggen"])
log("extra arrays: v4", {k: v.shape for k, v in X4.items()}, "| ext", {k: v.shape for k, v in XE.items()})


def pmale_voice(g3):
    g = np.asarray(g3, np.float64)
    return g[:, 1] / (g[:, 0] + g[:, 1] + 1e-9)


class Extra:
    """Extra level-0 streams, every constant from the fold's v4 train rows (+ the face gender model on train faces)."""
    def __init__(self, v4rows, ext_pool):
        from sklearn.linear_model import LogisticRegression
        rr = np.random.RandomState(0)
        rf, rv = rr.randint(len(v4rows), size=N_REF), rr.randint(len(v4rows), size=N_REF)
        F_, V_ = v4rows[rf], v4rows[rv]
        self.m = {}
        for k, (a, b) in dict(acl=("aclf", "aclv"), lb=("lbf", "lbv")).items():
            self.m[k] = (X4[a][v4rows].mean(0), X4[b][v4rows].mean(0))
        self.mib = (V4["ibf"][v4rows].mean(0), V4["iba"][v4rows].mean(0))
        self.mibal = X4["ibal"][v4rows].mean(0)
        # face gender: logistic regression on ImageBind faces of the fold's train persons (v4 labels + ext pool genders)
        ext_idx = np.where(FMETA.p.isin(ext_pool).values)[0]
        Xg = np.concatenate([V4["ibf"][v4rows], FA["ibv"][ext_idx]])
        yg = np.concatenate([[GMAP[s] == "m" for s in V4S[v4rows]], [GMAP.get(p) == "m" for p in FMETA.p.values[ext_idx]]])
        self.gclf = LogisticRegression(C=1.0, max_iter=3000).fit(l2n_(Xg), yg)
        ref = self.raw(dict(ibf=V4["ibf"][F_], iba=V4["iba"][V_], **{k: X4[k][F_] for k in ("aclf", "lbf", "ibal")},
                            **{k: X4[k][V_] for k in ("aclv", "lbv", "vgen")}))
        self.z = {k: Std(v) for k, v in ref.items()}

    def raw(self, T):
        out = {}
        for k, (a, b) in dict(acl=("aclf", "aclv"), lb=("lbf", "lbv")).items():
            out[k] = np.sum(l2n_(T[a] - self.m[k][0]) * l2n_(T[b] - self.m[k][1]), 1)
        fa = l2n_(l2n_(T["ibf"] - self.mib[0]) + l2n_(T["ibal"] - self.mibal))
        out["iba2"] = np.sum(fa * l2n_(T["iba"] - self.mib[1]), 1)
        out["ibal"] = np.sum(l2n_(T["ibal"] - self.mibal) * l2n_(T["iba"] - self.mib[1]), 1)
        pf = self.gclf.predict_proba(l2n_(T["ibf"]))[:, 1]; pv = pmale_voice(T["vgen"])
        out["gen"] = pf * pv + (1 - pf) * (1 - pv)
        out["pf_m"], out["pv_m"] = pf, pv
        return out

    def streams(self, T):
        r = self.raw(T)
        return {k: (self.z[k](v) if k in ("acl", "lb", "iba2", "ibal", "gen") else v) for k, v in r.items()}


def trial_extra(data, d):
    src = (X4 if data == "v4" else XE)
    T = dict(ibf=d["ibf"], iba=d["iba"])
    for k in ("aclf", "lbf", "ibal"):
        T[k] = src[k][d["fi"]]
    for k in ("aclv", "lbv", "vgen"):
        T[k] = src[k][d["vi"]]
    return T


def run_fold(s, out):
    t0 = time.time()
    fold = make_fold(s)
    rows = []
    mems = {}
    for name, ext in (("XALL", True), ("CTRL", False)):
        mems[name] = arm_ctrl(fold, ext=ext)[0]
        log(f"fold {s} {name} trained [{time.time() - t0:.0f}s]")
    sc = {n: L0(m, fold["v4tr"]) for n, m in mems.items()}
    ex = Extra(fold["v4tr"], fold["ext_pool"])
    for (data, prot), d in fold["files"].items():
        df = pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"], pf=d["pf"], pv=d["pv"], fi=d["fi"], vi=d["vi"]))
        for n in sc:
            S_ = sc[n].streams(d)
            for k in ("deep0", "cca", "member0", "ib", "age", "prod0"):
                df[f"{n}_{k}"] = S_[k]
        E = ex.streams(trial_extra(data, d))
        for k, v in E.items():
            df[k] = v
        rows.append(df)
    D = pd.concat(rows, ignore_index=True)
    D.to_parquet(out)
    log(f"fold {s}: {len(D)} trials -> {out} [{time.time() - t0:.0f}s]")


if __name__ == "__main__" and not os.environ.get("SAFE04_NOMAIN"):
    od = SP / "out" / "safe04"; od.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in os.environ.get("SAFE04_FOLDS", "1,2,3,4,5").split(",")]:
        f = od / f"streams_fold{s}.parquet"
        if not f.exists():
            run_fold(s, f)
