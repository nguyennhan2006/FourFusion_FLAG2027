"""BEST v7 (CB 23.28): self-contained library for the Bangla side (B3) of the level-0 system.

Every piece is a verbatim copy of code that was validated before (no import from Experiment/ or kaggle/):
  * Prep / branch / Model / Member.embed+score / RidgeCCA transforms / Std / l2n: LIB cell of kaggle/build_l0_notebooks.py
    (FLAG_18 ... 29, FLAG_24); the s007 XALL member is loaded from members/s007_xall.pt (export_members.py).
  * B3Scorer: DevL0 of the FLAG_24 cell (level-0 constants from 20 000 random train pairs, RandomState(0)) with the
    registered VER-01 weights FIXW, + 0.599 rel + 0.478 relg (Experiment/SAFE-04, log 102).
  * Side / zrow / RelMulti: Experiment/SAFE-04/rel.py.
  * external persons (MAV-Celeb v1/v2): the LOAD cell of build_l0_notebooks.py (persons merged by name, OVERLAP removed,
    genders from the v1 speaker list and the majority of the w2v2 predictions for v2).
A pair's score depends only on that face, that voice, and train constants (level 0).
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device("cpu")
N_REF = 20000
FIXW = np.array([0.298, 0.405, 0.948, 0.327])          # deep, CCA, ImageBind, age (VER-01, log 91)
W_REL, W_RELG = 0.599, 0.478                           # SAFE-04 (log 102): learned on true-label validation
AGE_MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)
OVERLAP = {"v1_complete:shaukat_tarin", "v2_complete:id0073"}   # persons also in v4 (SET-07)
REL_FACES, REL_VOICES = ("farl", "ibf", "ada"), ("ecapa", "btc")
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


class Std:
    """z-score with constants measured on the reference train pairs only."""
    def __init__(self, ref):
        self.m, self.s = float(np.mean(ref)), float(np.std(ref) + 1e-9)

    def __call__(self, x):
        return (x - self.m) / self.s


# ============================================================================ bridge (LIB cell, inference only)
class Prep:
    def __init__(self, d):
        self.mf, self.sf, self.mv, self.sv, self.Pf, self.Pv = (d[k] for k in ("mf", "sf", "mv", "sv", "Pf", "Pv"))

    @classmethod
    def fit(cls, Xf, Xv, pca_f=256, pca_v=None):
        """Standardise; PCA the face to 256 (fitted on training rows). = LIB Prep.__init__."""
        mf, sf = Xf.mean(0), Xf.std(0) + 1e-6
        mv, sv = Xv.mean(0), Xv.std(0) + 1e-6
        return cls(dict(mf=mf, sf=sf, mv=mv, sv=sv, Pf=cls._pca((Xf - mf) / sf, pca_f), Pv=cls._pca((Xv - mv) / sv, pca_v)))

    @staticmethod
    def _pca(Z, k):
        if not k or k >= Z.shape[1]:
            return None
        _, _, Vt = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
        return Vt[:k].T.astype(np.float32)

    def f(self, X):
        Z = (X - self.mf) / self.sf
        return (Z if self.Pf is None else Z @ self.Pf).astype(np.float32)

    def v(self, X):
        Z = (X - self.mv) / self.sv
        return (Z if self.Pv is None else Z @ self.Pv).astype(np.float32)


def branch(d_in, emb, drop, hid):
    return nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, emb))


class Model(nn.Module):
    def __init__(self, d_f, d_v, n_cls, cfg):
        super().__init__()
        self.bf = branch(d_f, cfg["emb"], cfg["drop"], cfg["hid"])
        self.bv = branch(d_v, cfg["emb"], cfg["drop"], cfg["hid"])
        self.W = nn.Parameter(torch.randn(n_cls, cfg["emb"]) * 0.01)
        self.gdisc = nn.Sequential(nn.Linear(cfg["emb"], 64), nn.ReLU(), nn.Linear(64, 2))
        self.ldisc = nn.Sequential(nn.Linear(cfg["emb"], 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, f, v):
        return F.normalize(self.bf(f), dim=1), F.normalize(self.bv(v), dim=1)

    def ef(self, x): return F.normalize(self.bf(x), dim=1)
    def ev(self, x): return F.normalize(self.bv(x), dim=1)
    def score(self, a, b): return (a * b).sum(1)


class XallMember:
    """The 5 s007 XALL nets + CCA-4 of CB 23.28, rebuilt from members/s007_xall.pt."""
    def __init__(self, units, cca):
        self.units, self.c = units, cca

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False, map_location="cpu")
        cfg = d["cfg"]["B_REC"]
        units = []
        for u in d["units"]:
            prep = Prep(u["prep"])
            d_f = prep.Pf.shape[1] if prep.Pf is not None else len(prep.mf)
            d_v = prep.Pv.shape[1] if prep.Pv is not None else len(prep.mv)
            net = Model(d_f, d_v, d["n_cls"], cfg)
            net.load_state_dict(u["state"]); net.eval()
            units.append((net, prep))
        return cls(units, d["cca"])

    @torch.no_grad()
    def embed(self, i, Xf, Xv, chunk=4096):
        net, prep = self.units[i]
        E = []
        for X, fn, pf in ((Xf, net.ef, prep.f), (Xv, net.ev, prep.v)):
            E.append(torch.cat([fn(torch.tensor(pf(X[j:j + chunk]), device=DEVICE)) for j in range(0, len(X), chunk)]))
        return E

    @torch.no_grad()
    def score(self, i, Ef, Ev, chunk=8192):
        net = self.units[i][0]
        return torch.cat([net.score(Ef[j:j + chunk], Ev[j:j + chunk]) for j in range(0, len(Ef), chunk)]).float().cpu().numpy()

    def pair(self, i, Xf, Xv):
        return self.score(i, *self.embed(i, Xf, Xv))

    def cca_x(self, X):
        c = self.c
        return l2n((((X - c["mx"]) / c["sx"]) @ c["Px"]) @ c["Wx"])

    def cca_y(self, Y):
        c = self.c
        return l2n((((Y - c["my"]) / c["sy"]) @ c["Py"]) @ c["Wy"])


# ============================================================================ relational bridge (Experiment/SAFE-04/rel.py)
class Side:
    """One modality / one encoder: identity centroids + per-identity column statistics from TRAIN samples of the other
    identities (removes hub centroids)."""
    def __init__(self, ids, rows_per_id, colnorm=True, max_rows=40, seed=0):
        self.ids = list(ids)
        C = np.stack([l2n(np.asarray(r, np.float32)).mean(0) for r in rows_per_id])
        self.c = C.mean(0); self.C = l2n(C - self.c)
        self.mu = self.sd = None
        if colnorm:
            rng = np.random.RandomState(seed)
            X, who = [], []
            for i, r in enumerate(rows_per_id):
                r = np.asarray(r, np.float32)
                take = r if len(r) <= max_rows else r[rng.choice(len(r), max_rows, replace=False)]
                X.append(take); who += [i] * len(take)
            P = self.raw(np.concatenate(X)); who = np.asarray(who)
            other = who[:, None] != np.arange(len(self.ids))[None, :]
            n = other.sum(0)
            self.mu = np.where(other, P, 0).sum(0) / n
            self.sd = np.sqrt(np.where(other, (P - self.mu) ** 2, 0).sum(0) / n) + 1e-6

    def raw(self, X):
        return l2n(l2n(np.asarray(X, np.float32)) - self.c) @ self.C.T

    def __call__(self, X):
        P = self.raw(X)
        return P if self.mu is None else (P - self.mu) / self.sd


def zrow(P, mask):
    n = mask.sum(1, keepdims=True)
    m = np.where(mask, P, 0).sum(1, keepdims=True) / n
    s = np.sqrt(np.where(mask, (P - m) ** 2, 0).sum(1, keepdims=True) / n) + 1e-9
    return np.where(mask, (P - m) / s, 0)


class RelMulti:
    """Profiles of several face encoders averaged (after per-sample z), same for voices; score = correlation."""
    def __init__(self, face_sides, voice_sides):
        self.F, self.V = face_sides, voice_sides
        self.ids = face_sides[0].ids
        assert all(s.ids == self.ids for s in face_sides + voice_sides)
        self.col = {p: i for i, p in enumerate(self.ids)}

    def _score(self, Xfs, Xvs, mask):
        Pf = zrow(np.mean([zrow(s(X), mask) for s, X in zip(self.F, Xfs)], 0), mask)
        Pv = zrow(np.mean([zrow(s(X), mask) for s, X in zip(self.V, Xvs)], 0), mask)
        return (Pf * Pv).sum(1) / mask.sum(1)

    def score(self, Xfs, Xvs, pf=None, pv=None, chunk=4096, keep=None):
        out = []
        n = len(Xfs[0])
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            m = np.ones((len(Xfs[0][sl]), len(self.ids)), dtype=bool) if keep is None else keep[sl].copy()
            if pf is not None:
                for j, (a, b) in enumerate(zip(pf[sl], pv[sl])):
                    for p in (a, b):
                        if p in self.col:
                            m[j, self.col[p]] = False
            out.append(self._score([X[sl] for X in Xfs], [X[sl] for X in Xvs], m))
        return np.concatenate(out)


def pmale(g3):
    """w2v2 audeering age-gender head, columns (female, male, child) -> P(male | adult)."""
    g = np.asarray(g3, np.float64)
    return g[:, 1] / (g[:, 0] + g[:, 1] + 1e-9)


# ============================================================================ external persons (LOAD cell)
def load_ext(find):
    """find(name) -> path of an external array / table. Returns FMETA, VMETA (one row per v1+v2 face / voice, v1 then
    v2), GEN (person -> 'm' / 'f') and the ext arrays needed here (face vgg / ibv / farl / adaface, voice btc / ecapa192 /
    w2v2aggen), all float32 after a float16 round trip (the external arrays of the eda bundle are float16, so the
    result does not depend on which copy is attached)."""
    FM_, VM_, GEN, arr = [], [], {}, {}
    names = {"face": {"vgg": ("face_vgg", "face"), "ibf": ("face_ibv",), "farl": ("face_farl",), "ada": ("face_adaface",)},
             "voice": {"btc": ("voice_btc", "voice"), "ecapa": ("voice_ecapa192",), "w2v2aggen": ("voice_w2v2aggen",)}}
    for src in ("v1_complete", "v2_complete"):
        fm, vm = pd.read_csv(find(f"ext_{src}_face_meta.csv")), pd.read_csv(find(f"ext_{src}_voice_meta.csv"))
        key_of = {}
        sp = find(f"ext_{src}_speakers.csv", required=False)
        if sp is not None:
            for r in pd.read_csv(sp).itertuples():
                k = f"{src}:" + str(r.name).strip().lower().replace(" ", "_")
                key_of[r.ids] = k
                GEN[k] = "m" if str(r.gender).lower().startswith("m") else "f"
        key = lambda x, key_of=key_of, src=src: key_of.get(x, f"{src}:{x}")
        FM_.append(fm.assign(p=fm.spk.map(key), src=src)); VM_.append(vm.assign(p=vm.spk.map(key), src=src))
        for kind, d in names.items():
            for k, alts in d.items():
                f = next((find(f"ext_{src}_{a}.npy", required=False) for a in alts if find(f"ext_{src}_{a}.npy", required=False)), None)
                assert f is not None, f"ext_{src}_{alts[0]}.npy not found"
                a = np.load(f).astype(np.float16).astype(np.float32)     # as stored in flag2027-eda-bundle
                assert len(a) == len(fm if kind == "face" else vm), (src, k, len(a))
                arr.setdefault((kind, k), []).append(a)
    FMETA, VMETA = pd.concat(FM_, ignore_index=True), pd.concat(VM_, ignore_index=True)
    arr = {k: np.concatenate(v) for k, v in arr.items()}
    male = arr[("voice", "w2v2aggen")][:, 1] > arr[("voice", "w2v2aggen")][:, 0]
    for p, g in VMETA.assign(m=male).groupby("p").m.mean().items():
        GEN.setdefault(p, "m" if g >= .5 else "f")
    if os.environ.get("LID_STRICT") == "1":      # CLEANCOMP-01 (3/11, opt-in): MAV voice files with a Whisper vote leave XALL training rows + rel anchors
        drop = set(strict_mav_files())           # (faces stay; GEN above was fixed from all calls; unset = unchanged bit for bit)
        key = VMETA.path if VMETA.path.str.startswith(("v1/", "v2/")).all() else VMETA.src.str[:2] + "/" + VMETA.path   # lid_strict.json names 'v1/voices/..'
        m = ~key.isin(drop).values
        print(f"[CLEANCOMP-01] LID_STRICT MAV voice files dropped: {int((~m).sum())} of {len(drop)} listed ({len(VMETA)} rows)", flush=True)
        assert 0 < int((~m).sum()), "LID_STRICT=1 but no listed MAV voice file matched VMETA.path"
        VMETA = VMETA[m].reset_index(drop=True)
        for k in [k for k in arr if k[0] == "voice"]:
            arr[k] = arr[k][m]
    return FMETA, VMETA, GEN, arr


def strict_mav_files():
    """lid_strict.json 'mav_files' (LIDSTRICT-01): next to the repo, or any attached dataset."""
    import json
    c = [Path(__file__).resolve().parents[2] / "Experiment/LIDSTRICT-01/lid_strict.json"]
    if Path("/kaggle/input").exists():
        c += list(Path("/kaggle/input").rglob("lid_strict.json"))
    hit = [q for q in c if q.exists()]
    assert hit, "LID_STRICT=1 needs lid_strict.json"
    return json.loads(hit[0].read_text())["mav_files"]


# ============================================================================ the Bangla-side scorer (B3)
class B3Scorer:
    """s007 XALL with the registered weights + relational bridge, level 0.

    TR: dict of train arrays, all v4 rows in organiser order: vgg, btc, ibf, iba, agef, agev, farl, ada, ecapa, vgen.
    spk / gmap: train persons and genders. ext = load_ext(...)."""
    def __init__(self, member, TR, spk, gmap, ext, log=print):
        self.mem, self.TR = member, TR
        FMETA, VMETA, GEN, EA = ext
        persons = sorted(set(VMETA.p) - OVERLAP)
        GM = dict(gmap); [GM.setdefault(k, v) for k, v in GEN.items()]
        n = len(spk)
        rr = np.random.RandomState(0)
        self.rf, self.rv = rf, rv = rr.randint(n, size=N_REF), rr.randint(n, size=N_REF)
        # ---- bridge streams: constants exactly as DevL0 (FLAG_24)
        self.sd = []
        for i in range(len(member.units)):
            Ef, Ev = member.embed(i, TR["vgg"], TR["btc"])
            self.sd.append(Std(member.score(i, Ef[rf], Ev[rv])))
        self.zc = Std(np.sum(member.cca_x(TR["vgg"])[rf] * member.cca_y(TR["btc"])[rv], 1))
        self.mf, self.mv = TR["ibf"].mean(0), TR["iba"].mean(0)
        self.zib = Std(np.sum(l2n(TR["ibf"][rf] - self.mf) * l2n(TR["iba"][rv] - self.mv), 1))
        self.zag = Std(-np.abs(TR["agef"][rf] - TR["agev"][rv]))
        if os.environ.get("NOREL") == "1":   # NOREL-01 (5/11, BTC Rules & FAQ: no train-identity cohort at test time): no anchors are
            self.rel = None                   # built, relational() returns zeros, the composite constants leave rel / relg out
            T_ref = {k: TR[k][rf] for k in ("vgg", "ibf", "agef")}
            T_ref.update({k: TR[k][rv] for k in ("btc", "iba", "agev")})
            b_ref = self.streams(T_ref) @ FIXW
            self.zb = Std(b_ref); self.zf = Std(self.zb(b_ref))
            log(f"B3 (NOREL=1): {len(member.units)} nets, no relational anchors (rel / relg columns = 0, never fused)")
            return
        # ---- relational bridge: anchors = every v4 person (all rows) + every v1/v2 person with faces and voices
        fp, vp = FMETA.p.values, VMETA.p.values
        ext_ok = [p for p in persons if (fp == p).any() and (vp == p).any()]
        tr_key = {"farl": "farl", "ibf": "ibf", "ada": "ada", "ecapa": "ecapa", "btc": "btc"}
        ext_key = {"farl": ("face", "farl"), "ibf": ("face", "ibf"), "ada": ("face", "ada"), "ecapa": ("voice", "ecapa"), "btc": ("voice", "btc")}

        def rows(k):
            kind = ext_key[k][0]
            meta_p = fp if kind == "face" else vp
            out = [(p, TR[tr_key[k]][spk == p]) for p in np.unique(spk)]
            out += [(p, EA[ext_key[k]][np.where(meta_p == p)[0]]) for p in ext_ok]
            return out
        sides = [[Side([a for a, _ in g], [b for _, b in g]) for g in [rows(k) for k in ks]] for ks in (REL_FACES, REL_VOICES)]
        self.rel = RelMulti(*sides)
        self.amale = np.array([GM.get(p) == "m" for p in self.rel.ids])
        ref_f = [TR[tr_key[k]][rf] for k in REL_FACES]; ref_v = [TR[tr_key[k]][rv] for k in REL_VOICES]
        r0 = self.rel.score(ref_f, ref_v, spk[rf], spk[rv])
        rg = self.rel.score(ref_f, ref_v, spk[rf], spk[rv], keep=self.keep(TR["vgen"][rv]))
        self.zrel, self.zrelg = Std(r0), Std(rg)
        # ---- final constants on the reference pairs
        T_ref = {k: TR[k][rf] for k in ("vgg", "ibf", "agef")}
        T_ref.update({k: TR[k][rv] for k in ("btc", "iba", "agev")})
        b_ref = self.streams(T_ref) @ FIXW + W_REL * self.zrel(r0) + W_RELG * self.zrelg(rg)
        self.zb = Std(b_ref); self.zf = Std(self.zb(b_ref))
        log(f"B3: {len(member.units)} nets, {len(self.rel.ids)} relational anchors ({int(self.amale.sum())} male), "
            f"{len(ext_ok)} external persons")

    def keep(self, vgen):
        return self.amale[None, :] == (pmale(vgen) > 0.5)[:, None]

    def streams(self, T):
        m = self.mem
        deep = np.mean([sd(m.pair(i, T["vgg"], T["btc"])) for i, sd in enumerate(self.sd)], 0)
        cca = self.zc(np.sum(m.cca_x(T["vgg"]) * m.cca_y(T["btc"]), 1))
        ib = self.zib(np.sum(l2n(T["ibf"] - self.mf) * l2n(T["iba"] - self.mv), 1))
        age = self.zag(-np.abs(T["agef"] - T["agev"]))
        return np.stack([deep, cca, ib, age], 1)

    def relational(self, T):
        if self.rel is None:                  # NOREL=1
            n = len(T["vgen"]); return np.zeros(n), np.zeros(n)
        Xf, Xv = [T[k] for k in REL_FACES], [T[k] for k in REL_VOICES]
        return self.zrel(self.rel.score(Xf, Xv)), self.zrelg(self.rel.score(Xf, Xv, keep=self.keep(T["vgen"])))

    def __call__(self, T):
        """T: row-aligned trial arrays vgg, btc, ibf, iba, agef, agev, farl, ada, ecapa, vgen. Higher = same."""
        r, rg = self.relational(T)
        return self.zf(self.zb(self.streams(T) @ FIXW + W_REL * r + W_RELG * rg))


def face_gender_model(ibf_train, spk, gmap, ext):
    """P(male) of a face: logistic regression on L2-normalised ImageBind faces of every train face (v4 rows + MAV-Celeb
    v1/v2 faces). Shared by CAFuse (Bangla) and CA-EN (English, BEST v9). -> (model, number of train faces)"""
    from sklearn.linear_model import LogisticRegression
    FMETA, VMETA, GEN, EA = ext
    GM = dict(gmap); [GM.setdefault(k, v) for k, v in GEN.items()]
    X = np.concatenate([ibf_train, np.asarray(EA[("face", "ibf")], np.float32)])
    y = np.concatenate([[GM.get(p) == "m" for p in spk], [GM.get(p) == "m" for p in FMETA.p.values]])
    return LogisticRegression(C=1.0, max_iter=3000).fit(l2n(X), y), len(y)


class CAFuse:
    """Condition-aware fusion of the B3 streams (Experiment/CAFUSE-01, log 112-113; CB dev Bangla 18.07 / 28.07).

    Logistic regression, fitted ONCE on the SAFE-04 true-label validation trials (cafuse_fuser.json), on
      6 B3 streams   [deep, cca, ib, age, rel, relg]   (B3Scorer.streams + B3Scorer.relational)
      7 conditions   [file language English, P(male) face, P(male) voice, |difference|, age face, age voice, |age gap|]
      their 42 products.
    P(male) face: logistic regression on L2-normalised ImageBind faces of every train face (v4 rows + MAV-Celeb v1/v2
    faces); P(male) voice: w2v2 gender head. Every input comes from the pair itself, its file's language and train
    constants, so the score stays level 0."""
    def __init__(self, b3, model, TR, spk, gmap, ext, log=print):
        self.b3, self.m = b3, model
        self.mean, self.scale, self.coef = (np.array(model[k], np.float64) for k in ("mean", "scale", "coef"))
        self.gclf, n = face_gender_model(TR["ibf"], spk, gmap, ext)
        log(f"CA-FUSE: {len(self.coef)} inputs, fitted on {model.get('n_trials')} validation trials; face gender model on {n} train faces")

    def conditions(self, T, is_en):
        pf, pv = self.gclf.predict_proba(l2n(T["ibf"]))[:, 1], pmale(T["vgen"])
        return np.stack([np.full(len(pf), float(is_en)), pf, pv, np.abs(pf - pv), T["agef"], T["agev"],
                         np.abs(T["agef"] - T["agev"])], 1)

    def __call__(self, T, is_en):
        """Higher = same."""
        r, rg = self.b3.relational(T)
        Z = np.concatenate([self.b3.streams(T), r[:, None], rg[:, None]], 1)
        C = self.conditions(T, is_en)
        X = np.concatenate([Z, C, (Z[:, :, None] * C[:, None, :]).reshape(len(Z), -1)], 1)
        return ((X - self.mean) / self.scale) @ self.coef + self.m["intercept"]



# ============================================================================ training (rebuild members from raw data)
# Verbatim from kaggle/build_l0_notebooks.py (LIB: REC, infonce, pk_batches, train_one, epochs_for, RidgeCCA; DATA: ext_rows;
# DEV_RUN: train_s007(ext=True)). Same RNG calls, so on the same machine and thread count the nets are bit-identical.
REC = dict(pca_f=256, pca_v=None, hid=512, emb=128, drop=0.3, lr=1e-3, wd=1e-2, tau=0.07, epochs=40)
P_PK, K_PK, CAP, STEPS, N_NETS = 32, 8, 150, 1200, 5
TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infonce_S(S, y):
    same = (y[:, None] == y[None, :]).float()
    return 0.5 * ((-(F.log_softmax(S, 1) * same).sum(1) / same.sum(1)).mean() +
                  (-(F.log_softmax(S, 0) * same).sum(0) / same.sum(0)).mean())


def infonce(u, w, y, tau):
    return infonce_S(u @ w.T / tau, y)


def pk_batches(spk_idx, P, K, n_rows, rng):
    speakers = list(spk_idx)
    out = []
    for _ in range(max(1, n_rows // (P * K))):
        chosen = rng.choice(len(speakers), size=min(P, len(speakers)), replace=False)
        out.append(np.concatenate([rng.choice(spk_idx[speakers[c]], size=K, replace=len(spk_idx[speakers[c]]) < K)
                                   for c in chosen]))
    return out


def train_one(TF, TV, spk, seed, epochs, P=P_PK, K=K_PK, device=None):
    """EXP-007 MLP, InfoNCE, PK sampler, AdamW + cosine (LIB train_one)."""
    DEV = device or TRAIN_DEVICE
    cfg = dict(REC, epochs=epochs)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    prep = Prep.fit(TF, TV, cfg["pca_f"], cfg["pca_v"])
    Ftr = torch.tensor(prep.f(TF), device=DEV)
    Vv = torch.stack([torch.tensor(prep.v(TV), device=DEV)])
    cls = {s: i for i, s in enumerate(np.unique(spk))}
    Y = torch.tensor([cls[s] for s in spk], device=DEV)
    net = Model(Ftr.shape[1], Vv.shape[2], len(cls), cfg).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    n = len(spk)
    spk_idx = {s: np.where(np.asarray(spk) == s)[0] for s in np.unique(spk)}
    P = min(P, len(spk_idx))
    for ep in range(cfg["epochs"]):
        net.train()
        rng.permutation(n); vsel = np.zeros(n, dtype=int)                 # same RNG calls as flag_v2 (full view)
        for b in pk_batches(spk_idx, P, K, n, rng):
            bt = torch.tensor(b, device=DEV); vt = torch.tensor(vsel[b], device=DEV)
            u, w = net(Ftr[bt], Vv[vt, bt])
            loss = infonce(u, w, Y[bt], cfg["tau"])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    return net.cpu(), prep


def epochs_for(n_rows, steps=None):
    bpe = max(1, n_rows // (P_PK * K_PK))
    return max(1, int(round((steps or STEPS) / bpe))), bpe


def cca_fit(X, Y, k=4, reg=1.0, pca_x=128, pca_y=192):
    """RidgeCCA(k=4, reg=1, pca 128 / 192).fit -> the parameter dict XallMember uses (LIB RidgeCCA)."""
    def _prep(Z0, n_pca):
        mu, sd = Z0.mean(0), Z0.std(0) + 1e-6
        Z = (Z0 - mu) / sd
        if n_pca is not None and n_pca < Z.shape[1]:
            _, _, Vt = np.linalg.svd(Z, full_matrices=False); P = Vt[:n_pca].T
        else:
            P = np.eye(Z.shape[1])
        return mu, sd, P
    mx, sx, Px = _prep(X, min(pca_x, X.shape[1]) if pca_x else None)
    my, sy, Py = _prep(Y, min(pca_y, Y.shape[1]) if pca_y else None)
    A_ = ((X - mx) / sx) @ Px; B_ = ((Y - my) / sy) @ Py
    n = len(A_)
    Cxx = A_.T @ A_ / n + reg * np.eye(A_.shape[1]); Cyy = B_.T @ B_ / n + reg * np.eye(B_.shape[1])
    Lxi = np.linalg.inv(np.linalg.cholesky(Cxx)); Lyi = np.linalg.inv(np.linalg.cholesky(Cyy))
    U, S_, Vt = np.linalg.svd(Lxi @ (A_.T @ B_ / n) @ Lyi.T, full_matrices=False)
    return dict(mx=mx, sx=sx, Px=Px, Wx=Lxi.T @ U[:, :k], my=my, sy=sy, Py=Py, Wy=Lyi.T @ Vt[:k].T)


def ext_rows(FMETA, VMETA, FA_vgg, VA_btc, persons, rng, cap=CAP):
    """Organiser-space training rows of external persons (DATA ext_rows, xvid=False): one per voice clip (<= cap per
    person), paired with a face of the SAME video (cycling; any face of the person if that video has none)."""
    vm = VMETA[VMETA.p.isin(persons)]
    fm = FMETA[FMETA.p.isin(persons)]
    by_vid = {k: np.asarray(g.index) for k, g in fm.groupby(["p", "lang", "video"])}
    by_p = {k: np.asarray(g.index) for k, g in fm.groupby("p")}
    Fi, Vi, S = [], [], []
    for p, g in vm.groupby("p"):
        if p not in by_p:
            continue
        idx = np.asarray(g.index)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        for j, i in enumerate(idx):
            cand = by_vid.get((p, VMETA.lang[i], VMETA.video[i]), by_p[p])
            Fi.append(cand[j % len(cand)])
            Vi.append(i); S.append(p)
    return FA_vgg[Fi], VA_btc[Vi], np.array(S)


def train_xall(v4_vgg, v4_btc, v4_spk, ext, log=print, device=None):
    """s007 XALL of CB 23.28 (FLAG_24 train_s007(ext=True)): 70 v4 + every v1/v2 person (OVERLAP removed), <= 150 rows
    each, PK 32 x 8, 1200 steps, 5 nets (seeds 1 + 100 i) + CCA-4 on the same rows."""
    import time as _t
    FMETA, VMETA, GEN, EA = ext
    persons = sorted(set(VMETA.p) - OVERLAP)
    EF, EV, ES = ext_rows(FMETA, VMETA, EA[("face", "vgg")], EA[("voice", "btc")], persons, np.random.RandomState(4000))
    TF, TV, TS = np.concatenate([v4_vgg, EF]), np.concatenate([v4_btc, EV]), np.concatenate([np.asarray(v4_spk), ES])
    ep, bpe = epochs_for(len(TS))
    t0 = _t.time()
    units = [train_one(TF, TV, TS, seed=1 + 100 * i, epochs=ep, device=device) for i in range(N_NETS)]
    log(f"s007 XALL: {len(set(TS))} persons, {len(TS)} rows, {ep * bpe} steps x {N_NETS} nets [{_t.time() - t0:.0f}s]")
    return XallMember(units, cca_fit(TF, TV))


def save_xall(member, path):
    """Same format as export_members.py."""
    units = [{"state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
              "prep": {k: getattr(prep, k) for k in ("mf", "sf", "mv", "sv", "Pf", "Pv")}} for net, prep in member.units]
    torch.save({"units": units, "cca": member.c, "cfg": dict(B_REC=REC), "n_cls": units[0]["state"]["W"].shape[0],
                "source": "safe_lib.train_xall"}, path)
