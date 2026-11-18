r"""System v11 on any test split (dev now, the Evaluation data on 11/11): per test file, the components of the current
system and the v9 / v11 scores (docs/PLAN_E2E_2026-10-10.md). Used by
  * kaggle/build_e2e_dev_bundle.py   (local: dev components from the arrays on this machine -> flag2027-e2e-dev)
  * FLAG_50_e2e_v11.ipynb            (Kaggle: freshly extracted arrays -> submission.zip + components_<split>.npz)
Components of file i: B3 (6 streams of BEST v9's Bangla side), CO (7 CA-FUSE conditions, is_en from the file language),
EN7 + ENCO (CA-EN streams + 6 conditions of v9's English side), T3 (anP512_deep, anP512_cca, t3_relg of v11's head; given),
w2v2 gender probabilities. Scores: v9 = CA-EN (English) / CA-FUSE (Bangla); v11 = v9 English / CA-FUSE + T3 (Bangla).
Every quantity of a pair depends only on that pair and train constants (level 0); `selftest` re-scores half of every file.
"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

R0 = Path(__file__).resolve().parents[2]
B3_COLS = ["deep", "cca", "ib", "age", "rel", "relg"]
CO_COLS = ["is_en", "pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]
EN7_COLS = ["s007_deep", "s007_cca", "s010B_deep", "s010B_cca", "ib", "age", "acl"]
ENCO_COLS = ["pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]
T3_COLS = ["anP512_deep", "anP512_cca", "t3_relg"]
design = lambda Z, C: np.concatenate([Z, C, (Z[:, :, None] * C[:, None, :]).reshape(len(Z), -1)], 1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def apply(model, Z, C):
    m = {k: np.array(model[k], np.float64) for k in ("mean", "scale", "coef")}
    return ((design(Z, C) - m["mean"]) / m["scale"]) @ m["coef"] + model["intercept"]


class System:
    """B3 + CA-EN component scorers on a FeatureStore (data = organiser csv / lists, feats = array folders)."""
    def __init__(self, data, feats, ext_dirs, xall_pt, v6_members, fuser_b9, threads=10, n_models=5):
        sys.path[:0] = [str(R0 / "BEST/v7_safe"), str(R0 / "BEST/v6_eval"), str(R0 / "kaggle")]
        import safe_lib as S
        import flag_v2 as V
        import torch
        torch.set_num_threads(threads)
        # the cached v6 members (s007 / s010B) are CPU tensors; flag_v2 sends inputs to cuda when a GPU exists (FLAG_50 on
        # Kaggle, 11/10: "mat1 is on cuda:0, other tensors on cpu"). Every component runs on CPU, as the submitted zips did.
        V.DEVICE = torch.device("cpu")
        if "flag_lib" in sys.modules:
            sys.modules["flag_lib"].DEVICE = V.DEVICE
        self.S, self.V = S, V
        P6 = _load("v6p", R0 / "BEST/v6_eval/pipeline.py")
        P6.CONFIG["members"] = {k: v for k, v in P6.CONFIG["members"].items() if k in ("s007", "s010B")}
        os.environ["FLAG_FEATS_EXTRA"] = os.pathsep.join(str(p) for p in feats[1:])
        self.store = store = V.FeatureStore(str(data), feats_dir=str(feats[0]))
        npy = self.npy = lambda stem, split: store._npy(stem, split).astype(np.float32)

        def find(name, required=True):
            for d in ext_dirs:
                for f in (Path(d) / name, *Path(d).glob(f"*/{name}")):
                    if f.exists():
                        return f
            assert not required, name
            return None
        ext = S.load_ext(find)
        self.b3 = S.B3Scorer(S.XallMember.load(xall_pt), self.arrays("train"), store.spk, store.gmap, ext, log=log)
        self.ca9 = S.CAFuse(self.b3, fuser_b9, self.arrays("train"), store.spk, store.gmap, ext, log=log)
        M = P6.train_members(V, store, n_models, Path(v6_members))
        rng = np.random.RandomState(0); n = len(store.spk)
        REF = (rng.randint(n, size=P6.CONFIG["ref_pairs"]), rng.randint(n, size=P6.CONFIG["ref_pairs"]))
        self.SC = {m: P6.Scorer(V, store, cfg, nets, REF) for m, (cfg, nets) in M.items()}
        self.pair, Std = P6.pair, P6.Std
        Fi_tr, Ai_tr = npy("face_ibv", "train"), npy("voice_iba", "train")
        self.mib = (Fi_tr.mean(0), Ai_tr.mean(0))
        self.age_f = lambda sp: npy("face_vitageprob", sp) @ P6.AGE_MIDS
        self.age_v = lambda sp: npy("voice_w2v2agage", sp)[:, 0] * 100
        self.zib = Std(self.ib0(Fi_tr[REF[0]], Ai_tr[REF[1]]))
        self.zag = Std(-np.abs(self.age_f("train")[REF[0]] - self.age_v("train")[REF[1]]))
        Fa_tr, Aa_tr = npy("face_aclip", "train"), npy("voice_aclip", "train")
        self.macl = (Fa_tr.mean(0), Aa_tr.mean(0))
        self.zacl = Std(self.acl0(Fa_tr[REF[0]], Aa_tr[REF[1]]))
        self.cells = [("/".join(k), k) for k in V.CELLS]

    def ib0(self, F_, A_):
        return self.pair(self.V.l2n(F_ - self.mib[0]), self.V.l2n(A_ - self.mib[1]))

    def acl0(self, F_, A_):
        return self.pair(self.V.l2n(F_ - self.macl[0]), self.V.l2n(A_ - self.macl[1]))

    def arrays(self, split):
        S, store, npy = self.S, self.store, self.npy
        return dict(vgg=store.face("vgg", split), btc=store.voice("given", split), ibf=npy("face_ibv", split),
                    iba=npy("voice_iba", split), agef=npy("face_vitageprob", split) @ S.AGE_MIDS,
                    agev=npy("voice_w2v2agage", split)[:, 0] * 100, farl=npy("face_farl", split),
                    ada=npy("face_adaface", split), ecapa=npy("voice_ecapa192", split), vgen=npy("voice_w2v2aggen", split))

    def b3_co(self, split, is_en, rows=None):
        T = self.arrays(split)
        if rows is not None:
            T = {k: v[rows] for k, v in T.items()}
        r, rg = self.b3.relational(T)
        return np.concatenate([self.b3.streams(T), r[:, None], rg[:, None]], 1), self.ca9.conditions(T, is_en=is_en)

    def en(self, split, rows=None):
        S, npy = self.S, self.npy
        tk = (lambda X: X) if rows is None else (lambda X: X[rows])
        Z = []
        for m in ("s007", "s010B"):
            Af, Av = map(tk, self.SC[m].rows(split))
            Z.append(np.mean([sd(self.pair(*self.SC[m].embed0(net, prep, Af, Av))) for (net, prep), sd in zip(self.SC[m].nets, self.SC[m].stds)], 0))
            Z.append(self.SC[m].cstd(self.pair(self.SC[m].cca.transform_x(Af), self.SC[m].cca.transform_y(Av))))
        af, av = tk(self.age_f(split)), tk(self.age_v(split))
        Z += [self.zib(self.ib0(tk(npy("face_ibv", split)), tk(npy("voice_iba", split)))), self.zag(-np.abs(af - av)),
              self.zacl(self.acl0(tk(npy("face_aclip", split)), tk(npy("voice_aclip", split))))]
        pf = self.ca9.gclf.predict_proba(S.l2n(tk(npy("face_ibv", split))))[:, 1]; pv = S.pmale(tk(npy("voice_w2v2aggen", split)))
        return np.stack(Z, 1), np.stack([pf, pv, np.abs(pf - pv), af, av, np.abs(af - av)], 1)

    def file_components(self, c, k):
        """Components of one test file (c = 'no_gender/English', k = its key)."""
        is_en = k[1] == "English"
        B3, CO = self.b3_co(c, is_en)
        EN7, ENCO = self.en(c)
        return dict(key=c, zip_name=self.V.NAMES[k], protocol=k[0], lang=k[1], is_en=is_en,
                    pid=self.store.dev[k][2].pair_id.to_numpy(dtype=str), b3=B3, co=CO, en7=EN7, enco=ENCO,
                    vgen=self.npy("voice_w2v2aggen", c))

    def selftest(self, c, k, seed=1):
        """Level 0: the components of a random half of the file, computed alone, equal the full-file ones."""
        n = len(self.store.dev[k][2]); half = np.sort(np.random.RandomState(seed).choice(n, n // 2, replace=False))
        B3, CO = self.b3_co(c, k[1] == "English"); B3h, COh = self.b3_co(c, k[1] == "English", half)
        EN7, ENCO = self.en(c); EN7h, ENCOh = self.en(c, half)
        return float(max(np.abs(B3h - B3[half]).max(), np.abs(COh - CO[half]).max(), np.abs(EN7h - EN7[half]).max(),
                         np.abs(ENCOh - ENCO[half]).max()))


def scores(comp, fusers):
    """v9 and v11 scores (higher = same) of one file's components; fusers = {v9_bangla, v9_english, v11_bangla}."""
    if comp["is_en"]:
        v9 = apply(fusers["v9_english"], comp["en7"], comp["enco"])
        return v9, v9
    v9 = apply(fusers["v9_bangla"], comp["b3"], comp["co"])
    v11 = apply(fusers["v11_bangla"], np.concatenate([comp["b3"], comp["t3"]], 1), comp["co"])
    return v9, v11


def v12_scores(comp, fusers):
    """v12 (EN-T3-01, CB 22.29): English = CA_EN_T3 (CA-EN streams + v11's T3 streams), Bangla = v11."""
    if comp["is_en"]:
        return apply(fusers["v12_english"], np.concatenate([comp["en7"], comp["t3"]], 1), comp["enco"])
    return scores(comp, fusers)[1]


def save_components(path, comps, extra=None):
    """comps = {i: component dict} -> one npz (keys <name>_<i>) in the layout e2e_lib.components() reads."""
    out = {}
    for i, c in comps.items():
        for k, v in c.items():
            if isinstance(v, np.ndarray):
                out[f"{k}_{i}"] = v.astype(str) if v.dtype == object else v      # no pickled object arrays (np.load without pickle)
    out.update(extra or {})
    np.savez(path, **out)


def write_zip(path, comps, which="v11"):
    import zipfile
    with zipfile.ZipFile(path, "w") as zf:
        for i, c in comps.items():
            s = np.asarray(c[which], np.float64)
            assert len(s) == len(c["pid"]) and np.isfinite(s).all(), c["key"]
            zf.writestr(c["zip_name"], "\n".join(f"{p} {v:.6f}" for p, v in zip(c["pid"], -s)) + "\n")
    return path
