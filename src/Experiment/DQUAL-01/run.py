r"""DQUAL-01 (criteria registered 14/10 in README.md before any result): person-level quality pool over every training source.
Reuses EDA-NEWDATA's loaders. G4 uses the AUXP-01 V4 final CC head (auxp01_final_models.pt of flag2027-sys2001-p).
    NEW_DIR=<feats_avs / lrs3 / avssamv parent> CC_MODELS=<auxp01_final_models.pt> VOX_LID=<lid_summary.json> python run.py
"""
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
_s = importlib.util.spec_from_file_location("edanew", EXP / "EDA-NEWDATA/run.py")
ED = importlib.util.module_from_spec(_s); _s.loader.exec_module(ED)
_t = importlib.util.spec_from_file_location("t3r", EXP / "T3RECIPE-01/t3r.py")
t3r = importlib.util.module_from_spec(_t); _t.loader.exec_module(t3r)
RNG = np.random.RandomState(0)
LABELLED = ("v4 train", "MAV v1/v2", "VoxCeleb2", "VoxBlink")


def cc_models():
    import torch
    ck = torch.load(os.environ["CC_MODELS"], map_location="cpu", weights_only=False)
    out = []
    for p, i, sd in ck["models"]:
        net = t3r.DualLinearProjector(1280, 512, ck["cfg"]["dim"], 0, 0); net.load_state_dict(sd); net.eval()
        out.append(net)
    return out


def project(nets, F, V):
    """Per model: L2 projected faces / voices; returns lists (one array per model)."""
    import torch
    with torch.no_grad():
        fe = [n.face(torch.from_numpy(F)).numpy() for n in nets]
        ve = [n.voice(torch.from_numpy(V)).numpy() for n in nets]
    return fe, ve


def main():
    from sklearn.linear_model import LogisticRegression
    S, R = ED.sources()
    g = dict(R.GMAP)
    vx = S["VoxCeleb2"]
    clf = {"voice": LogisticRegression(C=1.0, max_iter=3000).fit(vx["V"], vx["vmale"]),
           "face": LogisticRegression(C=1.0, max_iter=3000).fit(vx["F"], vx["fmale"])}
    nets = cc_models()
    l2 = ED.l2n
    rows = []
    for name, s in S.items():
        fspk, vspk = s["fspk"], s["vspk"]
        ids = np.intersect1d(np.unique(fspk), np.unique(vspk))
        fi = pd.Series(np.arange(len(fspk))).groupby(fspk).apply(np.array)
        vi = pd.Series(np.arange(len(vspk))).groupby(vspk).apply(np.array)
        pmf = pd.Series(clf["face"].predict_proba(s["F"])[:, 1]).groupby(fspk).mean()
        pmv = pd.Series(clf["voice"].predict_proba(s["V"])[:, 1]).groupby(vspk).mean()
        if name in ("VoxCeleb2", "VoxBlink"):
            lab = pd.Series(s["vmale"]).groupby(vspk).mean() > .5
        elif name in ("v4 train", "MAV v1/v2"):
            lab = pd.Series({p: g[p] == "m" for p in ids if p in g})
        else:
            lab = pmv > .5
        # G2: longest voice (pseudo identities: the whole clip = the two halves)
        if s.get("vsec") is None:
            vmax = pd.Series(np.inf, index=ids)
        elif name in ("AVSpeech", "LRS3", "AVSpeech Nam Á"):
            vmax = pd.Series(s["vsec"]).groupby(pd.Series(s["vvid"]).str.cat(vspk, sep="|").values).sum()
            vmax = pd.Series(vmax.values, index=[k.split("|", 1)[1] for k in vmax.index]).groupby(level=0).max()
        else:
            vmax = pd.Series(s["vsec"]).groupby(vspk).max()
        # G3: consistency with the identity's own centroid
        def cons(X, idx):
            c = l2(np.stack([X[a].mean(0) for a in idx.values]))
            return pd.Series([float(np.mean(X[a] @ c[k])) for k, a in enumerate(idx.values)], index=idx.index)
        cf, cv = cons(s["F"], fi), cons(s["V"], vi)
        # G4: own CC score vs the identity's face with 50 same-gender voices of the same source
        mf = np.stack([s["F"][fi[i]].mean(0) for i in ids]); mv = np.stack([s["V"][vi[i]].mean(0) for i in ids])
        fe, ve = project(nets, l2(mf).astype(np.float32), l2(mv).astype(np.float32))
        male = np.array([bool(lab.get(i, pmv.get(i, .5) > .5)) for i in ids])
        own = np.mean([np.sum(a * b, 1) for a, b in zip(fe, ve)], 0)
        conf = np.zeros(len(ids))
        for sex in (True, False):
            k = np.where(male == sex)[0]
            if len(k) < 2:
                continue
            J = k[RNG.randint(len(k), size=(len(k), 50))]
            imp = np.mean([np.einsum("nd,nkd->nk", a[k], b[J]) for a, b in zip(fe, ve)], 0)
            conf[k] = (imp < own[k][:, None]).mean(1)
        nvid = pd.Series(s["vvid"] if s["vvid"] is not None else np.full(len(vspk), "?")).groupby(vspk).nunique()
        T = pd.DataFrame(dict(source=name, spk=ids, n_face=[len(fi[i]) for i in ids], n_voice=[len(vi[i]) for i in ids],
                              n_videos=nvid.reindex(ids).values if s["vvid"] is not None else np.nan,
                              voice_sec_max=vmax.reindex(ids).values, p_male_face=pmf.reindex(ids).values,
                              p_male_voice=pmv.reindex(ids).values, male=male, cons_face=cf.reindex(ids).values,
                              cons_voice=cv.reindex(ids).values, cc_own=own, cc_conf=conf))
        T["G1"] = (T.p_male_face > .5) == T.male
        T["G2"] = T.voice_sec_max >= 4
        T["G3"] = (T.cons_face >= T.cons_face.quantile(.05)) & (T.cons_voice >= T.cons_voice.quantile(.05))
        T["G4"] = T.cc_conf > .5
        ok = T.G1 & T.G2 & T.G3 & T.G4
        multi = T.n_videos >= 2 if s["vvid"] is not None else pd.Series(True, index=T.index)   # v4: no video code, many clips
        T["tier"] = np.where(ok & multi, "A", np.where(ok, "B", "X"))
        T["fails"] = [",".join(c for c in ("G1", "G2", "G3", "G4") if not r[c]) for _, r in T.iterrows()]
        rows.append(T)
        print(name, T.tier.value_counts().to_dict(), flush=True)
    P = pd.concat(rows, ignore_index=True)
    P.to_parquet(HERE / "persons.parquet", index=False)
    summ = P.groupby("source").agg(identities=("spk", "size"), A=("tier", lambda t: int((t == "A").sum())),
                                   B=("tier", lambda t: int((t == "B").sum())), X=("tier", lambda t: int((t == "X").sum())),
                                   fail_G1=("G1", lambda x: int((~x).sum())), fail_G2=("G2", lambda x: int((~x).sum())),
                                   fail_G3=("G3", lambda x: int((~x).sum())), fail_G4=("G4", lambda x: int((~x).sum())))
    order = ["v4 train", "MAV v1/v2", "VoxCeleb2", "VoxBlink", "AVSpeech", "LRS3", "AVSpeech Nam Á"]
    summ = summ.reindex(order)
    summ.loc["total"] = summ.sum()
    print("\n" + summ.to_markdown())
    (HERE / "summary.md").write_text(summ.to_markdown() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
