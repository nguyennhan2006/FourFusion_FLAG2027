r"""EDA-NEWDATA (14/10): every training source of the CC head on one scale (face [FaRL16 | CLIP-L], voice [SimAM p | ft], each L2):
size, videos per identity, clip length / quality, label cleanliness (within-modality identity EER), gender balance and face-voice gender
agreement per identity, closeness to the South-Asian MAV persons (Urdu / Hindi). Descriptive only: nothing here selects data.
    NEW_DIR=<folder with feats_avs / feats_lrs3 / feats_avssamv> VOX_LID=<lid_summary.json> python run.py > report.md
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
os.environ["SAFE04_NOMAIN"] = "1"
sys.path[:0] = [str(EXP / "SAFE-04"), str(EXP / "STACK-02"), str(EXP / "T3ENC-01")]
NEW = Path(os.environ["NEW_DIR"])
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
RNG = np.random.RandomState(0)


def cat(d, kind, part=None):
    names = ("farl16", "clipL") if kind == "face" else ("sam100p", "sam100f")
    if part is None:
        return np.concatenate([l2n(np.load(d / f"{kind}_{n}.npy").astype(np.float32)) for n in names], 1)
    if part == "train":
        return np.concatenate([l2n(np.load(d / f"{kind}_{n}_train.npy").astype(np.float32)) for n in names], 1)
    return np.concatenate([l2n(np.concatenate([np.load(d / f"ext_{s}_{kind}_{n}.npy") for s in ("v1_complete", "v2_complete")]).astype(np.float32))
                           for n in names], 1)


def sources():
    """{name: dict(F, fspk, fvid, V, vspk, vvid, vsec, extra)}; identities as the training code keeps them."""
    import rel_sweep5 as R
    import importlib.util                                                    # T3ENC-01/run.py by path: `import run` would be this file
    _s = importlib.util.spec_from_file_location("t3enc01run", EXP / "T3ENC-01/run.py")
    T = importlib.util.module_from_spec(_s); _s.loader.exec_module(T)
    M = EXP / "T3ENC-01/feats_t3enc"
    out = {}
    v4 = np.asarray(R.V4S, str)
    out["v4 train"] = dict(F=cat(M, "face", "train"), fspk=v4, fvid=None, V=cat(M, "voice", "train"), vspk=v4, vvid=None, vsec=None)
    out["MAV v1/v2"] = dict(F=cat(M, "face", "ext"), fspk=R.FMETA.p.to_numpy(dtype=str), fvid=R.FMETA.video.to_numpy(dtype=str),
                            V=cat(M, "voice", "ext"), vspk=R.VMETA.p.to_numpy(dtype=str), vvid=R.VMETA.video.to_numpy(dtype=str),
                            vsec=R.VMETA.dur.to_numpy(dtype=float), lang=R.VMETA.lang.to_numpy(dtype=str))
    fm = pd.read_csv(T.VK / "face_meta.csv", dtype={"video": str, "utt": str}); vm = pd.read_csv(T.VK / "voice_meta.csv", dtype={"video": str, "utt": str})
    drop = T.dropped_speakers()
    kf, kv = ~fm.spk.isin(drop).values, ~vm.spk.isin(drop).values
    grp = {k: v[0] for k, v in json.loads((EXP / "DSEL-01/vox2_groups.json").read_text()).items()}
    out["VoxCeleb2"] = dict(F=cat(T.X_, "face")[kf], fspk=fm.spk.values[kf].astype(str), fvid=fm.video.values[kf].astype(str),
                            V=cat(T.X_, "voice")[kv], vspk=vm.spk.values[kv].astype(str), vvid=vm.video.values[kv].astype(str),
                            vsec=vm["dur"].values[kv].astype(float), vmale=(vm.gender.values[kv] == "m"), fmale=(fm.gender.values[kf] == "m"),
                            sa=lambda ids: float(np.mean([grp.get(i, "OTH") == "SA" for i in ids])))
    VB = EXP / "SCALE-VB/feats_vb"
    fm = pd.read_csv(VB / "face_meta.csv", dtype={"video": str, "utt": str}); vm = pd.read_csv(VB / "voice_meta.csv", dtype={"video": str, "utt": str})
    loc = vm.groupby("spk")["loc"].first().to_dict()
    sa_loc = {"India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal"}
    out["VoxBlink"] = dict(F=cat(VB, "face"), fspk=fm.spk.values.astype(str), fvid=fm.video.values.astype(str),
                           V=cat(VB, "voice"), vspk=vm.spk.values.astype(str), vvid=vm.video.values.astype(str),
                           vsec=vm["dur"].values.astype(float) if "dur" in vm else None,
                           vmale=(vm.gender.values == "Male"), fmale=(fm.gender.values == "Male"),
                           sa=lambda ids: float(np.mean([loc.get(i) in sa_loc for i in ids])))
    for name, sub in (("AVSpeech", "feats_avs"), ("LRS3", "feats_lrs3"), ("AVSpeech Nam Á", "feats_avssamv")):
        d = next(NEW.rglob(sub))
        dt = {"spk": str, "video": str, "utt": str, "clip_id": str, "yid": str}
        fm, vm = pd.read_csv(d / "face_meta.csv", dtype=dt), pd.read_csv(d / "voice_meta.csv", dtype=dt)
        ids = pd.read_csv(d / "identities.csv", dtype={"spk": str})
        keep = set(ids.spk[ids.keep.astype(bool)])
        kf, kv = fm.spk.isin(keep).values, (vm.spk.isin(keep) & (vm.half.astype(float) > 0)).values
        out[name] = dict(F=cat(d, "face")[kf], fspk=fm.spk.values[kf].astype(str), fvid=fm.video.values[kf].astype(str),
                         V=cat(d, "voice")[kv], vspk=vm.spk.values[kv].astype(str), vvid=vm.video.values[kv].astype(str),
                         vsec=vm.sec.values[kv].astype(float), sharp=fm.sharp.values[kf].astype(float),
                         snr=vm.snr_proxy.values[kv].astype(float), speech=vm.speech.values[kv].astype(float),
                         p_bn=vm.p_bn.values[kv].astype(float), n_ids_raw=len(ids))
    return out, R


def id_eer(X, spk, vid=None, n=20000):
    """Same-identity vs different-identity cosine EER on random row pairs (within one modality); with `vid`, positives come
    from two DIFFERENT videos of the identity (NaN when no identity has two videos)."""
    from lab_fuse import eer
    by = pd.Series(np.arange(len(spk))).groupby(spk).apply(np.array)
    multi = [a for a in by.values if len(a) > 1 and (vid is None or len(set(vid[a])) > 1)]
    if not multi:
        return np.nan
    pos = []
    while len(pos) < n:
        i, j = RNG.choice(multi[RNG.randint(len(multi))], 2, replace=False)
        if vid is None or vid[i] != vid[j]:
            pos.append((i, j))
    neg = []
    while len(neg) < n:
        i, j = RNG.randint(len(spk), size=2)
        if spk[i] != spk[j]:
            neg.append((i, j))
    P = np.array(pos + neg); y = np.r_[np.ones(n), np.zeros(n)]
    return eer(np.sum(X[P[:, 0]] * X[P[:, 1]], 1), y)


def main():
    from sklearn.linear_model import LogisticRegression
    S, R = sources()
    g = dict(R.GMAP)
    vx = S["VoxCeleb2"]                                                        # gender classifiers on VoxCeleb2's labels
    clf = {"voice": LogisticRegression(C=1.0, max_iter=3000).fit(vx["V"], vx["vmale"]),
           "face": LogisticRegression(C=1.0, max_iter=3000).fit(vx["F"], vx["fmale"])}
    acc = []                                                                   # identity-level accuracy where labels exist
    for name in ("VoxBlink", "v4 train", "MAV v1/v2"):
        s = S[name]
        for kind, X, spk, lab in (("voice", s["V"], s["vspk"], s.get("vmale")), ("face", s["F"], s["fspk"], s.get("fmale"))):
            pm = pd.Series(clf[kind].predict_proba(X)[:, 1]).groupby(spk).mean()
            truth = (pd.Series(lab).groupby(spk).mean() > .5) if lab is not None else pd.Series({p: g[p] == "m" for p in pm.index if p in g})
            k = pm.index.intersection(truth.index)
            acc.append(dict(source=name, kind=kind, identities=len(k), accuracy=float(((pm[k] > .5) == truth[k]).mean())))
    print(pd.DataFrame(acc).round(3).to_markdown(index=False))
    sa_known = {"v4 train": 1.0, "MAV v1/v2": 1.0, "AVSpeech Nam Á": 1.0}
    rows = []
    for name, s in S.items():
        fspk, vspk = s["fspk"], s["vspk"]
        ids = np.intersect1d(np.unique(fspk), np.unique(vspk))
        vids = pd.Series(s["vvid"] if s["vvid"] is not None else np.full(len(vspk), "?")).groupby(vspk).nunique()
        pmv = pd.Series(clf["voice"].predict_proba(s["V"])[:, 1]).groupby(vspk).mean()
        pmf = pd.Series(clf["face"].predict_proba(s["F"])[:, 1]).groupby(fspk).mean()
        both = pmv.index.intersection(pmf.index)
        if s.get("vmale") is not None:
            male = float((pd.Series(s["vmale"]).groupby(vspk).mean() > .5).mean())
        elif name in ("v4 train", "MAV v1/v2"):
            male = float(np.mean([g[p] == "m" for p in ids if p in g]))
        else:
            male = np.nan
        r = dict(source=name, identities=len(ids), face_rows=len(fspk), voice_rows=len(vspk),
                 voices_per_id=float(pd.Series(vspk).value_counts().median()),
                 multi_video_ids=float((vids >= 2).mean()) if s["vvid"] is not None else np.nan,
                 voice_sec_median=float(np.nanmedian(s["vsec"])) if s.get("vsec") is not None else np.nan,
                 voice_lt4s=float(np.nanmean(s["vsec"] < 4)) if s.get("vsec") is not None else np.nan,
                 voice_eer_any=id_eer(s["V"], vspk),
                 voice_eer_xvideo=id_eer(s["V"], vspk, s["vvid"]) if s["vvid"] is not None else np.nan,
                 face_eer_xvideo=id_eer(s["F"], fspk, s["fvid"]) if s["fvid"] is not None else np.nan,
                 male_share_label=male, male_share_pred=float((pmv > .5).mean()),
                 gender_mismatch_pred=float(((pmv[both] > .5) != (pmf[both] > .5)).mean()),
                 south_asian_share=s["sa"](ids) if "sa" in s else sa_known.get(name, np.nan))
        for q in ("sharp", "snr", "speech"):
            if q in s:
                r[f"{q}_median"] = float(np.nanmedian(s[q]))
        if "p_bn" in s:
            r["p_bn_max"] = float(np.nanmax(s["p_bn"]))
        rows.append(r)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(HERE / "summary.csv", index=False)
    print("\n" + T.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
