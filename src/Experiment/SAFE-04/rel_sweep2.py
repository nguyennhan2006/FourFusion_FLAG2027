"""SAFE-04 step 3b: profile-level ensembles (several face / voice encoders) and train column normalisation.

    FOLDS=1,2,3,4,5 python rel_sweep2.py   -> relonly_multi_fold<s>.parquet
"""
import os, time
os.environ["SAFE04_NOMAIN"] = "1"
from rel_sweep import *
from rel import Side, RelMulti

VARIANTS = {} if os.environ.get("SWEEP3") else {
    "relm_fia_e": (("farl", "ibf", "ada"), ("ecapa",), False),
    "relm_fia_eb": (("farl", "ibf", "ada"), ("ecapa", "btc"), False),
    "relm_fi_e": (("farl", "ibf"), ("ecapa",), False),
    "relc_f_e": (("farl",), ("ecapa",), True),
    "relc_fia_e": (("farl", "ibf", "ada"), ("ecapa",), True),
    "relc_fia_eb": (("farl", "ibf", "ada"), ("ecapa", "btc"), True),
    "relc_all_eb": (("farl", "ibf", "ada", "vgg", "siglip2"), ("ecapa", "btc"), True),
}


if os.environ.get("SWEEP3"):
    for post in ("rank", "top10", "top30", "top60"):
        VARIANTS[f"relc_fia_eb_{post}"] = (("farl", "ibf", "ada"), ("ecapa", "btc"), True, post)


def id_rows(fold, key, kind):
    """[(identity, rows of encoder `key`)] for the fold's train identities, in a fixed order (v4 train, then ext pool)."""
    M = FACE if kind == "face" else VOICE
    tr = fold["v4tr"]
    out = [(p, M[key][0][tr[V4S[tr] == p]]) for p in np.unique(V4S[tr])]
    meta = FMETA if kind == "face" else VMETA
    pv, pf = VMETA.p.values, FMETA.p.values
    drop = np.isin(VMETA.path.values, list(MAV_DROP)) if MAV_DROP else np.zeros(len(VMETA), bool)   # CLEANCOMP-01: LID_STRICT voice files leave the anchors
    ok = ~drop                                                                          # (MAV_DROP empty = unchanged bit for bit)
    for p in fold["ext_pool"]:
        if (pf == p).any() and (ok & (pv == p)).any():
            out.append((p, M[key][1][np.where((meta.p.values == p) & (ok if kind == "voice" else True))[0]]))
    return out


def run2(s):
    fold = make_fold(s)
    D = pd.concat([pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"])) for (data, prot), d in fold["files"].items()],
                  ignore_index=True)
    tr = fold["v4tr"]
    rr = np.random.RandomState(0)
    rf, rv = tr[rr.randint(len(tr), size=N_REF)], tr[rr.randint(len(tr), size=N_REF)]
    sides = {}
    for name, spec in VARIANTS.items():
        fks, vks, cn = spec[:3]; post = spec[3] if len(spec) > 3 else None
        t0 = time.time()
        for k, kind in [(f, "face") for f in fks] + [(v, "voice") for v in vks]:
            if (k, kind, cn) not in sides:
                g = id_rows(fold, k, kind)
                sides[(k, kind, cn)] = Side([a for a, _ in g], [b for _, b in g], colnorm=cn)
        rel = RelMulti([sides[(f, "face", cn)] for f in fks], [sides[(v, "voice", cn)] for v in vks], post=post)
        z = Std(rel.score([FACE[f][0][rf] for f in fks], [VOICE[v][0][rv] for v in vks], V4S[rf], V4S[rv]))
        col = []
        for (data, prot), d in fold["files"].items():
            j = 0 if data == "v4" else 1
            col.append(z(rel.score([FACE[f][j][d["fi"]] for f in fks], [VOICE[v][j][d["vi"]] for v in vks])))
        D[name] = np.concatenate(col)
        log(f"fold {s} {name}: {len(rel.ids)} ids [{time.time() - t0:.0f}s]")
    D.to_parquet(SP / f"out/safe04/relonly_{'post' if os.environ.get('SWEEP3') else 'multi'}_fold{s}.parquet")


if __name__ == "__main__":
    for s in [int(x) for x in os.environ.get("FOLDS", "1").split(",")]:
        run2(s)
