"""SAFE-04 step 3d: relational score inside the gender predicted from the trial's own voice.  FOLDS=... python rel_sweep5.py"""
import os, time
os.environ["SAFE04_NOMAIN"] = "1"
from rel_sweep2 import *
from rel import Side, RelMulti

VGEN4 = pmale_voice(np.load(FO / "feats_eval/voice_w2v2aggen_train.npy")) > 0.5
VGENE = pmale_voice(VA["w2v2aggen"]) > 0.5


def run5(s):
    fold = make_fold(s)
    D = pd.concat([pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"])) for (data, prot), d in fold["files"].items()],
                  ignore_index=True)
    tr = fold["v4tr"]
    rr = np.random.RandomState(0)
    rf, rv = tr[rr.randint(len(tr), size=N_REF)], tr[rr.randint(len(tr), size=N_REF)]
    fks, vks = ("farl", "ibf", "ada"), ("ecapa", "btc")
    sides = {}
    for k, kind in [(f, "face") for f in fks] + [(v, "voice") for v in vks]:
        g = id_rows(fold, k, kind); sides[k] = Side([a for a, _ in g], [b for _, b in g], colnorm=True)
    rel = RelMulti([sides[f] for f in fks], [sides[v] for v in vks])
    amale = np.array([GMAP.get(p) == "m" for p in rel.ids])
    keep = lambda vm: (amale[None, :] == vm[:, None])
    t0 = time.time()
    z = Std(rel.score([FACE[f][0][rf] for f in fks], [VOICE[v][0][rv] for v in vks], V4S[rf], V4S[rv], keep=keep(VGEN4[rv])))
    col = []
    for (data, prot), d in fold["files"].items():
        j = 0 if data == "v4" else 1
        vm = (VGEN4 if j == 0 else VGENE)[d["vi"]]
        col.append(z(rel.score([FACE[f][j][d["fi"]] for f in fks], [VOICE[v][j][d["vi"]] for v in vks], keep=keep(vm))))
    D["relg_fia_eb"] = np.concatenate(col)
    log(f"fold {s} relg_fia_eb: {len(rel.ids)} anchors, {amale.sum()} male [{time.time() - t0:.0f}s]")
    D.to_parquet(SP / f"out/safe04/relonly_g_fold{s}.parquet")


if __name__ == "__main__":
    for s in [int(x) for x in os.environ.get("FOLDS", "1").split(",")]:
        run5(s)
