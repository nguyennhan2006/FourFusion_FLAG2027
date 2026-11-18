r"""VBSEL-01 (registered 10/10 in README.md): VoxBlink used selectively. Population-matched subsets (South Asian VoxBlink by
metadata, South-Asian-language VoxCeleb2), single-source batches (SSB) and pretrain -> finetune, all with the T3 head
(InfoNCE, PK 512 x 2); gate vs v11. Reuses SCALE-VB/run.py (vb_rows, report) and ALLNEG-01/run.py (train_pk, make_loss).
    VOX_LID=<FLAG_42 lid_summary.json> FOLDS=1,2,3,4,5 python run.py      # VBSEL_ARMS=SA,... for a subset
    VOX_LID=... python run.py report > report.txt
"""
import os, sys, time, json, importlib.util
from contextlib import contextmanager
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
spec = importlib.util.spec_from_file_location("scalevb01run", EXP / "SCALE-VB/run.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)          # also loads ALLNEG-01 (S.A) and its sys.path
A = S.A
TAG = "vbsel01f"
P, K, P_VB = S.P, S.K, 0.15
SA_LOC = {"India", "Pakistan", "Nepal", "Sri Lanka", "Afghanistan", "Bhutan", "Maldives"}
SA_LAN = {"hindi", "urdu", "punjabi", "tamil", "telugu", "marathi", "gujarati", "nepali", "sinhala", "malayalam", "kannada"}
VX_SA = set(json.loads((HERE / "vox2_sa_speakers.json").read_text())["speakers"])
ARMS = {"SA": dict(vox="all", vb="sa"), "SA_SSB": dict(vox="all", vb="sa", ssb=True), "ALL_SSB": dict(vox="all", vb="all", ssb=True),
        "TWO_STAGE": dict(vox="all", vb="all", two_stage=True), "POP_SA": dict(vox="sa", vb="sa")}
DECISION = "SA"
if S._GPU or os.environ.get("VB_B") == "1":
    ARMS = {"B": dict(vox="all", vb="none"), **ARMS}
if os.environ.get("VBSEL_ARMS"):
    ARMS = {k: v for k, v in ARMS.items() if k in os.environ["VBSEL_ARMS"].split(",")}


def vb_sa_speakers():
    import pandas as pd
    vm = pd.read_csv(S.VB / "voice_meta.csv", dtype={"video": str})
    sp = vm.groupby("spk").agg(lan=("lan", "first"), loc=("loc", "first"))
    return set(sp.index[sp["loc"].isin(SA_LOC) | sp.lan.isin(SA_LAN)])


def ssb_batches(p_vb):
    """pk_batches replacement: every batch draws its P identities from ONE source ('vb:' labels vs the rest)."""
    def batches(spk_idx, P_, K_, n_rows, rng):
        src = {"vb": [s for s in spk_idx if str(s).startswith("vb:")], "in": [s for s in spk_idx if not str(s).startswith("vb:")]}
        out = []
        for _ in range(max(1, n_rows // (P_ * K_))):
            pool = src["vb"] if (src["vb"] and rng.rand() < p_vb) else src["in"]
            chosen = rng.choice(len(pool), size=min(P_, len(pool)), replace=False)
            out.append(np.concatenate([rng.choice(spk_idx[pool[c]], size=K_, replace=len(spk_idx[pool[c]]) < K_) for c in chosen]))
        return out
    return batches


@contextmanager
def patched_batches(R, fn):
    g = R.train_pk.__globals__; old = g["pk_batches"]; g["pk_batches"] = fn
    try:
        yield
    finally:
        g["pk_batches"] = old


def run(s):
    import pandas as pd, torch
    import rel_sweep5 as R
    import run as T                                                               # T3ENC-01/run.py
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "10")))
    t0 = time.time()
    fold = R.make_fold(s); files = list(fold["files"].items())
    D = pd.concat([pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"], fi=d["fi"], vi=d["vi"])) for (data, prot), d in files],
                  ignore_index=True)
    F4, V4v, FE, VE = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    arrs = (F4, V4v, FE, VE); X4 = dict(R.V4, vgg=F4, btc=V4v)
    trial = lambda d, data: dict(d, vgg=(F4 if data == "v4" else FE)[d["fi"]], btc=(V4v if data == "v4" else VE)[d["vi"]])
    TF0, TV0, TS0, _ = R.train_rows(fold, ext=True, arrays=arrs)
    FX, VX, SX = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    FB, VBv, SB, _ = S.vb_rows(R.log)
    vb_sa = {"vb:" + x for x in vb_sa_speakers()}
    mx_sa = np.isin(SX, ["vox2:" + x for x in VX_SA])
    mb_sa = np.isin(SB, list(vb_sa))
    R.log(f"fold {s}: VB-SA {len(set(SB[mb_sa]))} identities ({int(mb_sa.sum())} rows) | VX-SA {len(set(SX[mx_sa]))} identities "
          f"({int(mx_sa.sum())} rows)")
    sel = lambda o: ([(TF0, TV0, TS0)]
                     + ([(FX, VX, SX)] if o["vox"] == "all" else [(FX[mx_sa], VX[mx_sa], SX[mx_sa])] if o["vox"] == "sa" else [])
                     + ([(FB, VBv, SB)] if o["vb"] == "all" else [(FB[mb_sa], VBv[mb_sa], SB[mb_sa])] if o["vb"] == "sa" else []))
    stack = lambda parts: tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
    for arm, o in ARMS.items():
        TF, TV, TS = stack(sel(o))
        bpe = max(1, len(TS) // (P * K))
        if o.get("two_stage"):                                                    # A: all data, SSB; B: VoxCeleb2 + base, lr x 0.1
            TFb, TVb, TSb = stack([(TF0, TV0, TS0), (FX, VX, SX)])
            bpe_b = max(1, len(TSb) // (P * K))
            ep_a = max(1, int(round(R.STEPS * 2 / 3 / bpe))); ep_b = max(1, int(round(R.STEPS / 3 / bpe_b)))
            units = []
            for i in range(R.N_NETS):
                with patched_batches(R, ssb_batches(P_VB)):
                    net, prep = R.train_pk(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep_a, loss_fn=A.make_loss(0.0), P=P, K=K)
                units.append(R.train_pk(TFb, TVb, TSb, seed=fold["s"] + 100 * i + 7, epochs=ep_b, loss_fn=A.make_loss(0.0), prep=prep,
                                        net=net, lr=R.REC["lr"] * 0.1, P=P, K=K))
            mem, steps = R.Member(units, TFb, TVb), f"{ep_a * bpe} + {ep_b * bpe_b}"
        else:
            ep = max(1, int(round(R.STEPS / bpe)))
            with patched_batches(R, ssb_batches(P_VB) if o.get("ssb") else R.train_pk.__globals__["pk_batches"]):
                units = [R.train_pk(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep, loss_fn=A.make_loss(0.0), P=P, K=K)
                         for i in range(R.N_NETS)]
            mem, steps = R.Member(units, TF, TV), f"{ep * bpe}"
        sc = R.L0(mem, fold["v4tr"], X4=X4)
        S_ = [sc.streams(trial(d, data)) for (data, prot), d in files]
        D[f"vb{arm}_deep"] = np.concatenate([x["deep0"] for x in S_]); D[f"vb{arm}_cca"] = np.concatenate([x["cca"] for x in S_])
        nvb = len({x for x in set(TS) if str(x).startswith("vb:")})
        R.log(f"fold {s} {arm}: {len(set(TS))} identities ({nvb} VoxBlink), {len(TS)} rows, {steps} steps "
              f"(P {P} K {K}{', SSB p_vb ' + str(P_VB) if o.get('ssb') or o.get('two_stage') else ''}, {R.N_NETS} nets) [{time.time() - t0:.0f}s]")
        D.to_parquet(R.SP / f"out/safe04/{TAG}_fold{s}.parquet")
    R.log(f"fold {s} done [{time.time() - t0:.0f}s]")


def report():
    S.TAG, S.ARMS, S.DECISION = TAG, ARMS, DECISION          # the SCALE-VB report: gate vs v11 + same-device sensitivity vs B
    S.report()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            run(s)
