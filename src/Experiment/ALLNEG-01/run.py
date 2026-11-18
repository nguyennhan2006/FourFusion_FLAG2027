r"""ALLNEG-01 (registered 08/10 in README.md): the T3 head with ALL identities as negatives (learned-prototype AAM term),
large PK batches, and tail weight averaging + 5 nets, against T3ENC-01's arm B (= v10's head).
    FOLDS=1,2,3,4,5 python run.py ; python run.py report > report.txt
Env: VOX_LID (FLAG_42 lid_summary.json) applies the registered Bengali filter (then TAG allneg01f); AN_ARMS=B,AAM1,... subset.
"""
import os, sys, time, json, warnings, copy
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
os.environ["SAFE04_NOMAIN"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "SAFE-04")); sys.path.insert(0, str(HERE.parents[0] / "VOX2-01")); sys.path.insert(0, str(HERE.parents[0] / "T3ENC-01"))
LID = os.environ.get("VOX_LID")
TAG = "allneg01f" if LID else "allneg01"
T3TAG = "t3enc01f" if LID else "t3enc01"
ARMS = {"B": dict(), "AAM1": dict(aam=1.0), "AAM1L": dict(aam=1.0, steps=3600), "P256": dict(P=256, K=2), "SWA5": dict(swa=0.2, nets=5)}
if os.environ.get("AN_ARMS"):
    ARMS = {k: v for k, v in ARMS.items() if k in os.environ["AN_ARMS"].split(",")}
AAM_M, AAM_S = 0.2, 30.0
DECISION = "AAM1"                                     # the registered decision arm (BIGB-01 reuses this module)
REF_FROM_B = False                                    # sensitivity: reference = this run's own arm B (GPU runs) instead of v10


def aam_loss(u, W, y):
    """Additive angular margin softmax of L2-normalised embeddings u against L2-normalised prototypes W (one per identity)."""
    import torch, torch.nn.functional as F
    cos = u @ F.normalize(W, dim=1).t()
    cos_y = cos.gather(1, y[:, None])
    margin = torch.cos(torch.acos(cos_y.clamp(-1 + 1e-6, 1 - 1e-6)) + AAM_M)
    logits = AAM_S * (cos.scatter(1, y[:, None], margin))
    return F.cross_entropy(logits, y)


def make_loss(lam):
    import torch
    def loss_fn(net, f, v, y, g):
        import rel_sweep5 as R
        u, w = net(f, v)
        loss = R.infonce(u, w, y, R.REC["tau"])
        if lam:
            loss = loss + lam * 0.5 * (aam_loss(u, net.W, y) + aam_loss(w, net.W, y))
        return loss
    return loss_fn


def train_swa(R, TF, TV, spk, seed, epochs, tail, P, K):
    """= R.train_pk (InfoNCE, PK sampler, AdamW + cosine), plus a running average of the weights over the last `tail`
    share of the epochs (SWA-style, within one run; BatchNorm statistics averaged too)."""
    import torch
    cfg = dict(R.REC, epochs=epochs)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    prep = R.Prep(TF, TV, cfg["pca_f"], cfg["pca_v"])
    Ftr = torch.tensor(prep.f(TF), device=R.DEVICE); Vtr = torch.tensor(prep.v(TV), device=R.DEVICE)
    cls = {s: i for i, s in enumerate(np.unique(spk))}
    Y = torch.tensor([cls[s] for s in spk], device=R.DEVICE)
    net = R.Model(Ftr.shape[1], Vtr.shape[1], len(cls), cfg).to(R.DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    n = len(spk)
    spk_idx = {s: np.where(np.asarray(spk) == s)[0] for s in np.unique(spk)}
    P = min(P, len(spk_idx))
    start = int(round(cfg["epochs"] * (1 - tail))); avg, k = None, 0
    for ep in range(cfg["epochs"]):
        net.train()
        rng.permutation(n)
        for b in R.pk_batches(spk_idx, P, K, n, rng):
            bt = torch.tensor(b, device=R.DEVICE)
            u, w = net(Ftr[bt], Vtr[bt])
            loss = R.infonce(u, w, Y[bt], cfg["tau"])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep >= start:
            sd = {k_: v.detach().clone().float() for k_, v in net.state_dict().items()}
            avg = sd if avg is None else {k_: avg[k_] + sd[k_] for k_ in avg}; k += 1
    if avg is not None:
        net.load_state_dict({k_: (v / k).to(net.state_dict()[k_].dtype) for k_, v in avg.items()})
    net.eval()
    return net, prep


def run(s):
    import pandas as pd, torch
    import rel_sweep5 as R
    import run as T                                   # T3ENC-01/run.py: t3(), vox_rows(), dropped_speakers()
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
    TF, TV, TS, info = R.train_rows(fold, ext=True, arrays=arrs)
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    TF, TV, TS = np.concatenate([TF, F2]), np.concatenate([TV, V2]), np.concatenate([TS, S2])
    for arm, o in ARMS.items():
        P, K = o.get("P", R.P_PK), o.get("K", R.K_PK)
        bpe = max(1, len(TS) // (P * K)); ep = max(1, int(round(o.get("steps", R.STEPS) / bpe)))
        nets = o.get("nets", R.N_NETS)
        if "swa" in o:
            units = [train_swa(R, TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep, tail=o["swa"], P=P, K=K) for i in range(nets)]
        else:
            units = [R.train_pk(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep, loss_fn=make_loss(o.get("aam", 0.0)), P=P, K=K)
                     for i in range(nets)]
        sc = R.L0(R.Member(units, TF, TV), fold["v4tr"], X4=X4)
        S_ = [sc.streams(trial(d, data)) for (data, prot), d in files]
        D[f"an{arm}_deep"] = np.concatenate([x["deep0"] for x in S_]); D[f"an{arm}_cca"] = np.concatenate([x["cca"] for x in S_])
        R.log(f"fold {s} {arm}: {len(set(TS))} identities, {len(TS)} rows, {ep * bpe} steps (P {P} K {K}, {nets} nets) [{time.time() - t0:.0f}s]")
    D.to_parquet(R.SP / f"out/safe04/{TAG}_fold{s}.parquet")
    R.log(f"fold {s} done [{time.time() - t0:.0f}s]")


def report():
    import pandas as pd, importlib.util
    from stream_gate import load, S6
    from lab_fuse import SP, table
    sys.path.insert(0, str(HERE.parents[0] / "STACK-02"))   # STACK-02/run.py imports its sibling pools.py
    spec = importlib.util.spec_from_file_location("stack02run", HERE.parents[0] / "STACK-02/run.py")
    S2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S2)
    D = load(TAG)
    T3 = pd.concat([pd.read_parquet(f) for f in sorted((SP / "out/safe04").glob(f"{T3TAG}_fold*.parquet"))], ignore_index=True)
    assert (T3.y.values == D.y.values).all() and (T3.vi.values == D.vi.values).all()
    for c in ("t3B_deep", "t3B_cca", "t3_relg"):
        D[c] = T3[c].values
    if REF_FROM_B:
        D["t3B_deep"], D["t3B_cca"] = D.anB_deep.values, D.anB_cca.values
    R2 = pd.read_parquet(SP / "out/safe04/stack02_rows.parquet"); D["fp"] = R2.fp.values
    persons = {int(k): set(v) for k, v in json.loads((HERE.parents[0] / "STACK-02/fold_persons.json").read_text()).items()}
    arms = [a for a in ARMS if f"an{a}_deep" in D]
    if "B" in arms:
        print(f"wiring: corr(anB_deep, t3B_deep) = {np.corrcoef(D.anB_deep, D.t3B_deep)[0, 1]:.4f} (train_pk defaults must reproduce train_one: 1.0000)")
    print("\n== standalone EER of each head (mean of folds)")
    cols = ["t3B_deep"] + [f"an{a}_deep" for a in arms]
    print(table(D, {c: (lambda g, c=c: g[c].values) for c in cols}).groupby("cell").mean(numeric_only=True)[cols].round(2).to_string())
    sets = {"v10": S6 + ["t3B_deep", "t3B_cca", "t3_relg"], **{a: S6 + [f"an{a}_deep", f"an{a}_cca", "t3_relg"] for a in arms if a != "B"}}
    for regime, held in S2.HELD.items():
        for s in sorted(D.fold.unique()):
            te = D[(D.fold == s) & (D.data.isin(held) if held else True)]
            pool = S2.purge(D[(D.fold != s) & ~D.data.isin(held)], persons[int(s)])
            for name, st in sets.items():
                col = f"{name}|{regime}"
                if col not in D:
                    D[col] = np.nan
                D.loc[te.index, col] = S2.Design(pool, te, st).score(st, 1.0)
    T2 = table(D, {k: (lambda g, k=k: g[k].values) for k in D.columns if "|" in k})
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    print("\n== gate: CA-FUSE(S6 + arm deep, cca, t3_relg) vs v10 (S6 + t3B_*, t3_relg); purged; + = better; per cell (folds better)")
    for name in [n for n in sets if n != "v10"]:
        out = {}
        for regime in S2.HELD:
            sub = T2[["cell", f"v10|{regime}", f"{name}|{regime}"]].dropna()
            d = (sub[f"v10|{regime}"] - sub[f"{name}|{regime}"]).groupby(sub.cell)
            m, w = d.mean(), d.apply(lambda x: int((x > 0).sum()))
            g = [k for k in m.index if k.endswith(" g")]
            out[regime] = dict(all=m.mean(), g=m[g].mean() if g else np.nan, w4=int(sum(w[k] >= 4 for k in g)),
                               g_no_v4=m[[k for k in g if k != "v4 g"]].mean() if g else np.nan,
                               cells=" ".join(f"{k.replace(' ', '/')} {m[k]:+.2f}({w[k]})" for k in m.index))
        ok = out["oof"]["g"] >= 1.0 and out["oof"]["w4"] >= 2 and out["hold_urdu"]["all"] >= -0.2 and out["hold_hindi"]["all"] >= -0.2
        D2 = D.rename(columns={"v10|oof": "ref_", f"{name}|oof": "cand_"})
        lo, hi = S2.boot(D2, "ref_", "cand_", gc, "both")
        print(f"  {name:<6} oof all {out['oof']['all']:+.2f} | g {out['oof']['g']:+.2f} (>= 4/5 in {out['oof']['w4']} g cells; without v4 "
              f"{out['oof']['g_no_v4']:+.2f}) | hold urdu {out['hold_urdu']['all']:+.2f} hindi {out['hold_hindi']['all']:+.2f} | CI both sides "
              f"[{lo:+.2f}, {hi:+.2f}] -> {'PASSES' if ok else 'fails'}" + ("   <- DECISION" if name == DECISION else ""))
        print(f"         {out['oof']['cells']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            run(s)
