"""FUSENET-01 (README registered 15/11 before any number): classifier networks as the fuser of the two families (F1s' + RFX) on the
10 raw streams + 7 conditions of the one-fuser system J01 (FUSEARCH-01 / 02): MLP, residual net on the linear fuser, condition-gated
monotone net, self-attention over the stream set, and their logit average. Kaggle CPU (FLAG_219):
    python lab_entry.py FUSENET-01 report   (lab data + T3S job tables + rfr{c0r,c0s} of FLAG_213 + rfrc0t of FLAG_216)
Per RFX replicate x: control U_x = F1s' + R_x; rule A = nested choice among {U_x, nets, AVG} vs U_x (g cells >= +0.3, >= 4/5 folds,
held-out Urdu / Hindi >= -0.2; side: LB4-Bangla >= 0, worst g >= 0, ng mean >= -0.2); rule B = nested choice among the nets vs J01_x.
PASS = rule A on >= 2 of 3 replicates and no replicate negative. Fixed network hyper-parameters (README section 2), nothing tuned.
Nothing reads dev / CB / Evaluation. Writes fusenet_report.txt / fusenet_summary.json."""
import importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


FS = _load("fusearch01", EXPD / "FUSEARCH-01" / "run.py")
M, E, S6, table, OUT, KEY, say, FAST, NB = FS.M, FS.E, FS.S6, FS.table, FS.OUT, FS.KEY, FS.say, FS.FAST, FS.NB
TAG = "fusenet"
SRCS = [("rfr", "c0r"), ("rfr", "c0s"), ("rfr", "c0t")]
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")
NETS = ["MLP", "RES", "GATE", "SET"]
SEEDS, EPOCHS, BATCH, C_LIN = ((0,), 1, 1024, 0.1) if FAST else ((0, 1), 30, 1024, 0.1)   # FAST = smoke (wiring only)
LR, WD = 3e-3, 1e-2


def js(p):
    return dict(mean=round(float(np.mean(p)), 3), wins=int((np.asarray(p) > 0).sum()), folds=[round(float(v), 3) for v in p])


def make_net(kind, d, ns, k, sign, lin0):
    """Columns of X: [ns streams | k conditions | ns x k interactions] (= STACK-02 Design.idx order), standardised."""
    import torch, torch.nn as nn, torch.nn.functional as F

    class MLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.b = nn.Sequential(nn.Linear(ns + k, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

        def forward(s, X):
            return s.b(X[:, :ns + k]).squeeze(1)

    class RES(nn.Module):   # linear fuser on the full design (initialised at the J01 solution) + zero-initialised MLP residual
        def __init__(s):
            super().__init__(); s.lin = nn.Linear(d, 1); s.b = nn.Sequential(nn.Linear(ns + k, 16), nn.GELU(), nn.Linear(16, 1))
            with torch.no_grad():
                s.lin.weight.copy_(torch.tensor(np.asarray(lin0[0]), dtype=torch.float32)[None]); s.lin.bias.fill_(float(lin0[1]))
                s.b[2].weight.zero_(); s.b[2].bias.zero_()

        def forward(s, X):
            return (s.lin(X) + s.b(X[:, :ns + k])).squeeze(1)

    class GATE(nn.Module):  # conditions -> one positive weight per stream (times the stream's fixed sign) + a bias
        def __init__(s):
            super().__init__(); s.h = nn.Sequential(nn.Linear(k, 16), nn.GELU()); s.a = nn.Linear(16, ns); s.c = nn.Linear(16, 1)
            s.register_buffer("sg", torch.tensor(np.asarray(sign), dtype=torch.float32))

        def forward(s, X):
            h = s.h(X[:, ns:ns + k])
            return (F.softplus(s.a(h)) * s.sg * X[:, :ns]).sum(1) + s.c(h).squeeze(1)

    class SET(nn.Module):   # one token per stream [value | stream embedding | conditions] -> 1 Transformer layer -> mean -> logit
        def __init__(s):
            super().__init__(); s.e = nn.Parameter(torch.randn(ns, 8) * 0.1); s.i = nn.Linear(1 + 8 + k, 16)
            s.t = nn.TransformerEncoderLayer(16, 2, 32, dropout=0.1, batch_first=True); s.o = nn.Linear(16, 1)

        def forward(s, X):
            n = len(X)
            tok = torch.cat([X[:, :ns, None], s.e[None].expand(n, -1, -1), X[:, None, ns:ns + k].expand(-1, ns, -1)], 2)
            return s.o(s.t(s.i(tok)).mean(1)).squeeze(1)

    return {"MLP": MLP, "RES": RES, "GATE": GATE, "SET": SET}[kind]()


def net_scores(kind, Xtr, ytr, Xte, ns, k, keep=None):
    """Mean test logit of the SEEDS networks of one kind, trained on (Xtr, ytr) with the fixed recipe of README section 2."""
    import torch, torch.nn.functional as F
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    torch.set_num_threads(4)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(np.asarray(ytr, np.float32))
    Xe = torch.tensor(Xte, dtype=torch.float32); n, d = Xt.shape
    lin0 = sign = None
    if kind == "RES":
        lr = LogisticRegression(C=C_LIN, max_iter=8000).fit(Xtr, ytr); lin0 = (lr.coef_[0], lr.intercept_[0])
    if kind == "GATE":
        sign = [1.0 if np.nan_to_num(spearmanr(Xtr[:, i], ytr)[0]) >= 0 else -1.0 for i in range(ns)]
    out = np.zeros(len(Xte)); steps = EPOCHS * math.ceil(n / BATCH)
    for sd in SEEDS:
        torch.manual_seed(sd); g = torch.Generator().manual_seed(sd)
        m = make_net(kind, d, ns, k, sign, lin0)
        if kind == "RES":
            opt = torch.optim.AdamW([{"params": m.lin.parameters(), "lr": 1e-3, "weight_decay": 0.0},
                                     {"params": m.b.parameters(), "lr": LR, "weight_decay": WD}])
        else:
            opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
        m.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, BATCH):
                ix = perm[i:i + BATCH]
                loss = F.binary_cross_entropy_with_logits(m(Xt[ix]), yt[ix])
                if kind == "RES":                                     # the L2 penalty of LogisticRegression(C) in mean-loss form
                    loss = loss + (m.lin.weight ** 2).sum() / (2 * C_LIN * n)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval()
        if keep is not None:                                          # export: the fitted weights (JSON lists, float32 values)
            keep.append({a: b.tolist() for a, b in m.state_dict().items()})
        with torch.no_grad():
            out += np.concatenate([m(Xe[i:i + 8192]).numpy() for i in range(0, len(Xe), 8192)])
    assert np.isfinite(out).all(), f"{kind}: non-finite logits"
    return out / len(SEEDS)


def fuse_net(D, S2, persons, name, st, kind):
    """Same splits as FUSEARCH-01.fuse (nested per fold, purged, 3 regimes, pooled protocols); the model is a network."""
    t0 = time.time(); ns, k = len(st), len(S2.CO)
    for regime, held in S2.HELD.items():
        col = f"{name}|{regime}"; D[col] = np.nan
        for s in sorted(D.fold.unique()):
            mt = (D.fold == s) & (D.data.isin(held) if held else True)
            mp = (D.fold != s) & ~D.data.isin(held)
            te = D[mt]
            if len(te):
                Z = S2.Design(S2.purge(D[mp], persons[int(s)]), te, st); j = Z.idx(st)
                D.loc[te.index, col] = net_scores(kind, Z.Xtr[:, j], Z.ytr, Z.Xte[:, j], ns, k)
    say(f"  {name:<10} {kind} on {ns} streams + {k} conditions, seeds {SEEDS}, {EPOCHS} epochs [{time.time() - t0:.0f}s]")


def report():
    if os.environ.get("FUSENET_MODE") == "export":                    # FLAG_222: the entry point of the notebook is always `report`
        return export()
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    CT.merge_t3s()
    got = [(p, a) for p, a in SRCS if (OUT / f"{p}{a}lab_fold1.parquet").exists()]
    if not FAST:
        assert got == SRCS, f"tables {got} != registered {SRCS}"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for p, a in got:
            P = pd.read_parquet(OUT / f"{p}{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[c].astype(str).values == T[c].astype(str).values).all() for c in KEY), f"{p}{a} fold {s}: rows"
            for c in STR:
                T[f"{a}_{c}"] = P[f"{p}{a}_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    reps = [a for _, a in got]
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG); D["is_gp"] = D.cell.str.endswith(" g")
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials, RFX replicates {reps}; networks {NETS} + AVG, seeds {SEEDS}, {EPOCHS} epochs, batch {BATCH}")
    for a in reps:
        for c in STR:
            D[f"{a}_{c}"] = D[f"{a}_{c}"].fillna(float(np.nanmean(D[f"{a}_{c}"].values)))
    groups = {f"trRFS_{a}": [f"{a}_{c}" for c in STR] for a in reps}
    ok = np.asarray(~E.mav_flagged(D), dtype=bool)
    for g, cs in groups.items():
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    FS.fuse(D, S2, persons, "F1s", F1S)
    if os.environ.get("FUSENET_MODE") == "dumpv4":   # XJF-01 (16/11): base score U = F1s + R of the held-out TRAIN trials (v4 cells) -> scale constants
        def _ser(v):
            if isinstance(v, dict): return {str(a): _ser(b) for a, b in v.items()}
            if isinstance(v, (set, frozenset, list, tuple, np.ndarray)): return sorted(str(x) for x in v)
            return str(v)
        for a in reps:
            FS.fuse(D, S2, persons, f"R_{a}", [f"trRFS_{a}"]); D[f"U_{a}"] = D["F1s|oof"] + D[f"R_{a}|oof"]
        V = D[D.cell.str.startswith("v4")]
        keep = [c for c in V.columns if not pd.api.types.is_float_dtype(V[c])] + ["F1s|oof"] + [f"U_{a}" for a in reps]
        V[keep].to_parquet(OUT / "xjfdump_v4.parquet")
        (OUT / "xjfdump_persons.json").write_text(json.dumps(_ser(persons)))
        say(f"== dumpv4: {len(V)} v4 trials, columns {keep}")
        for c, g in V.groupby("cell"):
            y = g.y.values.astype(bool)
            for a in reps:
                u = g[f"U_{a}"].values.astype(float)
                say(f"  dumpv4 {c} {a}: n {len(u)} pos {int(y.sum())} | sd_all {u.std():.4f} sd_pos {u[y].std():.4f} sd_neg {u[~y].std():.4f} mean_pos {u[y].mean():.4f} mean_neg {u[~y].mean():.4f}")
        (OUT / "xjfdump_report.txt").write_text("\n".join(M.TXT) if hasattr(M, "TXT") else "")
        return
    for a in reps:
        FS.fuse(D, S2, persons, f"R_{a}", [f"trRFS_{a}"])
        for r in REG:
            D[f"U_{a}|{r}"] = D[f"F1s|{r}"] + D[f"R_{a}|{r}"]
        raw = F1S + [f"{a}_{c}" for c in STR]
        FS.fuse(D, S2, persons, f"J01_{a}", raw, C=C_LIN)
        for kd in NETS:
            fuse_net(D, S2, persons, f"{kd}_{a}", raw, kd)
        for r in REG:
            D[f"AVG_{a}|{r}"] = sum(D[f"{c}_{a}|{r}"] for c in ["J01"] + NETS) / (1 + len(NETS))

    lbb = lambda x: (((x["urdu ng"] + x["hindi ng"]) / 2 + x["urdu g"]) / 2).values
    summ, verdA, verdB, neg = {"replicates": reps, "recipe": dict(seeds=list(SEEDS), epochs=EPOCHS, batch=BATCH, lr=LR, wd=WD)}, {}, {}, {}
    for a in reps:
        ctl, j01 = f"U_{a}", f"J01_{a}"
        nets = [f"{kd}_{a}" for kd in NETS] + [f"AVG_{a}"]
        cols = [ctl, j01] + nets
        T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in cols})
        Tg = T[T.cell.str.endswith(" g")].groupby("fold")[cols].mean()
        Tn = T[T.cell.str.endswith(" ng")].groupby("fold")[cols].mean()
        piv = T.set_index(["fold", "cell"])
        say(f"\n== replicate {a}: EER (oof, mean of folds)\n" + T.groupby("cell").mean(numeric_only=True)[cols].round(2).T.to_string())
        picks = {}
        for sel, pool in ((f"SELA_{a}", [ctl] + nets), (f"SELB_{a}", nets)):
            for r in REG:
                D[f"{sel}|{r}"] = np.nan
            picks[sel] = {}
            for s in sorted(D.fold.unique()):
                pick = Tg.loc[Tg.index != s, pool].mean().idxmin(); picks[sel][int(s)] = pick
                m = (D.fold == s).values
                for r in REG:
                    D.loc[m, f"{sel}|{r}"] = D.loc[m, f"{pick}|{r}"]

        def stats(n, ref):
            p, h = EDA.cmp(D, n, ref)
            T2 = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in (ref, n)})
            x0, x1 = (T2.set_index(["fold", "cell"])[c].unstack("cell") for c in (ref, n))
            dl, dw = lbb(x0) - lbb(x1), (x0[gc].max(1) - x1[gc].max(1)).values
            ng = [c for c in x0.columns if c.endswith(" ng")]; dn = (x0[ng].mean(1) - x1[ng].mean(1)).values
            return p, h, dl, dw, dn

        out = {}
        say(f"  vs {ctl} (> 0 = candidate better): g cells | held-out | LB4-Bangla | worst g | ng mean")
        for n in [j01] + nets + [f"SELA_{a}"]:
            p, h, dl, dw, dn = stats(n, ctl)
            say(f"  {n:<10} g {EDA.fmt(p)} | held-out {h} | LB4-Bangla {dl.mean():+.2f} ({(dl > 0).sum()}/5) | worst g {dw.mean():+.2f} "
                f"({(dw > 0).sum()}/5) | ng {dn.mean():+.2f} ({(dn > 0).sum()}/5)")
            out[n] = dict(g=js(p), hold=h, lb4bn=js(dl), worst=js(dw), ng=js(dn))
        pA, hA, dl, dw, dn = stats(f"SELA_{a}", ctl)
        mainA = EDA.rule(pA, hA, thr=.3); sideA = bool(dl.mean() >= 0 and dw.mean() >= 0 and dn.mean() >= -.2)
        lo, hi = S2.boot(D.rename(columns={f"{ctl}|oof": "ref_", f"SELA_{a}|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
        say(f"  rule A: SEL_A picks {picks[f'SELA_{a}']}: main gate {'PASS' if mainA else 'FAIL'}, side conditions "
            f"{'PASS' if sideA else 'FAIL'}; CI95 person bootstrap [{lo:+.2f}, {hi:+.2f}]")
        say(f"  vs {j01} (the linear one-fuser system on the same inputs): g cells | held-out")
        outB = {}
        for n in nets + [f"SELB_{a}"]:
            p, h = EDA.cmp(D, n, j01)
            say(f"  {n:<10} g {EDA.fmt(p)} | held-out {h}")
            outB[n] = dict(g=js(p), hold=h)
        pB, hB = EDA.cmp(D, f"SELB_{a}", j01); ruleB = EDA.rule(pB, hB, thr=.3)
        say(f"  rule B: SEL_B picks {picks[f'SELB_{a}']}: {'PASS (network adds to the linear fuser)' if ruleB else 'FAIL'}")
        verdA[a], verdB[a] = bool(mainA and sideA), bool(ruleB)
        neg[a] = bool(pA.mean() < 0 or min(hA.values()) < -.2)
        summ[a] = dict(picks=picks, vs_control=out, vs_j01=outB, mainA=mainA, sideA=sideA, ci95=[round(lo, 3), round(hi, 3)], ruleB=ruleB,
                       eer_cells=T.groupby("cell").mean(numeric_only=True)[cols].round(3).to_dict())
    passed = sum(verdA.values()) >= 2 and not any(neg.values()) and len(reps) == len(SRCS)
    say(f"\n== DECISION (README section 3): {'PASS -> frozen Evaluation zip of the network fuser (not submitted)' if passed else 'FAIL -> keep FACTOR_ENFUSE'}"
        f" (rule A per replicate {verdA}, negative replicate {neg}); rule B per replicate {verdB}")
    summ.update(passed=passed, ruleA=verdA, ruleB=verdB, negative=neg)
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


def export():
    """README section 3 after PASS (16/11): J01 + the four networks fitted on EVERY SAFE-04 row (no purge, MAV-flagged rows out, as
    E.fit_fuser). RFX streams = oof rfdc0 of FLAG_195 (c5rl seeds 1-3 = the seeds of the Evaluation projector; the table of FUSEARCH-03).
    Kaggle CPU (FLAG_222, FUSENET_MODE=export) -> fusenet_summary.json (key export: J01, network weights, probe rows for the reload check).
    Nothing reads dev / CB / Evaluation."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    t00 = time.time(); XT = "fusenetx"; RAW = ["c0_t3", "c0_nf", "c0_nv"]
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py"); CT.merge_t3s()
    assert (OUT / "rfdc0lab_fold1.parquet").exists(), "attach the FLAG_195 part 1 run (rfdc0 tables)"
    parts = []
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet"); P = pd.read_parquet(OUT / f"rfdc0lab_fold{s}.parquet")
        assert len(P) == len(T) and all((P[c].astype(str).values == T[c].astype(str).values).all() for c in KEY), f"rfdc0 fold {s}: rows"
        for c in STR:
            T[f"c0_{c}"] = P[f"rfdc0_{c}"].values
        parts.append(T)
    allv = pd.concat(parts); fill = {c: float(np.nanmean(allv[c].values)) for c in RAW}
    for s, T in enumerate(parts, 1):
        for c in RAW:
            T[c] = T[c].fillna(fill[c])
        T.to_parquet(OUT / f"{XT}_fold{s}.parquet")
    F1S = S6 + E.V11_T3 + ["trCA"]; st = F1S + RAW; ns, k = len(st), len(E.CO)
    j01 = E.fit_fuser(XT, st, C=C_LIN, names=F1S + ["rfx_t3", "rfx_nf", "rfx_nv"])
    j01["fill"] = {f"rfx_{c[3:]}": v for c, v in fill.items()}
    D = E.safe04(XT); D = D[~np.asarray(E.mav_flagged(D), dtype=bool)].reset_index(drop=True)
    assert len(D) == j01["n_trials"], (len(D), j01["n_trials"])
    X = (E.design(D[st].values, D[E.CO].values) - np.array(j01["mean"])) / np.array(j01["scale"]); y = D.y.values.astype(int)
    lg = {"J01": X @ np.array(j01["coef"]) + j01["intercept"]}; nets = {}
    for kind in NETS:
        t0 = time.time(); nets[kind] = []
        lg[kind] = net_scores(kind, X, y, X, ns, k, keep=nets[kind])
        say(f"  {kind}: {len(nets[kind])} networks fitted on {len(X)} rows [{time.time() - t0:.0f}s]")
    lg["AVG"] = np.mean([lg[c] for c in ["J01"] + NETS], 0)
    chk = {c: dict(auc_in_sample=round(float(roc_auc_score(y, v)), 4), spearman_with_J01=round(float(spearmanr(v, lg["J01"])[0]), 4)) for c, v in lg.items()}
    say(f"== {TAG} export: {len(X)} trials, {X.shape[1]} design columns, streams {j01['streams']}, fill {j01['fill']}")
    say(f"  in-sample checks (wiring only, not a result): {chk}")
    ex = dict(j01=j01, nets=nets, ns=ns, k=k, d=int(X.shape[1]), kinds=NETS, seeds=list(SEEDS), epochs=EPOCHS, checks=chk,
              probe_x=X[:16].tolist(), probe_logit={c: v[:16].tolist() for c, v in lg.items()},
              note="FUSENET-01 export: AVG = mean logit of J01 (CA-FUSE C 0.1) + MLP + RES + GATE + SET, every SAFE-04 row (MAV-flagged out); RFX streams = oof rfdc0 (FLAG_195, c5rl seeds 1-3)")
    (OUT / f"{TAG}_summary.json").write_text(json.dumps({"export": ex}), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} export written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    (export if os.environ.get("FUSENET_MODE") == "export" else report)()
