"""RISKSIM-01 (README registered 12/11 ~19:00 before any number): which architecture is robust to new persons? Simulated
Evaluation-like files (17 face persons per cell, resampled from the SAFE-04 held-out persons) for every architecture with
validation tables. python lab_entry.py RISKSIM-01 report (Kaggle CPU, FLAG_179; same inputs as FLAG_178 / ENPROG-01).
Nothing reads dev / CB / Evaluation. Writes risksim_report.txt, risksim_summary.json."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P = _load("enprog01", EXPD / "ENPROG-01" / "run.py")               # merge(), ROUNDS, CTL, EXPERTS; P.M = RFSTUNE-01 (say / TXT, OUT)
M, E, S6, OUT, say, FAST = P.M, P.E, P.S6, P.OUT, P.say, P.FAST
TAG = "risksim"
NPER, B = 17, (20 if FAST else 500)
CELLS = [("en_ext ng", "oof"), ("en_ext g", "oof"), ("urdu ng", "hold_urdu"), ("urdu g", "hold_urdu")]
CELLS_V4 = [("v4 ng", "oof"), ("v4 g", "oof")]


def eer_fast(s, y):
    o = np.argsort(-s, kind="stable"); y = y[o]; npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    fnr = 1 - np.cumsum(y) / npos; fpr = np.cumsum(1 - y) / nneg
    i = int(np.argmin(np.abs(fnr - fpr))); return 100 * (fnr[i] + fpr[i]) / 2


def build():
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    arms = P.merge(CT)
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(P.TAG)
    ctl = P.CTL if P.CTL in arms else arms[0]
    cols = {x: [f"{x}_t3", f"{x}_nf", f"{x}_nv"] for x in arms}
    for x, cs in cols.items():
        for c in cs:
            D[c] = D[c].fillna(float(np.nanmean(D[c].values)))
    groups = {f"trRFS_{x}": cs for x, cs in cols.items()}
    ok = ~E.mav_flagged(D)
    for g, cs in groups.items():
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    ex = [f"rfe{a}" for a in P.EXPERTS if f"rfe{a}" in arms]
    fused = {"XALL": list(S6), "T3v11": list(E.V11_T3), "V11": S6 + E.V11_T3, "CA": ["trCA"], "F1s": S6 + E.V11_T3 + ["trCA"],
             **{f"R_{x}": [f"trRFS_{x}"] for x in arms}}
    if len(ex) >= 2:
        fused["R_mv"] = [f"trRFS_{x}" for x in ex]; fused["R_mvA"] = [f"trRFS_{ctl}"] + fused["R_mv"]
    say("\n== fused architectures (CA-FUSE C = 1, nested + purged, 3 regimes)")
    EDA.run_arms(D, S2, persons, fused)
    archs = {"XALL": "XALL", "T3v11": "T3v11", "V11": "V11", "CA": "CA", "F1s": "F1s", "RFS": f"R_{ctl}", "U'": None}
    EDA.add_sum(D, "U'", ["F1s", f"R_{ctl}"])
    for nm in ("mv", "mvA"):
        if f"R_{nm}" in fused:
            EDA.add_sum(D, f"U_{nm}", ["F1s", f"R_{nm}"]); archs[f"U_{nm}"] = None
    for x in arms:
        if x[3:] == "a0":
            continue
        EDA.add_sum(D, f"U_{x}", ["F1s", f"R_{x}"]); archs[f"U_{x}"] = None
    names = list(archs)
    col = {n: (archs[n] or n) for n in names}
    return D, names, col


def simulate(D, names, col, cells, rng):
    """-> array [B, n_arch] of simulated scores (mean EER over the cells' files) and per-cell arrays."""
    per = []
    for cell, reg in cells:
        g = D[D.cell == cell]
        if len(g) == 0:
            continue
        S = np.column_stack([g[f"{col[n]}|{reg}"].to_numpy(np.float64) for n in names]); y = g.y.to_numpy().astype(int)
        fp = g.fp.astype(str).to_numpy(); ps = np.unique(fp); idx = {p: np.nonzero(fp == p)[0] for p in ps}
        k = min(NPER, len(ps)); out = np.full((B, len(names)), np.nan)
        for b in range(B):
            r = np.concatenate([idx[p] for p in rng.choice(ps, k, replace=False)])
            for j in range(len(names)):
                s = S[r, j]; m = np.isfinite(s)
                out[b, j] = eer_fast(s[m], y[r][m])
        per.append((cell, len(ps), out))
    return per


def summarize(per, names, tagname):
    X = np.nanmean(np.stack([o for _, _, o in per]), 0)               # [B, n_arch]
    j0 = names.index("U'")
    rank = pd.DataFrame(X).rank(axis=1).to_numpy()
    rows = []
    for j, n in enumerate(names):
        d = X[:, j0] - X[:, j]                                         # > 0 = n better than U'
        rows.append(dict(arch=n, mean=np.nanmean(X[:, j]), sd=np.nanstd(X[:, j]), p10=np.nanpercentile(X[:, j], 10),
                         p90=np.nanpercentile(X[:, j], 90), d_vs_U=np.nanmean(d), p_win_U=np.nanmean(d > 0), mdd=1.28 * np.nanstd(d),
                         mean_rank=rank[:, j].mean(), p_rank1=(rank[:, j] == 1).mean()))
    T = pd.DataFrame(rows).set_index("arch")
    T["reading"] = np.where((T.p_win_U >= .8) & (T.d_vs_U >= .3), "robustly better than U'",
                            np.where((T.p_win_U <= .2) & (T.d_vs_U <= -.3), "robustly worse than U'", "not distinguishable from U'"))
    T.loc["U'", "reading"] = "reference"
    rng = np.random.default_rng(1); rh = []
    for _ in range(B):
        a, b = rng.integers(0, len(X), 2)
        if a != b:
            rh.append(pd.Series(X[a]).corr(pd.Series(X[b]), method="spearman"))
    say(f"\n== {tagname}: {len(per)} cells ({', '.join(f'{c} {n} persons' for c, n, _ in per)}), {NPER} persons per file, B {B}")
    say(T.sort_values("mean").round(3).to_string())
    stab = float(np.nanmean(rh))
    read = "rank fairly reliable" if stab >= .7 else "mostly person-sampling noise" if stab < .4 else "partly reliable"
    say(f"  rank stability (Spearman of the architecture ranking between two independent draws): {stab:.3f} -> {read}")
    return T, stab, read


def report():
    t00 = time.time()
    D, names, col = build()
    say(f"== {TAG}: {len(D)} trials, {len(names)} architectures: {names}")
    rng = np.random.default_rng(0); summ = {"architectures": names, "nper": NPER, "B": B}
    for tagname, cells in (("main (English + unheard Urdu)", CELLS), ("English cells only", CELLS[:2]), ("unheard cells only", CELLS[2:]),
                           ("v4 cells (secondary)", CELLS_V4)):
        per = simulate(D, names, col, cells, rng)
        if not per:
            continue
        T, stab, read = summarize(per, names, tagname)
        jU = names.index("U'"); csd = {c: float(np.nanstd(o[:, jU])) for c, _, o in per}
        summ[tagname] = dict(table=T.round(4).to_dict(orient="index"), rank_stability=round(stab, 4), reading=read, cell_sd=csd)
        say("  SD of the per-file EER of U' by cell: " + ", ".join(f"{c} {v:.2f}" for c, v in csd.items()))
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
