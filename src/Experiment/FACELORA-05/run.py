r"""FACELORA-05 step 2 (registered 16/10 in README.md before any result): the larger CLIP-L adapter C5 (`clipC5`, FLAG_65g: LoRA on
MLP + attention of blocks 12-23) in place of FACELORA-03's B3 (`clipB3`) in the CC head. FACELORA-03's run.py is the base (same rows, recipe AUXP-01 V4 with aux
VoxCeleb2 + VoxBlink, 3 seeds, same seeds); its adapted-array hooks are re-pointed to [clipB3 | clipC5] (as ENC-SCALE-02 does).
  C5  [farl16 | clipC5]  <- DECISION
  B3  [farl16 | clipB3]  control, B0 [farl16 | clipL] reference: FACELORA-03's per-arm tables (FLAG_86b / FLAG_86, same recipe, T4)
Faces: [farl16 0:512 | clipL 512:1280 | clipB3 1280:2048 | clipC5 2048:2816].
    FL5_TAG=facelora05_C5 FOLDS=1,2 python run.py ; python run.py merge ; python run.py report ; TR_SEEDS=5 python run.py final ; python run.py zip
Env: FL5_TAG (fold jobs write <FL5_TAG>_fold<s>), TR_SEEDS (3; the final head 5), TR_MAX_STEPS (smoke only), FL3_DIRS (local dirs
holding face_clipB3_train.npy / face_clipC5_train.npy; Kaggle: found under /kaggle/input).
"""
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
os.environ.setdefault("FL3_ARMS", "B0")                               # import FACELORA-03 without loading its adapted arrays
_s = importlib.util.spec_from_file_location("facelora03run", EXP / "FACELORA-03/run.py")
FL = importlib.util.module_from_spec(_s); _s.loader.exec_module(FL)
M, C1 = FL.M, FL.C1
M.TAG = C1.TAG = FL.TAG = TAG = os.environ.get("FL5_TAG", "facelora05")
FL.COLS = {"farl16": np.arange(0, 512), "clipL": np.arange(512, 1280), "clipB3": np.arange(1280, 2048), "clipC5": np.arange(2048, 2816)}
FL.ADAPTED, FL.LOAD = ("clipB3", "clipC5"), {"clipB3", "clipC5"}
FROZEN = {"clipB3": "clipL", "clipC5": "clipL"}                     # the frozen encoder each adapter started from (row checks)
DIMS = {"clipB3": 768, "clipC5": 768}
PE = os.environ.get("CT_PE") == "1"   # PEFACE-01 fast lane (10/11, README s8): + frozen PE-Core `peL` (1024, FLAG_129a) after clipC5;
if PE:                                # FC5 (VOICELORA-01) becomes [farl16 | clipC5 | peL]; unset = unchanged
    FL.COLS["peL"] = np.arange(2816, 3840); FL.ADAPTED = FL.ADAPTED + ("peL",); FL.LOAD = FL.LOAD | {"peL"}
    FROZEN["peL"], DIMS["peL"] = "clipL", 1024


def rsa(fz, a, n=400, seed=0):
    """Row-order check for an encoder of another width (peL vs clipL): Pearson r of the two pairwise cosine structures on n sampled
    finite rows (aligned rows ~0.4-0.7, shuffled rows ~0)."""
    ok = np.flatnonzero(np.isfinite(a).all(1) & np.isfinite(fz).all(1))
    r = np.random.RandomState(seed).choice(ok, min(n, len(ok)), replace=False)
    iu = np.triu_indices(len(r), 1)
    return float(np.corrcoef((a[r] @ a[r].T)[iu], (fz[r] @ fz[r].T)[iu])[0, 1])
ARMS = {"C5": ("farl16", "clipC5")}                                 # arm -> (FaRL columns, CLIP-L columns)
M.FACE = dict(M.FACE, **{a: np.r_[FL.COLS[f], FL.COLS[c]] for a, (f, c) in ARMS.items()})
M.PAIRS = dict(M.PAIRS, **{a: (a, "cat") for a in ARMS})
M.ARMS = {a: dict(aux=("vox", "vb"), pairs=[a], seeds=M.N_SEEDS, cfg=dict(M.t3r.SUB21A, **FL.V4CFG)) for a in ARMS}
M.DECISION = FL.DECISION = DECISION = "C5"
REF_ARMS = ("B0", "B3")                                             # FACELORA-03 per-arm tables merged in (`merge`)


def adapted(part):
    """[clipB3 | clipC5] of one part, per-encoder L2 (NaN rows stay NaN): v4 / ext (v1 then v2) / vox (FLAG_37) / vox2b / vb."""
    out = []
    for arr in FL.ADAPTED:
        d = FL.adir(arr)
        if part == "v4":
            a = np.load(d / f"face_{arr}_train.npy")
        elif part == "ext":
            a = np.concatenate([np.load(d / f"ext_{s}_face_{arr}.npy") for s in ("v1_complete", "v2_complete")])
        else:
            a = np.load(d / f"{part}_face_{arr}.npy")
        assert a.ndim == 2 and a.shape[1] == DIMS[arr], (arr, part, a.shape)
        out.append(FL.l2n(a.astype(np.float32)))
    assert len({len(x) for x in out}) == 1, (part, [len(x) for x in out])
    return np.concatenate(out, 1)


def with_adapted(Fz, part):
    """[farl16 | clipL] (1280) -> + [clipB3 | clipC5] (2816). Row checks: each adapter against its own frozen encoder (alignment
    >= 0.2, as FACELORA-03); both adapters must miss the SAME faces, else FLAG_86b's B3 tables are not this run's control."""
    Fa = adapted(part)
    assert len(Fz) == len(Fa), (part, len(Fz), len(Fa))
    al, fin, o = {}, {}, 0
    for arr in FL.ADAPTED:
        blk = Fa[:, o:o + DIMS[arr]]; o += DIMS[arr]
        al[arr] = (rsa if DIMS[arr] != 768 else FL.alignment)(Fz[:, FL.COLS[FROZEN[arr]]], blk)
        fin[arr] = np.isfinite(blk).all(1)
    diff = int(sum((fin["clipB3"] != fin[x]).sum() for x in FL.ADAPTED if x != "clipB3"))
    print(f"FACELORA-05 {part}: {len(Fa)} rows, finite clipB3 {fin['clipB3'].mean():.4f} clipC5 {fin['clipC5'].mean():.4f} "
          f"(rows that differ {diff}), alignment {al}", flush=True)
    assert min(al.values()) >= 0.2, f"adapted rows of {part} do not line up with the frozen rows: {al}"
    assert diff == 0, f"{part}: clipB3 and clipC5 miss different faces ({diff} rows): rerun B3 in this run as the control"
    return np.concatenate([Fz, Fa], 1)


FL.adapted, FL.with_adapted = adapted, with_adapted                 # FACELORA-03's patch / new_rows4 / vb_rows4 call these


def patch_test():
    """e2e_lib.t3_split (test files: [farl16 | clipL], 1280) -> + [clipB3 | clipC5] from the adapter arrays of the test lists
    (face_<arr>_<gender|no_gender>_<English|Bangla>.npy, one row per trial in list order), FACELORA-03's two row-order checks
    against each adapter's own frozen encoder."""
    import e2e_lib as E
    if getattr(E, "_facelora05", False):
        return
    split0 = E.t3_split

    def t3_split(c):
        F, V, m = split0(c)
        name = c["key"].replace("/", "_")
        parts = []
        for arr in FL.ADAPTED:
            a = FL.l2n(np.load(FL.adir(arr) / f"face_{arr}_{name}.npy").astype(np.float32))
            assert a.shape == (len(F), DIMS[arr]) and np.isfinite(a).all(), (c["key"], arr, a.shape, len(F))
            if DIMS[arr] != 768:          # PEFACE-01 (peL): structure check + duplicate faces (retrieval needs the same width)
                fz = F[:, FL.COLS[FROZEN[arr]]]; al = dict(rsa=round(rsa(fz, a), 3)); _, inv = np.unique(np.round(fz, 4), axis=0, return_inverse=True)
                first, same = {}, []
                for i, g in enumerate(inv.ravel()):
                    (same.append(float(a[i] @ a[first[g]]) >= .999) if g in first else first.__setitem__(g, i))
                al.update(dup_same=round(float(np.mean(same)), 4) if same else 1.0, retrieval=1.0 if al["rsa"] >= 0.2 else 0.0)
            else:
                al = FL.test_alignment(F[:, FL.COLS[FROZEN[arr]]], a)
            print(f"FACELORA-05 test {c['key']}: {arr} {len(a)} rows, {al}", flush=True)
            assert al["retrieval"] >= 0.2 and al["dup_same"] >= 0.99, (c["key"], arr, al)
            parts.append(a)
        return np.concatenate([F] + parts, 1), V, m

    E.t3_split, E._facelora05 = t3_split, True


def merge():
    """facelora03_B3 / facelora03_B0 (FLAG_86b / FLAG_86) + facelora05_C5 (this run) -> facelora05_fold<s> (trB3, trB0, trC5)."""
    from lab_fuse import SP
    for s in range(1, 6):
        parts = [SP / "out/safe04" / f"facelora03_{a}_fold{s}.parquet" for a in ("B3", "B0")] + [SP / "out/safe04" / f"facelora05_C5_fold{s}.parquet"]
        missing = [p.name for p in parts if not p.exists()]
        assert not missing, f"fold {s}: missing {missing}"
        base = None
        for f in parts:
            d = pd.read_parquet(f)
            if base is None:
                base = d
            else:
                assert len(d) == len(base) and (d.y.values == base.y.values).all() and (d.vi.values == base.vi.values).all(), f
                for c in [c for c in d.columns if c.startswith("tr") and c not in base]:
                    base[c] = d[c].values
        base.to_parquet(SP / "out/safe04" / f"facelora05_fold{s}.parquet")
        print(f"fold {s}: {[f.name for f in parts]} -> {[c for c in base.columns if c.startswith('tr')]}")


def report():
    """Rule 1 (standalone C5 - B3, mean of en_ext g + urdu g + v4 g per fold), rule 2 (default gate C5 vs v11), rule 3 (DECISION:
    CA-FUSE(v11 + trC5) - CA-FUSE(v11 + trB3) in the g cells >= +0.3 at >= 4/5 folds AND C5's held-out Urdu / Hindi gains not
    below B3's by more than 0.2)."""
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP, table
    base = S6 + E.V11_T3
    arms = list(REF_ARMS) + list(ARMS)
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    v = E.gate(TAG, {a: base + [f"tr{a}"] for a in arms}, DECISION, standalone=[f"tr{a}" for a in arms], out=out, ci_arms=[DECISION])
    D = E.safe04(TAG)
    S2, persons = E.stack02(), E.fold_persons()
    T_ = table(D, {a: (lambda g, a=a: g[f"tr{a}"].values) for a in arms})
    G = T_[T_.cell.isin(["en_ext g", "urdu g", "v4 g"])].groupby("fold")[arms].mean()
    txt = ["\n== RULE 1: standalone, mean of en_ext g + urdu g + v4 g per fold (lower = better)\n" + G.round(2).to_string()]
    d1 = G["C5"] - G["B3"]
    r1 = "YES" if (d1.mean() <= -0.3 and (d1 < 0).sum() >= 4) else ("HURTS" if d1.mean() >= 0.3 else "NO DIFFERENCE")
    txt.append(f"RULE 1 (C5 - B3 standalone): {d1.mean():+.2f} ({int((d1 < 0).sum())}/5 better) -> {r1} | C5 - B0 {(G['C5'] - G['B0']).mean():+.2f}")
    txt.append(f"RULE 2 (gate C5 vs v11): {'PASSES' if v['C5']['passes'] else 'fails'} (B3 {'PASSES' if v['B3']['passes'] else 'fails'})")
    per = FL.paired_g(D, S2, persons, base + ["trC5"], base + ["trB3"])          # EER(B3) - EER(C5): + = C5 better
    hold = {k: v["C5"][k]["all"] - v["B3"][k]["all"] for k in ("hold_urdu", "hold_hindi")}
    ok = bool(per.mean() >= 0.3 and (per > 0).sum() >= 4 and min(hold.values()) >= -0.2)
    txt.append("RULE 3 (CA-FUSE C5 - B3, g cells): " + ", ".join(f"fold {int(s)} {g:+.2f}" for s, g in per.items())
               + f" | mean {per.mean():+.2f}, {int((per > 0).sum())}/5 better | held-out C5 - B3: urdu {hold['hold_urdu']:+.2f}, "
               + f"hindi {hold['hold_hindi']:+.2f} -> {'YES: candidate, one CB run with the C5 zip' if ok else 'no'}")
    print("\n".join(txt))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    (HERE / "report.txt").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")


def zipit():
    """FACELORA-03's zip with C5 (Bangla: AUXP-01's fuser with trC5 in place of trV4; English: CA_EN_CC with trC5); also the
    per-file correlation with the B3 zip (CB 19.77)."""
    FL.zipit()
    od = M.out_dir()
    import e2e_lib as E
    zp = od / f"submission_{TAG}_{E.SPLIT}.zip"
    ref = FL.cmp_zip(zp, EXP / "FACELORA-03/submission_facelora03_dev.zip") if E.SPLIT == "dev" else None
    ck = od / f"{TAG}_zip_checks.json"
    j = json.loads(ck.read_text()); j["corr_vs_facelora03_b3"] = ref; ck.write_text(json.dumps(j, indent=1, default=str))
    print(f"correlation per file with the FACELORA-03 B3 zip (CB 19.77): {ref}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "merge":
        merge()
    elif cmd == "report":
        report()
    elif cmd == "final":
        FL.patch(); patch_test()
        print(f"FACELORA-04 final head: {DECISION}, {M.N_SEEDS} seeds, aux {M.ARMS[DECISION]['aux']}", flush=True)
        M.final()
    elif cmd == "zip":
        zipit()
    else:
        FL.patch()
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            M.run(s)
