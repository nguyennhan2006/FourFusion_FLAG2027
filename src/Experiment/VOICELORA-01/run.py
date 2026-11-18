r"""VOICELORA-01 step 2 (README: rule 15/10 step 2 + addendum 17/10 22:30 before any result): FLAG_67a's adapted voice `sam100fL`
in the CC head. FACELORA-05's run.py is the base (faces [farl16 0:512 | clipL 512:1280 | clipB3 1280:2048 | clipC5 2048:2816],
AUXP-01 V4 recipe, aux VoxCeleb2 + VoxBlink, 3 seeds); every voice carries [sam100p 0:256 | sam100f 256:512 | sam100fL 512:768]:
  C2   face B3, voice [p | f]          control of the B3 line, trained in THIS run on the same rows
  VL   face B3, voice [p | f | fL]     same recipe
  VLt  VL, voice_dropout .45           re-tuned to the wider voice input (CLAUDE.md §4.7; = ENC-SCALE-01's C4t)
  K5   face C5, voice [p | f]          control of the C5 line (CB 19.35 is the best system without CSLS)
  V5   face C5, voice [p | f | fL]
  V5t  V5, voice_dropout .45
  SELB / SELC  nested choice VL / VLt and V5 / V5t (SCALE-CC-01's nested_choice, vs v11)                     (`report`)
MAV voices: FLAG_67x arrays (voice_sam100fL_train.npy, ext_<v1|v2>_complete_voice_sam100fL.npy, voice_sam100fL_<list>.npy +
index_<list>.txt: FLAG_47b's layout). Auxiliary voices: FLAG_67a arrays (<vox|vox2b|vb>/voice_sam100fL.npy in the order of the
source's voice_meta.csv, NaN = not in the audio cache); auxiliary rows without sam100fL are left out in EVERY arm.
    FOLDS=1,2 python run.py ; python run.py merge ; python run.py report ; python run.py final ; python run.py zip
Env: VL2_TAG, TR_SEEDS (3), TR_MAX_STEPS (smoke), VL2_STANDIN=1 (smoke only: sam100fL = a copy of sam100f when FLAG_67a / FLAG_67x
outputs are not attached), VL2_FINAL_ARM (override of the arm the report picks), VL2_DIRS / VLX_DIRS (local dirs).
"""
import os, sys, json, zipfile, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
_s = importlib.util.spec_from_file_location("facelora05run", EXP / "FACELORA-05/run.py")
F5 = importlib.util.module_from_spec(_s); _s.loader.exec_module(F5)
FL = F5.FL
M, C1 = FL.M, FL.C1
M.TAG = C1.TAG = FL.TAG = F5.TAG = TAG = os.environ.get("VL2_TAG", "voicelora01")
l2n = FL.l2n
STANDIN = os.environ.get("VL2_STANDIN") == "1"
M.FACE = dict(M.FACE, FB3=np.r_[FL.COLS["farl16"], FL.COLS["clipB3"]], FC5=np.r_[FL.COLS["farl16"], FL.COLS["clipC5"]])
if "peL" in FL.COLS:                                                  # PEFACE-01 fast lane (CT_PE=1): FC5 = [farl16 | clipC5 | peL]
    M.FACE["FC5"] = np.r_[FL.COLS["farl16"], FL.COLS["clipC5"], FL.COLS["peL"]]
M.VOICE.update({"vl": np.r_[0:768]})
FACE_OF = {"C2": "FB3", "VL": "FB3", "VLt": "FB3", "K5": "FC5", "V5": "FC5", "V5t": "FC5"}
M.PAIRS.update({a: (f, "cat" if a in ("C2", "K5") else "vl") for a, f in FACE_OF.items()})
CFG = dict(M.t3r.SUB21A, **FL.V4CFG)
NS = M.N_SEEDS
M.ARMS = {a: dict(aux=("vox", "vb"), pairs=[a], seeds=NS, cfg=dict(CFG, **({"voice_dropout": .45} if a.endswith("t") else {})))
          for a in FACE_OF}
M.DECISION = FL.DECISION = F5.DECISION = DECISION = "V5"             # per-pair diagnostics only; the decisions are SELB / SELC
LINES = {"B": ("C2", ("VL", "VLt")), "C": ("K5", ("V5", "V5t"))}      # line -> (control, nested variants)
C5_ZIP = EXP / "FACELORA-05/submission_facelora05_dev.zip"            # CB 19.35: its English files go into the Bangla-only zip
_X = {}


def _find(name, env):
    c = [Path(p) for p in os.environ.get(env, "").split(os.pathsep) if p]
    if Path("/kaggle/input").exists():
        c += [f.parent for f in Path("/kaggle/input").rglob(name)]
    c = [d for d in c if (d / name).exists()]
    return c[0] if c else None


def vlx_dir():
    """FLAG_67x output (MAV rows + test lists), or None."""
    return _find("voice_sam100fL_train.npy", "VLX_DIRS")


def aux_dir():
    """FLAG_67a output (voicelora_manifest.json + <src>/voice_sam100fL.npy), or None."""
    d = _find("voicelora_manifest.json", "VL2_DIRS")
    return d if d is not None and (d / "vb/voice_sam100fL.npy").exists() else None


def _standin(V512, what):
    assert STANDIN, f"attach the {what} output: VL2_STANDIN is for smoke tests only"
    print(f"!! VOICELORA-01 SMOKE STAND-IN: {what} sam100fL = a copy of sam100f", flush=True)
    return l2n(np.asarray(V512[:, 256:512], np.float32))


def aux_extra(src, V512):
    """sam100fL (256) of one auxiliary source, rows = the source's voice_meta.csv; NaN rows = not in the audio cache."""
    if src not in _X:
        d = aux_dir()
        if d is None:
            _X[src] = _standin(V512, f"FLAG_67a ({src})")
        else:
            man = json.loads((d / "voicelora_manifest.json").read_text())
            assert not man.get("smoke") or STANDIN, "FLAG_67a output is a smoke run"
            _X[src] = l2n(np.load(d / src / "voice_sam100fL.npy").astype(np.float32))
        assert len(_X[src]) == len(V512), (src, len(_X[src]), len(V512))
        fin = np.isfinite(_X[src]).all(1)
        print(f"VOICELORA-01 {src}: {len(fin)} voice rows, with sam100fL {fin.mean():.4f}", flush=True)
    return _X[src]


def mav_extra(part, a):
    d = vlx_dir()
    if d is None:
        return _standin(a, f"FLAG_67x ({part})")
    x = (np.load(d / "voice_sam100fL_train.npy") if part == "v4" else
         np.concatenate([np.load(d / f"ext_{s}_voice_sam100fL.npy") for s in ("v1_complete", "v2_complete")]))
    assert len(x) == len(a), (part, len(x), len(a))
    x = l2n(x.astype(np.float32))
    assert np.isfinite(x).all(), f"MAV {part}: non-finite sam100fL"
    return x


def patch_voice():
    """T3ENC-01 t3('voice', v4 / ext / vox) -> 768 columns; FACELORA-03's vox_rows also drops voices without sam100fL."""
    T = FL.patch()
    if getattr(T, "_voicelora01", False):
        return T
    t3_0, vox_rows0 = T.t3, T.vox_rows

    def t3(kind, part):
        a = t3_0(kind, part)
        if kind != "voice":
            return a
        key = ("voice", part)
        if key not in _X:
            if part in ("v4", "ext"):
                ex = mav_extra(part, a)
            elif part == "vox":
                ex = aux_extra("vox", a)
            else:
                raise KeyError(part)
            _X[key] = np.concatenate([a, ex], 1)
        return _X[key]

    def vox_rows(F, V):
        F2, V2, S2 = vox_rows0(F, V)
        ok = np.isfinite(V2).all(1)
        print(f"VOICELORA-01 FLAG_37: {int((~ok).sum())} rows without sam100fL left out", flush=True)
        return F2[ok], V2[ok], S2[ok]

    T.t3, T.vox_rows, T._voicelora01 = t3, vox_rows, True
    return T


def new_rows_vl(kinds, log):
    """= FACELORA-03 new_rows4 (FLAG_53) with 768-column voices; voices without sam100fL left out."""
    VB2 = C1.vox2b_dir()
    fm = pd.read_csv(VB2 / "face_meta.csv", dtype=str)
    vm = pd.read_csv(VB2 / "voice_meta.csv", dtype={"spk": str, "video": str, "utt": str, "kind": str})
    Fz = np.concatenate([l2n(np.load(VB2 / f"face_{n}.npy").astype(np.float32)) for n in ("farl16", "clipL")], 1)
    F = FL.with_adapted(Fz, "vox2b")
    V = np.concatenate([l2n(np.load(VB2 / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)
    V = np.concatenate([V, aux_extra("vox2b", V)], 1)
    assert len(fm) == len(F) and len(vm) == len(V)
    fk = fm.kind.isin(kinds).values & np.isfinite(F).all(1)
    vk = vm.kind.isin(kinds).values & np.isfinite(V).all(1)
    fm, F, vm, V = fm[fk].reset_index(drop=True), F[fk], vm[vk].reset_index(drop=True), V[vk]
    keep = (set(fm.spk) & set(vm.spk)) - C1.drop53()
    face_of, fvid, vvid = fm.groupby("spk").indices, fm.video.values, vm.video.values
    Fi, Vi, S = [], [], []
    for s, idx in sorted(vm.groupby("spk").indices.items()):
        if s not in keep:
            continue
        for j, v in enumerate(sorted(idx)):
            cand = [f for f in face_of[s] if fvid[f] != vvid[v]] or list(face_of[s])
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vox2:" + s)
    log(f"VOICELORA-01 FLAG_53 rows {kinds}: {len(S)} rows, {len(set(S))} speakers")
    return F[Fi], V[Vi], np.array(S)


def vb_rows_vl(log):
    """= FACELORA-03 vb_rows4 (LID speaker filter, label-noise filter on sam100f, cross-video faces) with 768-column voices;
    voices without sam100fL left out."""
    import run as T
    VBM = M.vb_module()
    VB = VBM.VB
    fm = pd.read_csv(VB / "face_meta.csv", dtype={"video": str, "utt": str}); vm = pd.read_csv(VB / "voice_meta.csv", dtype={"video": str, "utt": str})
    Fz = np.concatenate([l2n(np.load(VB / f"face_{n}.npy").astype(np.float32)) for n in ("farl16", "clipL")], 1)
    F = FL.with_adapted(Fz, "vb")
    V = np.concatenate([l2n(np.load(VB / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)
    V = np.concatenate([V, aux_extra("vb", V)], 1)
    drop = set(json.loads((VB / "lid_summary.json").read_text())["speakers_dropped_ids"]) | T.strict_drop("vb")   # 3/11: LIDSTRICT-01, as vb_rows4
    xm = pd.read_csv(EXP / "VOX2-01/feats_vox2/voice_meta.csv", dtype={"video": str, "utt": str})
    XV = l2n(np.load(EXP / "T3ENC-01/feats_vox2t3/voice_sam100f.npy").astype(np.float32))
    keepx = ~xm.spk.isin(T.dropped_speakers()).values
    thr = float(np.nanquantile(VBM.loo_cos(XV[keepx], xm.spk.values[keepx]), 0.01))
    c = VBM.loo_cos(l2n(np.load(VB / "voice_sam100f.npy").astype(np.float32)), vm.spk.values)
    lid_ok = ~vm.spk.isin(drop).values
    ok = lid_ok & ~(lid_ok & (c < thr)) & np.isfinite(V).all(1)
    fok = np.isfinite(F).all(1)
    keep_spk = set(vm.spk[ok]) & set(fm.spk[~fm.spk.isin(drop) & fok])
    face_of = {s: g.index.values for s, g in fm[fok].groupby("spk")}
    fvid, vvid = fm.video.values, vm.video.values
    Fi, Vi, S = [], [], []
    vok = vm[ok]
    for s, idx in sorted(vok.groupby("spk").indices.items()):
        if s not in keep_spk:
            continue
        for j, v in enumerate(sorted(vok.index.values[idx])):
            cand = [f for f in face_of[s] if fvid[f] != vvid[v]] or list(face_of[s])
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vb:" + s)
    log(f"VOICELORA-01 VoxBlink rows: {len(S)} rows, {len(set(S))} speakers ({int((~np.isfinite(V).all(1)).sum())} voices without sam100fL)")
    return F[Fi], V[Vi], np.array(S)


FL.new_rows4, FL.vb_rows4 = new_rows_vl, vb_rows_vl                 # FACELORA-03's sources_fl3 looks them up at call time


def merge():
    """voicelora01_fold<s> (the six arms of this run) + FACELORA-03's trB3 (FLAG_86b, reference) when it lines up."""
    from lab_fuse import SP
    for s in range(1, 6):
        f = SP / "out/safe04" / f"{TAG}_fold{s}.parquet"
        d = pd.read_parquet(f)
        b = SP / "out/safe04" / f"facelora03_B3_fold{s}.parquet"
        if b.exists():
            r = pd.read_parquet(b)
            if len(r) == len(d) and (r.y.values == d.y.values).all() and (r.vi.values == d.vi.values).all():
                d["trB3"] = r["trB3"].values
            else:
                print(f"fold {s}: FACELORA-03 B3 table does not line up (reference skipped)", flush=True)
        d.to_parquet(f)
        print(f"fold {s}: {[c for c in d.columns if c.startswith('tr') and '_' not in c]}", flush=True)


def decision_file():
    from lab_fuse import SP
    return SP / "out/safe04" / f"{TAG}_decision.json"


def report():
    """Rules (README addendum 17/10 22:30): line B = SELB - C2, line C = SELC - K5, each >= +0.3 in the g cells, better at >= 4/5
    folds, held-out Urdu / Hindi not below the control by more than 0.2. Writes the final arm to <TAG>_decision.json."""
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP, table
    base = S6 + E.V11_T3
    S2, persons = E.stack02(), E.fold_persons()
    sel = {}

    def prep(D):
        for ln, (_, var) in LINES.items():
            ch, det = C1.nested_choice(D, S2, persons, var, base)
            D[f"trSEL{ln}"] = np.nan
            for s, a in ch.items():
                m = D.fold == s
                D.loc[m, f"trSEL{ln}"] = D.loc[m, f"tr{a}"]
            sel[ln] = dict(choice=ch, detail=det)
        sel["D"] = D
        return D
    cols = set(pd.read_parquet(SP / "out/safe04" / f"{TAG}_fold1.parquet").columns)
    arms = list(FACE_OF) + ["SELB", "SELC"] + (["B3"] if "trB3" in cols else [])
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    v = E.gate(TAG, {a: base + [f"tr{a}"] for a in arms}, "SELC", standalone=[f"tr{a}" for a in arms], out=out,
               ci_arms=["SELB", "SELC"], prep=prep)
    D = sel["D"]
    T_ = table(D, {a: (lambda g, a=a: g[f"tr{a}"].values) for a in arms})
    G = T_[T_.cell.isin(["en_ext g", "urdu g", "v4 g"])].groupby("fold")[arms].mean()
    txt = ["\n== standalone, mean of en_ext g + urdu g + v4 g per fold (lower = better)\n" + G.round(2).to_string()]
    fmt = lambda p: ", ".join(f"fold {int(s)} {g:+.2f}" for s, g in p.items()) + f" | mean {p.mean():+.2f}, {int((p > 0).sum())}/5 better"
    pg = lambda a, b: FL.paired_g(D, S2, persons, base + [f"tr{a}"], base + [f"tr{b}"])     # EER(b) - EER(a): + = a better
    res = {}
    for ln, (ctl, var) in LINES.items():
        for a in var:
            txt.append(f"CA-FUSE {a} - {ctl} (g cells): {fmt(pg(a, ctl))}")
        per = pg(f"SEL{ln}", ctl)
        hold = {k: v[f"SEL{ln}"][k]["all"] - v[ctl][k]["all"] for k in ("hold_urdu", "hold_hindi")}
        ok = bool(per.mean() >= .3 and (per > 0).sum() >= 4 and min(hold.values()) >= -0.2)
        ch = sel[ln]["choice"]
        major = max(var, key=lambda a: (sum(1 for x in ch.values() if x == a), a == var[0]))
        det = {int(k): {q: round(x, 3) for q, x in d.items()} for k, d in sel[ln]["detail"].items()}
        txt.append(f"SEL{ln} choice per fold {ch} (inner gains vs v11 {json.dumps(det)})")
        txt.append(f"RULE {ln} SEL{ln} - {ctl}: {fmt(per)} | held-out urdu {hold['hold_urdu']:+.2f}, hindi {hold['hold_hindi']:+.2f} "
                   f"| gate SEL{ln} vs v11 {'PASSES' if v[f'SEL{ln}']['passes'] else 'fails'} -> {'PASSES' if ok else 'no'}")
        res[ln] = dict(passes=ok, major=major, mean=round(float(per.mean()), 3), better=int((per > 0).sum()),
                       hold={k: round(float(x), 3) for k, x in hold.items()})
    txt.append(f"reference: CA-FUSE K5 - C2 (C5 vs B3 face, same rows): {fmt(pg('K5', 'C2'))}")
    if "B3" in arms:
        txt.append(f"reference: CA-FUSE C2 (this run, rows with sam100fL) - B3 (FLAG_86b): {fmt(pg('C2', 'B3'))}")
    if res["C"]["passes"]:
        arm, why = res["C"]["major"], "rule C passes"
    elif res["B"]["passes"]:
        arm, why = res["B"]["major"], "rule B passes (rule C fails)"
    else:
        arm, why = res["C"]["major"], "NO RULE PASSES: reference zip only, not a candidate"
    txt.append(f"DECISION: final arm {arm} ({why})")
    decision_file().write_text(json.dumps(dict(final_arm=arm, why=why, lines=res), indent=1))
    print("\n".join(txt))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    (HERE / "report.txt").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")


def final_arm():
    if os.environ.get("VL2_FINAL_ARM"):
        return os.environ["VL2_FINAL_ARM"]
    return json.loads(decision_file().read_text())["final_arm"]


def patch_test_voice():
    """e2e_lib.t3_split (test files: voice [sam100p | sam100f], 512) -> + sam100fL of the test lists (FLAG_67x arrays
    voice_sam100fL_<gender|no_gender>_<English|Bangla>.npy, one row per trial in list order; ids checked)."""
    import e2e_lib as E
    if getattr(E, "_voicelora01", False):
        return
    split0 = E.t3_split

    def t3_split(c):
        F, V, m = split0(c)
        name = c["key"].replace("/", "_")
        d = vlx_dir()
        if d is None:
            ex = _standin(V, f"FLAG_67x (test {c['key']})")
        else:
            idx = [l.split()[0] for l in (d / f"index_{name}.txt").read_text().splitlines() if l.strip()]
            assert len(idx) == len(V) and list(map(str, c["pid"])) == idx, f"{c['key']}: FLAG_67x rows do not match the trial list"
            ex = l2n(np.load(d / f"voice_sam100fL_{name}.npy").astype(np.float32))
        assert ex.shape == (len(V), 256) and np.isfinite(ex).all(), (c["key"], ex.shape)
        return F, np.concatenate([V, ex], 1), m

    E.t3_split, E._voicelora01 = t3_split, True


def final():
    """The arm the report picks (or VL2_FINAL_ARM) on every train person, 5 seeds, test streams of the split (T3RECIPE-01 final:
    level-0 self-test, models saved)."""
    FL.patch(); F5.patch_test(); patch_voice(); patch_test_voice()
    arm = final_arm()
    ns = int(os.environ.get("VL2_FINAL_SEEDS", "5"))
    M.N_SEEDS = max(M.N_SEEDS, ns)
    M.ARMS[arm] = dict(M.ARMS[arm], seeds=ns)
    M.DECISION = FL.DECISION = F5.DECISION = arm
    print(f"VOICELORA-01 final head: {arm}, {ns} seeds, aux {M.ARMS[arm]['aux']}", flush=True)
    M.final()


def bn_only(zp, en_src, dst):
    """Bangla files of zp + English files copied byte for byte from en_src (same member names)."""
    with zipfile.ZipFile(zp) as a, zipfile.ZipFile(en_src) as b, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        assert sorted(a.namelist()) == sorted(b.namelist()), (a.namelist(), b.namelist())
        for n in a.namelist():
            z.writestr(n, (b if "English" in n else a).read(n))


def zipit():
    """Dev zip of the final arm as FACELORA-03's zip (Bangla: AUXP-01's fuser with tr<arm>; English: CA_EN_CC with tr<arm>), then
    the zip to submit: Bangla of it + English byte for byte from the C5 zip (19.35); per-file correlation with the C5 zip."""
    arm = final_arm()
    M.DECISION = FL.DECISION = F5.DECISION = arm
    FL.zipit()
    import e2e_lib as E
    od = M.out_dir()
    zp = od / f"submission_{TAG}_{E.SPLIT}.zip"
    out = dict(arm=arm, decision=json.loads(decision_file().read_text()) if decision_file().exists() else None)
    if E.SPLIT == "dev" and C5_ZIP.exists():
        zb = od / f"submission_{TAG}_{arm}_bnonly_{E.SPLIT}.zip"
        bn_only(zp, C5_ZIP, zb)
        out.update(bnonly=zb.name, corr_vs_c5=FL.cmp_zip(zb, C5_ZIP), corr_full_vs_c5=FL.cmp_zip(zp, C5_ZIP))
    ck = od / f"{TAG}_zip_checks.json"
    j = json.loads(ck.read_text()) if ck.exists() else {}
    j.update(out); ck.write_text(json.dumps(j, indent=1, default=str))
    print(f"VOICELORA-01 zip: {json.dumps(out, default=str)}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "merge":
        merge()
    elif cmd == "report":
        report()
    elif cmd == "final":
        final()
    elif cmd == "zip":
        zipit()
    else:
        patch_voice()
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            M.run(s)
