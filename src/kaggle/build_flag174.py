r"""FLAG_174 (11/11): RFSTUNE-01 report (Experiment/RFSTUNE-01/README.md, registered before any number) on Kaggle CPU.
Same lab cells as FLAG_171 (FLAG_167's CPU report: ENV, embedded code, SAFE-04 data links, the 20 T3S job tables), plus the
rft<arm>lab_fold<s> tables of the two FLAG_173 runs (attached as kernel outputs), then `lab_entry.py RFSTUNE-01 report`.
    python kaggle/build_flag174.py   -> FLAG_174_rfstune_report.ipynb
Push (CPU, account of the FLAG_173 runs): datasets of sat_push 171 minus the EDA tables, --kernels <173 part 1>,<173 part 2>,
--env 'LAB_FOLDS=;LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=1'."""
import base64, gzip, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_flag167 as B167
from build_flag173 import ARMS, ARMS2, ARMS3, ARMS4, ARMS5, ARMS6, ARMS7, ARMS9

EXP = HERE.parent / "Experiment"

EXTRA = r'''
# RFSTUNE-01 tables: rft<arm>lab_fold<s> of the FLAG_173 runs -> <SP>/out/safe04
import shutil
ARMS = __ARMS__
got = {}
for f in sorted(IN.rglob("__GLOB__")):
    if not (SP / "out/safe04" / f.name).exists():
        shutil.copy(f, SP / "out/safe04" / f.name); got[f.name] = str(f.parent)
need = [f"__PFX__{a}lab_fold{s}.parquet" for a in ARMS for s in range(1, 6)]
miss = [n for n in need if not (SP / "out/safe04" / n).exists()]
if SMOKE and miss:                                                 # smoke only: noisy copies of FACELORA-03's B3 tables (wiring), 3 arms
    import numpy as np
    for n in [n for n in miss if n.startswith(tuple(f"__PFX__{a}lab" for a in ARMS[:__NSM__]))]:
        a, s = n[3:].split("lab_fold")[0], int(n[-9])
        d = pd.read_parquet(SP / "out/safe04" / f"facelora03_B3_fold{s}.parquet"); rng = np.random.RandomState(s + len(a))
        o = d[["fold", "data", "protocol", "y", "fi", "vi"]].copy()
        for c in ("t3", "nf", "nv", "ib", "age", "gender_agree"):
            o[f"__PFX__{a}_{c}"] = d["trB3"].values + 0.3 * rng.randn(len(d))
            if c in ("ib", "age"): o.loc[o.data != "v4", f"__PFX__{a}_{c}"] = np.nan
        o.to_parquet(SP / "out/safe04" / n); log(f"SMOKE: {n} = B3 stand-in")
    miss = []
log(f"RFSTUNE tables copied {len(got)}; missing {miss}")
assert not miss, "attach the outputs of the two FLAG_173 runs (rft<arm>lab_fold<s>.parquet)"
'''

RUN = r'''
# RFSTUNE-01 analysis (README sections 5-6) -> rfstune_report.txt / rfstune_summary.json
import base64, gzip, shutil, subprocess, sys
for d_, blob in (("EDA-3FAM-01", "__EDA__"), ("RFSTUNE-01", "__RUN__"), ("RFSTUNE-02", "__RUN2__"), ("ENPROG-01", "__RUN3__"), ("RISKSIM-01", "__RUN4__"), ("HARDNEG-RFS", "__RUN5__"), ("IBLORA-01", "__RUN6__"), ("FACTOR-01", "__RUN7__"), ("FACTOR-02", "__RUN8__"), ("TRANSL-01", "__RUN9__"), ("FVBIND-01", "__RUN10__"), ("FUSEDIM-01", "__RUN11__"), ("PEFACE-01", "__RUN12__"), ("FUSEDIM-02", "__RUN13__"), ("RFXARCH-01", "__RUN14__"), ("RFXFUSE-01", "__RUN15__"), ("FUSEARCH-01", "__RUN16__"), ("FUSEARCH-02", "__RUN17__"), ("FUSEARCH-03", "__RUN18__"), ("LANGLANE-01", "__RUN19__"), ("FUSENET-01", "__RUN20__"), ("RFXSTD-01", "__RUN21__")):
    p = LAB / "Experiment" / d_ / "run.py"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.decompress(base64.b64decode(blob)))
r = subprocess.Popen([sys.executable, "-u", str(LAB / "lab_entry.py"), "__EXP__", "report"], cwd=str(WORK), env=ENVV,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
for line in r.stdout:
    if "Warning" not in line: print(line, end="", flush=True)
assert r.wait() == 0, "RFSTUNE-01 report failed"
for n in ("__TAG___report.txt", "__TAG___summary.json"):
    shutil.copy(SP / "out/safe04" / n, OUTD / n)
print("__EXP__ outputs:", sorted(p.name for p in OUTD.glob("__TAG___*")), flush=True)
'''

EXTRA_G1 = r'''
# IBLORA-01 bac 2: the strict ct01g1 fold tables (trB0c = F0s / F1s' English head) -> <SP>/out/safe04; IBLORA-01s arrays stay in /kaggle/input
import shutil
for f in sorted(IN.rglob("ct01g1_fold[1-5].parquet")):
    if not (SP / "out/safe04" / f.name).exists():
        shutil.copy(f, SP / "out/safe04" / f.name); log(f"gathered {f.name} from {f.parent}")
have = sorted(p.name for p in (SP / "out/safe04").glob("ct01g1_fold[1-5].parquet"))
assert len(have) == 5, f"attach ngnnguynchn/flag2027-ct02s-g1-tables-p (ct01g1 tables): have {have}"
ibl = sorted(IN.rglob("face_ibLv_train.npy")); assert ibl, "attach phantrnanna/flag2027-feats-iblora1s-p"
ENVV["IBL_DIR"] = os.environ["IBL_DIR"] = str(ibl[0].parent); log(f"IBLORA-01s arrays at {ibl[0].parent}")
'''

INTRO = """
# FLAG 2027 - NB-174: RFSTUNE-01 report, nested choice of the RFS projector recipe on SAFE-04 (CPU)

**What (Experiment/RFSTUNE-01/README.md, registered 11/11 before any number).** U_<arm> = F1s + CA-FUSE(RFS streams of the arm) for the
15 registered arms of FLAG_173; nested arm choice SEL vs the control a0; rule: SEL - a0 on g cells >= +0.3, >= 4/5 folds, held-out
Urdu / Hindi >= -0.2. Validation only (SAFE-04, purged); no dev, no CB, no Evaluation, LID_STRICT rows, NOREL. Internet OFF.
"""


def b64(p):
    return base64.b64encode(gzip.compress(Path(p).read_bytes(), 9)).decode()


def cells(arms=ARMS, pfx="rft", exp="RFSTUNE-01", tag="rfstune", glob=None, nsm=3, extra=None):
    run = (RUN.replace("__EDA__", b64(EXP / "EDA-3FAM-01" / "run.py")).replace("__RUN__", b64(EXP / "RFSTUNE-01" / "run.py"))
           .replace("__RUN2__", b64(EXP / "RFSTUNE-02" / "run.py")).replace("__RUN3__", b64(EXP / "ENPROG-01" / "run.py")).replace("__RUN4__", b64(EXP / "RISKSIM-01" / "run.py")).replace("__RUN5__", b64(EXP / "HARDNEG-RFS" / "run.py")).replace("__RUN6__", b64(EXP / "IBLORA-01" / "run.py")).replace("__RUN7__", b64(EXP / "FACTOR-01" / "run.py")).replace("__RUN8__", b64(EXP / "FACTOR-02" / "run.py")).replace("__RUN9__", b64(EXP / "TRANSL-01" / "run.py")).replace("__RUN10__", b64(EXP / "FVBIND-01" / "run.py")).replace("__RUN11__", b64(EXP / "FUSEDIM-01" / "run.py")).replace("__RUN12__", b64(EXP / "PEFACE-01" / "run.py")).replace("__RUN13__", b64(EXP / "FUSEDIM-02" / "run.py")).replace("__RUN14__", b64(EXP / "RFXARCH-01" / "run.py")).replace("__RUN15__", b64(EXP / "RFXFUSE-01" / "run.py")).replace("__RUN16__", b64(EXP / "FUSEARCH-01" / "run.py")).replace("__RUN17__", b64(EXP / "FUSEARCH-02" / "run.py")).replace("__RUN18__", b64(EXP / "FUSEARCH-03" / "run.py")).replace("__RUN19__", b64(EXP / "LANGLANE-01" / "run.py")).replace("__RUN20__", b64(EXP / "FUSENET-01" / "run.py")).replace("__RUN21__", b64(EXP / "RFXSTD-01" / "run.py")).replace("__EXP__", exp).replace("__TAG__", tag))
    c = B167.fold_cells
    env, setup, resume, gather, standin = c[1], c[3], c[5], c[6], c[9]
    assert env[0] == setup[0] == resume[0] == gather[0] == standin[0] == "code" and "T3S_PARTS" in env[1] and "rglob" in gather[1] \
        and "def standin" in standin[1], "FLAG_167 cell layout changed"
    out = [("markdown", INTRO), env, ("markdown", "## 1. Code (embedded) + data links + SAFE-04 reference tables"), setup,
           ("markdown", "## 2. Resume + the 20 T3S job tables + the RFSTUNE tables"), resume, gather, standin,
           ("code", extra or EXTRA.replace("__GLOB__", glob or f"{pfx}*lab_fold[1-5].parquet").replace("__NSM__", str(nsm))
                     .replace("__ARMS__", repr(list(arms))).replace("__PFX__", pfx)),
           ("markdown", "## 3. RFSTUNE-01 report"), ("code", run),
           ("markdown", "## 4. Summary"), c[-1]]
    names = iter(["env", "setup: code + data links", "resume", "gather T3S job tables", "smoke stand-ins (SMOKE only)", "RFSTUNE tables",
                  "RFSTUNE-01 report", "summary"])
    return [(k, f'print("[cell] {next(names)}", flush=True)\n' + s if k == "code" else s) for k, s in out]


PE_EXTRA = r'''
# PEFACE-01 stage 1a: PE-Core CA tables (ngnnguynchn/flag2027-pefast-r2ca-tables-p) embedded as ctpe01r2_fold<s>
import base64 as _b, gzip as _g
PE_TABLES = __PE__
if not SMOKE:
    import numpy as _np
    for _s, (_tr, _y, _sha) in PE_TABLES.items():
        _np.savez(SP / "out/safe04" / f"ctpe01r2_fold{_s}.npz", trCA=_np.frombuffer(_g.decompress(_b.b64decode(_tr)), _np.float32),
                  y=_np.frombuffer(_g.decompress(_b.b64decode(_y)), _np.int8), sha=_np.array(_sha))
    print("PE-Core CA tables written:", sorted(PE_TABLES), flush=True)
'''


def _pe_tables():
    """compact embed of the 5 PE-Core CA tables: fold -> (trCA float32, y int8, sha1 of (fi, vi) int64 bytes)."""
    import base64 as _b, gzip as _g, hashlib
    import numpy as np, pandas as pd
    out = {}
    for s in range(1, 6):
        P = pd.read_parquet(EXP / "PEFACE-01" / "out" / "pefast_tables" / "tables" / f"ct01r2_fold{s}.parquet")
        enc = lambda a: _b.b64encode(_g.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()
        sha = hashlib.sha1(np.asarray(P.fi, np.int64).tobytes() + np.asarray(P.vi, np.int64).tobytes()).hexdigest()
        out[s] = (enc(P.trCA.to_numpy(np.float32)), enc(P.y.to_numpy().astype(np.int8)), sha)
    return out

if __name__ == "__main__":
    if sys.argv[1:] == ["22"]:                                      # FUSENET-01 export (FUSENET_MODE=export; rfdc0 of FLAG_195, as FUSEARCH-03)
        B167.BL.B12.nb(cells(["c0"], "rfd", "FUSENET-01", "fusenet", nsm=1), "FLAG_222_fusenet_export.ipynb")
        print("FLAG_222_fusenet_export.ipynb built")
    elif sys.argv[1:] == ["21"]:                                    # RFXSTD-01 (rfr c0r / c0s of FLAG_213 + rfk* of the two FLAG_220 runs, all required)
        arms21 = ["rfrc0r", "rfrc0s", "rfkk0r"] + [f"rfk{k}{x}" for k in ("cos", "ema", "ps", "wd", "dm", "all") for x in "rs"]
        ex = (EXTRA.replace("__GLOB__", "rf[rk]*lab_fold[1-5].parquet").replace("__NSM__", str(len(arms21))).replace("__ARMS__", repr(arms21))
              .replace("__PFX__", "").replace('a, s = n[3:].split("lab_fold")[0], int(n[-9])', 'a, s = n.split("lab_fold")[0], int(n[-9])'))
        assert 'a, s = n.split("lab_fold")[0], int(n[-9])' in ex
        B167.BL.B12.nb(cells(arms21, "", "RFXSTD-01", "rfxstd", extra=ex), "FLAG_221_rfxstd_report.ipynb")
        print("FLAG_221_rfxstd_report.ipynb built")
    elif sys.argv[1:] == ["20"]:                                    # FUSENET-01 (rfr c0r / c0s of FLAG_213 + rfr c0t of FLAG_216, all required)
        arms20 = ["rfrc0r", "rfrc0s", "rfrc0t"]
        ex = (EXTRA.replace("__GLOB__", "rfr*lab_fold[1-5].parquet").replace("__NSM__", str(len(arms20))).replace("__ARMS__", repr(arms20))
              .replace("__PFX__", "").replace('a, s = n[3:].split("lab_fold")[0], int(n[-9])', 'a, s = n.split("lab_fold")[0], int(n[-9])'))
        B167.BL.B12.nb(cells(arms20, "", "FUSENET-01", "fusenet", extra=ex), "FLAG_219_fusenet_report.ipynb")
        print("FLAG_219_fusenet_report.ipynb built")
    elif sys.argv[1:] == ["19"]:                                    # LANGLANE-01 part B (rfr c0r / c0s of FLAG_213 + rfr c0t of FLAG_216)
        arms19 = ["rfrc0r", "rfrc0s", "rfrc0t"]
        ex = (EXTRA.replace("__GLOB__", "rfr*lab_fold[1-5].parquet").replace("__NSM__", str(len(arms19))).replace("__ARMS__", repr(arms19))
              .replace("__PFX__", "").replace('a, s = n[3:].split("lab_fold")[0], int(n[-9])', 'a, s = n.split("lab_fold")[0], int(n[-9])')
              .replace('assert not miss,', 'assert not [n for n in miss if not n.startswith("rfrc0t")],'))   # c0t missing -> UNDECIDED
        B167.BL.B12.nb(cells(arms19, "", "LANGLANE-01", "langlane", extra=ex), "FLAG_217_langlane_report.ipynb")
        print("FLAG_217_langlane_report.ipynb built")
    elif sys.argv[1:] == ["18"]:                                    # FUSEARCH-03 (J01 export; rfdc0 of FLAG_195)
        B167.BL.B12.nb(cells(["c0"], "rfd", "FUSEARCH-03", "fusearch3", nsm=1), "FLAG_215_fusearch3_export.ipynb")
        print("FLAG_215_fusearch3_export.ipynb built")
    elif sys.argv[1:] == ["17"]:                                    # FUSEARCH-02 (rfdc0 of FLAG_195 + rfr* of FLAG_213)
        arms17 = ["rfdc0", "rfrc0r", "rfrc0s"]
        ex = (EXTRA.replace("__GLOB__", "rf[dr]*lab_fold[1-5].parquet").replace("__NSM__", str(len(arms17))).replace("__ARMS__", repr(arms17))
              .replace("__PFX__", "").replace('a, s = n[3:].split("lab_fold")[0], int(n[-9])', 'a, s = n.split("lab_fold")[0], int(n[-9])')
              .replace('assert not miss,', 'assert not [n for n in miss if n.startswith("rfr")],'))   # rfdc0 optional (reported only)
        B167.BL.B12.nb(cells(arms17, "", "FUSEARCH-02", "fusearch2", extra=ex), "FLAG_214_fusearch2_report.ipynb")
        print("FLAG_214_fusearch2_report.ipynb built")
    elif sys.argv[1:] == ["16"]:                                    # FUSEARCH-01 (rfd* of FLAG_195 + rfm* of FLAG_198 + rfl* of FLAG_207)
        arms16 = [f"rfd{a}" for a in ("c0", "d64", "d128", "d256", "d1024", "d2048")] + [f"rfm{a}" for a in ("m512", "m256", "m128")] + ["rflL1", "rflL2"]
        ex = (EXTRA.replace("__GLOB__", "rf[dml]*lab_fold[1-5].parquet").replace("__NSM__", str(len(arms16))).replace("__ARMS__", repr(arms16))
              .replace("__PFX__", "").replace('a, s = n[3:].split("lab_fold")[0], int(n[-9])', 'a, s = n.split("lab_fold")[0], int(n[-9])'))
        B167.BL.B12.nb(cells(arms16, "", "FUSEARCH-01", "fusearch", extra=ex), "FLAG_212_fusearch_report.ipynb")
        print("FLAG_212_fusearch_report.ipynb built")
    elif sys.argv[1:] == ["14"]:                                    # RFXARCH-01 (FLAG_205 rfa*)
        B167.BL.B12.nb(cells(["c0", "res", "id05", "id10", "resid05"], "rfa", "RFXARCH-01", "rfxarch", nsm=5), "FLAG_206_rfxarch_report.ipynb")
        print("FLAG_206_rfxarch_report.ipynb built")
    elif sys.argv[1:] == ["15"]:                                    # RFXFUSE-01 + LORAFIX-01 (rfdc0 of FLAG_195 required; rflL1 / rflL2 of FLAG_207 optional)
        arms15 = ["rfdc0", "rflL1", "rflL2"]
        ex = (EXTRA.replace("__GLOB__", "rf[dl]*lab_fold[1-5].parquet").replace("__NSM__", str(len(arms15))).replace("__ARMS__", repr(arms15))
              .replace("__PFX__", "").replace('a, s = n[3:].split("lab_fold")[0], int(n[-9])', 'a, s = n.split("lab_fold")[0], int(n[-9])')
              .replace('assert not miss,', 'assert not [n for n in miss if n.startswith("rfdc0")],'))
        B167.BL.B12.nb(cells(arms15, "", "RFXFUSE-01", "rfxfuse", extra=ex), "FLAG_208_rfxfuse_report.ipynb")
        print("FLAG_208_rfxfuse_report.ipynb built")
    elif sys.argv[1:] == ["13"]:                                    # FUSEDIM-02 (FLAG_195 rfd c0 / d128 / d256 + FLAG_198 rfm*)
        arms13 = ["rfdc0", "rfdd128", "rfdd256", "rfmm512", "rfmm256", "rfmm128"]
        ex = (EXTRA.replace("__GLOB__", "rf[dm]*lab_fold[1-5].parquet").replace("__NSM__", str(len(arms13))).replace("__ARMS__", repr(arms13))
              .replace("__PFX__", "").replace('a, s = n[3:].split("lab_fold")[0], int(n[-9])', 'a, s = n.split("lab_fold")[0], int(n[-9])'))
        B167.BL.B12.nb(cells(arms13, "", "FUSEDIM-02", "fusedim2", extra=ex), "FLAG_199_fusedim2_report.ipynb")
        print("FLAG_199_fusedim2_report.ipynb built")
    elif sys.argv[1:] == ["12"]:                                    # PEFACE-01 stage 1a (rfbc0 of FLAG_193 + embedded PE-Core CA tables)
        pe = _pe_tables()
        ex = (EXTRA.replace("__GLOB__", "rfb*lab_fold[1-5].parquet").replace("__NSM__", "1").replace("__ARMS__", repr(["c0"]))
              .replace("__PFX__", "rfb")
              + PE_EXTRA.replace("__PE__", repr(pe)))
        B167.BL.B12.nb(cells(["c0"], "rfb", "PEFACE-01", "pe1a", extra=ex), "FLAG_197_pe1a_report.ipynb")
        print("FLAG_197_pe1a_report.ipynb built")
    elif sys.argv[1:] == ["11"]:                                      # FUSEDIM-01 (FLAG_195 rfd c0 + 5 widths; fusions in run.py)
        B167.BL.B12.nb(cells(list(ARMS9), "rfd", "FUSEDIM-01", "fusedim", nsm=len(ARMS9)), "FLAG_196_fusedim_report.ipynb")
        print("FLAG_196_fusedim_report.ipynb built")
    elif sys.argv[1:] == ["10"]:                                    # FVBIND-01 gate 2 (FLAG_193 rfb c0 / fvb)
        B167.BL.B12.nb(cells(["c0", "fvb"], "rfb", "FVBIND-01", "fvbg2", nsm=2), "FLAG_194_fvbg2_report.ipynb")
        print("FLAG_194_fvbg2_report.ipynb built")
    elif sys.argv[1:] == ["9"]:                                       # TRANSL-01 (FLAG_183 rfxc5rl + FLAG_187 rfz*)
        arms9 = [f"rfz{a}" for a in ARMS7]                           # required; rfxc5rl (FLAG_183) is copied by the glob when attached
        ex = (EXTRA.replace("__GLOB__", "rf[xz]*lab_fold[1-5].parquet").replace("__NSM__", str(len(arms9))).replace("__ARMS__", repr(arms9))
              .replace("__PFX__", "").replace('a, s = n[3:].split("lab_fold")[0], int(n[-9])', 'a, s = n.split("lab_fold")[0], int(n[-9])'))
        B167.BL.B12.nb(cells(arms9, "", "TRANSL-01", "transl", extra=ex), "FLAG_188_transl_report.ipynb")
        print("FLAG_188_transl_report.ipynb built")
    elif sys.argv[1:] == ["8"]:                                       # FACTOR-02 (FLAG_183 rfx* + FLAG_185 rfy*; FACTOR2_PHASE=1 -> rfx only)
        arms8 = [f"rfx{a}" for a in ("c5rl", "ALLlo", "F1lo", "ALL")] + [f"rfy{a}" for a in ARMS6]
        ex = (EXTRA.replace("__GLOB__", "rf[xy]*lab_fold[1-5].parquet").replace("__NSM__", str(len(arms8)))
              .replace("ARMS = __ARMS__", "ARMS = __ARMS__\nimport os\nif os.environ.get('FACTOR2_PHASE', '') == '1': ARMS = [a for a in ARMS if a.startswith('rfx')]")
              .replace("__ARMS__", repr(arms8)).replace("__PFX__", "").replace('a, s = n[3:].split("lab_fold")[0], int(n[-9])', 'a, s = n.split("lab_fold")[0], int(n[-9])'))
        B167.BL.B12.nb(cells(arms8, "", "FACTOR-02", "factor2", extra=ex), "FLAG_186_factor2_report.ipynb")
        print("FLAG_186_factor2_report.ipynb built")
    elif sys.argv[1:] == ["7"]:                                       # FACTOR-01 (FLAG_183 tables rfx*; step 0 atlas + step 2 gate)
        B167.BL.B12.nb(cells(list(ARMS5), "rfx", "FACTOR-01", "factor", nsm=len(ARMS5)), "FLAG_184_factor_report.ipynb")
        print("FLAG_184_factor_report.ipynb built")
    elif sys.argv[1:] == ["6"]:                                       # IBLORA-01 bac 2 (English fuser of F1s' with ibLoRA)
        B167.BL.B12.nb(cells([], "ibl", "IBLORA-01", "iblora2", extra=EXTRA_G1), "FLAG_182_iblora2_report.ipynb")
        print("FLAG_182_iblora2_report.ipynb built")
    elif sys.argv[1:] == ["5"]:                                       # HARDNEG-RFS + EN-HEARD (FLAG_180 tables rfh*)
        B167.BL.B12.nb(cells(list(ARMS4), "rfh", "HARDNEG-RFS", "engx", nsm=len(ARMS4)), "FLAG_181_engx_report.ipynb")
        print("FLAG_181_engx_report.ipynb built")
    elif sys.argv[1:] == ["4"]:                                     # RISKSIM-01 (same inputs as FLAG_178)
        B167.BL.B12.nb(cells(list(ARMS3), "rfe", "RISKSIM-01", "risksim", glob="rf[etu]*lab_fold[1-5].parquet", nsm=len(ARMS3)),
                       "FLAG_179_risksim_report.ipynb"); print("FLAG_179_risksim_report.ipynb built")
    elif sys.argv[1:] == ["3"]:                                     # ENPROG-01 (FLAG_177 rfe + FLAG_173 rft + FLAG_175 rfu tables)
        B167.BL.B12.nb(cells(list(ARMS3), "rfe", "ENPROG-01", "enprog", glob="rf[etu]*lab_fold[1-5].parquet", nsm=len(ARMS3)),
                       "FLAG_178_enprog_report.ipynb"); print("FLAG_178_enprog_report.ipynb built")
    elif sys.argv[1:] == ["2"]:                                     # RFSTUNE-02 (FLAG_175 tables rfu<arm>lab_fold<s>)
        B167.BL.B12.nb(cells(list(ARMS2), "rfu", "RFSTUNE-02", "rfstune2"), "FLAG_176_rfstune2_report.ipynb"); print("FLAG_176_rfstune2_report.ipynb built")
    else:
        B167.BL.B12.nb(cells(), "FLAG_174_rfstune_report.ipynb"); print("FLAG_174_rfstune_report.ipynb built")
