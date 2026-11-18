"""FLAG_173 (11/11): RFSTUNE-01 lab (Experiment/RFSTUNE-01/README.md, registered before any number) = FLAG_170 (RFS-LAB, RFS_MODE=LAB)
with a loop over the registered arms: every arm = `rfs_train.py lab` with RFS_CFG (JSON overrides of CFG) and RFS_TAG=rft<arm>.
RFST_PART=1 / 2 runs every other arm (two GPU sessions, about the same cost); unset = all arms.
    python kaggle/build_flag173.py   -> FLAG_173_rfstune.ipynb
Push (GPU T4): datasets of FLAG_170 = ngnnguynchn/flag2027-rfs-bundle-p, ngnnguynchn/flag2027-lid-strict-p,
ngnnguynchn/flag2027-rfs-lorafix1-p, ngnnguynchn/flag2027-ct02s-r2ca-tables-p; --env 'RFST_PART=1' (or 2).
Smoke (Kaggle CPU): --smoke 'FLAG_SMOKE=1;RFS_STEPS=40;RFST_PART=1' (synthetic bundle + synthetic lab tables)."""
import json

import build_flag163 as B163

# README section 3 (registered); order = balance of the two parts (s2 = 2 x cost)
ARMS = {"a0": {}, "a0r": {"seeds": [4, 5, 6]}, "s1": {"steps": 1500}, "s2": {"steps": 6000}, "t1": {"tau": 0.03}, "t2": {"tau": 0.08},
        "d1": {"drop_f": 0.6, "drop_v": 0.3}, "d2": {"drop_v": 0.7}, "l1": {"lr": 1e-3}, "l2": {"lr": 2.5e-4},
        "p1": {"persons": 1024}, "p2": {"persons": 256}, "w1": {"v4w": 1.0}, "w2": {"v4w": 0.25}, "wd": {"wd": 1e-2}}
# RFSTUNE-02 (12/11, Experiment/RFSTUNE-02/README.md section 2, registered): round 2 around d1, tables rfu<arm>lab_fold<s> -> FLAG_175
D1 = {"drop_f": 0.6, "drop_v": 0.3}
ARMS2 = {"a0": {}, "d1": dict(D1), "d1r": dict(D1, seeds=[4, 5, 6]), "f6v5": {"drop_f": 0.6}, "f8v3": {"drop_v": 0.3},
         "f7v4": {"drop_f": 0.7, "drop_v": 0.4}, "f5v2": {"drop_f": 0.5, "drop_v": 0.2}, "f4v1": {"drop_f": 0.4, "drop_v": 0.1},
         "d1p1": dict(D1, persons=1024), "d1s2": dict(D1, steps=6000), "d1l1": dict(D1, lr=1e-3)}
# ENPROG-01 (12/11, Experiment/ENPROG-01/README.md section 2, registered): encoder-block ablation / single-voice-encoder experts,
# tables rfe<arm>lab_fold<s> -> FLAG_177. z<b> = block b set to 0; o<b> = only voice block b kept (both face blocks).
ARMS3 = {"a0": {}, "a0r": {"seeds": [4, 5, 6]}, "zwh": {"zero": ["v_whv3"]}, "zsam": {"zero": ["v_sam"]}, "zr221": {"zero": ["v_r221"]},
         "zxv": {"zero": ["v_xv"]}, "zclip": {"zero": ["f_clip"]}, "zibf": {"zero": ["f_ibf"]},
         "osam": {"zero": ["v_r221", "v_xv", "v_whv3"]}, "or221": {"zero": ["v_sam", "v_xv", "v_whv3"]},
         "oxv": {"zero": ["v_sam", "v_r221", "v_whv3"]}, "owh": {"zero": ["v_sam", "v_r221", "v_xv"]}}
# HARDNEG-RFS + EN-HEARD (12/11, READMEs section 3, registered): tables rfh<arm>lab_fold<s> -> FLAG_180; order balances the 2 parts
ARMS4 = {"a0": {}, "e3s2": {"mav_en": True, "src_w": {"vb": 0.25}, "steps": 6000}, "h025": {"hard": 0.25}, "h05": {"hard": 0.5},
         "h10": {"hard": 1.0}, "h05k": {"hard": 0.5, "hard_k": 8192}, "h05d1": dict(D1, hard=0.5), "e1": {"mav_en": True},
         "e2": {"src_w": {"vb": 0.25}}, "e3": {"mav_en": True, "src_w": {"vb": 0.25}}, "e3d1": dict(D1, mav_en=True, src_w={"vb": 0.25})}
# FACTOR-01 (13/11, Experiment/FACTOR-01/README.md section 3, registered): RFS projector with F1s' blocks, tables rfx<arm>lab_fold<s>
# -> FLAG_183. Order = balance of the two parts (wide arms alternate).
_F1 = ["f_c5s", "f_farl", "v_rl", "v_sf"]
_ALL = ["f_c5s", "f_farl", "f_pel", "v_rl", "v_sf", "v_ibv"]
ARMS5 = {"a0": {}, "ALL": {"add": _ALL}, "c5": {"add": ["f_c5s"]}, "F1": {"add": _F1}, "farl": {"add": ["f_farl"]},
         "ALLlo": {"add": _ALL, "drop_f": 0.6, "drop_v": 0.3}, "pel": {"add": ["f_pel"]}, "F1lo": {"add": _F1, "drop_f": 0.6, "drop_v": 0.3},
         "rl": {"add": ["v_rl"]}, "ALLhi": {"add": _ALL, "drop_f": 0.9, "drop_v": 0.6}, "sf": {"add": ["v_sf"]},
         "F1hi": {"add": _F1, "drop_f": 0.9, "drop_v": 0.6}, "ibv": {"add": ["v_ibv"]}, "c5rl": {"add": ["f_c5s", "v_rl"]},
         "c5x": {"add": ["f_c5s"], "zero": ["f_clip"]}}
# FACTOR-02 (13/11, Experiment/FACTOR-02/README.md section 1, registered): tables rfy<arm>lab_fold<s> -> FLAG_185. Part 1 = c5rllo, ALL41,
# ALLlor; part 2 = ALL52, ALLslo, ALLlob2k.
_LO = {"drop_f": 0.6, "drop_v": 0.3}
ARMS6 = {"c5rllo": dict(_LO, add=["f_c5s", "v_rl"]), "ALL52": {"add": _ALL, "drop_f": 0.5, "drop_v": 0.2},
         "ALL41": {"add": _ALL, "drop_f": 0.4, "drop_v": 0.1}, "ALLslo": dict(_LO, add=["f_c5s", "f_farl", "f_pel", "v_rl"]),
         "ALLlor": dict(_LO, add=_ALL, seeds=[4, 5, 6]), "ALLlob2k": dict(_LO, add=_ALL, persons=2048)}
# TRANSL-01 (14/11, Experiment/TRANSL-01/README.md section 1, registered): c5rl + one "translator" knob, tables rfz<arm>lab_fold<s>
# -> FLAG_187. Part 1 = c0, orth2, cyc2; part 2 = orth1, cyc1, mpos.
_C5RL = ["f_c5s", "v_rl"]
ARMS7 = {"c0": {"add": _C5RL}, "orth1": {"add": _C5RL, "orth": 0.1}, "orth2": {"add": _C5RL, "orth": 1.0},
         "cyc1": {"add": _C5RL, "cyc": 0.1}, "cyc2": {"add": _C5RL, "cyc": 1.0}, "mpos": {"add": _C5RL, "mpos": 2}}
# FVBIND-01 gate 2 (14/11, Experiment/FVBIND-01/README.md section 3 / 3b, registered): c5rl control + c5rl with the B2 voice block v_fvb
# (FLAG_192 output attached as a kernel input), tables rfb<arm>lab_fold<s> -> FLAG_194
ARMS8 = {"c0": {"add": _C5RL}, "fvb": {"add": _C5RL + ["v_fvb"]}}
# FUSEDIM-01 (14/11, Experiment/FUSEDIM-01/README.md section 1A, registered): c5rl with the shared-space width changed, tables
# rfd<arm>lab_fold<s> -> FLAG_195. Part 1 = c0, d64, d128; part 2 = d2048, d1024, d256.
ARMS9 = {"c0": {"add": _C5RL}, "d2048": {"add": _C5RL, "dim": 2048}, "d64": {"add": _C5RL, "dim": 64},
         "d1024": {"add": _C5RL, "dim": 1024}, "d128": {"add": _C5RL, "dim": 128}, "d256": {"add": _C5RL, "dim": 256}}
# FUSEDIM-02 (15/11, Experiment/FUSEDIM-02/README.md section 1A, registered): c5rl trained with the Matryoshka loss (prefixes 128 / 256
# + full 512), scored on 512 / 256 / 128 coordinates, tables rfm<arm>lab_fold<s> -> FLAG_198. Part 1 = m512, m256; part 2 = m128.
_MRL = [128, 256]
ARMS10 = {"m512": {"add": _C5RL, "mrl": _MRL}, "m128": {"add": _C5RL, "mrl": _MRL, "score_dim": 128},
          "m256": {"add": _C5RL, "mrl": _MRL, "score_dim": 256}}
# RFXARCH-01 (15/11, Experiment/RFXARCH-01/README.md section 1, registered): projector architecture / identity loss on c5rl, tables
# rfa<arm>lab_fold<s> -> FLAG_205. Part 1 = c0, id05, resid05; part 2 = res, id10.
ARMS11 = {"c0": {"add": _C5RL}, "res": {"add": _C5RL, "arch": "res"}, "id05": {"add": _C5RL, "idw": 0.5},
          "id10": {"add": _C5RL, "idw": 1.0}, "resid05": {"add": _C5RL, "arch": "res", "idw": 0.5}}
# LORAFIX-01 (15/11, Experiment/LORAFIX-01/README.md section 1, registered): c5rl lab run once per re-learned LoRA dataset (RFST_ONLY=L1 / L2),
# tables rfl<arm>lab_fold<s> -> FLAG_207; only the v4 ib column depends on the attached LoRA.
ARMS12 = {"L1": {"add": _C5RL}, "L2": {"add": _C5RL}}
# FUSEARCH-02 (15/11, Experiment/FUSEARCH-02/README.md section 1, registered): two fresh c5rl replicates (seeds 4-6 / 7-9), tables
# rfr<arm>lab_fold<s> -> FLAG_213. Part 1 = c0r, part 2 = c0s.
ARMS13 = {"c0r": {"add": _C5RL, "seeds": [4, 5, 6]}, "c0s": {"add": _C5RL, "seeds": [7, 8, 9]}}
# LANGLANE-01 (15/11, Experiment/LANGLANE-01/README.md section 0, registered): one fresh c5rl replicate (seeds 10-12), tables
# rfr<arm>lab_fold<s> -> FLAG_216 (one run, no PART).
ARMS16 = {"c0t": {"add": _C5RL, "seeds": [10, 11, 12]}}
# RFXSTD-01 (16/11, Experiment/RFXSTD-01/README.md section 1, registered): training-loop knobs of the c5rl projector, two replicates each
# (r = seeds 4-6, s = seeds 7-9) + k0r = rerun of the control c0r; tables rfk<arm>lab_fold<s> -> FLAG_220. Part 1 = the r arms + k0r, part 2 = the s arms.
_STD = {"cos": {"sched": "cos"}, "ema": {"ema": 0.999}, "ps": {"pairseed": 3}, "wd": {"wd_skip": 1}, "dm": {"dupmerge": 1},
        "all": {"sched": "cos", "ema": 0.999, "pairseed": 3, "wd_skip": 1, "dupmerge": 1}}
ARMS20 = {f"{k}{x}": {"add": _C5RL, "seeds": sd, **v} for k, v in _STD.items() for x, sd in (("r", [4, 5, 6]), ("s", [7, 8, 9]))}
ARMS20["k0r"] = {"add": _C5RL, "seeds": [4, 5, 6]}

INTRO = """
# FLAG 2027 - NB-173: RFSTUNE-01, RFS projector recipe retuned on our clean store (SAFE-04 lab, GPU)

**What (Experiment/RFSTUNE-01/README.md, registered 11/11 before any number).** For every registered arm (one knob of the RFS projector
recipe changed: steps, tau, dropout, lr, persons per step, v4 weight, wd; a0 = control, a0r = other seeds) the RFS-LAB of EDA-3FAM-01:
per SAFE-04 fold a 3-seed projector trained without the fold's test persons, then t3 / nf / nv on the fold's pairs.
Output `rft<arm>lab_fold<s>.parquet`. The decision (nested arm choice vs a0) is FLAG_174's CPU report. No dev, no CB.
"""

ARMLOOP = '''print("[cell] RFSTUNE-01: lab per arm", flush=True)
ARMS = json.loads(\'__ARMS__\')
PART = os.environ.get("RFST_PART", "")
names = list(ARMS)[int(PART) - 1::2] if PART else list(ARMS)
ONLY = [x for x in os.environ.get("RFST_ONLY", "").split(",") if x]      # smoke on real data: a few arms only (13/11)
if ONLY: names = [n for n in names if n in ONLY]
NFOLD = len([x for x in os.environ.get("RFS_LAB_FOLDS", "1,2,3,4,5").split(",") if x])
print("arms of this run:", names, flush=True)
ART.mkdir(parents=True, exist_ok=True)
(ART / f"__PFX___arms_part{PART or 'all'}.json").write_text(json.dumps({n: ARMS[n] for n in names}, indent=1))
t00 = time.time()
for i, n in enumerate(names):
    t0 = time.time(); print(f"[arm] {i + 1}/{len(names)} {n} {ARMS[n]}", flush=True)
    run(["lab", "--bundle", str(BUNDLE), "--work", str(WORK), "--art", str(ART), "--lid", LID],
        env={"RFS_LAB_TABLES": str(TABDIR), "RFS_CFG": json.dumps(ARMS[n]), "RFS_TAG": "__PFX__" + n})
    got = sorted(p.name for p in ART.glob(f"__PFX__{n}lab_fold*.parquet")); assert len(got) == NFOLD, (n, got)
    print(f"[arm] {n} done [{time.time() - t0:.0f}s, total {time.time() - t00:.0f}s]", flush=True)
print("LAB done:", sorted(p.name for p in ART.iterdir()), flush=True)
'''


def cells(arms=ARMS, pfx="rft"):
    out = B163.cells("LAB")
    i = [k for k, (t, s) in enumerate(out) if t == "code" and 'run(["lab"' in s]
    assert len(i) == 1, "FLAG_163 PIPE cell not found"
    assert "'" not in json.dumps(arms)
    out[i[0]] = ("code", ARMLOOP.replace("__ARMS__", json.dumps(arms)).replace("__PFX__", pfx))
    out[0] = ("markdown", INTRO)
    return out


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["20"]:                                      # RFXSTD-01
        B163.B12.nb(cells(ARMS20, "rfk"), "FLAG_220_rfxstd.ipynb"); print("FLAG_220_rfxstd.ipynb built")
    elif sys.argv[1:] == ["16"]:                                    # LANGLANE-01
        B163.B12.nb(cells(ARMS16, "rfr"), "FLAG_216_seedrep3.ipynb"); print("FLAG_216_seedrep3.ipynb built")
    elif sys.argv[1:] == ["13"]:                                    # FUSEARCH-02
        B163.B12.nb(cells(ARMS13, "rfr"), "FLAG_213_seedrep.ipynb"); print("FLAG_213_seedrep.ipynb built")
    elif sys.argv[1:] == ["11"]:                                    # RFXARCH-01
        B163.B12.nb(cells(ARMS11, "rfa"), "FLAG_205_rfxarch.ipynb"); print("FLAG_205_rfxarch.ipynb built")
    elif sys.argv[1:] == ["12"]:                                    # LORAFIX-01
        B163.B12.nb(cells(ARMS12, "rfl"), "FLAG_207_lorafix.ipynb"); print("FLAG_207_lorafix.ipynb built")
    elif sys.argv[1:] == ["10"]:                                    # FUSEDIM-02 (Matryoshka)
        B163.B12.nb(cells(ARMS10, "rfm"), "FLAG_198_mrl.ipynb"); print("FLAG_198_mrl.ipynb built")
    elif sys.argv[1:] == ["9"]:                                     # FUSEDIM-01
        B163.B12.nb(cells(ARMS9, "rfd"), "FLAG_195_fusedim.ipynb"); print("FLAG_195_fusedim.ipynb built")
    elif sys.argv[1:] == ["8"]:                                     # FVBIND-01 gate 2
        B163.B12.nb(cells(ARMS8, "rfb"), "FLAG_193_fvbg2.ipynb"); print("FLAG_193_fvbg2.ipynb built")
    elif sys.argv[1:] == ["7"]:                                     # TRANSL-01
        B163.B12.nb(cells(ARMS7, "rfz"), "FLAG_187_transl.ipynb"); print("FLAG_187_transl.ipynb built")
    elif sys.argv[1:] == ["6"]:                                     # FACTOR-02
        B163.B12.nb(cells(ARMS6, "rfy"), "FLAG_185_factor2.ipynb"); print("FLAG_185_factor2.ipynb built")
    elif sys.argv[1:] == ["5"]:                                       # FACTOR-01
        B163.B12.nb(cells(ARMS5, "rfx"), "FLAG_183_factor.ipynb"); print("FLAG_183_factor.ipynb built")
    elif sys.argv[1:] == ["4"]:                                       # HARDNEG-RFS + EN-HEARD
        B163.B12.nb(cells(ARMS4, "rfh"), "FLAG_180_engx.ipynb"); print("FLAG_180_engx.ipynb built")
    elif sys.argv[1:] == ["3"]:                                     # ENPROG-01
        B163.B12.nb(cells(ARMS3, "rfe"), "FLAG_177_enprog.ipynb"); print("FLAG_177_enprog.ipynb built")
    elif sys.argv[1:] == ["2"]:                                     # RFSTUNE-02
        B163.B12.nb(cells(ARMS2, "rfu"), "FLAG_175_rfstune2.ipynb"); print("FLAG_175_rfstune2.ipynb built")
    else:
        B163.B12.nb(cells(), "FLAG_173_rfstune.ipynb"); print("FLAG_173_rfstune.ipynb built")
