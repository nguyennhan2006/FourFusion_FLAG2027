"""EVALPROBE-01: dựng zip dò cho các lượt Evaluation còn lại + ghép zip cuối theo từng ô.

Bốn file trong zip độc lập nhau, nên mỗi lượt nộp đo được bốn ứng viên (một ứng viên mỗi ô).
Trộn chỉ bằng trung bình điểm thô (không dùng thống kê của file test: không z-score, không hạng).

    python Experiment/EVALPROBE-01/make_probes.py probes          # dựng zip dò P1, P2
    python Experiment/EVALPROBE-01/make_probes.py final ngEn=<cand> ngBn=<cand> gEn=<cand> gBn=<cand>
"""
import glob, os, sys, zipfile
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zips")
CELLS = {"ngEn": "no_gender/sub_score_v4_English_heard.txt",
         "ngBn": "no_gender/sub_score_v4_Bangla_unheard.txt",
         "gEn": "gender/sub_score_v4_English_heard.txt",
         "gBn": "gender/sub_score_v4_Bangla_unheard.txt"}
PE = "Experiment/POSTEVAL-01/zips/posteval_"
POOL = {  # tên ngắn -> zip Evaluation đã có
    "FACTOR_ENFUSE": PE + "FACTOR_ENFUSE_c5rl.zip",
    "FVBIND_ENFUSE": "Experiment/FVBIND-01/zips/posteval_FVBIND_ENFUSE_fvb.zip",
    "FVBIND_U": "Experiment/FVBIND-01/zips/posteval_FVBIND_U_fvb.zip",
    "FUSEDIM2_ENFUSE": PE + "FUSEDIM2_ENFUSE_m256w60.zip",
    "ENFUSE_U": PE + "ENFUSE_U.zip",
    "U_prime": PE + "U_prime.zip",
    "RFS": PE + "RFS.zip", "RFSG": PE + "RFSG.zip", "F1s_prime": PE + "F1s_prime.zip",
    "RFX": "Experiment/FACTOR-01/out/rfsfxc5rl_eval.zip",
    **{f"CELL4_{t}": PE + f"CELL4_ENFUSE_{t}.zip" for t in ("g1", "g2", "g3", "g4", "e1", "h1", "lf", "b1", "b2", "n1")
       if os.path.exists(os.path.join(ROOT, PE + f"CELL4_ENFUSE_{t}.zip"))},   # CELL4-01 arms that are finished
    "J01": PE + "FUSEARCH3_J01.zip",   # FUSEARCH-03: one CA-FUSE fuser on 7 F1s' scores + raw RFX streams (other session)
    **{f"EVS_ENFUSE_{v}": PE + f"EVS_ENFUSE_{v}.zip"
       for v in ["RFSG", "d1", "d1s2", "e3", "e3d1", "f4v1", "f5v2", "h05", "h05d1"]},
}
# Điểm Evaluation đã biết theo ô (ngEn, ngBn, gEn, gBn); docs/EVAL_SUBMISSIONS.md
KNOWN = {"U_prime": (15.31, 13.72, 24.04, 20.95), "ENFUSE_U": (14.80, 13.72, 22.12, 20.95),
         "FACTOR_ENFUSE": (13.78, 11.97, 22.12, 20.11),
         "FVBIND_lượt8": (14.29, 12.22, 22.12, 18.99)}


def load(path):
    z = zipfile.ZipFile(os.path.join(ROOT, path))
    d = {}
    for s, c in CELLS.items():
        t = z.read(c).decode().split()
        d[s] = (t[0::2], np.array(t[1::2], dtype=float))
    return d


DATA = {k: load(v) for k, v in POOL.items()}
ORDER = {s: DATA["FACTOR_ENFUSE"][s][0] for s in CELLS}


def vec(name, s):
    ids, v = DATA[name][s]
    m = dict(zip(ids, v))
    return np.array([m[k] for k in ORDER[s]])


ENS = {  # ứng viên trộn: trung bình điểm thô của các thành viên
    "ENS_ALL": ["FACTOR_ENFUSE", "FVBIND_ENFUSE", "FUSEDIM2_ENFUSE", "ENFUSE_U"]
               + [f"EVS_ENFUSE_{v}" for v in ["RFSG", "d1", "d1s2", "e3", "e3d1", "f4v1", "f5v2", "h05", "h05d1"]],
    "ENS_RFX2": ["FACTOR_ENFUSE", "FVBIND_ENFUSE"],
}


def comp(name, s):
    """Thành phần tách đúng từng bit: FACTOR_ENFUSE = F1BM + RFX; FVBIND_ENFUSE = F1BM + RFXF.
    F1BM = họ F1s′ chấm bằng bộ trộn Bangla (ở file Bangla trùng F1s′)."""
    f1bm = vec("FACTOR_ENFUSE", s) - vec("RFX", s)
    return {"F1BM": f1bm, "RFX": vec("RFX", s), "RFXF": vec("FVBIND_ENFUSE", s) - f1bm}[name]


def cand(name, s):
    if name in ENS:
        return np.mean([vec(m, s) for m in ENS[name]], axis=0)
    if name.startswith("W"):  # W<w>_<RFX|RFXF> = 2·[w·F1BM + (1−w)·thành phần thứ hai]; W0.50_RFX ≡ FACTOR_ENFUSE
        w, sec = name[1:].split("_", 1)
        w = float(w)
        return 2 * (w * comp("F1BM", s) + (1 - w) * comp(sec, s))
    return vec(name, s)


def write_zip(path, pick):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for s, c in CELLS.items():
            v = cand(pick[s], s)
            assert np.isfinite(v).all() and len(v) == len(ORDER[s])
            z.writestr(c, "".join(f"{k} {x:.8f}\n" for k, x in zip(ORDER[s], v)))
    print("wrote", os.path.relpath(path, ROOT), pick)


# Quét tỉ lệ w của F1BM so với thành phần thứ hai, mỗi ô một thứ tự khác nhau (w = 0.50 đã biết điểm).
# Thành phần thứ hai: RFX cho ba ô (FACTOR_ENFUSE tốt hơn ở ngEn / ngBn), RFXF cho gBn (FVBIND tốt hơn 1.12).
SWEEP = {"gEn": [0.35, 0.20, 0.00, 0.65, 0.80, 0.42],
         "ngEn": [0.65, 0.35, 0.80, 0.20, 0.42, 0.58],
         "ngBn": [0.65, 0.35, 0.20, 0.80, 0.58, 0.42],
         "gBn": [0.35, 0.65, 0.20, 0.80, 0.42, 0.58]}
SEC = {"gEn": "RFX", "ngEn": "RFX", "ngBn": "RFX", "gBn": "RFXF"}
PROBES = {f"P{i + 1}": {s: f"W{SWEEP[s][i]:.2f}_{SEC[s]}" for s in CELLS} for i in range(6)}
# English-first probes (15/11 20:30): structurally different candidates for the two English files; Bangla files ride along.
# ENH = EVS_ENFUSE_e3 (EN-HEARD: MAV voices of English clips only + VoxBlink down-weighted, RFS base, Bangla fuser).
EPROBES = {
    "E2": {"ngEn": "W0.35_RFX", "ngBn": "W0.35_RFX", "gEn": "J01", "gBn": "J01"},
    "E3": {"ngEn": "EVS_ENFUSE_e3", "ngBn": "W0.20_RFX", "gEn": "EVS_ENFUSE_e3", "gBn": "W0.65_RFXF"},
    "E4": {"ngEn": "J01", "ngBn": "W0.80_RFX", "gEn": "W0.00_RFX", "gBn": "W0.20_RFXF"},
    # 16/11: CELL4-01 g2 (hard-negative weights + v4 loss x3 + English voices) and g1 (one-gender batches + English voices)
    "E5": {"ngEn": "CELL4_g2", "ngBn": "W0.58_RFX", "gEn": "CELL4_g2", "gBn": "W0.42_RFXF"},
    "E6": {"ngEn": "CELL4_g1", "ngBn": "W0.42_RFX", "gEn": "CELL4_g1", "gBn": "W0.58_RFXF"},
}
BEST_NOW = {"ngEn": "FACTOR_ENFUSE", "ngBn": "FACTOR_ENFUSE", "gEn": "FACTOR_ENFUSE", "gBn": "FVBIND_ENFUSE"}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probes"
    if mode == "probes":
        for s in CELLS:  # kiểm tách thành phần: w = 0.50 phải trùng zip đã nộp
            assert np.abs(cand("W0.50_RFX", s) - vec("FACTOR_ENFUSE", s)).max() < 1e-9
            assert np.abs(cand("W0.50_RFXF", s) - vec("FVBIND_ENFUSE", s)).max() < 1e-9
        print("Spearman với ứng viên tốt nhất hiện tại của từng ô:")
        for c in list(ENS) + ["RFX", "F1s_prime", "W0.00_RFX", "W0.20_RFX", "W0.35_RFX", "W0.65_RFX", "W0.80_RFX"]:
            print(f"  {c:18s}", " ".join(f"{s}={spearmanr(cand(c, s), cand(BEST_NOW[s], s)).statistic:.4f}" for s in CELLS))
        for n, pick in PROBES.items():
            write_zip(os.path.join(OUT, f"evalprobe_{n}.zip"), pick)
    elif mode == "eprobes":
        for n, pick in EPROBES.items():
            if not all(c in POOL or c in ENS or c.startswith("W") for c in pick.values()): print("  skip", n, "(zip not built yet)"); continue
            for sc in CELLS:
                print(f"  {n} {sc}: {pick[sc]:16s} rho với bản tốt nhất đã biết {spearmanr(cand(pick[sc], sc), cand(BEST_NOW[sc], sc)).statistic:.4f}")
            write_zip(os.path.join(OUT, f"evalprobe_{n}.zip"), pick)
    elif mode == "final":
        pick = dict(BEST_NOW)
        pick.update(dict(a.split("=", 1) for a in sys.argv[2:]))
        write_zip(os.path.join(OUT, "evalprobe_FINAL.zip"), pick)
