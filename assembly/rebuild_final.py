r"""Rebuild the submitted Evaluation zip FACTOR_ENFUSE (c5rl) from the outputs of the Kaggle runs (CPU, seconds, numpy + pandas).

    python rebuild_final.py [IN_DIR] [OUT_ZIP] [REF_ZIP]
      IN_DIR  (default ./in):  f1s_eval.zip, rfx_eval.zip, components_eval.npz, e2e_manifest.json,
                               cleantune01_fuser_bangla_CA.json, ct01r2_final_eval.npz
      OUT_ZIP (default ./FACTOR_ENFUSE_c5rl_rebuilt.zip);  REF_ZIP (default ../submission/posteval_FACTOR_ENFUSE_c5rl.zip)

Stored score = -logit (CodaBench: low = same person).
  U_X     = F1s' + RFX, pair by pair (1 : 1).
  English = -(F1s' Bangla fuser applied to the English trials) + RFX.
  Bangla  = U_X.
The F1s' Bangla fuser is CA-FUSE: logistic regression on [Z, C, Z x C] (Z = 7 streams, C = 7 conditions), weights in the JSON.
Every file of OUT_ZIP is compared with REF_ZIP (contents, not zip bytes: zip headers hold timestamps)."""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else H / "in"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else H / "FACTOR_ENFUSE_c5rl_rebuilt.zip"
REF = Path(sys.argv[3]) if len(sys.argv) > 3 else H.parent / "submission" / "posteval_FACTOR_ENFUSE_c5rl.zip"


def read_zip(p):
    with zipfile.ZipFile(p) as z:
        return {n: z.read(n) for n in z.namelist()}


def table(b):
    return pd.read_csv(io.BytesIO(b), sep=" ", header=None, names=["pid", "s"], dtype={"pid": str})


def main():
    f1z, rfz = read_zip(IN / "f1s_eval.zip"), read_zip(IN / "rfx_eval.zip")
    assert sorted(f1z) == sorted(rfz), "F1s' and RFX zips must hold the same four files"
    # 1. U_X = F1s' + RFX (F1s' pair order)
    ux = {}
    for n, b in f1z.items():
        f1 = [l.split() for l in b.decode().splitlines() if l.strip()]
        rf = dict(l.split() for l in rfz[n].decode().splitlines() if l.strip())
        assert len(rf) == len(f1) and all(p in rf for p, _ in f1), n
        ux[n] = "".join(f"{p} {float(rf[p]) + float(s):.8f}\n" for p, s in f1).encode()
    # 2. English files: F1s' Bangla fuser on the English trials + RFX
    man = json.loads((IN / "e2e_manifest.json").read_text())
    comp = np.load(IN / "components_eval.npz", allow_pickle=False)
    fu = json.loads((IN / "cleantune01_fuser_bangla_CA.json").read_text())
    assert fu["streams"] == ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age", "anP512_deep", "anP512_cca", "trCA"], fu["streams"]
    head = np.load(IN / "ct01r2_final_eval.npz")
    mean, scale, coef = (np.array(fu[k], np.float64) for k in ("mean", "scale", "coef"))
    out = {}
    for f in man["files"]:
        i, n = f["i"], f["zip_name"]
        if not f["is_en"]:
            out[n] = ux[n]; continue
        Z = np.concatenate([comp[f"b3_{i}"][:, :4], comp[f"t3_{i}"][:, [0, 1]], np.asarray(head[f"trCA_{i}"], np.float64)[:, None]], 1)
        C = comp[f"co_{i}"]
        X = np.concatenate([Z, C, (Z[:, :, None] * C[:, None, :]).reshape(len(Z), -1)], 1)
        s = ((X - mean) / scale) @ coef + fu["intercept"]
        assert np.isfinite(s).all(), n
        rf = table(rfz[n]); rf = dict(zip(rf.pid, rf.s))
        pid = [str(p) for p in comp[f"pid_{i}"]]
        new = dict(zip(pid, -s + np.array([rf[p] for p in pid])))
        order = [l.split()[0] for l in ux[n].decode().splitlines() if l.strip()]
        assert set(order) == set(pid), n
        out[n] = "".join(f"{p} {new[p]:.8f}\n" for p in order).encode()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as w:
        for n in sorted(f1z, key=list(f1z).index):
            w.writestr(n, out[n])
    print("wrote", OUT)
    if REF.exists():
        ref = read_zip(REF); bad = 0
        for n in ref:
            same = ref[n] == out.get(n)
            bad += not same
            print(f"  {n}: {len(ref[n].splitlines())} pairs, {'identical' if same else 'DIFFERENT'}")
        print("OK: identical to the submitted zip" if bad == 0 and sorted(ref) == sorted(out) else "MISMATCH")
        sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
