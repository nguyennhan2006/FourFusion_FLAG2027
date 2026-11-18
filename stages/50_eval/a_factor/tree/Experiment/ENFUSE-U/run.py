r"""ENFUSE-U (README registered 12/11 ~20:30 before the zip): the two English Evaluation files scored by F1s' own Bangla fuser
(CA-FUSE S4 + v11 T3 (2) + trCA, NOREL) + RFS (1 : 1 as U'); Bangla files byte-identical to U'. Frozen, NOT submitted.
    python Experiment/ENFUSE-U/run.py      (local CPU, seconds) -> Experiment/POSTEVAL-01/zips/posteval_ENFUSE_U.zip, out/enfuse_u.txt
Inputs (outputs of the Evaluation runs, downloaded to in/): f1/e2e_out/{cleantune01_fuser_bangla_CA.json, ct01r2_final_eval.npz},
c50/e2e_eval/{components_eval.npz, e2e_manifest.json}; zips EVALPATH-U/eval_out/{f1s_eval, rfs_eval, submission_U_eval}.zip."""
import io, json, os, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
os.environ["E2E_DIR"] = str(HERE / "in/c50/e2e_eval"); os.environ["E2E_SPLIT"] = "eval"; os.environ["NOREL"] = "1"
sys.path.insert(0, str(EXP / "E2E"))
import e2e_lib as E                                                   # noqa: E402  (E2E_DIR / E2E_SPLIT / NOREL read at import)

EV = EXP / "EVALPATH-U/eval_out"
# EVALSWEEP-01 (13/11): optional overrides; unset = the original ENFUSE_U build, byte for byte.
#   ENF_RFS = RFS-family Evaluation zip for the English sum; ENF_U = zip whose Bangla bytes are copied; ENF_OUT = output zip.
RFZ = Path(os.environ.get("ENF_RFS", EV / "rfs_eval.zip"))
UZ = Path(os.environ.get("ENF_U", EV / "submission_U_eval.zip"))
OUTZ = Path(os.environ.get("ENF_OUT", EXP / "POSTEVAL-01/zips/posteval_ENFUSE_U.zip"))
ENF_W = float(os.environ["ENF_W"]) if os.environ.get("ENF_W") else None   # FUSEDIM-02 (15/11): English = 2 [w F1s' + (1 - w) RFS]; unset = 1:1


def zscores(p):
    with zipfile.ZipFile(p) as z:
        return {n: pd.read_csv(io.BytesIO(z.read(n)), sep=" ", header=None, names=["pid", "s"], dtype={"pid": str}) for n in z.namelist()}, \
               {n: z.read(n) for n in z.namelist()}


def main():
    assert E.SPLIT == "eval" and E.NOREL
    bn = json.loads((HERE / "in/f1/e2e_out/cleantune01_fuser_bangla_CA.json").read_text())
    want = ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age", "anP512_deep", "anP512_cca", "trCA"]
    assert bn["streams"] == want, bn["streams"]
    head = np.load(HERE / "in/f1/e2e_out/ct01r2_final_eval.npz")
    comps = E.components()
    F1, _ = zscores(EV / "f1s_eval.zip"); RF, _ = zscores(RFZ); U, ubytes = zscores(UZ)
    lines, out = [], {}
    for i, c in comps.items():
        n = c["zip_name"]
        Z = np.concatenate([c["b3"][:, :4], c["t3"][:, [0, 1]], np.asarray(head[f"trCA_{i}"], np.float64)[:, None]], 1)
        fn = lambda r, Z=Z, C=c["co"]: E.apply_fuser(bn, Z if r is None else Z[r], C if r is None else C[r])
        s, dl0 = E.level0_selftest(fn, len(c["pid"]))
        assert dl0 < 1e-9 and np.isfinite(s).all(), (c["key"], dl0)
        pid = [str(p) for p in c["pid"]]
        f1 = dict(zip(F1[n].pid, F1[n].s)); rf = dict(zip(RF[n].pid, RF[n].s)); u = dict(zip(U[n].pid, U[n].s))
        assert set(pid) == set(f1) == set(rf) == set(u), n
        if c["is_en"]:
            rfa = np.array([rf[p] for p in pid])                      # stored score = -logit (CB: low = same), as U'
            new = -s + rfa if ENF_W is None else 2 * (ENF_W * -s + (1 - ENF_W) * rfa)   # FUSEDIM-02: ENF_W = share of the F1s' fuser
            uo = np.array([u[p] for p in pid])
            rho = pd.Series(new).corr(pd.Series(uo), method="spearman"); out[n] = (pid, new)
            lines.append(f"{c['key']}: Bangla fuser on English + RFS, level-0 |diff| {dl0:.1e}; Spearman with U' English {rho:.4f}")
        else:
            d = float(np.abs(-s - np.array([f1[p] for p in pid])).max())
            lines.append(f"{c['key']}: wiring, F1s' Bangla scores reproduced, max |diff| {d:.1e}")
            assert d <= 1e-5, (c["key"], d)
            assert set(pid) == set(RF[n].pid) == set(U[n].pid), n
    with zipfile.ZipFile(OUTZ, "w", zipfile.ZIP_DEFLATED) as w:
        for c in comps.values():
            n = c["zip_name"]
            if c["is_en"]:
                pid, new = out[n]
                order = [str(p) for p in U[n].pid]; m = dict(zip(pid, new))
                w.writestr(n, "".join(f"{p} {m[p]:.8f}\n" for p in order))
            else:
                w.writestr(n, ubytes[n])
    with zipfile.ZipFile(OUTZ) as a:
        for c in comps.values():
            same = a.read(c["zip_name"]) == ubytes[c["zip_name"]]
            assert same != c["is_en"], c["key"]
    lines.append(f"-> {OUTZ.name} (English: Bangla fuser + {RFZ.name}; Bangla: {UZ.name} bytes). NOT submitted; scored only if labels are released.")
    (HERE / "out").mkdir(exist_ok=True)
    rep = "enfuse_u.txt" if "ENF_OUT" not in os.environ else f"{OUTZ.stem}.txt"
    (HERE / "out" / rep).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "ENF_OUT" in os.environ:                                        # EVALSWEEP-01: freeze in the POSTEVAL-01 manifest
        import hashlib
        m = EXP / "POSTEVAL-01/manifest.json"; M = json.loads(m.read_text(encoding="utf-8"))
        M["zips"][OUTZ.name] = dict(sha256=hashlib.sha256(OUTZ.read_bytes()).hexdigest(),
                                    note=f"ENFUSE-U/run.py ENF_RFS={RFZ.name} ENF_U={UZ.name}, frozen before any label")
        m.write_text(json.dumps(M, indent=1), encoding="utf-8")
        lines.append(f"manifest: {OUTZ.name} {M['zips'][OUTZ.name]['sha256'][:16]}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
