r"""CLEANCOMP-01 step 3 (local CPU, about 10 min, README section 4): the dev components of F1s' Bangla files with the strict S6 member and the
strict v11 T3 head. English files, conditions, v9 / v11 reference scores and every other array are copied from the old bundle.
    python build_dev.py <final dir: s007_xall.pt + v11_head/t3_streams.npz> <bundle dir made by build_tables.py> [--check-old]
  --check-old   first rebuilds the Bangla b3 with BEST/v7_safe/members/s007_xall.pt and LID_STRICT unset: must equal the old b3 (wiring check)."""
import json, os, sys, time
from pathlib import Path
import numpy as np

R0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0 / "Experiment/E2E")); sys.path.insert(0, str(R0 / "kaggle"))
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
FEATS = [R0 / "kaggle/output/feats_eval", R0 / "kaggle/output/feats_av", R0 / "kaggle/output/feats_models"]
EXTD = [R0 / "kaggle/output/feats_ext_full", R0 / "kaggle/output/feats_models", R0 / "kaggle/output/feats_eval"]


def system(xall_pt, strict):
    os.environ["LID_STRICT"] = "1" if strict else "0"
    import system_v11 as SY
    fus = json.loads((R0 / "BEST/v7_safe/cafuse_fuser.json").read_text())
    return SY, SY.System(R0 / "kaggle_upload", FEATS, EXTD, xall_pt, R0 / "BEST/v6_eval/_members", fus, threads=10)


def main(fin, bundle, check_old=False):
    fin, bundle = Path(fin), Path(bundle)
    z = np.load(bundle / "e2edev/components_dev.npz", allow_pickle=False)
    man = json.loads((bundle / "e2edev/e2e_manifest.json").read_text())
    bn = [f for f in man["files"] if not f["is_en"]]
    if check_old:
        SY, sy = system(R0 / "BEST/v7_safe/members/s007_xall.pt", False)
        for f in bn:
            B3, CO = sy.b3_co(f["key"], False)
            d = float(max(np.abs(B3 - z[f"b3_{f['i']}"]).max(), np.abs(CO - z[f"co_{f['i']}"]).max()))
            log(f"{f['key']}: old member / old lists reproduce the bundle's b3 + co, max |diff| {d:.2e}")
            assert d < 1e-4, d
    SY, sy = system(fin / "s007_xall.pt", True)
    t3 = np.load(fin / "v11_head/t3_streams.npz", allow_pickle=False)
    out, rep = {k: z[k] for k in z.files}, []
    for f in bn:
        i, tag = f["i"], f["key"].replace("/", "_")
        B3, CO = sy.b3_co(f["key"], False)
        dco = float(np.abs(CO - z[f"co_{i}"]).max())
        assert dco < 1e-4, (f["key"], "conditions changed", dco)
        T3 = t3[tag].astype(np.float64)
        assert B3.shape == z[f"b3_{i}"].shape and T3.shape == z[f"t3_{i}"].shape and np.isfinite(B3).all() and np.isfinite(T3).all()
        rho = lambda a, b: float(np.corrcoef(a, b)[0, 1])
        rep.append(f"{f['key']}: corr(old, new) b3 " + " ".join(f"{rho(z[f'b3_{i}'][:, j], B3[:, j]):.3f}" for j in range(6)) +
                   " | t3 " + " ".join(f"{rho(z[f't3_{i}'][:, j], T3[:, j]):.3f}" for j in range(3)))
        out[f"b3_{i}"], out[f"t3_{i}"] = B3, T3
        log(rep[-1])
    np.savez(bundle / "e2edev/components_dev.npz", **out)
    man["cleancomp"] = dict(created=time.strftime("%Y-%m-%d %H:%M"), note="Bangla b3 + t3 under LID_STRICT=1; v11 / v9 keys = the old submitted scores (reference only)")
    (bundle / "e2edev/e2e_manifest.json").write_text(json.dumps(man, indent=1))
    (bundle / "e2edev/CLEANCOMP_dev.txt").write_text("\n".join(rep) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--check-old" in sys.argv)
