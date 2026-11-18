r"""CLEANCOMP-01 step 2 (CPU, a minute): the five fold tables of FLAG_164 + the old e2e-dev bundle -> the SAFE-04 reference tables of
the replacement bundle (README section 4). Same file names and columns as the old ones; only the CLEANCOMP-01 columns change:
  streams_fold<s>        XALL_deep0 / XALL_cca / XALL_ib / XALL_age        relonly_multi_fold<s>  relc_fia_eb       relonly_g_fold<s>  relg_fia_eb
  bigb01f_fold<s>        anP512_deep / anP512_cca                          t3enc01f_fold<s>       t3_relg
    python build_tables.py <dir with cleancomp01_fold1..5.parquet> <out dir>   (copies the rest of the old bundle: fusers, ref, components_dev.npz)"""
import io, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

R0 = Path(__file__).resolve().parents[2]
BASE = R0 / "kaggle/flag2027-e2e-dev.zip"
NEW = {"streams": ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age"], "relonly_multi": ["relc_fia_eb"], "relonly_g": ["relg_fia_eb"],
       "bigb01f": ["anP512_deep", "anP512_cca"], "t3enc01f": ["t3_relg"]}


def main(src, out):
    src, out = Path(src), Path(out)
    z = zipfile.ZipFile(BASE)
    rep = []
    for info in z.infolist():
        dst = out / info.filename; dst.parent.mkdir(parents=True, exist_ok=True)
        data = z.read(info.filename)
        stem = Path(info.filename).stem
        for t, cols in NEW.items():
            if stem.startswith(f"{t}_fold"):
                s = int(stem.split("fold")[-1])
                old, new = pd.read_parquet(io.BytesIO(data)), pd.read_parquet(src / f"cleancomp01_fold{s}.parquet")
                key = [k for k in ("y", "fi", "vi") if k in old.columns]
                assert len(old) == len(new) and all((old[k].values == new[k].values).all() for k in key), (t, s, "row alignment")
                if "data" in old.columns:
                    assert (old.data.values == new.data.values).all(), (t, s)
                d = {c: float(np.corrcoef(old[c].values, new[c].values)[0, 1]) for c in cols}
                for c in cols:
                    old[c] = new[c].values
                b = io.BytesIO(); old.to_parquet(b); data = b.getvalue()
                rep.append(f"fold {s} {t}: corr(old, new) " + ", ".join(f"{c} {v:.4f}" for c, v in d.items()))
        dst.write_bytes(data)
    print("\n".join(sorted(rep)))
    (out / "e2edev/CLEANCOMP.txt").write_text("CLEANCOMP-01: SAFE-04 tables with S6 / rel / relg / v11 T3 columns relearned under LID_STRICT=1\n" + "\n".join(sorted(rep)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
