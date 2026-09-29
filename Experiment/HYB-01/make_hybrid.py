"""HYB-01 = per-cell merge of three submitted zips (each cell is scored as an independent file, so the result is
known exactly from their CodaBench per-cell EERs):  ng/En <- SET-08 12.30, ng/Bn <- SET-07 17.21,
g/En <- SET-07 27.49, g/Bn <- SET-02 24.59  ->  expected 20.40 (SET-02: 21.68)."""
import sys, zipfile
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle"))
import flag_v2 as V
SRC = {"SET-02": R / "Experiment/SET-01/out/submission_SET02.zip", "SET-07": R / "Experiment/SET-07/out/submission_SET07.zip",
       "SET-08": R / "Experiment/ARCH-01/out/submission_SET08.zip"}
PICK = {"no_gender/sub_score_v4_English_heard.txt": "SET-08", "no_gender/sub_score_v4_Bangla_unheard.txt": "SET-07",
        "gender/sub_score_v4_English_heard.txt": "SET-07", "gender/sub_score_v4_Bangla_unheard.txt": "SET-02"}
out = Path(__file__).parent / "submission_HYB01.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
    for fn, src in PICK.items():
        zo.writestr(fn, zipfile.ZipFile(SRC[src]).read(fn))
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
V.check_zip(store, out)
for fn, src in PICK.items():
    assert zipfile.ZipFile(out).read(fn) == zipfile.ZipFile(SRC[src]).read(fn)
print("byte-identical to sources; expected CB", round((12.30 + 17.21 + 27.49 + 24.59) / 4, 2))
