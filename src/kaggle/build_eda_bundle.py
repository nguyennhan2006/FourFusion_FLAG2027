"""Pack every array FLAG_13_ext_curation needs into ONE Kaggle dataset: kaggle/flag2027-eda-bundle.zip -> upload as
`flag2027-eda-bundle` (private). Sources: kaggle_upload (organiser v4 train features), feats_ext_full (v1/v2 rows in the
organiser feature space + meta), feats_models (ImageBind, ages, FaRL), feats_v2 / feats_eval (ArcFace, ECAPA-192).
v4 arrays stay float32 (the bridge must match flag_v2 exactly); v1/v2 arrays are float16 (as the Kaggle outputs are).
    python build_eda_bundle.py
"""
import json, time, zipfile, io
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1]
UP, FX, FM, FV, FE = (R / "kaggle_upload", R / "kaggle/output/feats_ext_full", R / "kaggle/output/feats_models",
                      R / "kaggle/output/feats_v2", R / "kaggle/output/feats_eval")
OUT = R / "kaggle/flag2027-eda-bundle.zip"
arrays, tables = {}, {}

# ---- v4 train (organiser row order)
face = pd.read_csv(UP / "train/train_English_faces.csv", header=None)
voice = pd.read_csv(UP / "train/train_English_voices.csv", header=None)
assert (face.iloc[:, -1].values == voice.iloc[:, -1].values).all()
arrays["v4_face_vgg"] = face.iloc[:, :-1].values.astype(np.float32)
arrays["v4_voice_btc"] = voice.iloc[:, :-1].values.astype(np.float32)
txt = pd.read_csv(UP / "train/train_English.txt", sep=" ", header=None, names=["pair_id", "label", "voice", "face", "spk", "spk_int"])
meta = pd.read_csv(UP / "train/meta_file_train_set.csv", header=None, names=["spk", "gender"])
tables["v4_rows.csv"] = txt[["pair_id", "spk"]]
tables["v4_gender.csv"] = meta
for name, path in [("v4_face_ibv", FM / "face_ibv_train.npy"), ("v4_voice_iba", FM / "voice_iba_train.npy"),
                   ("v4_face_vitageprob", FM / "face_vitageprob_train.npy"), ("v4_voice_w2v2agage", FM / "voice_w2v2agage_train.npy"),
                   ("v4_face_farl", FM / "face_farl_train.npy"), ("v4_face_arcface", FV / "face_arcface_train.npy"),
                   ("v4_voice_ecapa192", FV / "voice_ecapa192_train.npy")]:
    arrays[name] = np.load(path).astype(np.float32)

# ---- MAV-Celeb v1 / v2 (rows of feats_ext_full meta)
for src in ["v1_complete", "v2_complete"]:
    tables[f"ext_{src}_face_meta.csv"] = pd.read_csv(FX / f"ext_{src}_face_meta.csv")
    tables[f"ext_{src}_voice_meta.csv"] = pd.read_csv(FX / f"ext_{src}_voice_meta.csv")
    sp = FX / f"ext_{src}_speakers.csv"
    if sp.exists():
        tables[f"ext_{src}_speakers.csv"] = pd.read_csv(sp)
    for name, path in [("face_vgg", FX / f"ext_{src}_face.npy"), ("voice_btc", FX / f"ext_{src}_voice.npy"),
                       ("face_ibv", FM / f"ext_{src}_face_ibv.npy"), ("voice_iba", FM / f"ext_{src}_voice_iba.npy"),
                       ("face_vitageprob", FM / f"ext_{src}_face_vitageprob.npy"),
                       ("voice_w2v2agage", FM / f"ext_{src}_voice_w2v2agage.npy"),
                       ("voice_w2v2aggen", FM / f"ext_{src}_voice_w2v2aggen.npy"), ("face_farl", FM / f"ext_{src}_face_farl.npy"),
                       ("face_arcface", FE / f"ext_{src}_face_arcface.npy"), ("voice_ecapa192", FE / f"ext_{src}_voice_ecapa192.npy")]:
        a = np.load(path).astype(np.float16)
        n_meta = len(tables[f"ext_{src}_{name.split('_')[0]}_meta.csv"])
        assert len(a) == n_meta, (src, name, len(a), n_meta)
        arrays[f"ext_{src}_{name}"] = a

manifest = {"created": time.strftime("%Y-%m-%d %H:%M"), "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()},
            "tables": {k: len(v) for k, v in tables.items()},
            "note": "FourFusion private bundle for kaggle/FLAG_13_ext_curation.ipynb; do not redistribute (organiser features)."}
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for k, v in arrays.items():
        b = io.BytesIO(); np.save(b, v); z.writestr(f"eda_bundle/{k}.npy", b.getvalue())
    for k, v in tables.items():
        z.writestr(f"eda_bundle/{k}", v.to_csv(index=False))
    z.writestr("eda_bundle/eda_bundle_manifest.json", json.dumps(manifest, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB): {len(arrays)} arrays, {len(tables)} tables")
