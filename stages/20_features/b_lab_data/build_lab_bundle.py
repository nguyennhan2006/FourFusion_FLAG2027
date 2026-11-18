"""Pack every feature file the SAFE-04 lab opens (audited 08/10) into ONE Kaggle dataset for FLAG_48_lab.ipynb:
kaggle/flag2027-lab-data.zip -> upload ONCE as `flag2027-lab-data` (private). The code is NOT here: build_lab_notebook.py
embeds it in the notebook, so a new experiment only needs a rebuilt notebook. The organiser eda bundle is NOT inside: attach the existing
dataset `flag2027-eda-bundle` (created 2026-09-29 08:06, the version the local lab uses; the notebook checks the stamp).

Layout inside the zip = the repo layout the code expects (paths are relative to the repo root); the notebook links
these data folders next to the code it writes:
  lab/lab_manifest.json
  lab/Experiment/T3ENC-01/feats_t3enc, feats_vox2t3, feats_vox2lid/lid_summary.json; VOX2-01/feats_vox2/*_meta.csv
  lab/kaggle/output/feats_av, feats_eval, feats_models  (only the files the lab opens)
The three ECAPA-6144 arrays (654 MB, used only by SAFE-04's relational sweeps, never by the T3-head experiments) are
written as (rows, 0) placeholders so the import-time row checks pass; the notebook's lab_entry.py removes that stream.
    python build_lab_bundle.py          # flag2027-lab-data.zip (upload once)
    python build_lab_bundle.py vb       # flag2027-lab-vb.zip: the VoxBlink features of FLAG_46 for SCALE-VB (upload once)
    python build_lab_bundle.py e6144    # flag2027-lab-e6144.zip: the REAL ECAPA-6144 arrays (s010B) for ENBASE-01 (upload once)
"""
import io, json, sys, time, zipfile
import os  # FF path (CHANGES.md): FF_REPO_ROOT
import numpy as np
from pathlib import Path

R = Path(os.environ["FF_REPO_ROOT"]) if os.environ.get("FF_REPO_ROOT") else Path(__file__).resolve().parents[1]   # FF path (CHANGES.md): repository root given by the stage
OUT = R / "kaggle/flag2027-lab-data.zip"
E, FO = R / "Experiment", R / "kaggle/output"
T3N = [("face", "farl16"), ("face", "clipL"), ("voice", "sam100p"), ("voice", "sam100f")]
DATA_DIRS = []
T3 = ([f"Experiment/T3ENC-01/feats_t3enc/{k}_{n}_train.npy" for k, n in T3N]
      + [f"Experiment/T3ENC-01/feats_t3enc/ext_{s}_{k}_{n}.npy" for s in ("v1_complete", "v2_complete") for k, n in T3N]
      + [f"Experiment/T3ENC-01/feats_vox2t3/{k}_{n}.npy" for k, n in T3N])
DATA = T3 + ["Experiment/T3ENC-01/feats_vox2lid/lid_summary.json", "Experiment/VOX2-01/feats_vox2/face_meta.csv",
        "Experiment/VOX2-01/feats_vox2/voice_meta.csv"]
AV = [f"ext_{s}_{k}_{n}.npy" for s in ("v1_complete", "v2_complete") for k, n in
      [("face", "aclip"), ("face", "ibalign"), ("face", "lbind"), ("face", "rows"), ("voice", "aclip"), ("voice", "lbind"),
       ("voice", "rows")]] + ["face_aclip_train.npy", "face_ibalign_train.npy", "face_lbind_train.npy", "voice_aclip_train.npy",
                              "voice_lbind_train.npy"]
EVAL = [f"ext_{s}_{k}.npy" for s in ("v1_complete", "v2_complete") for k in ("face_arcface", "voice_ecapa192")] + \
       ["face_ibv_train.npy", "voice_w2v2aggen_train.npy"]
MODELS = [f"ext_{s}_{k}.npy" for s in ("v1_complete", "v2_complete") for k in
          ("face_adaface", "face_farl", "face_siglip2", "face_vitage", "voice_w2v2ag")] + \
         ["face_adaface_train.npy", "face_siglip2_train.npy", "face_vitage_train.npy", "voice_w2v2ag_train.npy"]
DATA += [f"kaggle/output/feats_av/{n}" for n in AV] + [f"kaggle/output/feats_eval/{n}" for n in EVAL] + \
        [f"kaggle/output/feats_models/{n}" for n in MODELS]
VB_FILES = [f"Experiment/SCALE-VB/feats_vb/{n}" for n in ("face_farl16.npy", "face_clipL.npy", "voice_sam100p.npy", "voice_sam100f.npy",
                                                       "face_meta.csv", "voice_meta.csv", "lid_summary.json", "feats_manifest.json")]
PLACEHOLDERS = ["kaggle/output/feats_eval/voice_ecapa6144_train.npy", "kaggle/output/feats_eval/ext_v1_complete_voice_ecapa6144.npy",
                "kaggle/output/feats_eval/ext_v2_complete_voice_ecapa6144.npy"]

def build_vb():
    """VoxBlink-clean features (FLAG_46; CC BY-NC-SA 4.0, no redistribution: keep the dataset PRIVATE)."""
    out = R / "kaggle/flag2027-lab-vb.zip"
    missing = [f for f in VB_FILES if not (R / f).exists()]
    assert not missing, missing
    man = {"created": time.strftime("%Y-%m-%d %H:%M"), "kind": "data-vb", "files": {f: (R / f).stat().st_size for f in VB_FILES},
           "note": "FourFusion private: VoxBlink-clean features (FLAG_46), CC BY-NC-SA 4.0, no redistribution."}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in VB_FILES:
            z.write(R / f, f"labvb/{f}")
        z.writestr("labvb/lab_manifest.json", json.dumps(man, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB): {len(VB_FILES)} files ({sum(man['files'].values()) / 1e6:.0f} MB raw)")


E6_FILES = ["voice_ecapa6144_train.npy", "ext_v1_complete_voice_ecapa6144.npy", "ext_v2_complete_voice_ecapa6144.npy"]


def build_e6144():
    """Real ECAPA-6144 (organiser-pipeline FLAG_12 output) under kaggle/output/feats_e6144/ (the lab-data bundle only has
    (rows, 0) placeholders in feats_eval/). Used by ENBASE-01's s010B analog."""
    out = R / "kaggle/flag2027-lab-e6144.zip"
    man = {"created": time.strftime("%Y-%m-%d %H:%M"), "kind": "data-e6144", "files": {}, "shapes": {},
           "note": "FourFusion private: ECAPA-6144 of the organiser data (v4 train) and MAV-Celeb v1/v2; do not redistribute."}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in E6_FILES:
            src = FO / "feats_eval" / f; a = np.load(src, mmap_mode="r")
            assert a.ndim == 2 and a.shape[1] == 6144 and np.isfinite(np.asarray(a[:1000])).all(), (f, a.shape)
            z.write(src, f"labe6144/kaggle/output/feats_e6144/{f}"); man["files"][f] = src.stat().st_size; man["shapes"][f] = list(a.shape)
        z.writestr("labe6144/lab_manifest.json", json.dumps(man, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB): {man['shapes']}")


if __name__ == "__main__" and sys.argv[1:] == ["vb"]:
    build_vb()
elif __name__ == "__main__" and sys.argv[1:] == ["e6144"]:
    build_e6144()
elif __name__ == "__main__":
    files = list(DATA)
    for d, pat in DATA_DIRS:
        files += [str(p.relative_to(R)).replace("\\", "/") for p in sorted((R / d).glob(pat))]
    missing = [f for f in files + PLACEHOLDERS if not (R / f).exists()]
    assert not missing, missing
    man = {"created": time.strftime("%Y-%m-%d %H:%M"), "eda_bundle_created": "2026-09-29 08:06", "files": {}, "placeholders": {},
           "kind": "data", "note": "FourFusion private lab data for kaggle/FLAG_48_lab.ipynb; organiser features, do not redistribute."}
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(R / f, f"lab/{f}"); man["files"][f] = (R / f).stat().st_size
        for f in PLACEHOLDERS:
            n = len(np.load(R / f, mmap_mode="r"))
            b = io.BytesIO(); np.save(b, np.zeros((n, 0), np.float32)); z.writestr(f"lab/{f}", b.getvalue())
            man["placeholders"][f] = n
        z.writestr("lab/lab_manifest.json", json.dumps(man, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB): {len(files)} files + {len(PLACEHOLDERS)} placeholders "
          f"({sum(man['files'].values()) / 1e6:.0f} MB raw)")
