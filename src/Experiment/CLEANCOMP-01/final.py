r"""CLEANCOMP-01 final artefacts (README section 4), LID_STRICT=1, run by lab_entry.py CLEANCOMP-01 with CC_MODE=final:
  s007_xall.pt                 the S6 bridge member (safe_lib.train_xall = the training path of BEST/v7_safe/pipeline.py) with the MAV voice
                               files of lid_strict.json 'mav_files' left out of its training rows (safe_lib.load_ext, opt-in)
  v11_head/v11_t3head.pt       v11's T3 head (E2E/v11_head.py train: BIGB-01 P512, 3 nets) on the VoxCeleb2 rows of the LIDSTRICT-01 list
  v11_head/t3_streams.npz      its three streams [anP512_deep, anP512_cca, t3_relg] on the four dev files (E2E/v11_head.py score)
  final_info.json              counts for the A0 audit
The dev components of F1s (S6 b3 + t3 columns) are then rebuilt on the local machine from these files (build_cleancomp_dev.py)."""
import os, sys, json, time, importlib.util
from pathlib import Path
import numpy as np

os.environ["SAFE04_NOMAIN"] = "1"
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
TAGS = ["gender_Bangla", "gender_English", "no_gender_Bangla", "no_gender_English"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    assert os.environ.get("LID_STRICT") == "1" and os.environ.get("VOX_LID"), "CLEANCOMP-01 final runs under LID_STRICT=1 (+ VOX_LID)"
    import torch
    out = Path(os.environ["CC_OUT"]); out.mkdir(parents=True, exist_ok=True)
    A = _load("allneg01run", "ALLNEG-01/run.py")
    import rel_sweep5 as R
    sys.path.insert(0, str(EXP.parent / "BEST/v7_safe"))
    import safe_lib as S
    t0 = time.time()
    if os.environ.get("CC_SMOKE") == "1":                      # smoke only: tiny XALL (safe_lib constants), not a result
        S.STEPS, S.N_NETS = int(os.environ.get("L0_STEPS", "10")), int(os.environ.get("L0_NETS", "1"))
    info = dict(created=time.strftime("%Y-%m-%d %H:%M"), lid_strict=True)
    # 1. S6 bridge member
    IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
    dirs = [next(IN.rglob("eda_bundle_manifest.json")).parent, EXP.parent / "kaggle/output/feats_models"]

    def find(name, required=True):
        for d in dirs:
            for f in (Path(d) / name, *Path(d).glob(f"*/{name}")):
                if f.exists():
                    return f
        assert not required, name
        return None
    ext = S.load_ext(find)                                   # LID_STRICT: MAV_DROP voice rows already left out
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mem = S.train_xall(R.V4["vgg"], R.V4["btc"], R.V4S, ext, log=R.log, device=dev)
    S.save_xall(mem, out / "s007_xall.pt")
    info["xall"] = dict(voice_rows_left=int(len(ext[1])), device=str(dev), persons=len(set(ext[1].p) - S.OVERLAP))
    R.log(f"-> s007_xall.pt [{time.time() - t0:.0f}s]")
    # 2. v11's T3 head + its dev streams
    H = _load("v11head", "E2E/v11_head.py")
    hd = out / "v11_head"; hd.mkdir(exist_ok=True)
    H.train(hd)
    st = out / "_feats_stage"; st.mkdir(exist_ok=True)
    # the dev T3 inputs (raw face / voice T3 arrays + w2v2 gender of every trial) travel in flag2027-e2e-dev-p's components_dev.npz
    cz = next(Path("/kaggle/input").rglob("components_dev.npz")) if Path("/kaggle/input").exists() else next(IN.rglob("components_dev.npz"))
    man = json.loads((cz.parent / "e2e_manifest.json").read_text()); z = np.load(cz, allow_pickle=False)
    for f in man["files"]:
        tag = f["key"].replace("/", "_"); assert tag in TAGS, tag
        for n in ("face_farl16", "face_clipL", "voice_sam100p", "voice_sam100f"):
            np.save(st / f"{n}_{tag}.npy", z[f"{n}_{f['i']}"])
        np.save(st / f"voice_w2v2aggen_{tag}.npy", z[f"vgen_{f['i']}"])
    (st / "t3_ready.json").write_text(json.dumps({"splits": TAGS}))
    H.score(hd, st, TAGS, wait_h=0.01)
    import shutil; shutil.rmtree(st)
    ck = torch.load(hd / "v11_t3head.pt", map_location="cpu", weights_only=False)["info"]
    info["head"] = {k: ck[k] for k in ("identities", "vox_ids", "rows", "steps", "nets", "device", "torch", "export_check_max_abs_diff")}
    info["head"]["dropped_n"] = len(ck["dropped"])
    (out / "final_info.json").write_text(json.dumps(info, indent=1))
    R.log(f"CLEANCOMP-01 final done [{time.time() - t0:.0f}s] {json.dumps(info)}")


if __name__ == "__main__":
    main()
