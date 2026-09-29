"""SET-04 members: EXP-007 recipe with n=10 models (as submitted), full face x voice matrices."""
import sys, time, torch
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(10)
import flag_v2 as V
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(head="mlp", drop=0.3, emb=128, epochs=40)
for name, cfg in {"s007": dict(REC), "r2mix": dict(REC, voice="ecapa192", voice_view="mix"), "s010B": dict(REC, voice="ecapa6144")}.items():
    out = Path(f"mats10/{name}_matrix.npz")
    if out.exists(): print("skip", out); continue
    t = time.time(); V.dev_scores(store, cfg, n_models=10, cca_w=0.25, matrix_path=out); print(name, f"{time.time() - t:.0f}s", flush=True)
