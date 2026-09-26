"""Retrain the recipes behind the current submission locally (CPU) and save full face x voice matrices.
  s007  : EXP-007 recipe, organiser 192 voice            (English member)
  s010B : EXP-007 recipe, ECAPA-6144 voice               (English member)
  r2mix : EXP-007 recipe, own ECAPA-192 + Bangla crops   (Bangla member)
n_models=5 (submissions used 10) to keep CPU time sane; fidelity is checked against the submitted zips."""
import os, sys, time, torch
K = r"D:/Sinh viên CNhan/FLAG_2027_FourFusion/kaggle"
sys.path.insert(0, K); torch.set_num_threads(10)
import flag_v2 as V
store = V.FeatureStore(K + "/../kaggle_upload", feats_dir=K + "/output/feats_v2")
REC = dict(head="mlp", drop=0.3, emb=128, epochs=40)
for name, cfg in {"s007": dict(REC), "s010B": dict(REC, voice="ecapa6144"),
                  "r2mix": dict(REC, voice="ecapa192", voice_view="mix")}.items():
    out = f"mats/{name}_matrix.npz"
    if os.path.exists(out): print("skip", out); continue
    t = time.time(); V.dev_scores(store, cfg, n_models=5, cca_w=0.25, matrix_path=out)
    print(name, f"{time.time() - t:.0f}s", flush=True)
