"""SET-05 part B extraction: clustering-candidate voice embeddings for NON-English MAV-Celeb clips (true labels).
Sample: <=15 Urdu (v1) / Hindi (v2) clips per speaker, taken from the rows already embedded in organiser space
(ext_<src>_voice_meta.csv) so every clip also has its organiser-192 bridge embedding. Same 12 s centre crop."""
import sys, io, time, zipfile, numpy as np, pandas as pd, torch
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(10)
import flag_extract as X
D = Path(r"D:/Sinh viên CNhan/download/04092026"); FE = R / "kaggle/output/feats_ext"
OUT = Path(__file__).parent / "feats_b"; OUT.mkdir(exist_ok=True)
SRC = {"v1_complete": ("mavceleb_v1_complete-001.zip", "urdu"), "v2_complete": ("mavceleb_v2_complete-002.zip", "hindi")}
crop = lambda w: w if len(w) <= 12 * 16000 else w[(len(w) - 12 * 16000) // 2:(len(w) - 12 * 16000) // 2 + 12 * 16000]
ecapa = X.Ecapa(R / "kaggle/output/_ecapa_sb")                 # speechbrain spkrec-ecapa-voxceleb (192 + 6144)
rdn = X.load_redimnet2("b6", "vox2")
for src, (zf, lang) in SRC.items():
    if (OUT / f"{src}_rdn6vox.npy").exists(): print("skip", src); continue
    m = pd.read_csv(FE / f"ext_{src}_voice_meta.csv"); m["row"] = np.arange(len(m))
    m = m[m.lang == lang]
    rng = np.random.RandomState(0); keep = []
    for spk_, idx in m.groupby("spk").groups.items():
        idx = np.asarray(idx)
        if len(idx) >= 5:
            keep += list(rng.choice(idx, min(len(idx), 15), replace=False))
    m = m.loc[sorted(keep)].reset_index(drop=True)
    z = zipfile.ZipFile(D / zf); t = time.time(); E, Rv = [], []
    for i, p in enumerate(m.path):
        w = crop(X.load_audio(io.BytesIO(z.read(p))))
        E.append(ecapa([w])[0][0])
        with torch.no_grad():
            e = rdn(torch.from_numpy(np.ascontiguousarray(w)).float()[None]); e = e[0] if isinstance(e, (tuple, list)) else e
        Rv.append(e.reshape(-1).numpy())
        if i % 50 == 0: print(src, i, len(m), f"{time.time() - t:.0f}s", flush=True)
    np.save(OUT / f"{src}_ecapa192.npy", np.stack(E)); np.save(OUT / f"{src}_rdn6vox.npy", np.stack(Rv))
    m.to_csv(OUT / f"{src}_meta.csv", index=False); print(src, "done", len(m), flush=True)
print("ALL DONE", flush=True)
