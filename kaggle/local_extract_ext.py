"""Local (CPU) extraction of MAV-Celeb v1/v2 with the organiser encoders, capped to what training uses:
<=150 utterances per speaker (round-robin over language x video), clips centre-cropped to 12 s, 8 face frames/video.
Resumable: finished arrays are skipped."""
import time, zipfile, torch, pandas as pd
from pathlib import Path
torch.set_num_threads(10)
import flag_extract as X
D = Path(r"D:/Sinh viên CNhan/download/04092026")
OUT = Path(r"D:/Sinh viên CNhan/FLAG_2027_FourFusion/kaggle/output/feats_ext"); OUT.mkdir(parents=True, exist_ok=True)
X.write_speaker_meta(D / "mavceleb_v1_complete-001.zip", "v1_complete", OUT)
ve = X.OrganiserVoice(Path(r"D:/Sinh viên CNhan/FLAG_2027_FourFusion/kaggle/output/_ecapa_btc"))
fe = X.VGGFace(Path(r"C:/Users/ASUS/AppData/Local/Temp/vggface"))
for name, f in [("v1_complete", "mavceleb_v1_complete-001.zip"), ("v2_complete", "mavceleb_v2_complete-002.zip")]:
    t = time.time()
    X.extract_external(D / f, name, OUT, ve, fe, frames_per_video=8, max_bs=16, max_wav_per_spk=150, max_sec=12)
    print(name, f"{time.time() - t:.0f}s", flush=True)
X.write_manifest(OUT, dict(sources=["v3_train", "v1_complete", "v2_complete"], where="local CPU",
                           cap="150 wav/speaker, 12 s centre crop, 8 frames/video"))
print("ALL DONE", flush=True)
