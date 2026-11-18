"""FLAG_189 (14/11): FVBIND-01 (Experiment/FVBIND-01/README.md, registered before any number) = FLAG_163's setup / inputs cells
(RFS_MODE=LAB: bundle, lid_strict, SAFE-04 fold tables, lora_C1 / v4_extra) + one cell running Experiment/FVBIND-01/fvbind.py
(embedded next to rfs_train.py) on the arms of FVB_ARMS. Output: fvb<arm>_fold<s>.parquet, fvb<arm>_emb.npz, fvb<arm>_state.pt.
    python kaggle/build_flag189.py   -> kaggle/FLAG_189_fvbind.ipynb
Push: bash kaggle/sat_push.sh 189 (ACC, ARMS=B0,B1 | IB0,B2; SMOKE=1 synthetic on Kaggle CPU; SMOKE=real = real data + stand-in model)."""
import base64, gzip
from pathlib import Path
import build_flag163 as B163

HERE = Path(__file__).resolve().parent
FVB = HERE.parent / "Experiment" / "FVBIND-01" / "fvbind.py"

INTRO = """
# FLAG 2027 - NB-189: FVBIND-01, ImageBind audio tower re-trained the ImageBind way (big batch, identity face anchor), GPU

**What (Experiment/FVBIND-01/README.md, registered 14/11 before any number).** LoRA C1 recipe on the ImageBind-huge audio tower,
trained only on the auxiliary VoxCeleb2 + VoxBlink persons of the RFS pool (clean_ext / lid_strict, BN42-clean audio cache):
B0 = LoRA C1 recipe (64 persons), B1 = 2 048 persons per step (GradCache), B2 = B1 + anchor [C5s | FaRL] + linear voice head,
IB0 = frozen ImageBind (reference). Standalone stream on the SAFE-04 fold tables; gate 1 is read locally (`run.py gate`).
No dev, no CB, no Evaluation, no MAV / v4 in training.
"""

FVBCELL = '''print("[cell] FVBIND-01: train arms + SAFE-04 streams", flush=True)
p = CODE / "train" / "fvbind.py"; p.write_bytes(gzip.decompress(base64.b64decode("__FVB__")))
STANDIN = os.environ.get("FVB_STANDIN") == "1"
FOUT = W / "fvb_out"; FOUT.mkdir(exist_ok=True)
ck, cache, mav, v4 = "", "", "", ""
if not SMOKE:
    if not STANDIN:
        for c in ("pip -q install iopath ftfy regex fvcore", "pip -q install --no-deps git+https://github.com/facebookresearch/ImageBind"):
            print(subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-300:], flush=True)
        ck = W / "imagebind_huge.pth"
        if not ck.exists():
            torch.hub.download_url_to_file("https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth", str(ck) + ".part", progress=False); os.replace(str(ck) + ".part", ck)
        import hashlib
        h = hashlib.sha256()
        with open(ck, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
        assert h.hexdigest().startswith("d6f6c22b"), "ImageBind checkpoint SHA-256 mismatch"; print("imagebind sha256 ok", flush=True)
    tars = [q for q in find_all("tar0000.npz") if q.parent.name == "vox"]; assert tars, "thieu audio cache (duongphihoang/flag2027-audio-cache-p)"
    cache = tars[0].parent.parent; assert (cache / "vb").is_dir(), "audio cache without vb/"
    for d, dirs, files in os.walk(IN):
        if Path(d).name in ("v1", "v2") and {"faces", "voices"} <= set(dirs): mav = Path(d).parent; break
        dirs[:] = [x for x in dirs if x not in ("faces", "voices")] if len(Path(d).relative_to(IN).parts) < 7 else []
    assert mav, "thieu MAV raw (chnnhnnguyn/external-data-mavceleb)"
    for d, dirs, files in os.walk(IN):
        if "train_English.txt" in files and Path(d, "faces").is_dir(): v4 = Path(d); break
        dirs[:] = [x for x in dirs if x not in ("faces", "voices")] if len(Path(d).relative_to(IN).parts) < 7 else []
    assert v4, "thieu raw-v4 (ngnnguynchn/flag2027-raw-v4-p)"
t0 = time.time()
r = subprocess.Popen([sys.executable, "-u", str(p), "--bundle", str(BUNDLE), "--work", str(WORK), "--out", str(FOUT), "--lid", LID,
                      "--tables", str(TABDIR), "--ib_ckpt", str(ck), "--cache", str(cache), "--mav", str(mav), "--v4root", str(v4),
                      "--arms", os.environ.get("FVB_ARMS", "B0,B1")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in r.stdout: print(line, end="", flush=True)
assert r.wait() == 0, "fvbind.py failed"
print(f"FVBIND done [{time.time() - t0:.0f}s]:", sorted(q.name for q in FOUT.iterdir()), flush=True)
'''


def cells():
    out = B163.cells("LAB")
    i = [k for k, (t, s) in enumerate(out) if t == "code" and 'run(["lab"' in s]
    assert len(i) == 1, "FLAG_163 PIPE cell not found"
    out[i[0]] = ("code", FVBCELL.replace("__FVB__", base64.b64encode(gzip.compress(FVB.read_bytes(), 9)).decode()))
    out[0] = ("markdown", INTRO)
    return out


if __name__ == "__main__":
    B163.B12.nb(cells(), "FLAG_189_fvbind.ipynb"); print("FLAG_189_fvbind.ipynb built")
