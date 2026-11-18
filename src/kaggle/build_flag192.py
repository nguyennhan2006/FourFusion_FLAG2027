"""FLAG_192 (14/11): FVBIND-01 gate 2, step 1 (Experiment/FVBIND-01/README.md section 3 / 3b) = FLAG_163's setup / inputs cells
(RFS_MODE=LAB) + one cell running Experiment/FVBIND-01/fvb_extract.py (embedded next to fvbind.py and rfs_train.py): block v_fvb of
every bundle voice row, every v4 voice row and every voice of the four Evaluation lists, from the B2 state of FLAG_189.
Inputs besides FLAG_189's: kernel outputs thanhhuyen11/flag-2027-nb-189-fvbind-b2-ib0 (fvbB2_state.pt, fvbB2_emb.npz) and
thanhhuyen11/flag-2027-nb-47rs-screen2-eval (index_<proto>_<lang>.txt of the Evaluation lists), dataset ngnnguynchn/flag2027-evalphase-p.
Output fvbx_out/ (layout of rfs_train.XBLK '_AV' + eval/), read by FLAG_193 (lab), FLAG_163 P and FLAG_163i (Evaluation path).
    python kaggle/build_flag192.py   -> kaggle/FLAG_192_fvbx.ipynb
Push: bash kaggle/sat_push.sh 192 (ACC=thanhhuyen11; SMOKE=1 synthetic; SMOKE=real = real data + stand-in model, CPU)."""
import base64, gzip
from pathlib import Path
import build_flag163 as B163

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "Experiment" / "FVBIND-01"

INTRO = """
# FLAG 2027 - NB-192: FVBIND-01 gate 2, block v_fvb (B2 voice embedding) for the bundle, v4 and the Evaluation lists (GPU)

**What (Experiment/FVBIND-01/README.md section 3b, registered 14/11 before any gate-2 number).** B2 of FLAG_189 (ImageBind audio + LoRA,
anchor [C5s | FaRL], linear head) passed gate 1. This notebook writes v_fvb = head(ImageBind_B2(voice)) - mu_v for every voice row the
RFX lab / final projector reads and for the voices of the Evaluation lists. No training, no dev, no CB.
"""

XCELL = '''print("[cell] FVBIND-01 gate 2: extract v_fvb", flush=True)
SRC = __SRC__
for nm in ("fvbind.py", "fvb_extract.py"):
    (CODE / "train" / nm).write_bytes(gzip.decompress(base64.b64decode(SRC[nm])))
p = CODE / "train" / "fvb_extract.py"
STANDIN = os.environ.get("FVB_STANDIN") == "1"
FOUT = W / "fvbx_out"; FOUT.mkdir(exist_ok=True)
ck = cache = v4 = state = emb = evidx = evroot = ""
def _walk_files(pred, skip=("faces", "voices")):
    hits = []
    for d, dirs, files in os.walk(IN):
        hits += [Path(d) / f for f in files if pred(Path(d) / f)]
        dirs[:] = [x for x in dirs if x not in skip] if len(Path(d).relative_to(IN).parts) < 8 else []
    return hits
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
        st = _walk_files(lambda q: q.name in ("fvbB2_state.pt", "fvbB2_emb.npz"))
        state = [q for q in st if q.name == "fvbB2_state.pt"]; emb = [q for q in st if q.name == "fvbB2_emb.npz"]
        assert state and emb, "attach kernel output thanhhuyen11/flag-2027-nb-189-fvbind-b2-ib0"; state, emb = state[0], emb[0]
        print("B2 state", state, flush=True)
    tars = [q for q in find_all("tar0000.npz") if q.parent.name == "vox"]; assert tars, "thieu audio cache (duongphihoang/flag2027-audio-cache-p)"
    cache = tars[0].parent.parent; assert (cache / "vb").is_dir(), "audio cache without vb/"
    for d, dirs, files in os.walk(IN):
        if "train_English.txt" in files and Path(d, "faces").is_dir(): v4 = Path(d); break
        dirs[:] = [x for x in dirs if x not in ("faces", "voices")] if len(Path(d).relative_to(IN).parts) < 7 else []
    assert v4, "thieu raw-v4 (ngnnguynchn/flag2027-raw-v4-p)"
    ix = _walk_files(lambda q: q.name.startswith("index_") and q.suffix == ".txt" and ("47rs" in q.as_posix() or "fvbx-in" in q.as_posix()))
    assert ix, "attach kernel output thanhhuyen11/flag-2027-nb-47rs-screen2-eval (or dataset *flag2027-fvbx-in-p)"; evidx = ix[0].parent
    ev = [Path(d) for d, dirs, files in os.walk(IN) if Path(d).name == "flag2027-evalphase-p"]
    assert ev, "attach dataset ngnnguynchn/flag2027-evalphase-p"; evroot = ev[0]
    print("eval index", evidx, sorted(q.name for q in ix), "| eval root", evroot, flush=True)
t0 = time.time()
r = subprocess.Popen([sys.executable, "-u", str(p), "--bundle", str(BUNDLE), "--out", str(FOUT), "--ib_ckpt", str(ck), "--cache", str(cache),
                      "--mavbase", str(IN), "--v4root", str(v4), "--state", str(state), "--emb", str(emb), "--evidx", str(evidx),
                      "--evroot", str(evroot)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in r.stdout: print(line, end="", flush=True)
assert r.wait() == 0, "fvb_extract.py failed"
print(f"FVBX done [{time.time() - t0:.0f}s]:", sorted(str(q.relative_to(FOUT)) for q in FOUT.rglob("*.np*")), flush=True)
'''


def cells():
    out = B163.cells("LAB")
    i = [k for k, (t, s) in enumerate(out) if t == "code" and 'run(["lab"' in s]
    assert len(i) == 1, "FLAG_163 PIPE cell not found"
    src = {nm: base64.b64encode(gzip.compress((EXP / nm).read_bytes(), 9)).decode() for nm in ("fvbind.py", "fvb_extract.py")}
    out[i[0]] = ("code", XCELL.replace("__SRC__", repr(src)))
    out[0] = ("markdown", INTRO)
    return out


if __name__ == "__main__":
    B163.B12.nb(cells(), "FLAG_192_fvbx.ipynb"); print("FLAG_192_fvbx.ipynb built")
