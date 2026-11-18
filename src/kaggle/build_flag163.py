"""Generate FLAG_163_rfs.ipynb (RFS-01 buoc 2, Experiment/RFS-01/README.md '## Buoc 2: hoc lai', nhat ky 361).
Mot notebook, hai luot (bien RFS_MODE):
  L : LoRA C1 tren thap audio ImageBind-huge (+ lora / tuoi cho cac hang giong v4)  -> dataset `RFS_UPLOAD` (lora_C1.pt, v4_extra.npz)   GPU, Internet
  P : split + oof (projector tung fold) + projector cuoi + bo tron + release + FinalScorer tu kiem -> artifact (final_params.npz, projector_Q.pt, lora_C1.pt, manifest_final.json)  GPU
Dau vao bundle (ext.npz, v4.npz) theo 'Hop dong bundle' cua README; lid_strict.json; (L) audio cache, MAV raw, raw-v4, (P) dataset cua luot L.
    python build_flag163.py
Smoke Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;RFS_STEPS=40;RFS_NTRIAL=150   (bundle gia, lora gia; chay het P)"""
import base64, gzip
from pathlib import Path

import build_eval_notebooks as B12

HERE = Path(__file__).resolve().parent
RFS = HERE.parent / "Experiment" / "RFS-01"


def b64(p):
    return base64.b64encode(gzip.compress(Path(p).read_bytes(), 9)).decode()


INTRO = """
# FLAG 2027 - NB-163: RFS-01 buoc 2, hoc lai cong thuc RF-092 tren kho nguoi sach cua nhom

Code hoc viet lai theo bao cao RF (khong dung artifact RF). `RFS_MODE=L`: LoRA C1 ImageBind-audio. `RFS_MODE=P`: projector QV-Whisper (3 seed) theo fold + cuoi,
luong t3 / ib / nf / nv / age, bo tron g va ng, artifact nap boi `FinalScorer` cua flag_final_core_092 (khong sua core). Luat doc CB: README `Experiment/RFS-01/`.
"""

SETUP = '''print("[cell] setup: files + deps", flush=True)
import os, sys, json, gzip, base64, subprocess, shutil, time
from pathlib import Path
MODE = os.environ.get("RFS_MODE", "__DEFMODE__").upper(); SMOKE = os.environ.get("FLAG_SMOKE") == "1"
W = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./rfs_work"); W.mkdir(exist_ok=True)
CODE = W / "rfs"
for rel, blob in (("train/assemble.py", "__ASM__"), ("train/gender.csv", "__GEN__"), ("train/rfs_train.py", "__TRAIN__"), ("ref/flag_final_core_092.py", "__CORE__"), ("ref/manifest_final.json", "__MAN__")):
    p = CODE / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(gzip.decompress(base64.b64decode(blob)))
TRAIN = CODE / "train" / "rfs_train.py"
def run(args, env=None):
    """chay rfs_train.py, chuyen stdout tung dong ra log"""
    p = subprocess.Popen([sys.executable, "-u", str(TRAIN), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, **(env or {})})
    for line in p.stdout: print(line, end="", flush=True)
    assert p.wait() == 0, "rfs_train.py failed"
if MODE == "L" and not SMOKE:
    for c in ("pip -q install iopath ftfy regex fvcore", "pip -q install --no-deps git+https://github.com/facebookresearch/ImageBind"):
        print(subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-300:], flush=True)
import torch
print("mode", MODE, "smoke", SMOKE, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count(), flush=True)
'''

INPUTS = '''print("[cell] inputs: bundle, lid, roots", flush=True)
IN = Path("/kaggle/input")
def find_all(name):
    return sorted(p for p in IN.rglob(name)) if IN.exists() else []
WORK = W / "work"; WORK.mkdir(exist_ok=True); ART = W / "rfs_art"; BUNDLE = None; LID = ""
if MODE == "B":      # ghep FLAG_162 (a, b, c) + FLAG_160 (whv3 a, b, c) thanh bundle (CPU, vai phut)
    def dsdir(slug_part):
        r = [p for p in IN.rglob("*") if p.is_dir() and p.name.startswith("flag2027-feats-") and slug_part in p.name]
        assert r, f"thieu dataset *{slug_part}*"; return r[0]
    Q = lambda n: str(find_all(n)[0].parent)
    BUNDLE = W / "rfs_bundle"
    cmd = [str(CODE / "train" / "assemble.py")]
    p = subprocess.Popen([sys.executable, "-u", *cmd, "--out", str(BUNDLE), "--rfs_a", str(dsdir("rfs-a")), "--rfs_b", str(dsdir("rfs-b")), "--rfs_c", str(dsdir("rfs-c")),
                          "--wh_a", str(dsdir("whv3-a")), "--wh_b", str(dsdir("whv3-b")), "--wh_c", str(dsdir("whv3-c")), "--extmeta", Q("ext_v1_complete_face_meta.csv"),
                          "--rawv4", Q("train_English.txt"), "--gender_csv", str(CODE / "train" / "gender.csv")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout: print(line, end="", flush=True)
    assert p.wait() == 0, "assemble failed"
elif SMOKE:
    BUNDLE = W / "bundle_synth"; run(["synth", "--bundle", str(BUNDLE)]); LID = str(BUNDLE / "lid_strict.json")
    if MODE == "LAB":
        TABDIR = W / "lab_tables_synth"; run(["synthlab", "--bundle", str(BUNDLE), "--work", str(TABDIR)])
    torch.save({"params": [torch.zeros(2)]}, WORK / "lora_C1.pt")          # smoke: lora gia (luot L that chay tren Kaggle GPU)
elif True:
    cand = [p.parent for p in find_all("ext.npz") if (p.parent / "v4.npz").exists()]
    assert cand, "thieu bundle (ext.npz + v4.npz): gan dataset dac trung RFS-01 (FLAG_162 + adapter, README 'Hop dong bundle')"
    BUNDLE = cand[0]; l = sorted(find_all("lid_strict.json"), key=lambda q: "lid-strict" not in str(q));   # 13/11: the lid-strict dataset first (other inputs may hold a copy)
    assert l, "thieu lid_strict.json (ngnnguynchn/flag2027-lid-strict-p)"; LID = str(l[0])
    if MODE == "LAB":   # RFS-LAB-01: SAFE-04 pair tables of F1s (ngnnguynchn/flag2027-ct02s-r2ca-tables-p)
        t = find_all(os.environ.get("RFS_LAB_GLOB", "ct01r2_fold*.parquet").replace("*", "1")); assert t, "thieu bang fold SAFE-04 (flag2027-ct02s-r2ca-tables-p)"; TABDIR = t[0].parent
    if MODE in ("P", "LAB"):
        for n in ("lora_C1.pt", "v4_extra.npz"):
            f = find_all(n); assert f, f"thieu {n}: gan dataset dau ra cua luot L"; shutil.copy(f[0], WORK / n)
print("bundle", BUNDLE, "| lid", LID, flush=True)
'''

LORA = '''print("[cell] lora C1 (ImageBind-audio) + v4 lora/tuoi", flush=True)
if MODE == "L":
    import torch as _t
    ck = W / "imagebind_huge.pth"
    if not ck.exists():
        _t.hub.download_url_to_file("https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth", str(ck) + ".part", progress=False); os.replace(str(ck) + ".part", ck)
    tars = [p for p in find_all("tar0000.npz") if p.parent.name == "vox"]; assert tars, "thieu audio cache (duongphihoang/flag2027-audio-cache-p)"
    mav = None
    for d, dirs, files in os.walk(IN):
        if Path(d).name in ("v1", "v2") and {"faces", "voices"} <= set(dirs): mav = Path(d).parent; break
        dirs[:] = [x for x in dirs if x not in ("faces", "voices")] if len(Path(d).relative_to(IN).parts) < 7 else []
    assert mav, "thieu MAV raw (chnnhnnguyn/external-data-mavceleb)"
    v4 = None
    for d, dirs, files in os.walk(IN):
        if "train_English.txt" in files and Path(d, "faces").is_dir(): v4 = Path(d); break
        dirs[:] = [x for x in dirs if x not in ("faces", "voices")] if len(Path(d).relative_to(IN).parts) < 7 else []
    assert v4, "thieu raw-v4 (ngnnguynchn/flag2027-raw-v4-p)"
    run(["lora", "--bundle", str(BUNDLE), "--work", str(WORK), "--lid", LID, "--ib_ckpt", str(ck), "--cache", str(tars[0].parent.parent), "--mav", str(mav), "--v4root", str(v4)])
    OUTD = W / "rfs_lora"; OUTD.mkdir(exist_ok=True)
    for n in ("lora_C1.pt", "v4_extra.npz"): shutil.copy(WORK / n, OUTD / n)
else:
    print("mode P: lora tu dataset cua luot L", flush=True)
'''

PIPE = '''print("[cell] split + oof + proj + fuse + release + selfcheck", flush=True)
if MODE == "P":
    t0 = time.time()
    run(["all", "--bundle", str(BUNDLE), "--work", str(WORK), "--art", str(ART), "--lid", LID, "--lora", str(WORK / "lora_C1.pt")])
    print(f"P done [{time.time() - t0:.0f}s]; artifact:", sorted(p.name for p in ART.iterdir()), flush=True)
    print((ART / "fuse_report.json").read_text(), flush=True)
if MODE == "LAB":   # RFS-LAB-01 (Experiment/EDA-3FAM-01): RFS streams on the SAFE-04 folds, purge per fold
    t0 = time.time()
    run(["lab", "--bundle", str(BUNDLE), "--work", str(WORK), "--art", str(ART), "--lid", LID], env={"RFS_LAB_TABLES": str(TABDIR)})
    print(f"LAB done [{time.time() - t0:.0f}s]:", sorted(p.name for p in ART.iterdir()), flush=True)
'''

UP = '''print("[cell] upload", flush=True)
UPL = os.environ.get("RFS_UPLOAD", "")
OUT = {"L": W / "rfs_lora", "B": W / "rfs_bundle"}.get(MODE, ART)
if SMOKE:
    print("smoke: no upload; artifact", sorted(p.name for p in OUT.iterdir()) if OUT.exists() else None, flush=True)
elif UPL:
    try:
        import kagglehub
        kagglehub.dataset_upload(UPL, str(OUT), version_notes=f"RFS-01 FLAG_163 mode {MODE}")
        print("uploaded ->", UPL, flush=True)
    except Exception as e:
        print(f"upload to {UPL} failed ({e!r}); files are in the notebook output", flush=True)
print("ALL CELLS OK", flush=True)
'''


def cells(defmode="P"):
    setup = (SETUP.replace("__DEFMODE__", defmode).replace("__TRAIN__", b64(RFS / "train" / "rfs_train.py")).replace("__CORE__", b64(RFS / "ref" / "flag_final_core_092.py"))
             .replace("__MAN__", b64(RFS / "ref" / "manifest_final.json")).replace("__ASM__", b64(RFS / "train" / "assemble.py"))
             .replace("__GEN__", b64(HERE.parent / "Input" / "meta_file_train_set.csv")))
    out = [("markdown", INTRO), ("code", setup), ("code", INPUTS), ("code", LORA), ("code", PIPE), ("code", UP)]
    for _, s in [c for c in out if c[0] == "code"]:
        assert s.lstrip().startswith('print("[cell]'), s[:40]
    return out


if __name__ == "__main__":
    B12.nb(cells(), "FLAG_163_rfs.ipynb")
    B12.nb(cells("LAB"), "FLAG_170_rfslab.ipynb")   # RFS-LAB-01 / EDA-3FAM-01 (5/11): same code, RFS_MODE defaults to LAB
