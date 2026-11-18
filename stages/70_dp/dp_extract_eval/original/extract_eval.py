"""XJF-01 step 1: recognition-grade features of the Evaluation files, with the SAME code that made kaggle/output/feats_v2,
feats_eval (ArcFace buffalo_l, speechbrain ECAPA 192 / 6144) and feats_models (AdaFace IR-101).
    python Experiment/XJF-01/extract_eval.py <eval_set.zip> <work dir for the unzipped media>
Output: Experiment/XJF-01/feats_evalphase/{face_arcface,face_arcface_detok,face_adaface,voice_ecapa192,voice_ecapa6144}_<split>.npy,
ids_<split>.txt, verify.json (20 train rows re-extracted and compared with the stored train arrays)."""
import io, json, sys, time, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "kaggle"))
import flag_extract as X
import flag_models as M

zpath, work = Path(sys.argv[1]), Path(sys.argv[2])
OUT = HERE / "feats_evalphase"; OUT.mkdir(exist_ok=True)
KO = REPO / "kaggle" / "output"
t00 = time.time()
if not (work / ".done").exists():
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z: z.extractall(work)
    (work / ".done").touch()
S = {}
for prot in ("no_gender", "gender"):
    for lang in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("English", "Bangla")):   # English only is enough for blend.py
        k = f"{prot}/{lang}"; S[k] = X.resolve_split(work, f"{lang}_test.txt", 1, 2, must_contain=f"/{prot}/")
        S[k]["ids"] = [l.split()[0] for l in Path(S[k]["txt"]).read_text().splitlines() if l.strip()]
        S[k]["dur"] = np.array([X.sf.info(str(S[k]["root"] / p)).duration for p in S[k]["wav"]], dtype=np.float32)
        (OUT / f"ids_{X.tag(k)}.txt").write_text("\n".join(S[k]["ids"]) + "\n")
        X.log(f"{k}: {len(S[k]['ids'])} rows")
ver = {}

# ---------------------------------------------------------------- ArcFace (+ check on 20 train rows read from the train zip)
import cv2
af = X.ArcFace()
tz = zipfile.ZipFile(REPO / "Input" / "train_set.zip")
trows = [l.split() for l in tz.read("train_set/train_set/train_English.txt").decode().splitlines() if l.strip()]
pick = list(range(0, len(trows), len(trows) // 20))[:20]
ref = np.load(KO / "feats_v2" / "face_arcface_train.npy", mmap_mode="r")
cs = []
for i in pick:
    img = cv2.imdecode(np.frombuffer(tz.read("train_set/train_set/" + trows[i][3]), np.uint8), cv2.IMREAD_COLOR)
    a, ok, _ = X.align_face(af.app.det_model, img); e = af.rec.get_feat(a).flatten()
    cs.append(float(e @ ref[i] / (np.linalg.norm(e) * np.linalg.norm(ref[i]) + 1e-9)))
ver["arcface_train_cos"] = [min(cs), float(np.mean(cs))]; X.log("ArcFace check vs stored train rows: min / mean cos", ver["arcface_train_cos"])
for k, sp in S.items():
    f = OUT / f"face_arcface_{X.tag(k)}.npy"
    if f.exists(): continue
    E, OK = [], []
    for p in sp["jpg"]:
        e, ok, _ = af.embed(sp["root"] / p); E.append(e); OK.append(ok)
    np.save(f, np.stack(E)); np.save(OUT / f"face_arcface_detok_{X.tag(k)}.npy", np.array(OK))
    X.log(f"arcface {k}: detected {100 * np.mean(OK):.1f} % [{time.time() - t00:.0f}s]")

# ---------------------------------------------------------------- AdaFace
try:
    from PIL import Image
    import torch
    ad = M.AdaFace(None)
    ref = np.load(KO / "feats_models" / "face_adaface_train.npy", mmap_mode="r"); cs = []
    for i in pick:
        im = Image.open(io.BytesIO(tz.read("train_set/train_set/" + trows[i][3]))).convert("RGB")
        e = ad(ad.prep(im)[None])[""][0]
        cs.append(float(e @ ref[i] / (np.linalg.norm(e) * np.linalg.norm(ref[i]) + 1e-9)))
    ver["adaface_train_cos"] = [min(cs), float(np.mean(cs))]; X.log("AdaFace check: min / mean cos", ver["adaface_train_cos"])
    for k, sp in S.items():
        f = OUT / f"face_adaface_{X.tag(k)}.npy"
        if f.exists(): continue
        E = []
        for lo in range(0, len(sp["jpg"]), 32):
            x = torch.stack([ad.prep(Image.open(sp["root"] / p).convert("RGB")) for p in sp["jpg"][lo:lo + 32]])
            E.append(ad(x)[""])
        np.save(f, np.concatenate(E).astype(np.float32)); X.log(f"adaface {k} done [{time.time() - t00:.0f}s]")
except Exception as e:   # AdaFace is optional (the HF release needs omegaconf / yaml)
    ver["adaface_error"] = repr(e)[:300]; X.log("AdaFace skipped:", ver["adaface_error"])

# ---------------------------------------------------------------- speechbrain ECAPA 192 / 6144
ec = X.Ecapa(KO / "_ecapa_sb")
r192 = np.load(KO / "feats_v2" / "voice_ecapa192_train.npy", mmap_mode="r"); cs = []
for i in pick:
    e = ec([X.load_audio(io.BytesIO(tz.read("train_set/train_set/" + trows[i][2])))])[0][0]
    cs.append(float(e @ r192[i] / (np.linalg.norm(e) * np.linalg.norm(r192[i]) + 1e-9)))
ver["ecapa192_train_cos"] = [min(cs), float(np.mean(cs))]; X.log("ECAPA check: min / mean cos", ver["ecapa192_train_cos"])
for k, sp in S.items():
    f1, f2 = OUT / f"voice_ecapa192_{X.tag(k)}.npy", OUT / f"voice_ecapa6144_{X.tag(k)}.npy"
    if f1.exists() and f2.exists(): continue
    paths = [sp["root"] / p for p in sp["wav"]]
    res = X.run_batched(paths, lambda ps: ec([X.read_wav(p) for p in ps]), sp["dur"], 240.0, 32,
                        cpu_fn=lambda ps: ec([X.read_wav(p) for p in ps], device="cpu"), label=f"ecapa {k}")
    np.save(f1, np.stack([r[0] for r in res]).astype(np.float32)); np.save(f2, np.stack([r[1] for r in res]).astype(np.float32))
    X.log(f"ecapa {k} done [{time.time() - t00:.0f}s]")
(OUT / "verify.json").write_text(json.dumps(ver, indent=1))
X.log(f"XJF extract done [{time.time() - t00:.0f}s]", sorted(p.name for p in OUT.iterdir()))
