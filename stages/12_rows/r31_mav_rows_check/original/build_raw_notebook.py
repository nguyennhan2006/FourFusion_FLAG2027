"""Generate FLAG_31_e2e_raw.ipynb: the BEST v9 system (CodaBench dev 22.63) from RAW data only (v7 until 04/10, log 134).

Inputs: the organiser's raw v4 data (train + dev, or the Evaluation data) and the raw MAV-Celeb v1 / v2 releases
(mavceleb_v1_complete / v2_complete, zip or the folder Kaggle extracts). Nothing else: every array and every trained net
is made here. Optional: an earlier output of this notebook (resume; finished steps are skipped).

Chain = FLAG_30 (build_best_notebook.py) plus:
  * EXTRAW: MAV-Celeb meta + organiser-space arrays (VGGFace fc7 faces, 8 frames per video; yangwang825 ECAPA voices,
    every wav, uncropped) = kaggle/flag_extract.extract_external as run for kaggle/output/feats_ext_full (the eda bundle);
  * the FLAG_12 EXT loop for ImageBind / FaRL / AdaFace faces and ECAPA-192 / w2v2 voices of v1 / v2 (12 s crop, as the
    bundle);
  * BEST/v7_safe/pipeline.py (--bangla-fuser --english-fuser) trains what is missing: v6 s007 / s010B (5 seeds; r2mix
    dropped) and the 5 s007 XALL nets.
    python build_raw_notebook.py
"""
from pathlib import Path
import build_eval_notebooks as B12
import build_av_notebook as B14
import build_e2e_notebooks as E
import build_best_notebook as BB

INTRO = """
# FLAG 2027 — NB-31: BEST v9 from RAW data only (level 0, CodaBench dev **22.63** with the original nets)

Same system as NB-30 (`FLAG_30_e2e_best`): English files ← **CA-EN** (condition-aware fusion of SET-17's level-0 streams, `fuser_en.json`); Bangla files ← **CA-FUSE** (the same fusion on the B3 streams: s007 XALL + relational bridge over 221 train identities, `cafuse_fuser.json`). Every pair is scored on its own, and every constant comes from train data. The two fusers were fitted once on labelled validation trials and are written verbatim by this notebook.

**Difference from NB-30:** nothing pre-built is attached. This notebook makes, from the raw media:
- the features of the v4 train set and of every test file (FLAG_12 encoders + FaRL / AdaFace + AudioCLIP);
- the MAV-Celeb v1 / v2 metadata and features (organiser VGGFace / ECAPA, ImageBind, FaRL, AdaFace, ECAPA-192, w2v2);
- the trained nets: s007 / s010B (5 seeds each) and the 5 s007 XALL nets.

The nets are retrained, so scores are not bit-identical to the submitted v9 zip. GPU and thread count change the last digits; expect rank-corr ≈ 0.999 and the same EER within seed noise. On the CPU machine that trained the originals, the same code rebuilds them exactly (checked).

**Inputs (Add Input), only data:**
1. the organiser's raw v4 data: `train_set` + `dev_set` (on 11/11 also the Evaluation data; then set `ONLY_UNDER`);
2. the raw MAV-Celeb releases `mavceleb_v1_complete` (contains `meta_v1.csv`: names and genders) and `mavceleb_v2_complete`. Zip or the folder Kaggle extracts; they are external training data, declared;
3. optional: an earlier output of THIS notebook (resume).

**Settings:** GPU T4, **Internet ON** (checkpoints about 9 GB). **Time on a T4:** about 4.5-5.5 h:
- v4 features: about 1 h 45;
- v1 / v2: about 2-2.5 h (25 GB of media);
- training: about 10 min;
- pipeline: about 20-30 min.

If the session ends, attach this notebook's output and run again: arrays and nets that exist are skipped.
**Submit:** `submission_best_v9.zip`. Scores are written as -score (lower = same).
"""

ENV = BB.ENV.replace('''EXT_SOURCES, EXT_ENCODERS = [], []                                  # no MAV-Celeb here''',
                     '''EXT_SOURCES = ["v1_complete", "v2_complete"]                         # MAV-Celeb releases (raw, attached)
EXT_ENCODERS = ["ecapa", "ibv", "farl", "adaface", "w2v2ag"]          # v1/v2 streams the B3 system needs''')
assert 'EXT_ENCODERS = ["ecapa", "ibv"' in ENV

RESUME = r'''
# ============================================================================ resume: arrays / nets of an earlier run of THIS notebook
OUT_EXT = WORKDIR / "feats_ext"; OUT_EXT.mkdir(exist_ok=True)
MEM6, MEMX = WORKDIR / "members_v6", WORKDIR / "members_xall"
# SCAN (FLAG_12 scan_inputs) lists every input file outside faces/ and voices/: no rglob over the raw media
for pat, dst in ((r"/feats[-_]ext$", OUT_EXT), (r"/members[-_]v6$", MEM6), (r"/members[-_]xall$", MEMX)):
    n = 0
    for name_, ps in SCAN["files"].items():
        for p in ps:
            if re.search(pat, p.parent.as_posix()) and WORKDIR not in p.parents and not (dst / name_).exists():
                dst.mkdir(exist_ok=True); shutil.copy(p, dst / name_); n += 1; break
    log(f"resume: {n} files copied into {dst.name}")
'''

EXTRAW = r'''
# ============================================================================ MAV-Celeb v1 / v2 from the raw release (kaggle/flag_extract.py, FLAG_08)
# Rows exactly as the eda bundle (kaggle/output/feats_ext_full): faces = 8 evenly spaced frames per (id, language, video);
# voices = every wav, UNCROPPED; row order = file paths sorted part by part (case-sensitive, as on Kaggle's Linux).
# ext_<src>_face.npy (VGGFace fc7, fp16), ext_<src>_voice.npy (yangwang825/ecapa-tdnn-vox2, fp32), *_meta.csv, speakers.
import io as _io
from pathlib import PurePosixPath


class VGGFace:
    """VGGFace (Parkhi 2015) fc7 post-ReLU, the organiser's 4096-d face feature (cos 1.0000 with their csv)."""
    MEAN_BGR = np.array([93.594, 104.762, 129.186], dtype=np.float32)

    def __init__(self, cache: Path):
        import importlib.util, urllib.request
        cache.mkdir(parents=True, exist_ok=True)
        base = "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/"
        for f in ["vgg_face_dag.py", "vgg_face_dag.pth"]:
            if not (cache / f).exists():
                log("downloading", f)
                urllib.request.urlretrieve(base + f, cache / f)
        spec = importlib.util.spec_from_file_location("vgg_face_dag", cache / "vgg_face_dag.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.m = mod.vgg_face_dag(str(cache / "vgg_face_dag.pth")).to(DEV).eval()
        self._o = {}
        self.m.relu7.register_forward_hook(lambda mod_, i, o: self._o.__setitem__("x", o))

    def prep(self, data):
        from PIL import Image
        im = (data if isinstance(data, Image.Image) else Image.open(_io.BytesIO(data))).convert("RGB")
        if im.size != (224, 224):
            im = im.resize((224, 224), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)[..., ::-1] - self.MEAN_BGR
        return a.transpose(2, 0, 1).copy()

    @torch.no_grad()
    def __call__(self, arrays):
        self.m(torch.from_numpy(np.stack(arrays)).to(DEV))
        return self._o["x"].float().cpu().numpy()


class OrganiserVoice:
    """The organiser's voice encoder (yangwang825/ecapa-tdnn-vox2), batched like their script (relative lengths)."""

    def __init__(self, savedir):
        speechbrain_import()
        from speechbrain.inference.interfaces import Pretrained

        class Encoder(Pretrained):
            MODULES_NEEDED = ["compute_features", "mean_var_norm", "embedding_model"]

        kw = {}
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
        self.m = Encoder.from_hparams(source="yangwang825/ecapa-tdnn-vox2", savedir=str(savedir),
                                      run_opts={"device": str(DEV)}, **kw).eval()

    @torch.no_grad()
    def __call__(self, sigs):
        lens = [len(s) for s in sigs]
        L = max(lens)
        x = torch.zeros(len(sigs), L)
        for i, s in enumerate(sigs):
            x[i, :len(s)] = torch.from_numpy(s)
        rel = torch.tensor([n / L for n in lens], dtype=torch.float32, device=DEV)
        f = self.m.mods.compute_features(x.to(DEV))
        f = self.m.mods.mean_var_norm(f, rel)
        return list(self.m.mods.embedding_model(f, rel).squeeze(1).float().cpu().numpy())


def _parse(member):
    """faces|voices/<id>/<lang>/<video>/<file> (optionally under a version folder) -> (kind, id, lang, video)."""
    p = member.split("/")
    for kind in ("voices", "faces"):
        if kind in p:
            k = p.index(kind)
            if len(p) >= k + 5:
                return kind, p[k + 1], p[k + 2].lower(), p[k + 3]
    return None


class RawSource:
    """Media of one MAV-Celeb release, zip or extracted folder. names: 'faces/...' / 'voices/...' keys (version folder
    dropped, as the eda bundle meta), sorted part by part; read(key) -> bytes; size(key) -> bytes on disk."""

    def __init__(self, src):
        src = Path(src)
        if src.is_dir():
            files = {}
            for dp, _, fns in os.walk(src):
                for f in fns:
                    q = Path(dp) / f
                    k = media_key(q.relative_to(src).as_posix())
                    if k:
                        files[k] = q
            self.read = lambda k: files[k].read_bytes()
            self.size = lambda k: files[k].stat().st_size
            self.root, self.files = src, files
        else:
            z = zipfile.ZipFile(src)
            files = {media_key(i.filename): i for i in z.infolist() if not i.is_dir() and media_key(i.filename)}
            self.read = lambda k: z.read(files[k].filename)
            self.size = lambda k: files[k].file_size
            self.root, self.files, self.zip = src, files, z
        self.names = sorted(files, key=lambda k: PurePosixPath(k).parts)

    def speakers_csv(self):
        """The release's own identity table (v1: meta_v1.csv = ids, name, split, gender); v2 has none."""
        if isinstance(self.root, Path) and self.root.is_dir():
            metas = [p for d in (self.root, self.root.parent) for p in sorted(d.glob("*.csv")) if "meta" in p.name.lower()]
            return pd.read_csv(metas[0]) if metas else None
        metas = [n for n in self.zip.namelist() if n.lower().endswith(".csv") and "meta" in n.lower()]
        return pd.read_csv(self.zip.open(metas[0])) if metas else None


def extract_external(rs, name, out, voice_enc, face_enc, frames_per_video=8, max_padded_sec=240.0, max_bs=32, face_bs=64):
    fv, ff = out / f"ext_{name}_voice.npy", out / f"ext_{name}_face.npy"
    wavs, faces = [], {}
    for k in rs.names:
        r = _parse(k)
        if r is None:
            continue
        if r[0] == "voices" and k.lower().endswith(".wav"):
            wavs.append((k,) + r[1:])
        elif r[0] == "faces" and k.lower().endswith(".jpg"):
            faces.setdefault(r[1:], []).append(k)
    log(f"{name}: {len(wavs)} wav, {sum(map(len, faces.values()))} jpg in {len(faces)} videos, {len({w[1] for w in wavs})} ids")
    if not fv.exists():
        durs = np.array([rs.size(f) / 32000 for f, *_ in wavs], dtype=np.float32)      # sort key only
        res = run_batched(list(range(len(wavs))), lambda ii: voice_enc([load_audio(_io.BytesIO(rs.read(wavs[i][0]))) for i in ii]),
                          durs, max_padded_sec, max_bs, label=f"voice {name}")
        pd.DataFrame(dict(path=[w[0] for w in wavs], spk=[w[1] for w in wavs], lang=[w[2] for w in wavs],
                          video=[w[3] for w in wavs], dur=durs)).to_csv(out / f"ext_{name}_voice_meta.csv", index=False)
        np.save(fv, np.stack(res).astype(np.float32))                      # the array last: it marks the step as done
    if not ff.exists():
        pick = []
        for key, fs in sorted(faces.items()):
            fs = sorted(fs)
            sel = np.unique(np.linspace(0, len(fs) - 1, min(frames_per_video, len(fs))).round().astype(int))
            pick += [(fs[j],) + key for j in sel]
        F_ = []
        for s in range(0, len(pick), face_bs):
            F_.append(face_enc([face_enc.prep(rs.read(p[0])) for p in pick[s:s + face_bs]]))
            if (s // face_bs) % 50 == 0:
                log(f"  face {name} {s}/{len(pick)}")
        pd.DataFrame(dict(path=[p[0] for p in pick], spk=[p[1] for p in pick], lang=[p[2] for p in pick],
                          video=[p[3] for p in pick])).to_csv(out / f"ext_{name}_face_meta.csv", index=False)
        np.save(ff, np.concatenate(F_).astype(np.float16))
    log(name, "done:", np.load(fv, mmap_mode="r").shape, np.load(ff, mmap_mode="r").shape)


SRC = {n: ext_source(SCAN, n) for n in EXT_SOURCES}
assert all(SRC.values()), f"attach the raw MAV-Celeb releases (zip or folder): {SRC}"
RS = {}
raw = lambda n: RS.setdefault(n, RawSource(SRC[n]))                      # one listing per release
todo = [n for n in EXT_SOURCES if not ((OUT_EXT / f"ext_{n}_voice.npy").exists() and (OUT_EXT / f"ext_{n}_face.npy").exists())]
if todo:
    ve, fe = OrganiserVoice(CACHE / "ecapa_btc"), VGGFace(CACHE / "vggface")
    for n in todo:
        t0 = time.time()
        extract_external(raw(n), n, OUT_EXT, ve, fe)
        log(f"{n}: organiser-space arrays in {time.time() - t0:.0f}s")
    del ve, fe; release()
for n in EXT_SOURCES:
    if not (OUT_EXT / f"ext_{n}_speakers.csv").exists():
        sp = raw(n).speakers_csv() if n == "v1_complete" else None
        if sp is not None:
            sp.to_csv(OUT_EXT / f"ext_{n}_speakers.csv", index=False); log(f"{n}: speaker table saved ({len(sp)} ids)")
assert (OUT_EXT / "ext_v1_complete_speakers.csv").exists(), "meta_v1.csv (names + genders) not found in the v1 release"

# the FLAG_12 EXT loop reads these rows (12 s voice crop, as the eda bundle)
EXT = {n: ExtRows(SRC[n], n, OUT_EXT) for n in EXT_SOURCES}
'''

RUN7 = r'''
# ============================================================================ BEST v9 pipeline (trains the missing nets)
import importlib.util
for m in ("flag_lib", "flag_v2", "pipeline", "safe_lib", "v7_pipeline", "v6_pipeline"):
    sys.modules.pop(m, None)
os.environ["FLAG_DATA"] = str(ORG)
spec = importlib.util.spec_from_file_location("v7_pipeline", WORKDIR / "repo/BEST/v7_safe/pipeline.py")
P7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P7)
members = [str(d) for d in (MEM6, MEMX) if d.exists() and any(d.glob("*.pt"))]
FUSERS = WORKDIR / "repo/BEST/v7_safe"
argv = ["pipeline.py", "--data", str(ORG), "--feats", str(OUT), str(OUT_AV), "--ext", str(OUT_EXT), str(OUT),
        "--out", str(SUB), "--n-models", str(N_MODELS), "--threads", str(os.cpu_count() or 4),
        "--bangla-fuser", str(FUSERS / "cafuse_fuser.json"), "--english-fuser", str(FUSERS / "fuser_en.json")]
if members:
    argv += ["--members-from", *members]
if any(k.startswith("no_gender/") for k in S) and not ONLY_UNDER:                 # dev run: compare with the submitted v9 zip
    argv += ["--reference", f"v9={REF_ZIP}"]
print(" ".join(argv))
sys.argv = argv
t0 = time.time()
P7.main()
for src, dst in ((SUB / "members_v6", MEM6), (SUB / "members_xall", MEMX)):        # keep the trained nets for a resume
    if src.exists():
        dst.mkdir(exist_ok=True)
        for f in src.glob("*.pt"):
            if not (dst / f.name).exists():
                shutil.copy(f, dst / f.name)
log(f"v9 pipeline done in {(time.time() - t0) / 60:.1f} min")
'''


AV_NOEXT = """_EXT_ALL, EXT = EXT, {}              # AudioCLIP is used on the v4 files only: no v1 / v2 jobs (FLAG_14 RUN adds them)
"""
AV_NOEXT_END = """EXT = _EXT_ALL
"""


def build(path):
    cells = [("markdown", INTRO), ("code", ENV), ("code", E.PIP), ("code", BB.PIP_EXTRA),
             ("markdown", "## 1. FLAG_12: find the data (row order = the organiser .txt), resume from an earlier output"), ("code", B12.DATA12),
             ("code", RESUME),
             ("markdown", "## 2. Organiser csv + lists for the pipeline"), ("code", E.ORGDATA),
             ("markdown", "## 3. FLAG_12: audio loading, batching, helpers"), ("code", B12.AUDIO12),
             ("markdown", "## 4. FLAG_12: encoders"), ("code", B12.ENC12),
             ("markdown", "## 5. FaRL + AdaFace (FLAG_10 code, FLAG_12 interface)"), ("code", BB.FACE2),
             ("markdown", "## 6. MAV-Celeb v1 / v2: metadata + organiser-space arrays from the raw releases"), ("code", EXTRAW),
             ("markdown", "## 7. Extract every other array (v4 splits + v1 / v2 rows; skipped where it exists)"), ("code", BB.RUN12),
             ("markdown", "## 8. FLAG_12: check against attached reference arrays (optional)"), ("code", B12.CHECK12),
             ("markdown", "## 9. FLAG_14: AudioCLIP (v4 train + test files; resampler pinned to polyphase, log 133)"),
             ("code", E.AV + AV_NOEXT), ("code", B14.AUDIO[len(B12.AUDIO12):]), ("code", B14.ENC), ("code", B14.RUN),
             ("code", E.AV_END + AV_NOEXT_END),
             ("markdown", "## 10. Libraries, verbatim, written to `repo/`"),
             ("code", BB.lib_file("kaggle/flag_lib.py")), ("code", BB.lib_file("kaggle/flag_v2.py")),
             ("code", BB.lib_file("BEST/v6_eval/pipeline.py")), ("code", BB.lib_file("BEST/v7_safe/safe_lib.py")),
             ("code", BB.lib_file("BEST/v7_safe/pipeline.py")),
             ("code", BB.lib_file("BEST/v7_safe/cafuse_fuser.json")), ("code", BB.lib_file("BEST/v7_safe/fuser_en.json")),
             ("code", BB.REFZIP),
             ("markdown", "## 11. Train the missing nets + run BEST v9"), ("code", RUN7),
             ("markdown", "## 12. Result"), ("code", BB.DONE7)]
    E.nb(cells, path)


if __name__ == "__main__":
    build("FLAG_31_e2e_raw.ipynb")
