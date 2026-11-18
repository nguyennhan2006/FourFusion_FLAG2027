"""Generate FLAG_30_e2e_best.ipynb: raw organiser data -> the BEST v9 submission (CodaBench dev 22.63, log 116 / 134).

Same chain as FLAG_16 (build_e2e_notebooks.py), extended for v7, then v9 (04/10: --bangla-fuser + --english-fuser):
    FLAG_12 feature cells (verbatim) + FaRL / AdaFace (code of kaggle/flag_models.py, interface of FLAG_12)
    -> FLAG_14 AudioCLIP cells (verbatim)
    -> kaggle/flag_lib.py, kaggle/flag_v2.py, BEST/v6_eval/pipeline.py, BEST/v7_safe/{safe_lib,pipeline}.py and the two
       fuser files BEST/v7_safe/{cafuse_fuser,fuser_en}.json written to <working>/repo/ with the repository layout (verbatim)
    -> the submitted v9 zip (dev reference), embedded in base64
    -> BEST/v7_safe/pipeline.main() --bangla-fuser --english-fuser  ->  submission zip (+ comparison with v9 on dev)
Inputs on Kaggle: raw v4 data (train_set + dev_set, or the Evaluation data), `flag2027-members` (v6 members),
`flag2027-eda-bundle` (MAV-Celeb v1/v2 meta + arrays), `flag2027-best-v7` (kaggle/flag2027-best-v7.zip: s007_xall.pt,
ext AdaFace arrays; made by build_best_v7_dataset.py). Optional: an earlier output of this notebook (feats_eval /
feats_av arrays are reused). Inputs are found through the FLAG_12 input scan (SCAN), never by rglob over /kaggle/input.
    python build_best_notebook.py
"""
from pathlib import Path
import build_eval_notebooks as B12
import build_av_notebook as B14
import build_e2e_notebooks as E
import base64, hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V9_ZIP = ROOT / "Experiment/CAFUSE-EN/submission_CA_EN_BN.zip"           # BEST v9 as submitted (CodaBench dev 22.63)
V9_MD5 = "f6d77418f3a44f2273ab5fa921d8059d"

INTRO = """
# FLAG 2027 — NB-30: end to end, raw data -> BEST v9 submission (level 0, CodaBench dev **22.63**)

| File language | Recipe | dev EER (ng / g) |
|---|---|---|
| English | **CA-EN**: logistic regression (`fuser_en.json`, fitted once on 12 896 labelled validation trials) on SET-17's own level-0 streams (s007 + s010B deep / CCA, 5 members each; ImageBind; age; AudioCLIP) + 6 pair conditions (P(male) face / voice, their difference, age face / voice, age gap) + their products | 16.87 / 27.49 |
| Bangla | **CA-FUSE**: the same kind of fusion (`cafuse_fuser.json`) on the six B3 streams: s007 XALL (5 nets, 70 v4 + 151 MAV-Celeb v1/v2 persons) deep / CCA, ImageBind, age, and the relational bridge `rel` / `relg` over 221 train identities (FaRL + ImageBind + AdaFace faces, ECAPA-192 + BTC voices) | 18.07 / 28.07 |

Every pair (face, voice) is scored on its own, and every constant comes from train data (organiser ruling of 30/09: no clusters, no test statistics). Choosing the recipe per file language (organiser file names) is declared. The pipeline re-scores half of each file and checks that no score moves.

**Steps:**
1. **FLAG_12 cells**: ArcFace, ECAPA-192 / 6144, ViT-age, wav2vec2 age-gender, ImageBind. **+ FaRL, AdaFace**, the code of FLAG_10 (`kaggle/flag_models.py`) with the FLAG_12 interface. Arrays of an attached earlier output (folder `feats_eval`) are reused.
2. **FLAG_14 cells**: AudioCLIP.
3. **Library cells**: verbatim source of `flag_lib`, `flag_v2`, v6 `pipeline`, v7 `safe_lib` + `pipeline`, the two fuser files, and the submitted v9 zip (base64, md5-checked) as the dev reference.
4. **Pipeline** (`BEST/v7_safe/pipeline.py --bangla-fuser --english-fuser`): CA-EN for the English files, CA-FUSE for the Bangla files. On dev it compares every file with the submitted v9 zip.

**Inputs (Add Input):**
- raw organiser data (on 11/11 also the Evaluation data; then set `ONLY_UNDER`);
- `flag2027-members`;
- `flag2027-eda-bundle`;
- `flag2027-best-v7`;
- optional: an earlier output of this notebook.

**Settings:** GPU T4, **Internet ON** (checkpoints about 8 GB: ImageBind, AudioCLIP, CLIP ViT-B/16 + FaRL, AdaFace, w2v2, ECAPA).
**Time on a T4:** FLAG_12 extraction about 1 h 15 min; FaRL + AdaFace about 10-15 min; AudioCLIP about 20 min; pipeline about 20-30 min on CPU (members cached).
**Submit:** `submission_best_v9.zip`. Scores are written as -score (lower = same). On new data, if the first result is > 50, flip the sign.
**AudioCLIP:** the 44.1 kHz resampler is pinned to `polyphase` (log 133), so arrays extracted on the current Kaggle image match the September arrays that v9 was built on (cos >= 0.9999).
**Known limit:** file names / trial lists follow the dev layout (`flag_lib.NAMES`), as in FLAG_16. Check them against the Evaluation data before the run.
"""

PIP_EXTRA = r'''
# FaRL = OpenAI CLIP ViT-B/16 + FaRL visual weights; AdaFace = the CVLFace release (yaml / omegaconf). --no-deps: CLIP pins old torch.
if not LOCAL:
    !pip -q install --no-deps "git+https://github.com/openai/CLIP.git" 2>&1 | tail -2
    !pip -q install ftfy regex pyyaml omegaconf 2>&1 | tail -2
'''

FACE2 = r'''
# ============================================================================ FaRL + AdaFace (kaggle/flag_models.py, FLAG_10), FLAG_12 interface
import urllib.request


def _download(url, cache):
    f = Path(cache) / url.rsplit("/", 1)[-1]
    if not f.exists():
        log("download", url)
        urllib.request.urlretrieve(url, f)
    return f


class FaRL:
    """FaRL ViT-B/16 (LAION-Face 20M, ep64): CLIP ViT-B/16 with the FaRL visual weights; CLIP preprocessing on the raw jpg
    (no detection, as FLAG_10). 512, fp16 autocast."""
    kind = "face"
    URL = ("https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/"
           "FaRL-Base-Patch16-LAIONFace20M-ep64.pth")

    def __init__(self):
        import clip
        m, self.pre = clip.load("ViT-B/16", device="cpu", download_root=str(CACHE))
        ck = _download(self.URL, CACHE)
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        vis = [k for k in m.state_dict() if k.startswith("visual.")]
        hit = [k for k in vis if k in sd and sd[k].shape == m.state_dict()[k].shape]
        assert len(hit) == len(vis), f"FaRL checkpoint covers {len(hit)}/{len(vis)} visual tensors"
        m.load_state_dict(sd, strict=False)
        self.m = m.visual.float().to(DEV).eval()

    def prep(self, im):
        return self.pre(im)

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        with amp(use_amp):
            return {"": self.m(x.to(DEV)).float().cpu().numpy()}


def align_face(det_model, img):
    """BGR image -> 112x112 BGR crop on the ArcFace 5-point template (flag_extract.align_face: the big central face;
    15 % margin centre crop when nothing is detected)."""
    import cv2
    from insightface.utils import face_align
    faces = det_model.detect(img, max_num=0, metric="default")
    bboxes, kpss = faces if isinstance(faces, tuple) else (faces, None)
    if bboxes is not None and len(bboxes) and kpss is not None:
        h, w = img.shape[:2]
        ctr = np.array([w / 2, h / 2])
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        dist = np.linalg.norm((bboxes[:, :2] + bboxes[:, 2:4]) / 2 - ctr, axis=1)
        j = int(np.argmax(area - 2.0 * dist ** 2))
        return face_align.norm_crop(img, landmark=kpss[j], image_size=112), True
    h, w = img.shape[:2]; m = int(min(h, w) * 0.15)
    return cv2.resize(img[m:h - m, m:w - m], (112, 112)), False


class AdaFace:
    """CVLFace AdaFace IR-101 (WebFace12M), built with the release's own factory (its transformers wrapper never calls
    post_init(), which transformers >= 5 requires). Input: RGB 112x112 on the ArcFace template, scaled to [-1, 1]. 512, fp32."""
    kind = "face"
    ID = "minchul/cvlface_adaface_ir101_webface12m"

    def __init__(self):
        import yaml
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf
        from insightface.app import FaceAnalysis
        path = snapshot_download(self.ID)
        cwd = os.getcwd()
        os.chdir(path); sys.path.insert(0, path)
        try:
            from models import get_model
            m = get_model(OmegaConf.create(yaml.safe_load(open("pretrained_model/model.yaml"))))
            m.load_state_dict_from_path("pretrained_model/model.pt")
        finally:
            os.chdir(cwd); sys.path.remove(path)
        self.m = m.to(DEV).eval()
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        app.prepare(ctx_id=0 if DEV.type == "cuda" else -1, det_size=(224, 224), det_thresh=0.3)
        self.det = app.det_model

    def prep(self, im):
        bgr = np.asarray(im)[..., ::-1].copy()
        a, _ = align_face(self.det, bgr)
        rgb = torch.from_numpy(a[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return (rgb - 0.5) / 0.5

    @torch.no_grad()
    def __call__(self, x, use_amp=False):                             # small net: always fp32 (as FLAG_10)
        out = self.m(x.to(DEV))
        out = out[0] if isinstance(out, (tuple, list)) else out
        return {"": out.float().cpu().numpy()}


_build12 = build


def build(name):
    return {"farl": FaRL, "adaface": AdaFace}[name]() if name in ("farl", "adaface") else _build12(name)
'''

# FLAG_12 run cell with the two face encoders registered (same output names as FLAG_10: face_farl_<split>, face_adaface_<split>)
RUN12 = B12.RUN12.replace('''"ibv": ("face", {"": "ibv"}), "iba": ("voice", {"": "iba"})}''',
                          '''"ibv": ("face", {"": "ibv"}), "iba": ("voice", {"": "iba"}),
          "farl": ("face", {"": "farl"}), "adaface": ("face", {"": "adaface"})}''')
RUN12 = RUN12.replace('''"ibv": "", "iba": ""}''', '''"ibv": "", "iba": "", "farl": "", "adaface": ""}''')
assert RUN12.count("farl") == 3, "FLAG_12 run cell changed: update the FaRL / AdaFace registration"

ENV = E.ENV.replace("__MODE__", "indep")
ENV = ENV.replace('''SYSTEM = os.environ.get("E2E_SYSTEM", "set17")''', '''SYSTEM = "set17"                                      # v7: the English files are SET-17 level 0''')
ENV = ENV.replace('''N_MODELS = int(os.environ.get("E2E_N_MODELS", "15"))''', '''N_MODELS = int(os.environ.get("E2E_N_MODELS", "5"))    # 5 = the members of the submitted 23.28''')
ENV = ENV.replace('''ENCODERS = ["arcface", "ecapa", "vitage", "w2v2ag", "ibv", "iba"]''',
                  '''ENCODERS = ["arcface", "ecapa", "vitage", "w2v2ag", "ibv", "iba", "farl", "adaface"]''')
assert ENV.count('"farl"') == 1 and 'SYSTEM = "set17"' in ENV and '"E2E_N_MODELS", "5"' in ENV


def lib_file(rel):
    """Verbatim source of a repository file, written to <working>/repo/<rel> (keeps the layout the v7 pipeline expects)."""
    src = (ROOT / rel).read_text(encoding="utf-8")
    q = chr(39) * 3
    assert q not in src, rel
    var = "SRC_" + rel.replace("/", "_").replace(".", "_").upper()
    nl = chr(10)
    return (nl + "# ============================================================================ " + rel + " (verbatim)" + nl
            + var + " = r" + q + src + q + nl
            + "_p = WORKDIR / 'repo' / " + repr(rel) + "; _p.parent.mkdir(parents=True, exist_ok=True)" + nl
            + "_p.write_text(" + var + ", encoding='utf-8'); print(" + repr(rel + ":") + ", len(" + var + ".splitlines()), 'lines')" + nl)


RUN7 = r'''
# ============================================================================ BEST v9 pipeline (BEST/v7_safe/pipeline.py + both fusers)
import importlib.util
for m in ("flag_lib", "flag_v2", "pipeline", "safe_lib", "v7_pipeline", "v6_pipeline"):
    sys.modules.pop(m, None)
os.environ["FLAG_DATA"] = str(ORG)
spec = importlib.util.spec_from_file_location("v7_pipeline", WORKDIR / "repo/BEST/v7_safe/pipeline.py")
P7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P7)


def input_dir(name):
    """Folder of an attached file, from the FLAG_12 input scan (SCAN): no rglob over /kaggle/input."""
    ps = sorted(SCAN["files"].get(name, []), key=lambda p: len(p.as_posix()))
    return ps[0].parent if ps else None


v6_members, xall = input_dir("s007_seed1.pt"), input_dir("s007_xall.pt")
bundle, adaext = input_dir("eda_bundle_manifest.json"), input_dir("ext_v1_complete_face_adaface.npy")
assert v6_members is not None, "attach flag2027-members (v6 members: s007 / s010B seeds)"
assert xall is not None and adaext is not None, "attach flag2027-best-v7 (s007_xall.pt + ext AdaFace arrays)"
assert bundle is not None, "attach flag2027-eda-bundle (MAV-Celeb v1/v2 meta + arrays)"
FUSERS = WORKDIR / "repo/BEST/v7_safe"
argv = ["pipeline.py", "--data", str(ORG), "--feats", str(OUT), str(OUT_AV), "--ext", str(bundle), str(adaext),
        "--members-from", str(v6_members), str(xall), "--out", str(SUB), "--n-models", str(N_MODELS),
        "--threads", str(os.cpu_count() or 4),
        "--bangla-fuser", str(FUSERS / "cafuse_fuser.json"), "--english-fuser", str(FUSERS / "fuser_en.json")]
if any(k.startswith("no_gender/") for k in S) and not ONLY_UNDER:                 # dev run: compare with the submitted v9 zip
    argv += ["--reference", f"v9={REF_ZIP}"]
print(" ".join(argv))
sys.argv = argv
t0 = time.time()
P7.main()
log(f"v9 pipeline done in {(time.time() - t0) / 60:.1f} min")
'''

DONE7 = r'''
# ============================================================================ result
print(json.dumps(json.loads((SUB / "checks.json").read_text()), indent=1))
(SUB / "broken.json").write_text(json.dumps(BROKEN, indent=1))            # unreadable media (stand-ins were scored)
print("unreadable media:", {k: len(set(v)) for k, v in BROKEN.items()}, "-> submission/broken.json")
final = WORKDIR / "submission_best_v9.zip"
shutil.copy(SUB / "submission.zip", final)
with zipfile.ZipFile(final) as z:
    for n in z.namelist():
        d = pd.read_csv(z.open(n), sep=" ", header=None)
        print(f"{n:45s} {len(d):5d} rows, score mean {d[1].mean():+.3f}")
shutil.rmtree(WORKDIR / "raw", ignore_errors=True)
print(f"\nSUBMIT: {final}. On dev, every 'vs v9' line above should read rank changes 0 (= the EER of the submitted v9 "
      "zip). Arrays re-extracted on another Kaggle image may move a few English ranks (AudioCLIP cos >= 0.9999): "
      "rank-corr >= 0.999 is then the pass mark.")
'''


REFZIP = (
    "\n# ============================================================================ the submitted BEST v9 zip (dev reference)\n"
    "import base64, hashlib\n"
    "REF_B64 = (\n" + "".join(f'    "{c}"\n' for c in (lambda b: [b[i:i + 120] for i in range(0, len(b), 120)])(
        base64.b64encode(V9_ZIP.read_bytes()).decode())) + ")\n"
    "REF_ZIP = WORKDIR / 'repo/reference/submission_CA_EN_BN.zip'; REF_ZIP.parent.mkdir(parents=True, exist_ok=True)\n"
    "REF_ZIP.write_bytes(base64.b64decode(REF_B64))\n"
    f"assert hashlib.md5(REF_ZIP.read_bytes()).hexdigest() == {V9_MD5!r}, 'reference zip corrupted'\n"
    "print('reference: submitted BEST v9 zip (CodaBench dev 22.63) ->', REF_ZIP)\n")
assert hashlib.md5(V9_ZIP.read_bytes()).hexdigest() == V9_MD5


def build(path):
    cells = [("markdown", INTRO), ("code", ENV), ("code", E.PIP), ("code", PIP_EXTRA),
             ("markdown", "## 1. FLAG_12: find the data (row order = the organiser .txt), resume from an earlier output"), ("code", B12.DATA12),
             ("markdown", "## 2. Organiser csv + lists for the pipeline"), ("code", E.ORGDATA),
             ("markdown", "## 3. FLAG_12: audio loading, batching, helpers"), ("code", B12.AUDIO12),
             ("markdown", "## 4. FLAG_12: encoders"), ("code", B12.ENC12),
             ("markdown", "## 5. FaRL + AdaFace (FLAG_10 code, FLAG_12 interface): needed by the relational bridge of B3"), ("code", FACE2),
             ("markdown", "## 6. Extract every array the pipeline reads (FLAG_12 run cell + FaRL / AdaFace; skipped where it exists)"), ("code", RUN12),
             ("markdown", "## 7. FLAG_12: check against attached reference arrays (optional)"), ("code", B12.CHECK12),
             ("markdown", "## 8. FLAG_14: AudioCLIP (train + test files; resampler pinned to polyphase, log 133)"),
             ("code", E.AV), ("code", B14.AUDIO[len(B12.AUDIO12):]), ("code", B14.ENC), ("code", B14.RUN), ("code", E.AV_END),
             ("markdown", "## 9. Libraries, verbatim, written to `repo/` with the repository layout"),
             ("code", lib_file("kaggle/flag_lib.py")), ("code", lib_file("kaggle/flag_v2.py")),
             ("code", lib_file("BEST/v6_eval/pipeline.py")), ("code", lib_file("BEST/v7_safe/safe_lib.py")),
             ("code", lib_file("BEST/v7_safe/pipeline.py")),
             ("code", lib_file("BEST/v7_safe/cafuse_fuser.json")), ("code", lib_file("BEST/v7_safe/fuser_en.json")),
             ("code", REFZIP),
             ("markdown", "## 10. Run BEST v9"), ("code", RUN7),
             ("markdown", "## 11. Result"), ("code", DONE7)]
    E.nb(cells, path)


if __name__ == "__main__":
    build("FLAG_30_e2e_best.ipynb")
