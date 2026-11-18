print("[cell] r08: optional local copy of the VGG-Face weights (new cell; .pth checked by sha256, .py still from the original link)", flush=True)
# New cell (FourFusion reproduction repo). The original cell (flag_extract.VGGFace) downloads vgg_face_dag.py (the network
# definition, exec'd) and vgg_face_dag.pth (the weights) from www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/ into
# WORKDIR/_vggface, each only when absent. This cell pre-seeds the weights from a local copy when one with the known sha256 exists:
#   $FLAG_CACHE/vgg_face_dag.pth            (the runner seeds FLAG_CACHE from FF_DATA/06_pretrained/cache/flag = tools/fetch_models.py dest)
#   FF_DATA/06_pretrained/cache/flag/vgg_face_dag.pth   FF_DATA/06_pretrained/rfs_models/vgg_face.pt (same file, fetch_models also_as)
#   FF_DATA/06_pretrained/vgg_face_dag/vgg_face_dag.pth (older layout)
# vgg_face_dag.py is copied only when its sha256 is recorded (VGG_PY_SHA256; not measured yet: stage.json todo); otherwise the
# original code downloads it (a few kB). Numbers are unchanged: the weights are the same file, checked by sha256.
import hashlib, os, shutil
from pathlib import Path

VGG_PTH_SHA256 = "f070b241e3faf17a08c78f8700d334413230bac5de7fbcfd01b3cf05ce10de1a"   # 580 015 466 bytes (E: package + RFS cache)
VGG_PY_SHA256 = None                                                                     # todo: record from a first download


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


_roots = [Path(d) for d in os.environ.get("FF_DATA", "").split(os.pathsep) if d]
_pth = ([Path(os.environ["FLAG_CACHE"]) / "vgg_face_dag.pth"] if os.environ.get("FLAG_CACHE") else []) \
    + [r / "06_pretrained" / rel for r in _roots for rel in ("cache/flag/vgg_face_dag.pth", "rfs_models/vgg_face.pt",
                                                              "vgg_face_dag/vgg_face_dag.pth")] \
    + ([Path(os.environ["FF_MODELS"]) / "cache/flag/vgg_face_dag.pth"] if os.environ.get("FF_MODELS", "auto") not in ("auto", "off") else [])
_dst = WORKDIR / "_vggface"
if (_dst / "vgg_face_dag.pth").exists():
    print("VGG-Face weights already in", _dst, flush=True)
else:
    for _c in _pth:
        if not _c.is_file():
            continue
        if _sha256(_c) != VGG_PTH_SHA256:
            print("!! ignored (another sha256):", _c, flush=True)
            continue
        _dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_c, _dst / "vgg_face_dag.pth")
        print("VGG-Face weights copied from", _c, flush=True)
        _py = _c.parent / "vgg_face_dag.py"
        if VGG_PY_SHA256 and _py.is_file() and _sha256(_py) == VGG_PY_SHA256 and not (_dst / "vgg_face_dag.py").exists():
            shutil.copy2(_py, _dst / "vgg_face_dag.py")
            print("vgg_face_dag.py copied from", _py, flush=True)
        break
    else:
        print("no local VGG-Face weights: the original cell downloads them from www.robots.ox.ac.uk", flush=True)
if not (_dst / "vgg_face_dag.py").exists():
    print("vgg_face_dag.py: downloaded by the original cell from www.robots.ox.ac.uk (no recorded sha256 for a local copy)", flush=True)
