print("[cell] r08 smoke: with FF_SMOKE=1 process only a few MAV speakers (new cell; does nothing otherwise)", flush=True)
# New cell (FourFusion reproduction repo), before the original extraction cell. A full run lists ~714 000 MAV files and encodes
# 40 271 wav + 16 533 frames (about 9-10 h on a T4), far beyond a CPU smoke session. With FF_SMOKE=1 this cell arranges that, when
# the extraction cell imports the flag_extract module written by cell 2, two of its functions are replaced right after the import:
#   find_attached_source(IN, name) -> the v1 / v2 root found by the guard cell (no rglob over the whole input tree);
#   open_archive(folder)           -> a DirArchive over the first FF_R08_SMOKE_SPEAKERS speakers (default 2) and the first
#                                     FF_R08_SMOKE_VIDEOS videos (default 2) of each speaker / language, faces and voices.
# The patch is applied by an import hook, so the module (and torch) are still imported after the original `!pip install` line.
# Everything else (pip install, encoders, batching, the meta / array writers) runs as in the original. The tables then differ from
# ngnnguynchn/flag2027-ext-meta-p by design: smoke_env sets FF_R08_ALLOW_MISMATCH=1 so the verify cell only warns.
import os, sys
from pathlib import Path

if os.environ.get("FF_SMOKE") == "1":
    import importlib.abc, importlib.machinery
    _NS = int(os.environ.get("FF_R08_SMOKE_SPEAKERS", "2"))
    _NV = int(os.environ.get("FF_R08_SMOKE_VIDEOS", "2"))
    _MAVROOTS = {"v1_complete": _roots[0], "v2_complete": _roots[1]}          # _roots: sorted v1 / v2 roots of the guard cell

    def _smoke_patch(X):
        class _SmokeArchive(X.DirArchive):
            def __init__(self, root):
                self.root = Path(root)
                files = []
                for kind in ("faces", "voices"):
                    for spk in sorted(p for p in (self.root / kind).iterdir() if p.is_dir())[:_NS]:
                        for lang in sorted(p for p in spk.iterdir() if p.is_dir()):
                            for vid in sorted(p for p in lang.iterdir() if p.is_dir())[:_NV]:
                                files += [p for p in vid.iterdir() if p.is_file()]
                self._infos = [self._Info(self.root, p) for p in sorted(files)]
                self._by = {i.filename: i for i in self._infos}

        _orig_open_archive = X.open_archive
        X.find_attached_source = lambda inp, name: _MAVROOTS.get(name)
        X.open_archive = lambda path: _SmokeArchive(path) if Path(path).is_dir() else _orig_open_archive(path)
        print(f"SMOKE: flag_extract patched: {_NS} speakers x {_NV} videos per language from",
              {k: str(v) for k, v in _MAVROOTS.items()}, flush=True)

    class _SmokeHook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "flag_extract":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            _exec = spec.loader.exec_module

            def exec_module(module):
                _exec(module)
                _smoke_patch(module)
            spec.loader.exec_module = exec_module
            return spec

    if "flag_extract" in sys.modules:
        _smoke_patch(sys.modules["flag_extract"])
    else:
        sys.meta_path.insert(0, _SmokeHook())
        print("SMOKE: flag_extract will be patched when the extraction cell imports it", flush=True)
else:
    print("not a smoke run: the original extraction runs on the full MAV-Celeb v1 / v2 trees", flush=True)
