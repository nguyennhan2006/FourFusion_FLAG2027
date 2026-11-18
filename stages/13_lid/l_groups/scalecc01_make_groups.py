r"""SCALE-CC-01: language group of every NEW VoxCeleb2 speaker of FLAG_53 (`kind == new`), with DSEL-01's rule (majority
VoxLingua107 top-1 language over the speaker's clips; FLAG_53 used the same model, so the label order of FLAG_42 applies) ->
vox2b_groups.json {speaker: [group, majority language, share]}. Speakers of FLAG_37 keep their DSEL-01 group (vox2_groups.json).
Also vox2_gender.json {speaker: m / f} from VoxCeleb2's official vox2_meta.csv (gender column only).
Written 12/10 before any run.   python make_groups.py

FourFusion reproduction repo (stage l_groups): copy of Experiment/SCALE-CC-01/make_groups.py (md5 803aaac544e8ad8b81a45bf3988a1f45).
Only change: input / output paths can be set with FF_DSEL01_SCRIPT, FF_FLAG53_DIR, FF_FLAG42_LABELS, FF_VOX2_META, FF_OUT; the
defaults are the original team-machine paths. CHANGES.md lists it.
"""
import json, importlib.util, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
_s = importlib.util.spec_from_file_location("dsel01groups", os.environ.get("FF_DSEL01_SCRIPT", EXP / "DSEL-01/make_groups.py"))
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)          # SA / EU / EN sets of DSEL-01
VB2 = Path(os.environ.get("FF_FLAG53_DIR", EXP / "SCALE-ID-01/feats_vox2b"))                       # original: EXP / "SCALE-ID-01/feats_vox2b"
LABELS = Path(os.environ.get("FF_FLAG42_LABELS", EXP / "T3ENC-01/feats_vox2lid/lid_labels.json"))   # original: inline in main
VOX2_META = Path(os.environ.get("FF_VOX2_META", HERE / "vox2_meta.csv"))                           # original: HERE / "vox2_meta.csv"
OUT = Path(os.environ.get("FF_OUT", HERE))                                                         # original: outputs next to this file


def group(lang):
    return "SA" if lang in G.SA else "EN" if lang in G.EN else "EU" if lang in G.EU else "OTH"


if __name__ == "__main__":
    labs = [l.split(":")[0] for l in json.loads(LABELS.read_text())]
    vm = pd.read_csv(VB2 / "voice_meta.csv", dtype={"spk": str, "video": str, "utt": str, "kind": str})
    P = np.load(VB2 / "voice_lid_probs.npy")
    assert len(P) == len(vm) and P.shape[1] == len(labs) == 107
    vm["lid"] = np.array(labs)[P.astype(np.float32).argmax(1)]
    out = {}
    for s, g in vm[vm.kind == "new"].groupby("spk"):
        vc = g.lid.value_counts()
        out[s] = [group(vc.index[0]), vc.index[0], round(float(vc.iloc[0] / len(g)), 3)]
    (OUT / "vox2b_groups.json").write_text(json.dumps(out, indent=0, sort_keys=True))
    M = pd.read_csv(VOX2_META, dtype=str)
    M.columns = [c.strip().lstrip("﻿") for c in M.columns]
    M = M.apply(lambda c: c.str.strip())
    gen = dict(zip(M["VoxCeleb2 ID"], M["Gender"].str.lower()))
    (OUT / "vox2_gender.json").write_text(json.dumps(gen, indent=0, sort_keys=True))
    c = pd.Series([v[0] for v in out.values()]).value_counts().to_dict()
    sa = [s for s, v in out.items() if v[0] == "SA"]
    print(f"new speakers {len(out)}: groups {c} | SA gender {pd.Series([gen.get(s) for s in sa]).value_counts().to_dict()} | "
          f"gender known for {sum(s in gen for s in out)}/{len(out)}")
