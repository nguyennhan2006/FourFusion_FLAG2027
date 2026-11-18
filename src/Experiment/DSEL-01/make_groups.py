r"""DSEL-01: language group of every VoxCeleb2 speaker of v11's T3 head (training data only; written 11/10 before any run).
The majority VoxLingua107 top-1 language over the speaker's clips (FLAG_42 probabilities) -> EN (en, and cy: 'Welsh' calls are
British English, p(cy) 0.79 with en second on 87 % of them), EU (other European languages), SA (South-Asian languages; Bengali
speakers are already removed by the registered LID filter), OTH (the rest).
    python make_groups.py   -> vox2_groups.json  {speaker: [group, majority language, share of clips with that language]}
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
LD = HERE.parents[0] / "T3ENC-01/feats_vox2lid"
SA = {"hi", "ur", "pa", "ta", "te", "ml", "kn", "mr", "gu", "ne", "si", "sd", "as", "or", "bn"}
EU = {"de", "fr", "it", "es", "nl", "ca", "gl", "pt", "sv", "pl", "da", "no", "nn", "fi", "ro", "cs", "sk", "hu", "el", "bg", "hr",
      "sl", "sr", "bs", "uk", "ru", "lt", "lv", "et", "is", "eu", "br", "oc", "lb", "sq", "mk", "be", "fo", "gv", "la", "mt", "yi"}
EN = {"en", "cy"}

if __name__ == "__main__":
    vm = pd.read_csv(LD / "voice_meta.csv", dtype={"video": str, "utt": str})
    labs = [l.split(":")[0] for l in json.loads((LD / "lid_labels.json").read_text())]
    P = np.load(LD / "voice_lid_probs.npy")
    assert len(P) == len(vm)
    vm["lid"] = np.array(labs)[P.argmax(1)]
    out = {}
    for s, g in vm.groupby("spk"):
        vc = g.lid.value_counts()
        lang = vc.index[0]
        grp = "EN" if lang in EN else "EU" if lang in EU else "SA" if lang in SA else "OTH"
        out[s] = [grp, lang, round(float(vc.iloc[0] / len(g)), 3)]
    (HERE / "vox2_groups.json").write_text(json.dumps(out, sort_keys=True))
    print(pd.Series([v[0] for v in out.values()]).value_counts().to_dict(), "->", HERE / "vox2_groups.json")
