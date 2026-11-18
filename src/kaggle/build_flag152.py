r"""FLAG_152 (LID-AUDIT-01, 3/11): MMS-LID-4017 language ID on every MAV-Celeb v1 / v2 voice file (30 s windows), the third,
independent vote next to FLAG_150's Whisper-large-v3 and VoxLingua107 (Experiment/LID-AUDIT-01/README.md). Output mms.parquet
(one row per file: m_top1s, max p(ben) / p(asm) over windows, m_flag); the decision is made locally by LID-AUDIT-01/run.py.

    python kaggle/build_flag152.py      -> kaggle/FLAG_152_mmslid.ipynb
GPU T4 x 2, Internet ON (HF model). Smoke: --smoke 'WL_MAX=6'. MAV file listing / reading reused from build_flag150.
"""
import json
from pathlib import Path
import build_flag150 as B150

ROOT = Path(__file__).resolve().parents[1]
MARK = "# VoxLingua107"
assert MARK in B150.MAV
SETUP = (B150.SETUP.replace("__RISK__", "{}").replace("/kaggle/working/lidw", "/kaggle/working/mmslid")
         .replace("FLAG_150 LIDSTRICT-01 layer 3 (Whisper-large-v3 + VoxLingua107 on MAV)", "FLAG_152 LID-AUDIT-01 (MMS-LID-4017 on MAV)"))
LIST = B150.MAV.split(MARK)[0].replace("VoxLingua107 (whole file) + Whisper (30 s windows)", "file listing")

MMS = r'''print("[cell] MMS-LID-4017 on every MAV-Celeb v1 / v2 voice file (30 s windows)", flush=True)
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
MM = os.environ.get("MMS_MODEL", "facebook/mms-lid-4017")
DEVS = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
DT = torch.float16 if DEVS[0] != "cpu" else torch.float32
FE = AutoFeatureExtractor.from_pretrained(MM)
MODS = [Wav2Vec2ForSequenceClassification.from_pretrained(MM, torch_dtype=DT).to(d).eval() for d in DEVS]
LAB = MODS[0].config.id2label
MBN, MAS = [[i for i, l in LAB.items() if l == c] for c in ("ben", "asm")]
assert len(MBN) == 1 and len(MAS) == 1, (MBN, MAS)
MBN, MAS = MBN[0], MAS[0]
log("MMS ready:", MM, len(LAB), "labels on", DEVS)
def mlid(seg, k):
    x = FE(seg, sampling_rate=SR, return_tensors="pt").input_values.to(DEVS[k], DT)
    with torch.no_grad():
        return torch.softmax(MODS[k](x).logits.float(), -1)[0].cpu().numpy()
def shard(k):
    out, t = [], time.time()
    for j in range(k, len(items), len(DEVS)):
        v, sid, lang, vid, fn, op = items[j]
        r = dict(src="mav_" + v, spk=sid, lang=lang, video=vid, file=fn)
        try:
            w = read(op)
        except Exception as e:
            r["err"] = repr(e)[:120]; out.append(r); continue
        tops, pb, pa, fl = [], 0.0, 0.0, False
        for s in range(0, max(len(w), 1), 30 * SR):
            seg = w[s:s + 30 * SR]
            if len(seg) < SR // 2 and s > 0:
                continue
            p = mlid(np.pad(seg, (0, max(0, SR // 2 - len(seg)))), k); top = int(p.argmax())
            tops.append(LAB[top]); pb = max(pb, float(p[MBN])); pa = max(pa, float(p[MAS]))
            fl = fl or LAB[top] in ("ben", "asm") or p[MBN] + p[MAS] >= 0.2
        r.update(dur=len(w) / SR, m_top1s=",".join(tops), m_pbn=pb, m_pas=pa, m_flag=fl)
        out.append(r)
        if len(out) % 500 == 0:
            log(f"GPU {k}: {len(out)} files ({len(out) / (time.time() - t):.1f} files/s)")
    return out
with ThreadPoolExecutor(len(DEVS)) as ex:
    R = [r for part in ex.map(shard, range(len(DEVS))) for r in part]
Q = pd.DataFrame(R)
Q.to_parquet(OUT / "mms.parquet")
fl = Q.m_flag.fillna(False).astype(bool) if "m_flag" in Q else pd.Series(False, index=Q.index)
log("MMS flagged files:", int(fl.sum()), Q[fl].src.value_counts().to_dict(), "| speakers", Q[fl].groupby("src").spk.nunique().to_dict(),
    "| read errors", int(Q.err.notna().sum()) if "err" in Q else 0)
log("top-1 languages:", Q.m_top1s.dropna().str.split(",").explode().value_counts().head(12).to_dict())
log("ALL CELLS OK")
'''


def main():
    cells = [SETUP, LIST, MMS]
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "cells": [{"cell_type": "markdown", "metadata": {}, "source": __doc__}] +
                   [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in cells]}
    out = ROOT / "kaggle/FLAG_152_mmslid.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    for c in cells:
        compile(c, "cell", "exec")
    print(out, f"{out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
