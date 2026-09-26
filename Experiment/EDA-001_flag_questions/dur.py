"""EDA-001 Q1-Q3: counts per split/speaker and audio duration (read from WAV headers inside the zips)."""
import zipfile, struct, json, collections, numpy as np, pandas as pd
from pathlib import Path
IN = Path(__file__).resolve().parents[2] / "Input"
def wav_info(z, name):
    with z.open(name) as f:
        h = f.read(4096)
    assert h[:4] == b"RIFF", name
    i, sr, bps, ch, n = 12, None, None, None, None
    while i < len(h) - 8:
        cid, sz = h[i:i+4], struct.unpack("<I", h[i+4:i+8])[0]
        if cid == b"fmt ":
            ch, sr = struct.unpack("<HI", h[i+10:i+16]); bps = struct.unpack("<H", h[i+22:i+24])[0]
        if cid == b"data":
            n = sz; break
        i += 8 + sz + (sz & 1)
    if n is None:  # data chunk beyond header window: fall back to file size
        n = z.getinfo(name).file_size - 44
    return sr, ch, bps, n / (sr * ch * bps / 8)
rows = []
for zn in ["train_set.zip", "dev_set.zip"]:
    z = zipfile.ZipFile(IN / zn)
    for name in z.namelist():
        if name.endswith(".wav"):
            sr, ch, bps, d = wav_info(z, name)
            parts = name.split("/")
            if zn.startswith("train"):
                split, lang, spk = "train", "English", parts[-2]
            else:
                split, lang, spk = f"dev/{parts[1]}", parts[2].split("_")[0], None
            rows.append(dict(split=split, lang=lang, spk=spk, name=name, sr=sr, ch=ch, bps=bps, dur=d))
df = pd.DataFrame(rows)
df.to_csv("wav_durations.csv", index=False)
print("sample rates", df.sr.value_counts().to_dict(), "channels", df.ch.value_counts().to_dict(), "bits", df.bps.value_counts().to_dict())
q = lambda s: s.quantile([0, .05, .25, .5, .75, .95, .99, 1]).round(2).tolist() + [round(s.sum() / 3600, 2)]
print("\nduration quantiles [min,p5,p25,p50,p75,p95,p99,max] + total hours")
for k, g in df.groupby(["split", "lang"]):
    print(k, len(g), q(g.dur))
tr = df[df.split == "train"]
per = tr.groupby("spk").size()
print("\ntrain unique wav per speaker: min", per.min(), "median", per.median(), "max", per.max(), "n_spk", len(per))
print("train wav files > 20s:", int((tr.dur > 20).sum()), "| dev > 20s:", int((df[df.split != 'train'].dur > 20).sum()))
