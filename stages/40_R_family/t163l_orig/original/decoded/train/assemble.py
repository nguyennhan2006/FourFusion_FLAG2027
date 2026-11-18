"""RFS-01: ghep dau ra FLAG_162 (a / b / c) + FLAG_160 (whv3 a / b / c) + meta thanh bundle cua rfs_train.py (ext.npz, v4.npz).
    python assemble.py --out DIR --rfs_a D --rfs_b D --rfs_c D --wh_a D --wh_b D --wh_c D --extmeta D --rawv4 D
Bo cuc dau vao (README 'Buoc 1'): a: face_<rfclip|rfibf|rfagef>_<split>.npy, voice_<rfsam|rfr221|rfxv|rfibv|rfagev>_<split>.npy (split 'train' = hang train_English.txt),
ext_<v1_complete|v2_complete>_<face|voice>_<mang>.npy (thu tu ext_*_meta.csv); b / c: <vox|vox2b|vb>/{face,voice}_<mang>.npy theo face_meta.csv / voice_meta.csv
(hang khong thuoc luot do = NaN; hop b + c). whv3: voice_whv3_<split>.npy, ext_<src>_voice_whv3.npy, <src>/voice_whv3.npy."""
import argparse, io
from pathlib import Path
import numpy as np, pandas as pd

F32 = lambda X: np.asarray(X, np.float32)


def merge(*arrs):
    """hop cac luot: hang NaN cua luot truoc duoc thay boi luot sau (hang dau tien huu han thang)."""
    out = F32(arrs[0]).copy()
    for a in arrs[1:]:
        if a is None: continue
        bad = ~np.isfinite(out).all(1); out[bad] = F32(a)[bad]
    return out


def find(root, pat):
    r = sorted(Path(root).rglob(pat)); assert r, f"thieu {pat} trong {root}"; return r[0]


def load(root, rel):
    p = Path(root) / rel
    return np.load(p) if p.exists() else None


def aux_tables(a):
    """VoxCeleb2 (vox = FLAG_37 + vox2b = FLAG_53) va VoxBlink (vb): moi nguon mot meta; mat va giong rieng."""
    F, V = {k: [] for k in "spk vid src clip ibf".split()}, {k: [] for k in "spk vid src utt sam r221 xv whv3".split()}
    for s in ("vox", "vox2b", "vb"):
        src = "vb" if s == "vb" else "vox"
        fm = pd.read_csv(find(a.rfs_b, f"{s}/face_meta.csv"), dtype=str); vm = pd.read_csv(find(a.rfs_b, f"{s}/voice_meta.csv"), dtype=str)
        g = lambda ds, rel: load(ds, rel)
        for blk, tab, nm in (("rfclip", F, "clip"), ("rfibf", F, "ibf")):
            X = merge(*[x for x in (g(a.rfs_b, f"{s}/face_{blk}.npy"), g(a.rfs_c, f"{s}/face_{blk}.npy")) if x is not None])
            tab.setdefault("_" + nm, []).append(X)
        for blk, nm in (("rfsam", "sam"), ("rfr221", "r221"), ("rfxv", "xv")):
            V.setdefault("_" + nm, []).append(merge(*[x for x in (g(a.rfs_b, f"{s}/voice_{blk}.npy"), g(a.rfs_c, f"{s}/voice_{blk}.npy")) if x is not None]))
        wh = [x for x in (g(a.wh_b, f"{s}/voice_whv3.npy"), g(a.wh_c, f"{s}/voice_whv3.npy")) if x is not None]
        V.setdefault("_whv3", []).append(merge(*wh) if wh else np.full((len(vm), 1280), np.nan, np.float32))
        F["spk"] += list(fm.spk); F["vid"] += [f"{k}/{v}" for k, v in zip(fm.spk, fm.video)]; F["src"] += [src] * len(fm)
        V["spk"] += list(vm.spk); V["vid"] += [f"{k}/{v}" for k, v in zip(vm.spk, vm.video)]; V["src"] += [src] * len(vm); V["utt"] += list(vm.utt)
    return F, V


def mav_tables(a):
    F, V = dict(spk=[], vid=[], clip=[], ibf=[]), dict(spk=[], vid=[], path=[], sam=[], r221=[], xv=[], whv3=[])
    for v in ("v1", "v2"):
        fm = pd.read_csv(find(a.extmeta, f"ext_{v}_complete_face_meta.csv")); vm = pd.read_csv(find(a.extmeta, f"ext_{v}_complete_voice_meta.csv"))
        ext = lambda kind, blk, ds: load(ds, f"ext_{v}_complete_{kind}_{blk}.npy")
        F["spk"] += [f"{v}_{s}" for s in fm.spk]; F["vid"] += [f"{v}_{s}" for s in fm.video]
        for blk, nm in (("rfclip", "clip"), ("rfibf", "ibf")): F[nm].append(F32(ext("face", blk, a.rfs_a)))
        V["spk"] += [f"{v}_{s}" for s in vm.spk]; V["vid"] += [f"{v}_{s}" for s in vm.video]; V["path"] += [f"{v}/{p}" for p in vm.path]
        for blk, nm in (("rfsam", "sam"), ("rfr221", "r221"), ("rfxv", "xv")): V[nm].append(F32(ext("voice", blk, a.rfs_a)))
        w = ext("voice", "whv3", a.wh_a); V["whv3"].append(F32(w) if w is not None else np.full((len(vm), 1280), np.nan, np.float32))
    return F, V


def v4_table(a):
    """70 nguoi v4 train: moi hang train_English.txt la mot clip (mat + giong); vid = so hang (cac hang khac nhau la 'video' khac nhau)."""
    txt = pd.read_csv(find(a.rawv4, "train_English.txt"), sep=" ", header=None)
    wav, jpg, spk = txt[2].astype(str), txt[3].astype(str), txt[4].astype(str)
    gmap = dict(l.strip().split(",")[:2] for l in open(a.gender_csv, encoding="utf-8") if l.strip()) if a.gender_csv else {}
    gend = np.array([1 if gmap.get(s) == "m" else 0 for s in spk], np.int8)
    # BTC train_set.zip: CSV o train_set/features/{faces,voices}/, khong cung thu muc train_English.txt -> tim len toi 3 cap cha
    up = lambda n: next((r[0] for p in [Path(a.rawv4), *Path(a.rawv4).parents][:4] for r in [sorted(p.rglob(n))] if r), None) or find(a.rawv4, n)
    rd = lambda n: pd.read_csv(up(n), header=None).to_numpy(np.float32)
    xf, xv = rd("train_English_faces.csv"), rd("train_English_voices.csv"); assert len(xf) == len(xv) == len(txt), (xf.shape, xv.shape, len(txt))
    L = lambda kind, blk: F32(np.load(find(a.rfs_a, f"{kind}_{blk}_train.npy")))
    wv = find(a.wh_a, "voice_whv3_train.npy")
    af, av = L("face", "rfagef"), L("voice", "rfagev")
    return dict(f_spk=spk.to_numpy(str), f_vid=np.array([f"r{i}" for i in range(len(txt))]), f_gender=gend, f_path=jpg.to_numpy(str),
                v_spk=spk.to_numpy(str), v_vid=np.array([f"r{i}" for i in range(len(txt))]), v_gender=gend, v_path=wav.to_numpy(str),
                f_clip=L("face", "rfclip"), f_ibf=L("face", "rfibf"), f_xf=xf, v_sam=L("voice", "rfsam"), v_r221=L("voice", "rfr221"), v_xv=xv,
                v_whv3=F32(np.load(wv)), f_age=af[:, 0], v_age=av[:, 0])


def main():
    ap = argparse.ArgumentParser()
    for k in ("out", "rfs_a", "rfs_b", "rfs_c", "wh_a", "wh_b", "wh_c", "extmeta", "rawv4", "gender_csv"): ap.add_argument("--" + k)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Fa, Va = aux_tables(a); Fm, Vm = mav_tables(a); cat = lambda xs: np.concatenate(xs)
    E = dict(f_spk=np.array(Fa["spk"] + Fm["spk"]), f_vid=np.array(Fa["vid"] + Fm["vid"]), f_src=np.array(Fa["src"] + ["mav"] * len(Fm["spk"])),
             f_clip=cat(Fa["_clip"] + Fm["clip"]), f_ibf=cat(Fa["_ibf"] + Fm["ibf"]),
             v_spk=np.array(Va["spk"] + Vm["spk"]), v_vid=np.array(Va["vid"] + Vm["vid"]), v_src=np.array(Va["src"] + ["mav"] * len(Vm["spk"])),
             v_utt=np.array(Va["utt"] + ["0"] * len(Vm["spk"])), v_path=np.array([""] * len(Va["spk"]) + Vm["path"]),
             v_sam=cat(Va["_sam"] + Vm["sam"]), v_r221=cat(Va["_r221"] + Vm["r221"]), v_xv=cat(Va["_xv"] + Vm["xv"]), v_whv3=cat(Va["_whv3"] + Vm["whv3"]).astype(np.float16))
    np.savez(out / "ext.npz", **{k: (v.astype(np.float16) if k in ("f_clip", "f_ibf", "v_sam", "v_r221", "v_xv") else v) for k, v in E.items()})
    np.savez(out / "v4.npz", **v4_table(a))
    print("bundle:", {k: v.shape for k, v in E.items()}, flush=True)


if __name__ == "__main__":
    main()
