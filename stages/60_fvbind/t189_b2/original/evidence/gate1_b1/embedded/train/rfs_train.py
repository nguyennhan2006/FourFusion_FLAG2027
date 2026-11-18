"""RFS-01 buoc 2 (FLAG_163): hoc lai code hoc cua he RF-092 theo bao cao, ra artifact ma FinalScorer cua flag_final_core_092 nap duoc.
Lenh: synth | split | lora | oof | proj | fuse | release | all (= split oof proj fuse release).
Dau vao (bundle, xem README 'Hop dong bundle'): ext.npz (nguoi phu: f_*, v_*), v4.npz (70 nguoi v4 train), v4_extra.npz (lora, tuoi; do lenh lora sinh ra).
Cau hinh qua bien moi truong: RFS_STEPS (3000), RFS_NTRIAL (3000), RFS_RESERVE_IN_FINAL (1), RFS_LORA_STEPS (1200), RFS_LORA_BATCH (64), RFS_DEV."""
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SEED = 20261103
CFG = dict(dim=512, drop_f=0.8, drop_v=0.5, lr=5e-4, wd=1e-4, steps=3000, warmup=100, persons=512, tau=0.05, seeds=[1, 2, 3])
# RFSTUNE-01 (11/11): opt-in JSON overrides of CFG, plus v4w (weight of the v4 same-gender term, 0.5) and v4p (v4 persons per
# step, 32). Unset = unchanged. RFS_STEPS still wins over CFG["steps"] (smoke).
_OV = json.loads(os.environ.get("RFS_CFG") or "{}")
# ENPROG-01 (12/11): "zero" = list of encoder blocks (f_clip, f_ibf, v_sam, v_r221, v_xv, v_whv3) set to 0 in the lab only
# (projector input keeps its size; a zero block carries no information). Unset = unchanged.
# ENGLISH-02 (12/11, HARDNEG-RFS / EN-HEARD): "hard" = share of steps whose persons are the hard_k face-centroid neighbours of a
# random anchor (same look -> same gender / near age negatives), "src_w" = person sampling weight per source, "mav_en" = MAV voice
# rows of English clips only. Unset = unchanged (same random stream).
# FACTOR-01 (13/11, Experiment/FACTOR-01/README.md): "add" = extra encoder blocks (XBLK: f_c5s, f_farl, f_pel, v_rl, v_sf, v_ibv)
# appended to the projector input, each unit-normalised like the blocks above. Lab only; unset = unchanged (core scoring).
# TRANSL-01 (14/11, Experiment/TRANSL-01/README.md): "orth" = weight of ||W W^T - I||_F^2 / d_out on the two Linear layers (Procrustes-like),
# "cyc" = weight of the back-translation loss (shared space -> each side's input, linear decoders outside ProjQ), "mpos" = 2 pairs per
# person in the batch with multi-positive symmetric InfoNCE. Unset = unchanged (same random stream).
assert set(_OV) <= set(CFG) | {"v4w", "v4p", "zero", "hard", "hard_k", "src_w", "mav_en", "add", "orth", "cyc", "mpos"}, f"RFS_CFG keys {sorted(_OV)}"
CFG.update(_OV)
XF = [k for k in CFG.get("add", []) if k.startswith("f_")]
XV = [k for k in CFG.get("add", []) if k.startswith("v_")]
STREAMS_G, STREAMS_NG = ["t3", "ib", "nf", "nv"], ["t3", "ib", "nf", "nv", "age"]
COND = ["p_face_male", "p_voice_male", "gender_agree", "protocol_gender"]
DEV = torch.device(os.environ.get("RFS_DEV") or ("cuda" if torch.cuda.is_available() else "cpu"))
log = lambda *a: print(*a, flush=True)
_C = None


def core():
    global _C
    if _C is None:
        d = os.environ.get("RFS_CORE_DIR") or str(HERE.parent / "ref")
        sys.path.insert(0, d)
        import flag_final_core_092 as C
        _C = C
    return _C


def U(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def face_mat(clip, ibf, extra=()):
    return np.hstack([U(clip), U(ibf)] + [U(np.nan_to_num(np.asarray(e, np.float32))) for e in extra]).astype(np.float32)


NOWH = os.environ.get("RFS_NO_WHV3") == "1"   # RFS-NOWH (ablation, 4/11): projector without the Whisper block; release skipped


def voice_mat(sam, r2, xv, wh, extra=()):   # thu tu cua t3_scores: sam, r221, ecapa, whisper (+ FACTOR-01 extra blocks)
    return np.hstack([U(sam), U(r2), U(xv)] + ([] if NOWH else [U(wh)]) +
                     [U(np.nan_to_num(np.asarray(e, np.float32))) for e in extra]).astype(np.float32)


# ---------------------------------------------------------------- FACTOR-01 extra blocks (opt-in through RFS_CFG "add")
# part -> (file pattern, parent hint). Rows: vox = FLAG_37, vox2b = FLAG_53, vb = FLAG_46 (each source's face_meta / voice_meta order,
# = the aux order of assemble.py), ext = MAV v1 then v2 (= the bundle's mav rows), v4 = train_English.txt rows. Hint "=x": parent
# folder named x; another string: substring of the path. Several files for one part (two lanes, or one file in two datasets) are
# merged row-wise (first finite row wins). Missing rows -> 0 block (carries no information); coverage and an identity check are logged.
_A = lambda h=None: dict(vox=("vox_{k}_{a}.npy", h), vox2b=("vox2b_{k}_{a}.npy", h), vb=("vb_{k}_{a}.npy", h), ext=("ext_{s}_{k}_{a}.npy", h),
                         v4=("{k}_{a}_train.npy", h))
_T3 = dict(vox=("{k}_{a}.npy", "feats_vox2t3"), vox2b=("{k}_{a}.npy", "feats_vox2b"), vb=("{k}_{a}.npy", "feats_vb"),
           ext=("ext_{s}_{k}_{a}.npy", "feats_t3enc"), v4=("{k}_{a}_train.npy", "feats_t3enc"))
_AV = dict(vox=("{k}_{a}.npy", "=vox"), vox2b=("{k}_{a}.npy", "=vox2b"), vb=("{k}_{a}.npy", "=vb"), ext=("ext_{s}_{k}_{a}.npy", None),
           v4=("{k}_{a}_train.npy", None))
XBLK = {"f_c5s": (["clipC5"], _A()),            # CLIP-L + C5s LoRA adapter (quynhnhu82/flag2027-feats-ls-c5s-p)
        "f_pel": (["peL"], _A()),               # PE-Core-L14-336 frozen (trngsnam/flag2027-feats-peface01-pe0-p)
        "f_farl": (["farl16"], _T3),            # FaRL ep16 of the T3 head (lab-data / feats-vox2b / lab-vb)
        "v_sf": (["sam100f"], _T3),             # SimAM-ResNet100 ft VoxCeleb2 of the T3 head
        "v_rl": (["rn293", "ecapa1024"], _AV),  # CAT4 extras: ResNet293-LM + ECAPA1024-LM (hthlin/flag2027-aux-voice-feats-p, feats-screen2-p)
        "v_ibv": (["rfibv"], _AV)}              # ImageBind voice of FLAG_162 (feats-rfs-a / b / c), not in the RFS projector so far
_XI = None


def _xload(name, hint):
    global _XI
    if _XI is None:
        roots = [Path(x) for x in os.environ.get("RFS_XDIRS", "").split(os.pathsep) if x] or [Path("/kaggle/input")]
        _XI = {}
        for r in roots:
            for q in (r.rglob("*.npy") if r.exists() else []):
                _XI.setdefault(q.name, []).append(q)
    c = sorted(set(_XI.get(name, [])))
    if hint:
        c = [q for q in c if (q.parent.name == hint[1:] if hint.startswith("=") else hint in q.as_posix())]
    assert c, f"FACTOR-01: {name} [{hint}] not found: attach its dataset"
    A = [np.load(q).astype(np.float32) for q in c]
    assert len({a.shape for a in A}) == 1, (name, [str(q) for q in c], [a.shape for a in A])
    out = A[0].copy()
    for a in A[1:]:
        bad = ~np.isfinite(out).all(1); out[bad] = a[bad]
    log(f"  {name} [{hint}]: {out.shape}, {len(c)} file(s), finite {np.isfinite(out).all(1).mean():.4f} ({c[0].parent})")
    return out


def _xpart(key, kind, part):
    arrs, spec = XBLK[key]; pat, hint = spec[part]
    subs = ("v1_complete", "v2_complete") if part == "ext" else (None,)
    cols = []
    for a in arrs:
        X = np.concatenate([_xload(pat.format(k=kind, a=a, s=s_), hint) for s_ in subs])
        cols.append(X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8))      # NaN rows stay NaN
    return np.concatenate(cols, 1)


def _id_eer(X, spk, vid, seed=0, n=4000):
    """identity check of one block on one part: cos of same-person rows (other video when possible) vs different persons."""
    rng = np.random.default_rng(seed); ok = np.isfinite(X).all(1) & (np.abs(np.nan_to_num(X)).sum(1) > 0); idx = np.nonzero(ok)[0]
    by = {}
    for i in idx:
        by.setdefault(spk[i], []).append(i)
    multi = [k for k, v in by.items() if len(v) >= 2]
    if len(multi) < 20:
        return float("nan")
    a = np.array([rng.choice(by[multi[rng.integers(len(multi))]], 2, replace=False) for _ in range(n)])
    b = idx[rng.integers(0, len(idx), (n, 2))]; b = b[spk[b[:, 0]] != spk[b[:, 1]]]
    Xn = np.nan_to_num(X)
    sc = np.r_[(Xn[a[:, 0]] * Xn[a[:, 1]]).sum(1), (Xn[b[:, 0]] * Xn[b[:, 1]]).sum(1)]
    return eer(sc, np.r_[np.ones(len(a)), np.zeros(len(b))])


def load_extra(E, V4, keys):
    """E[k] / V4[k] for every k in keys (rows of the bundle); asserts the row layout and the identity check (EER < 35 on every part)."""
    stand = os.environ.get("FLAG_SMOKE") == "1" or os.environ.get("RFS_X_STANDIN") == "1"
    for k in keys:
        assert k in XBLK, f"unknown FACTOR-01 block {k} (known {sorted(XBLK)})"
        kind, pre = ("face", "f") if k.startswith("f_") else ("voice", "v")
        if stand:
            rng = np.random.default_rng(len(k)); log(f"!! FACTOR-01 SMOKE STAND-IN: {k} (person code + noise)")
            for D in (E, V4):
                sp = np.asarray(D[f"{pre}_spk"]).astype(str); u, inv = np.unique(sp, return_inverse=True)
                D[k] = (rng.normal(size=(len(u), 48))[inv] + rng.normal(size=(len(sp), 48))).astype(np.float32)
            continue
        parts = {p_: _xpart(k, kind, p_) for p_ in ("vox", "vox2b", "vb", "ext")}
        n = {p_: len(x) for p_, x in parts.items()}; src = np.asarray(E[f"{pre}_src"]).astype(str)
        assert sum(n.values()) == len(src), f"{k}: parts {n} != bundle {kind} rows {len(src)}"
        a1, a2 = n["vox"] + n["vox2b"], n["vox"] + n["vox2b"] + n["vb"]
        assert (src[:a1] == "vox").all() and (src[a1:a2] == "vb").all() and (src[a2:] == "mav").all(), f"{k}: source boundaries"
        X = np.concatenate([parts[p_] for p_ in ("vox", "vox2b", "vb", "ext")])
        X4 = _xpart(k, kind, "v4"); assert len(X4) == len(V4[f"{pre}_spk"]), f"{k}: v4 rows {len(X4)} != {len(V4[f'{pre}_spk'])}"
        spk, vid = np.asarray(E[f"{pre}_spk"]).astype(str), np.asarray(E[f"{pre}_vid"]).astype(str)
        base = "f_clip" if pre == "f" else "v_sam"
        for nm, sl in (("vox", slice(0, n["vox"])), ("vox2b", slice(n["vox"], a1)), ("vb", slice(a1, a2)), ("mav", slice(a2, None))):
            e_new, e_ref = _id_eer(X[sl], spk[sl], vid[sl]), _id_eer(U(np.asarray(E[base][sl], np.float32)), spk[sl], vid[sl])
            cov = float(np.isfinite(X[sl]).all(1).mean())
            log(f"  {k} {nm:5s}: coverage {cov:.4f}, identity EER {e_new:.2f} (bundle {base} {e_ref:.2f})")
            assert np.isnan(e_new) or e_new < 35, f"{k} {nm}: identity EER {e_new:.1f} >= 35 -> rows do not match the bundle"
        e4 = _id_eer(X4, np.asarray(V4[f"{pre}_spk"]).astype(str), np.asarray(V4[f"{pre}_vid"]).astype(str))
        log(f"  {k} v4   : coverage {np.isfinite(X4).all(1).mean():.4f}, identity EER {e4:.2f}")
        assert np.isfinite(X4).all(), f"{k}: v4 rows must be complete"
        E[k], V4[k] = np.nan_to_num(X).astype(np.float32), X4.astype(np.float32)


def t3x(M, Fx, Vx):
    f, v = torch.tensor(Fx), torch.tensor(Vx)
    with torch.no_grad():
        return np.mean([(F.normalize(m.f(f), dim=-1) * F.normalize(m.v(v), dim=-1)).sum(1).numpy() for m in M], 0)


def qfx(M, Fx, Vx):
    f, v = torch.tensor(Fx), torch.tensor(Vx)
    with torch.no_grad():
        return (np.mean([torch.log(m.f(f).norm(dim=1)).numpy() for m in M], 0), np.mean([torch.log(m.v(v).norm(dim=1)).numpy() for m in M], 0))


# >>> XSCORER (FACTOR-01 step 3, 13/11): copied verbatim into FLAG_163i by kaggle/patch_163i_factor.py; uses only L (the core module).
def make_xscorer(L):
    """FinalScorer of the core with the projector input widened by the FACTOR-01 blocks listed in manifest['add']
    ({'face': [[name, dim], ...], 'voice': [...]}, appended after the core blocks, each unit-normalised). Same fills, ib / gender /
    age streams and fusers as L.FinalScorer.score; only t3 / nf / nv read the wider input. No 'add' -> L.FinalScorer unchanged."""
    import numpy as np, torch, torch.nn.functional as F
    from pathlib import Path

    def _u(X):
        X = np.nan_to_num(np.asarray(X, np.float32)); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    class XScorer(L.FinalScorer):
        def __init__(self, art):
            art = Path(art); self.man = L.read_json(art / 'manifest_final.json')
            with np.load(art / 'final_params.npz') as z:
                self.p = {k: z[k] for k in z.files}
            self.vx = [b['block'] for b in self.man['t3_voice_extra']]
            add = self.man.get('add') or {}
            self.xf = [k for k, _ in add.get('face', [])]; self.xv = [k for k, _ in add.get('voice', [])]
            df = 768 + 1024 + sum(int(d) for _, d in add.get('face', []))
            dv = 256 + 256 + 192 + sum(int(b['dim']) for b in self.man['t3_voice_extra']) + sum(int(d) for _, d in add.get('voice', []))
            self.proj = []
            for sd in torch.load(art / 'projector_Q.pt', map_location='cpu'):
                m = L.ProjQ(df, dv, self.man['projector']); m.load_state_dict(sd); self.proj.append(m.eval())

        def score(self, x, proto):
            if not (self.xf or self.xv):
                return L.FinalScorer.score(self, x, proto)
            p = self.p; fill = lambda k, X: L.fill_nonfinite(X, p[f'fill_{k}'])[0]
            for k in self.xf + self.xv:
                assert np.isfinite(np.asarray(x[k], np.float32)).all(), f'extra block {k}: non-finite rows'
            xf, xv, ibf, r2 = fill('xf', x['xf']), fill('xv', x['xv']), fill('ib_face', x['ibf']), fill('r221', x['r221'])
            cl, sam, vext = fill('clip', x['clip']), fill('sam', x['sam']), [fill(b, x[b]) for b in self.vx]
            f = torch.tensor(np.hstack([_u(cl), _u(ibf)] + [_u(x[k]) for k in self.xf]))
            v = torch.tensor(np.hstack([_u(sam), _u(r2), _u(xv)] + [_u(e) for e in vext] + [_u(x[k]) for k in self.xv]))
            with torch.no_grad():
                s = {'t3': np.mean([(F.normalize(m.f(f), dim=-1) * F.normalize(m.v(v), dim=-1)).sum(1).numpy() for m in self.proj], 0),
                     'nf': np.mean([torch.log(m.f(f).norm(dim=1)).numpy() for m in self.proj], 0),
                     'nv': np.mean([torch.log(m.v(v).norm(dim=1)).numpy() for m in self.proj], 0)}
            s['ib'] = (L._unit(ibf - p['mu_ibf']) * L._unit(fill('lora', x['lora']) - p['fill_lora'])).sum(1)
            pf = 1 / (1 + np.exp(-(((xf - p['prep_mf']) / p['prep_sf']) @ p['prep_P']).astype(np.float32) @ p['gf_coef'] - p['gf_b']))
            pv = 1 / (1 + np.exp(-((xv - p['prep_mv']) / p['prep_sv']).astype(np.float32) @ p['gv_coef'] - p['gv_b']))
            s.update(p_face_male=pf, p_voice_male=pv, gender_agree=pf * pv + (1 - pf) * (1 - pv), protocol_gender=np.full(len(xf), float(proto == 'gender')))
            if proto == 'gender':
                out = -self._lr('fg', np.stack([s[k] for k in self.man['streams_g']], 1))
            else:
                a_f = np.where(np.isfinite(x['age_f']), x['age_f'], p['age_fill'][0]); a_v = np.where(np.isfinite(x['age_v']), x['age_v'], p['age_fill'][1])
                s['age'] = -np.abs(a_f - np.polyval(p['age_b'], a_v))
                out = -self._lr('fn', L.fuser_matrix4(s, self.man['streams_ng']))
            if not np.isfinite(out).all(): raise ValueError('non-finite scores')
            return out
    return XScorer
# <<< XSCORER


def eer(score, y):   # score cao = cung nguoi
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, score)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return 100 * float((fpr[i] + 1 - tpr[i]) / 2)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ du lieu
def load_bundle(bd, work=None):
    bd = Path(bd)
    E, V4 = dict(np.load(bd / "ext.npz")), dict(np.load(bd / "v4.npz"))
    for D in (E, V4):   # float16 -> float32 (norm / tong binh phuong khong tran)
        for k, v in D.items():
            if v.dtype == np.float16 and k != "v_whv3": D[k] = v.astype(np.float32)
    for p in ([Path(work) / "v4_extra.npz"] if work else []) + [bd / "v4_extra.npz"]:
        if p.exists():
            V4.update(dict(np.load(p))); break
    return E, V4


def clean_ext(E, lid):
    """Pool sach: bo giong cua nguoi trong lid_strict (vox2, vb) va file MAV trong mav_files; hang khong huu han bi bo. Mat cua nguoi bi bo giong van duoc giu."""
    drop = {"vox": set(lid.get("vox2", [])), "vb": set(lid.get("vb", []))}
    bad = set(lid.get("mav_files", []))
    keep = np.array([not ((s in drop and k in drop[s]) or (s == "mav" and p in bad)) for s, k, p in zip(E["v_src"], E["v_spk"], E["v_path"])])
    keep &= np.isfinite(E["v_sam"]).all(1) & np.isfinite(E["v_r221"]).all(1) & np.isfinite(E["v_xv"]).all(1) & np.isfinite(E["v_whv3"].astype(np.float32)).all(1)
    fk = np.isfinite(E["f_clip"]).all(1) & np.isfinite(E["f_ibf"]).all(1)
    log(f"pool sach: giong {int(keep.sum())}/{len(keep)}, mat {int(fk.sum())}/{len(fk)}")
    return keep, fk


def pair_table(fkey, fvid, vkey, vvid, K, rng, strict=True):
    """Moi nguoi: K cap (hang mat, hang giong) ngau nhien, mat va giong tu hai video khac nhau (strict: bo nguoi khong co cap nhu vay)."""
    fr, vr = {}, {}
    for i, k in enumerate(fkey): fr.setdefault(k, []).append(i)
    for i, k in enumerate(vkey): vr.setdefault(k, []).append(i)
    names, PF, PV = [], [], []
    for k in sorted(fr.keys() & vr.keys()):
        fi, vi = np.array(fr[k]), np.array(vr[k])
        ok = fvid[fi][:, None] != vvid[vi][None, :]
        if not ok.any():
            if strict: continue
            ok = np.ones_like(ok)
        a, b = np.nonzero(ok); s = rng.integers(0, len(a), K)
        names.append(k); PF.append(fi[a[s]]); PV.append(vi[b[s]])
    return names, np.array(PF).reshape(len(names), K), np.array(PV).reshape(len(names), K)


class Pool:
    def __init__(self, E, lid, K=64, seed=SEED):
        keep, fk = clean_ext(E, lid)
        if CFG.get("mav_en"):                                            # EN-HEARD: MAV voices of English clips only (opt-in)
            keep = keep & np.array([s != "mav" or "/English/" in str(p) for s, p in zip(E["v_src"], E["v_path"])])
        fkey = np.array([f"{s}|{k}" for s, k in zip(E["f_src"], E["f_spk"])]); vkey = np.array([f"{s}|{k}" for s, k in zip(E["v_src"], E["v_spk"])])
        fk_i, vk_i = np.nonzero(fk)[0], np.nonzero(keep)[0]
        names, PF, PV = pair_table(fkey[fk_i], E["f_vid"][fk_i], vkey[vk_i], E["v_vid"][vk_i], K, np.random.default_rng(seed))
        self.names, self.PF, self.PV = names, fk_i[PF], vk_i[PV]
        self.Fm = torch.tensor(face_mat(E["f_clip"], E["f_ibf"], [E[k] for k in XF]))
        self.Vm = torch.tensor(voice_mat(E["v_sam"], E["v_r221"], E["v_xv"], E["v_whv3"].astype(np.float32), [E[k] for k in XV]))
        log(f"pool: {len(names)} nguoi co cap mat/giong khac video (K={K})")


class V4Set:
    def __init__(self, V4, persons, K=64, seed=SEED):
        ps = set(map(str, persons))
        fi = np.nonzero([str(s) in ps for s in V4["f_spk"]])[0]; vi = np.nonzero([str(s) in ps for s in V4["v_spk"]])[0]
        names, PF, PV = pair_table(V4["f_spk"][fi], V4["f_vid"][fi], V4["v_spk"][vi], V4["v_vid"][vi], K, np.random.default_rng(seed), strict=False)
        self.names, self.PF, self.PV = names, fi[PF], vi[PV]
        g = {str(s): int(x) for s, x in zip(V4["f_spk"], V4["f_gender"])}
        self.gender = np.array([g[str(n)] for n in names])
        self.Fm = torch.tensor(face_mat(V4["f_clip"], V4["f_ibf"], [V4[k] for k in XF]))
        self.Vm = torch.tensor(voice_mat(V4["v_sam"], V4["v_r221"], V4["v_xv"], V4["v_whv3"].astype(np.float32), [V4[k] for k in XV]))


# ------------------------------------------------------------------ projector QV-Whisper
LOSS = os.environ.get("RFS_LOSS", "nce")   # OBJFAM-01 (10/11): "sig" = pairwise sigmoid loss for the projector (LoRA keeps InfoNCE); unset = unchanged
SIG_T, SIG_B = 10.0, -10.0                  # fixed SigLIP init values (no learnable params: the core's ProjQ state dict stays loadable)


def nce(f, v, tau):
    f, v = F.normalize(f, dim=-1), F.normalize(v, dim=-1)
    lg = f @ v.T / tau; y = torch.arange(len(f), device=f.device)
    return (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2


def sig(f, v):
    f, v = F.normalize(f, dim=-1), F.normalize(v, dim=-1)
    z = 2 * torch.eye(len(f), device=f.device) - 1   # +1 matched pair, -1 every other pair in the batch
    return -F.logsigmoid(z * (SIG_T * (f @ v.T) + SIG_B)).sum() / len(f)


def proj_loss(f, v, tau):
    return sig(f, v) if LOSS == "sig" else nce(f, v, tau)


def mpos_nce(f, v, lab, tau):
    """TRANSL-01 mpos: symmetric InfoNCE where every voice row of the same person is a positive of a face row (and back)."""
    f, v = F.normalize(f, dim=-1), F.normalize(v, dim=-1)
    lg = f @ v.T / tau; pos = (lab[:, None] == lab[None, :]).float()
    lf = -(pos * F.log_softmax(lg, 1)).sum(1) / pos.sum(1)
    lv = -(pos.T * F.log_softmax(lg.T, 1)).sum(1) / pos.sum(0)
    return (lf.mean() + lv.mean()) / 2


def train_proj(pool, v4, seed, cfg=CFG, steps=None, dev=DEV):
    C = core(); steps = steps or cfg["steps"]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = C.ProjQ(pool.Fm.shape[1], pool.Vm.shape[1], cfg).to(dev).train()
    orth, cyc, mpos = float(cfg.get("orth", 0.0)), float(cfg.get("cyc", 0.0)), int(cfg.get("mpos", 1))
    assert mpos in (1, 2) and (mpos == 1 or LOSS == "nce"), "TRANSL-01: mpos is 1 or 2, InfoNCE only"
    dec = None
    if cyc > 0:   # TRANSL-01 back-translation decoders (outside ProjQ: the projector state dict stays loadable by the core)
        dec = nn.ModuleDict({"f": nn.Linear(cfg["dim"], pool.Fm.shape[1]), "v": nn.Linear(cfg["dim"], pool.Vm.shape[1])}).to(dev).train()
    eye = torch.eye(cfg["dim"], device=dev) if orth > 0 else None
    opt = torch.optim.AdamW(list(m.parameters()) + (list(dec.parameters()) if dec is not None else []), lr=cfg["lr"], weight_decay=cfg["wd"])
    if orth or cyc or mpos > 1: log(f"  TRANSL-01: orth {orth}, cyc {cyc}, mpos {mpos}")
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / cfg["warmup"]))
    Fp, Vp, Fv, Vv = pool.Fm.to(dev), pool.Vm.to(dev), v4.Fm.to(dev), v4.Vm.to(dev)
    n, npp = len(pool.names), min(cfg["persons"], len(pool.names)); K = pool.PF.shape[1]
    gi = [x for x in (np.nonzero(v4.gender == g)[0] for g in (0, 1)) if len(x) >= 2]
    w = None
    if cfg.get("src_w"):                                                 # EN-HEARD: person sampling weight per source (opt-in)
        w = np.array([float(cfg["src_w"].get(nm.split("|", 1)[0], 1.0)) for nm in pool.names]); w = w / w.sum()
    hard, Cn = float(cfg.get("hard", 0.0)), None
    if hard > 0:                                                         # HARDNEG-RFS: unit face centroid per person (8 pairs)
        with torch.no_grad():
            Cn = torch.cat([torch.nn.functional.normalize(Fp[pool.PF[i:i + 2048, :8]].mean(1), dim=-1) for i in range(0, n, 2048)])
        hk = min(max(int(cfg.get("hard_k", 4 * npp)), npp), n); log(f"  hard batches: share {hard}, neighbourhood {hk} of {n} persons")
    t0 = time.time()
    for t in range(steps):
        if Cn is not None and rng.random() < hard:
            a = int(rng.integers(n)); nb = torch.topk(Cn @ Cn[a], hk).indices.cpu().numpy(); p = rng.choice(nb, npp, replace=False)
        else:
            p = rng.choice(n, npp, replace=False, p=w)
        k = rng.integers(0, K, npp)
        if mpos == 2:                                                    # TRANSL-01: a second, different pair of every person
            k2_ = (k + rng.integers(1, K, npp)) % K; pi, ki = np.concatenate([p, p]), np.concatenate([k, k2_])
            xf, xv = Fp[pool.PF[pi, ki]], Vp[pool.PV[pi, ki]]; zf, zv = m.f(xf), m.v(xv)
            loss = mpos_nce(zf, zv, torch.as_tensor(np.concatenate([np.arange(npp)] * 2), device=dev), cfg["tau"])
        else:
            xf, xv = Fp[pool.PF[p, k]], Vp[pool.PV[p, k]]; zf, zv = m.f(xf), m.v(xv)
            loss = proj_loss(zf, zv, cfg["tau"])
        extra = {}
        if eye is not None:                                              # TRANSL-01 orth: rows of each Linear stay orthonormal
            extra["orth"] = sum(((L.weight @ L.weight.T - eye) ** 2).sum() / L.weight.shape[0] for L in (m.f[2], m.v[2]))
            loss = loss + orth * extra["orth"]
        if dec is not None:                                              # TRANSL-01 cyc: face -> shared -> face, voice -> shared -> voice
            extra["cyc"] = 0.5 * ((1 - F.cosine_similarity(dec["f"](zf), xf, dim=-1)).mean() + (1 - F.cosine_similarity(dec["v"](zv), xv, dim=-1)).mean())
            loss = loss + cyc * extra["cyc"]
        if gi:   # 32 nguoi v4 train cung gioi
            ids = gi[rng.integers(len(gi))]; q = rng.choice(ids, min(int(cfg.get("v4p", 32)), len(ids)), replace=False); k2 = rng.integers(0, v4.PF.shape[1], len(q))
            loss = loss + float(cfg.get("v4w", 0.5)) * proj_loss(m.f(Fv[v4.PF[q, k2]]), m.v(Vv[v4.PV[q, k2]]), cfg["tau"])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if t % 500 == 0 or t == steps - 1:
            log(f"  proj seed {seed} step {t}/{steps} loss {loss.item():.4f}" + "".join(f" {a} {b.item():.4f}" for a, b in extra.items()) + f" [{time.time() - t0:.0f}s]")
    return {k: v.detach().cpu() for k, v in m.eval().cpu().state_dict().items()}


def load_models(sds, dv, df=768 + 1024):
    C = core(); ms = []
    for sd in sds:
        m = C.ProjQ(df, dv, CFG); m.load_state_dict(sd); ms.append(m.eval())
    return ms


def train_set(pool, v4, steps):
    return [train_proj(pool, v4, s, steps=steps) for s in CFG["seeds"]]


# ------------------------------------------------------------------ chia fold, probe, luong
def split(V4, seed=SEED):
    """70 nguoi v4 = 10 giu rieng + 60 chia 5 fold, can gioi (xep theo gioi roi chia vong tron)."""
    rng = np.random.default_rng(seed); pers = {}
    for s, g in zip(V4["f_spk"], V4["f_gender"]): pers[str(s)] = int(g)
    byg = {g: sorted(p for p in pers if pers[p] == g) for g in (0, 1)}
    for g in byg: rng.shuffle(byg[g])
    n0 = round(10 * len(byg[0]) / len(pers)) if len(pers) >= 20 else 0
    nres = {0: n0, 1: 10 - n0 if len(pers) >= 20 else 0}
    reserve = byg[0][:nres[0]] + byg[1][:nres[1]]
    rest = byg[0][nres[0]:] + byg[1][nres[1]:]
    order = sorted(rest, key=lambda p: pers[p])
    folds = [[] for _ in range(5)]
    for i, p in enumerate(order): folds[i % 5].append(p)
    return dict(seed=seed, reserve=sorted(reserve), folds=[sorted(f) for f in folds], gender=pers)


def fit_probes(V4, persons):
    """Probe gioi: mat = chuan hoa + PCA 256 + logistic tren VGG 4096; giong = chuan hoa + logistic tren ECAPA 192. Nhan 1 = nam."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    ps = set(map(str, persons)); fm = np.array([str(s) in ps for s in V4["f_spk"]]); vm = np.array([str(s) in ps for s in V4["v_spk"]])
    xf, xv = V4["f_xf"][fm].astype(np.float32), V4["v_xv"][vm].astype(np.float32)
    yf, yv = (V4["f_gender"][fm] == 1).astype(int), (V4["v_gender"][vm] == 1).astype(int)
    P = {"prep_mf": xf.mean(0), "prep_sf": xf.std(0), "prep_mv": xv.mean(0), "prep_sv": xv.std(0)}
    P["prep_sf"][P["prep_sf"] == 0] = 1; P["prep_sv"][P["prep_sv"] == 0] = 1
    zf = (xf - P["prep_mf"]) / P["prep_sf"]
    P["prep_P"] = PCA(n_components=min(256, len(zf) - 1, zf.shape[1]), svd_solver="randomized", random_state=0).fit(zf).components_.T.astype(np.float32)
    lr = LogisticRegression(max_iter=2000).fit(zf @ P["prep_P"], yf); P["gf_coef"], P["gf_b"] = lr.coef_[0].astype(np.float32), np.float32(lr.intercept_[0])
    lr = LogisticRegression(max_iter=2000).fit((xv - P["prep_mv"]) / P["prep_sv"], yv); P["gv_coef"], P["gv_b"] = lr.coef_[0].astype(np.float32), np.float32(lr.intercept_[0])
    return P


def fit_age(V4, persons):
    """age_b: duong thang tuoi mat ~ b(tuoi giong), tren trung binh tung nguoi v4 train."""
    ps = set(map(str, persons)); a = {}
    for s, x in zip(V4["f_spk"], V4["f_age"]): a.setdefault(str(s), [[], []])[0].append(x)
    for s, x in zip(V4["v_spk"], V4["v_age"]): a.setdefault(str(s), [[], []])[1].append(x)
    pts = np.array([[np.nanmean(v[0]), np.nanmean(v[1])] for s, v in a.items() if s in ps and len(v[0]) and len(v[1])])
    pts = pts[np.isfinite(pts).all(1)]
    return np.polyfit(pts[:, 1], pts[:, 0], 1).astype(np.float64)


def const_params(V4, persons):
    """Hang so tu v4 train: trung binh lap cho hang loi (fill_*) va tam tru (mu_ibf, fill_lora)."""
    ps = set(map(str, persons)); fm = np.array([str(s) in ps for s in V4["f_spk"]]); vm = np.array([str(s) in ps for s in V4["v_spk"]])
    mean = lambda X: np.nanmean(X.astype(np.float32), 0).astype(np.float32)
    P = dict(fill_xf=mean(V4["f_xf"][fm]), fill_xv=mean(V4["v_xv"][vm]), fill_ib_face=mean(V4["f_ibf"][fm]), fill_clip=mean(V4["f_clip"][fm]),
             fill_r221=mean(V4["v_r221"][vm]), fill_sam=mean(V4["v_sam"][vm]), fill_whv3=mean(V4["v_whv3"][vm]), fill_lora=mean(V4["v_lora"][vm]),
             age_fill=np.array([np.nanmean(V4["f_age"][fm]), np.nanmean(V4["v_age"][vm])], np.float64))
    P["mu_ibf"] = P["fill_ib_face"].copy(); P["fill_ib_voice"] = P["fill_lora"].copy()
    return P


def trial_streams(M, V4, fi, vi, proto, P):
    """Cac luong cua danh sach cap (hang mat fi, hang giong vi) cua v4; cung phep tinh voi FinalScorer.score truoc buoc tron."""
    C = core(); n = lambda X: np.asarray(X, np.float32)
    cl, ibf, sam, r2, xv, wh = n(V4["f_clip"][fi]), n(V4["f_ibf"][fi]), n(V4["v_sam"][vi]), n(V4["v_r221"][vi]), n(V4["v_xv"][vi]), n(V4["v_whv3"][vi])
    vx = [] if NOWH else [wh]
    if XF or XV:                                                       # FACTOR-01: same maths as the core, wider input
        Fx = face_mat(cl, ibf, [n(V4[k][fi]) for k in XF]); Vx = voice_mat(sam, r2, xv, wh, [n(V4[k][vi]) for k in XV])
        s = {"t3": t3x(M, Fx, Vx)}; s["nf"], s["nv"] = qfx(M, Fx, Vx)
    else:
        s = {"t3": C.t3_scores(M, cl, ibf, sam, r2, xv, vextra=vx)}
        s["nf"], s["nv"] = C.qf_norms(M, cl, ibf, sam, r2, xv, vx)
    s["ib"] = (C._unit(ibf - P["mu_ibf"]) * C._unit(n(V4["v_lora"][vi]) - P["fill_lora"])).sum(1)
    xf = n(V4["f_xf"][fi])
    pf = 1 / (1 + np.exp(-(((xf - P["prep_mf"]) / P["prep_sf"]) @ P["prep_P"]).astype(np.float32) @ P["gf_coef"] - P["gf_b"]))
    pv = 1 / (1 + np.exp(-((xv - P["prep_mv"]) / P["prep_sv"]).astype(np.float32) @ P["gv_coef"] - P["gv_b"]))
    s.update(p_face_male=pf, p_voice_male=pv, gender_agree=pf * pv + (1 - pf) * (1 - pv), protocol_gender=np.full(len(fi), float(proto == "gender")))
    s["age"] = -np.abs(V4["f_age"][fi] - np.polyval(P["age_b"], V4["v_age"][vi]))
    return s


def make_trials(V4, persons, gender, n, proto, rng):
    """n cap dung (cung nguoi, uu tien khac video) + n cap sai (gender: cung gioi; no_gender: bat ky) tren cac hang v4 cua `persons`."""
    S = V4Set(V4, persons, K=64, seed=int(rng.integers(1 << 30)))
    q = rng.integers(0, len(S.names), n); k = rng.integers(0, 64, n)
    fi, vi, y = list(S.PF[q, k]), list(S.PV[q, k]), [1] * n
    fr = {nm: np.nonzero(V4["f_spk"] == nm)[0] for nm in S.names}; vr = {nm: np.nonzero(V4["v_spk"] == nm)[0] for nm in S.names}
    for _ in range(n):
        a = S.names[rng.integers(len(S.names))]
        cand = [b for b in S.names if b != a and (proto != "gender" or gender[str(b)] == gender[str(a)])] or [b for b in S.names if b != a]
        b = cand[rng.integers(len(cand))]
        fi.append(rng.choice(fr[a])); vi.append(rng.choice(vr[b])); y.append(0)
    return np.array(fi), np.array(vi), np.array(y)


# ------------------------------------------------------------------ cac buoc
def read_lid(a):
    p = Path(a.lid) if a.lid else None
    return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else {}


def stage_split(a):
    E, V4 = load_bundle(a.bundle, a.work); sp = split(V4)
    Path(a.work).mkdir(parents=True, exist_ok=True); (Path(a.work) / "split.json").write_text(json.dumps(sp, indent=1))
    log("split: giu rieng", len(sp["reserve"]), "fold", [len(f) for f in sp["folds"]])


def stage_oof(a):
    work = Path(a.work); sp = json.loads((work / "split.json").read_text()); E, V4 = load_bundle(a.bundle, work)
    if XF or XV:                                                       # FACTOR-01 step 3: final RFX with extra blocks (opt-in)
        load_extra(E, V4, XF + XV)
    pool = Pool(E, read_lid(a))
    steps = int(os.environ.get("RFS_STEPS", CFG["steps"])); NT = int(os.environ.get("RFS_NTRIAL", 3000)); dv = pool.Vm.shape[1]; df = pool.Fm.shape[1]
    rng = np.random.default_rng(SEED + 1); gender = sp["gender"]; out = {}
    const = const_params(V4, [p for f in sp["folds"] for p in f]); (work / "fold_proj").mkdir(exist_ok=True)
    rt = {pr: make_trials(V4, sp["reserve"], gender, NT, pr, rng) for pr in ("gender", "no_gender")} if sp["reserve"] else {}
    for k, fold in enumerate(sp["folds"]):
        tr = [p for j, f in enumerate(sp["folds"]) if j != k for p in f]
        pth = work / "fold_proj" / f"fold{k}.pt"
        if pth.exists(): sds = torch.load(pth)
        else: sds = train_set(pool, V4Set(V4, tr), steps); torch.save(sds, pth)
        M = load_models(sds, dv, df); P = dict(const); P.update(fit_probes(V4, tr)); P["age_b"] = fit_age(V4, tr)
        for pr in ("gender", "no_gender"):
            fi, vi, y = make_trials(V4, fold, gender, NT, pr, rng)
            for nm, v in trial_streams(M, V4, fi, vi, pr, P).items(): out[f"f{k}_{pr}_{nm}"] = v
            out[f"f{k}_{pr}_y"] = y
            if pr in rt:   # nguoi giu rieng cham bang projector / probe cua fold nay (chua tung thay chung)
                for nm, v in trial_streams(M, V4, rt[pr][0], rt[pr][1], pr, P).items(): out[f"r{k}_{pr}_{nm}"] = v
                out[f"r{k}_{pr}_y"] = rt[pr][2]
        log(f"fold {k}: EER t3 g {eer(out[f'f{k}_gender_t3'], out[f'f{k}_gender_y']):.2f}  ng {eer(out[f'f{k}_no_gender_t3'], out[f'f{k}_no_gender_y']):.2f}")
    np.savez(work / "oof_streams.npz", **out)


def stage_proj(a):
    work = Path(a.work); sp = json.loads((work / "split.json").read_text()); E, V4 = load_bundle(a.bundle, work)
    if XF or XV:                                                       # FACTOR-01 step 3 (opt-in)
        load_extra(E, V4, XF + XV)
        (work / "add_dims.json").write_text(json.dumps({"face": [[k, int(V4[k].shape[1])] for k in XF], "voice": [[k, int(V4[k].shape[1])] for k in XV]}))
    pool = Pool(E, read_lid(a))
    steps = int(os.environ.get("RFS_STEPS", CFG["steps"]))
    tr = [p for f in sp["folds"] for p in f] + (sp["reserve"] if int(os.environ.get("RFS_RESERVE_IN_FINAL", 1)) else [])
    torch.save(train_set(pool, V4Set(V4, tr), steps), work / "projector_Q.pt")
    (work / "final_persons.json").write_text(json.dumps(tr))


def stage_lab(a):
    """RFS-LAB-01 (Experiment/EDA-3FAM-01): RFS streams on the SAFE-04 pairs of F1s / T3S (tables `RFS_LAB_GLOB` under `RFS_LAB_TABLES`).
    Fold s: 3-seed projector trained WITHOUT every person (face or voice side) of fold s's test pairs (= SAFE-04 purge): v4 rows -> V4Set
    persons, MAV rows -> pool keys mav|<spk>. Then t3 / nf / nv on every pair; ib / age / gender_agree on v4 pairs only (the LoRA C1 saw MAV
    clips, and MAV rows have no lora / VGG / age block). Lab ext row i = i-th mav row of the bundle (both v1 then v2 meta order); checked
    on every pair (positive <=> same person)."""
    import pandas as pd
    E, V4 = load_bundle(a.bundle, a.work)
    if XF or XV:                                                       # FACTOR-01 extra blocks (opt-in)
        load_extra(E, V4, XF + XV)
    for k in CFG.get("zero", []):                                     # ENPROG-01 block ablation (opt-in)
        assert k in ("f_clip", "f_ibf", "v_sam", "v_r221", "v_xv", "v_whv3") + tuple(XBLK) and k in E and k in V4, k
        E[k], V4[k] = np.zeros_like(E[k]), np.zeros_like(V4[k]); log(f"zero block {k}")
    pool = Pool(E, read_lid(a)); dv = pool.Vm.shape[1]; C = core()
    steps = int(os.environ.get("RFS_STEPS", CFG["steps"])); tag = os.environ.get("RFS_TAG") or (("rfnw" if NOWH else "rf") + ("sg" if LOSS == "sig" else ""))
    log(f"lab tag {tag}, CFG {json.dumps(CFG)}, steps {steps}")
    tabs = sorted(Path(os.environ["RFS_LAB_TABLES"]).rglob(os.environ.get("RFS_LAB_GLOB", "ct01r2_fold*.parquet")))
    mf, mv = np.nonzero(E["f_src"] == "mav")[0], np.nonzero(E["v_src"] == "mav")[0]
    allp = sorted(set(map(str, V4["f_spk"]))); od = Path(a.art); od.mkdir(parents=True, exist_ok=True)
    n = lambda X: np.asarray(X, np.float32)
    grp = {}   # MAV duplicate identities (e.g. v1 id0001 = id0004): the lab pairs them as positives -> purge the whole group
    for p in tabs:
        A = pd.read_parquet(p, columns=["data", "y", "fi", "vi"]); A = A[(A.data != "v4") & (A.y == 1)]
        for f_, v_ in set(zip(E["f_spk"][mf[A.fi.to_numpy().astype(int)]], E["v_spk"][mv[A.vi.to_numpy().astype(int)]])):
            if f_ != v_:
                g = grp.get(f_, {f_}) | grp.get(v_, {v_})
                for k in g: grp[k] = g
    log(f"MAV duplicate-identity groups: {sorted(sorted(g) for g in {frozenset(g) for g in grp.values()})}")
    for s in [int(x) for x in os.environ.get("RFS_LAB_FOLDS", "1,2,3,4,5").split(",")]:
        t0 = time.time(); src = [p for p in tabs if p.stem.endswith(f"fold{s}")]; assert len(src) == 1, (s, tabs)
        T = pd.read_parquet(src[0])[["fold", "data", "protocol", "y", "fi", "vi"]].reset_index(drop=True)
        y = T.y.to_numpy().astype(int); m4 = (T.data == "v4").to_numpy(); mx = ~m4
        fi4, vi4 = T.fi.to_numpy()[m4].astype(int), T.vi.to_numpy()[m4].astype(int)
        fie, vie = mf[T.fi.to_numpy()[mx].astype(int)], mv[T.vi.to_numpy()[mx].astype(int)]
        c4 = ((V4["f_spk"][fi4] == V4["v_spk"][vi4]) == (y[m4] == 1)).mean() if m4.any() else 1.0
        cx = ((E["f_spk"][fie] == E["v_spk"][vie]) == (y[mx] == 1)).mean() if mx.any() else 1.0
        log(f"fold {s}: {len(T)} pairs (v4 {int(m4.sum())}, MAV {int(mx.sum())}); row mapping check v4 {c4:.4f} MAV {cx:.4f}")
        same = lambda f_, v_: f_ == v_ or v_ in grp.get(f_, ())
        cx = (np.array([same(f_, v_) for f_, v_ in zip(E["f_spk"][fie], E["v_spk"][vie])]) == (y[mx] == 1)).mean() if mx.any() else 1.0
        log(f"fold {s}: mapping check with duplicate-identity groups: MAV {cx:.4f}")
        assert c4 > 0.999 and cx > 0.999, "lab rows do not map to bundle rows"
        test4 = set(map(str, V4["f_spk"][fi4])) | set(map(str, V4["v_spk"][vi4]))
        testx = {f"mav|{k}" for s_ in set(E["f_spk"][fie]) | set(E["v_spk"][vie]) for k in grp.get(s_, {s_})}
        tr4 = [p for p in allp if p not in test4]; keep = np.array([nm not in testx for nm in pool.names])
        sub = type("SubPool", (), {})(); sub.Fm, sub.Vm = pool.Fm, pool.Vm
        sub.names = [nm for nm, k in zip(pool.names, keep) if k]; sub.PF, sub.PV = pool.PF[keep], pool.PV[keep]
        log(f"fold {s}: purge v4 {len(test4)} persons (train {len(tr4)}), MAV {len(testx)} persons out of the pool ({len(sub.names)}/{len(pool.names)} kept)")
        M = load_models(train_set(sub, V4Set(V4, tr4), steps), dv, sub.Fm.shape[1])
        P = const_params(V4, tr4); P.update(fit_probes(V4, tr4)); P["age_b"] = fit_age(V4, tr4)
        cols = {k: np.full(len(T), np.nan, np.float32) for k in ("t3", "nf", "nv", "ib", "age", "gender_agree")}
        if m4.any():
            S4 = trial_streams(M, V4, fi4, vi4, "gender", P)
            for k in cols: cols[k][m4] = S4[k]
        if mx.any():
            vx = [] if NOWH else [n(E["v_whv3"][vie])]
            args = (n(E["f_clip"][fie]), n(E["f_ibf"][fie]), n(E["v_sam"][vie]), n(E["v_r221"][vie]), n(E["v_xv"][vie]))
            if XF or XV:                                               # FACTOR-01: same maths as the core, wider input
                Fx = face_mat(args[0], args[1], [n(E[k][fie]) for k in XF])
                Vx = voice_mat(args[2], args[3], args[4], n(E["v_whv3"][vie]), [n(E[k][vie]) for k in XV])
                cols["t3"][mx] = t3x(M, Fx, Vx); cols["nf"][mx], cols["nv"][mx] = qfx(M, Fx, Vx)
            else:
                cols["t3"][mx] = C.t3_scores(M, *args, vextra=vx); cols["nf"][mx], cols["nv"][mx] = C.qf_norms(M, *args, vx)
        for k, v in cols.items(): T[f"{tag}_{k}"] = v
        for (d, pr), g in T.groupby(["data", "protocol"]):
            sc = g[f"{tag}_t3"].to_numpy(); ok = np.isfinite(sc)
            log(f"  fold {s} {d:6s} {pr:9s}: n {len(g)}  finite {ok.mean():.3f}  EER t3 {eer(sc[ok], g.y.to_numpy()[ok]):.2f}")
        T.to_parquet(od / f"{tag}lab_fold{s}.parquet"); log(f"fold {s} done [{time.time() - t0:.0f}s]")


def synth_lab(bundle, out, seed=0):
    """Fake SAFE-04 tables over the synthetic bundle (smoke): 5 folds, v4 and MAV persons split 5 ways, fi / vi in lab numbering."""
    import pandas as pd
    rng = np.random.default_rng(seed); E, V4 = load_bundle(bundle); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    mf, mv = np.nonzero(E["f_src"] == "mav")[0], np.nonzero(E["v_src"] == "mav")[0]
    p4, px = sorted(set(map(str, V4["f_spk"]))), sorted(set(E["f_spk"][mf]))
    for s in range(1, 6):
        rows = []
        for data, prs, fsp, vsp, persons in (("v4", ("gender", "no_gender"), V4["f_spk"], V4["v_spk"], p4[s - 1::5]),
                                             ("en_ext", ("gender", "no_gender"), E["f_spk"][mf], E["v_spk"][mv], px[s - 1::5])):
            fr = np.nonzero(np.isin(fsp, persons))[0]; vr = np.nonzero(np.isin(vsp, persons))[0]
            for pr in prs:
                for _ in range(60):
                    a_ = fr[rng.integers(len(fr))]; same = rng.random() < .5
                    c = vr[(vsp[vr] == fsp[a_]) == same]; b_ = c[rng.integers(len(c))]
                    rows.append(dict(fold=s, data=data, protocol=pr, y=int(fsp[a_] == vsp[b_]), fi=int(a_), vi=int(b_)))
        pd.DataFrame(rows).to_parquet(out / f"ct01r2_fold{s}.parquet")
    log("fake lab tables ->", out)


def _lr(X, y):   # tron logistic: chuan hoa (mean/scale; do lech 0 -> 1) roi hoi quy; _apply cho logit, diem nop = -logit
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X); lr = LogisticRegression(max_iter=5000).fit(sc.transform(X), y)
    return sc.mean_.astype(np.float64), sc.scale_.astype(np.float64), lr.coef_[0].astype(np.float64), np.float64(lr.intercept_[0])


def _apply(p, name, X):
    return ((X - p[f"{name}_mean"]) / p[f"{name}_scale"]) @ p[f"{name}_coef"] + p[f"{name}_b"]


def stage_fuse(a):
    C = core(); work = Path(a.work); E, V4 = load_bundle(a.bundle, work)
    O = dict(np.load(work / "oof_streams.npz")); tr = json.loads((work / "final_persons.json").read_text())
    P = const_params(V4, tr); P.update(fit_probes(V4, tr)); P["age_b"] = fit_age(V4, tr)
    names = STREAMS_NG + COND
    cat = lambda pr, nm: np.concatenate([O[f"f{k}_{pr}_{nm}"] for k in range(5)])
    sg = {nm: cat("gender", nm) for nm in names}; sn = {nm: cat("no_gender", nm) for nm in names}
    yg, yn = cat("gender", "y"), cat("no_gender", "y")
    Xg = np.stack([sg[k] for k in STREAMS_G], 1); Xn = C.fuser_matrix4(sn, STREAMS_NG)
    for nm, (X, y) in dict(fg=(Xg, yg), fn=(Xn, yn)).items():
        m, s, c, b = _lr(X, y); P.update({f"{nm}_mean": m, f"{nm}_scale": s, f"{nm}_coef": c, f"{nm}_b": b})
    rep = dict(oof_eer={}, reserve_eer={}, note="oof_eer: fuser trong mau (6-30 tham so), t3 la EER rieng; reserve_eer: 10 nguoi giu rieng cham bang projector/probe tung fold")
    for pr, X, y, nm in (("gender", Xg, yg, "fg"), ("no_gender", Xn, yn, "fn")):
        rep["oof_eer"][pr] = dict(t3=eer(cat(pr, "t3"), y), fused=eer(_apply(P, nm, X), y))
        if f"r0_{pr}_y" in O:
            e = []
            for k in range(5):
                s = {n_: O[f"r{k}_{pr}_{n_}"] for n_ in names}
                e.append(eer(_apply(P, nm, np.stack([s[q] for q in STREAMS_G], 1) if pr == "gender" else C.fuser_matrix4(s, STREAMS_NG)), O[f"r{k}_{pr}_y"]))
            rep["reserve_eer"][pr] = dict(mean=float(np.mean(e)), per_projector=e)
    np.savez(work / "final_params.npz", **{k: np.asarray(v) for k, v in P.items()}); (work / "fuse_report.json").write_text(json.dumps(rep, indent=1))
    log("fuse:", json.dumps(rep))


def stage_release(a):
    work, art = Path(a.work), Path(a.art); art.mkdir(parents=True, exist_ok=True)
    man = json.loads((HERE.parent / "ref" / "manifest_final.json").read_text()); man.pop("source_artifact_manifest", None)
    shutil.copy(work / "final_params.npz", art / "final_params.npz"); shutil.copy(work / "projector_Q.pt", art / "projector_Q.pt")
    shutil.copy(Path(a.lora) if a.lora else work / "lora_C1.pt", art / "lora_C1.pt")
    man["system"] = "RFS-01 (FLAG_163): RF-L2-092 recipe re-learned on our clean person pool, learning code written from the RF report: " + man["system"].split(": ", 1)[1] + (" [OBJFAM-01: projector trained with a pairwise sigmoid loss, t = 10, b = -10]" if LOSS == "sig" else "")
    man["declared"] = ("RFS-01. External pool (person level; voice never learned for speakers flagged by the team's strict Bengali rule): VoxCeleb2 dev, VoxBlink, MAV-Celeb v1/v2, "
                       "filtered with Experiment/LIDSTRICT-01/lid_strict.json (VoxLingua107 + Whisper-large-v3 votes, bn or as); MAV voice files with a Whisper Bengali vote removed. "
                       "LoRA C1 on the ImageBind-huge audio tower trained on cleaned VoxCeleb2 + MAV clips (centre 3 s). Probes, age map and fusers fitted on FLAG train (70 persons, English). "
                       "Written from the RF-092 report (not RF's training code); LoRA batch, fold split and pool differ from RF. No score normalisation, no cohort, no test statistics. "
                       "Frozen pretrained models: CLIP-L, ImageBind-huge, SimAM-ResNet100 (VoxBlink2), WeSpeaker ResNet221, SpeechBrain ECAPA, Whisper-large-v3 encoder, ViT-age, audeering age-gender.")
    if XF or XV:                                                       # FACTOR-01 step 3: wider projector input, read by make_xscorer
        man["add"] = json.loads((work / "add_dims.json").read_text())
        man["system"] += f" [FACTOR-01: projector input + {XF + XV}]"
        man["declared"] += (" FACTOR-01 extra blocks: " + ", ".join(XF + XV) + " (f_c5s = CLIP-L + LoRA adapter C5s trained on cleaned faces; "
                            "v_rl = WeSpeaker ResNet293-LM + ECAPA1024-LM, frozen pretrained).")
    man["files"] = {n: sha(art / n) for n in ("final_params.npz", "lora_C1.pt", "projector_Q.pt")}
    (art / "manifest_final.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    for n in ("fuse_report.json", "split.json"):
        if (work / n).exists(): shutil.copy(work / n, art / n)
    selfcheck(art, a)


def selfcheck(art, a):
    """FinalScorer nap artifact va cham duoc; muc 0 (cham nua = cham ca); khop trial_streams + bo tron voi cung tham so."""
    C = core(); E, V4 = load_bundle(a.bundle, a.work); rng = np.random.default_rng(0)
    if XF or XV:                                                       # FACTOR-01 step 3: the scorer FLAG_163i will use
        load_extra(E, V4, XF + XV); sc = make_xscorer(C)(art)
        assert sc.xf == XF and sc.xv == XV, (sc.xf, sc.xv)
    else:
        sc = C.FinalScorer(art)
    n = 40; fi, vi = rng.integers(0, len(V4["f_spk"]), n), rng.integers(0, len(V4["v_spk"]), n); f32 = lambda k: V4[k].astype(np.float32)
    x = dict(xf=f32("f_xf")[fi], ibf=f32("f_ibf")[fi], clip=f32("f_clip")[fi], age_f=V4["f_age"][fi], xv=f32("v_xv")[vi], lora=f32("v_lora")[vi],
             r221=f32("v_r221")[vi], sam=f32("v_sam")[vi], whv3=f32("v_whv3")[vi], age_v=V4["v_age"][vi],
             **{k: f32(k)[fi] for k in XF}, **{k: f32(k)[vi] for k in XV})
    res = {}
    for pr in ("gender", "no_gender"):
        full = sc.score(x, pr); part = sc.score({k: v[:n // 2] for k, v in x.items()}, pr)
        assert np.allclose(full[:n // 2], part, atol=1e-5), f"muc 0 {pr}: {np.abs(full[:n // 2] - part).max()}"
        s = trial_streams(sc.proj, V4, fi, vi, pr, sc.p)
        ref = -_apply(sc.p, "fg", np.stack([s[k] for k in STREAMS_G], 1)) if pr == "gender" else -_apply(sc.p, "fn", C.fuser_matrix4(s, STREAMS_NG))
        assert np.allclose(full, ref, atol=1e-4), f"khop luong {pr}: {np.abs(full - ref).max()}"
        res[pr] = float(np.abs(full - ref).max())
    log("selfcheck OK: FinalScorer nap + cham; muc 0 (nua = ca); khop trial_streams", res); return res


# ------------------------------------------------------------------ LoRA C1 tren thap audio ImageBind-huge
def train_lora(ib, MT, params, fbanks, pid, face, steps, lr=1e-4, tau=0.05, B=64, seed=1, warm=60, scale=256.0, dev=None):
    """fbanks: list tensor (n_chunk,1,128,204) fp16, moi clip mot tensor; pid: nguoi cua tung clip; face: (n_nguoi,1024) ImageBind-mat (dong bang).
    InfoNCE doi xung tau giua mat va audio co LoRA; moi buoc B nguoi khac nhau, moi nguoi mot clip ngau nhien. Mo hinh fp16: loss nhan `scale` roi chia lai, buoc co grad vo han bi bo."""
    dev = dev or DEV; rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); pid = np.asarray(pid); byp = {}
    for i, p in enumerate(pid): byp.setdefault(int(p), []).append(i)
    people = np.array(sorted(byp)); face = torch.tensor(U(face).astype(np.float32)).to(dev); B = min(B, len(people))
    for blk in getattr(ib.modality_trunks[MT.AUDIO], "blocks", []):   # tiet kiem bo nho: checkpoint tung khoi khi co gradient
        if not getattr(blk, "_rfs_ck", False):
            def fw(*a, _f=blk.forward, **k):
                from torch.utils.checkpoint import checkpoint
                return checkpoint(_f, *a, use_reentrant=False, **k) if torch.is_grad_enabled() else _f(*a, **k)
            blk.forward = fw; blk._rfs_ck = True
    t0 = time.time(); skipped = 0
    for t in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (t + 1) / warm)
        ps = rng.choice(people, B, replace=False); ci = [rng.choice(byp[int(p)]) for p in ps]
        x = torch.stack([fbanks[i][:3] for i in ci]).to(dev)
        a = ib({MT.AUDIO: x})[MT.AUDIO].float()
        loss = nce(face[torch.tensor(ps, device=dev)], a, tau)
        opt.zero_grad(); (loss * scale).backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params): skipped += 1; continue
        for p in params: p.grad /= scale
        opt.step()
        if t % 100 == 0 or t == steps - 1: log(f"  lora step {t}/{steps} loss {loss.item():.4f} skipped {skipped} [{time.time() - t0:.0f}s]")
    return skipped


def lora_data(E, lid, a, max_per=3, cap=None, seed=1):
    """Clip + nguoi cho LoRA: VoxCeleb2 (audio cache, tam 3 s) + MAV (tam 3 s tu wav); bo nguoi lid_strict, file mav_files; toi da max_per video moi nguoi."""
    C = core(); rng = np.random.default_rng(seed)
    keep, _ = clean_ext(E, lid); byp = {}
    for i in np.nonzero(keep)[0]:
        if E["v_src"][i] in ("vox", "mav"): byp.setdefault((str(E["v_src"][i]), str(E["v_spk"][i])), []).append(i)
    sel = []
    for k, rows in sorted(byp.items()):
        rng.shuffle(rows); seen = set()
        for i in rows:
            if E["v_vid"][i] not in seen and len(seen) < max_per: seen.add(E["v_vid"][i]); sel.append(i)
    if cap: sel = [int(i) for i in rng.permutation(sel)[:cap]]
    wavs, cache_idx, by_file = {}, {}, {}
    if a.cache:
        for p in sorted((Path(a.cache) / "vox").glob("tar*.npz")):
            Z = np.load(p)
            for r, k in enumerate(zip(Z["spk"].astype(str), Z["video"].astype(str), Z["utt"].astype(str))): cache_idx[k] = (p, r)
    for i in sel:
        if E["v_src"][i] == "vox":
            k = (str(E["v_spk"][i]), str(E["v_vid"][i]), str(E["v_utt"][i]))
            if k in cache_idx: by_file.setdefault(cache_idx[k][0], []).append((i, cache_idx[k][1]))
        else:
            w, _ = C.load_wave(str(Path(a.mav) / E["v_path"][i]))
            if w is not None: c0 = max(0, (len(w) - 48000) // 2); wavs[i] = w[c0:c0 + 48000].astype(np.float32)
    for p, items in by_file.items():
        Z = np.load(p); W = Z["wav"]; n = Z["n"] if "n" in Z.files else None
        for i, r in items: wavs[i] = W[r][:max(int(n[r]) if n is not None else W.shape[1], 8000)].astype(np.float32) / 32768.0
    rows = sorted(wavs); names = {}
    pid = [names.setdefault((str(E["v_src"][i]), str(E["v_spk"][i])), len(names)) for i in rows]
    log(f"lora data: {len(rows)} clip, {len(names)} nguoi (vox {sum(E['v_src'][i] == 'vox' for i in rows)}, mav {sum(E['v_src'][i] == 'mav' for i in rows)})")
    return [wavs[i] for i in rows], np.array(pid), names


def face_by_person(E, names):
    out = np.zeros((len(names), E["f_ibf"].shape[1]), np.float32); cnt = np.zeros(len(names))
    for s, k, x in zip(E["f_src"], E["f_spk"], E["f_ibf"]):
        j = names.get((str(s), str(k)))
        if j is not None and np.isfinite(x).all(): out[j] += U(x[None])[0]; cnt[j] += 1
    ok = cnt > 0
    return out[ok] / cnt[ok, None], ok


def stage_lora(a):
    C = core(); work = Path(a.work); work.mkdir(parents=True, exist_ok=True); E, V4 = load_bundle(a.bundle, work)
    steps, B = int(os.environ.get("RFS_LORA_STEPS", 1200)), int(os.environ.get("RFS_LORA_BATCH", 64))
    wavs, pid, names = lora_data(E, read_lid(a), a, cap=int(os.environ.get("RFS_LORA_CAP", 0)) or None)
    face, ok = face_by_person(E, names); remap = -np.ones(len(names), int); remap[ok] = np.arange(ok.sum())
    m = remap[pid] >= 0; fb = [C.ib_audio_clips(w).half() for w, k in zip(wavs, m) if k]; pid = remap[pid][m]
    log(f"lora: {len(fb)} clip co mat, {len(face)} nguoi, B={B}, {steps} buoc")
    ib, MT = C.load_imagebind(a.ib_ckpt); params = C.add_lora(ib, MT, 8, 16)
    sk = train_lora(ib, MT, params, fb, pid, face, steps, B=B)
    torch.save({"params": [p.data.detach().cpu() for p in params], "steps": steps, "batch": B, "clips": len(fb), "persons": int(len(face)), "skipped": sk}, work / "lora_C1.pt")
    pv = [str(Path(a.v4root) / p) for p in V4["v_path"]]
    ex = {"v_lora": C.embed_audio(ib, MT, pv).astype(np.float32)}; del ib
    if "f_age" not in V4:   # bundle tu FLAG_162 da co tuoi (rfagef / rfagev); khong co thi trich bang AgeExtractor cua core
        ag = C.AgeExtractor(); ex["f_age"] = ag.faces([str(Path(a.v4root) / p) for p in V4["f_path"]])[:, 0]; ex["v_age"] = ag.voices(pv)[:, 0]
    np.savez(work / "v4_extra.npz", **ex); log("lora + v4_extra xong")


# ------------------------------------------------------------------ bundle gia (smoke / test)
def synth(out, n_ext=150, n_v4=70, seed=0):
    rng = np.random.default_rng(seed); out = Path(out); out.mkdir(parents=True, exist_ok=True); D = 24
    W = {k: rng.standard_normal((D, d)).astype(np.float32) for k, d in dict(clip=768, ibf=1024, xf=4096, sam=256, r221=256, xv=192, whv3=1280, lora=1024).items()}
    Wg = rng.standard_normal(D).astype(np.float32); nz = lambda X: (X + 0.5 * rng.standard_normal(X.shape)).astype(np.float32)
    E = {k: [] for k in "f_spk f_vid f_src v_spk v_vid v_src v_utt v_path".split()}; zs = []
    for i in range(n_ext):
        src = ("vox", "vb", "mav")[i % 3]; z = rng.standard_normal(D).astype(np.float32); spk = f"{src[0]}{i:04d}"
        for v in range(2 if src == "vb" else 3):
            E["f_spk"].append(spk); E["f_vid"].append(f"{spk}_{v}"); E["f_src"].append(src)
            E["v_spk"].append(spk); E["v_vid"].append(f"{spk}_{v}"); E["v_src"].append(src); E["v_utt"].append("0")
            E["v_path"].append(f"v1/voices/{spk}/English/{v}/0.wav" if src == "mav" else ""); zs.append(z)
    zs = np.array(zs); A = {k: np.array(v) for k, v in E.items()}
    A.update(f_clip=nz(zs @ W["clip"]), f_ibf=nz(zs @ W["ibf"]), v_sam=nz(zs @ W["sam"]), v_r221=nz(zs @ W["r221"]), v_xv=nz(zs @ W["xv"]), v_whv3=nz(zs @ W["whv3"]).astype(np.float16))
    np.savez(out / "ext.npz", **A)
    V = {k: [] for k in "f_spk f_vid f_gender f_path v_spk v_vid v_gender v_path".split()}; zf, af, av = [], [], []
    for i in range(n_v4):
        z = rng.standard_normal(D).astype(np.float32); g = int(z @ Wg > 0); spk = f"id{i:05d}"; age = float(rng.uniform(20, 60))
        for v in range(4):
            for _ in range(3):
                V["f_spk"].append(spk); V["f_vid"].append(f"{spk}_{v}"); V["f_gender"].append(g); V["f_path"].append(f"faces/{spk}/{v}.jpg")
                V["v_spk"].append(spk); V["v_vid"].append(f"{spk}_{v}"); V["v_gender"].append(g); V["v_path"].append(f"voices/{spk}/{v}.wav")
                zf.append(z); af.append(age + rng.normal(0, 3)); av.append((age - 9.16) / 0.584 + rng.normal(0, 5))
    zf = np.array(zf); B4 = {k: np.array(v) for k, v in V.items()}
    B4.update(f_gender=B4["f_gender"].astype(np.int8), v_gender=B4["v_gender"].astype(np.int8), f_clip=nz(zf @ W["clip"]), f_ibf=nz(zf @ W["ibf"]), f_xf=nz(zf @ W["xf"]),
              v_sam=nz(zf @ W["sam"]), v_r221=nz(zf @ W["r221"]), v_xv=nz(zf @ W["xv"]), v_whv3=nz(zf @ W["whv3"]).astype(np.float16))
    np.savez(out / "v4.npz", **B4)
    np.savez(out / "v4_extra.npz", v_lora=nz(zf @ W["lora"]), f_age=np.array(af, np.float32), v_age=np.array(av, np.float32))
    (out / "lid_strict.json").write_text(json.dumps(dict(vox2=["v0000"], vb=["b0001"], mav_files=["v1/voices/m0002/English/0/0.wav"])))
    log("bundle gia ->", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    for k in ("bundle", "work", "art", "lid", "lora", "ib_ckpt", "cache", "mav", "v4root"): ap.add_argument("--" + k)
    a = ap.parse_args()
    if a.cmd == "synth": return synth(a.bundle)
    if a.cmd == "synthlab": return synth_lab(a.bundle, a.work)
    fn = dict(split=stage_split, lora=stage_lora, oof=stage_oof, proj=stage_proj, fuse=stage_fuse, release=stage_release, lab=stage_lab)
    for c in ((["split", "oof", "proj", "fuse"] + ([] if NOWH else ["release"])) if a.cmd == "all" else [a.cmd]):
        log(f"== {c} =="); fn[c](a)
    if NOWH and a.cmd == "all":   # ablation only: no release; the notebook reads art/fuse_report.json
        import shutil; Path(a.art).mkdir(parents=True, exist_ok=True); shutil.copy(Path(a.work) / "fuse_report.json", Path(a.art) / "fuse_report.json")


if __name__ == "__main__":
    main()
