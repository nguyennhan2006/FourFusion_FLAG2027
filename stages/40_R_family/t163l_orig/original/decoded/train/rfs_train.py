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


def face_mat(clip, ibf):
    return np.hstack([U(clip), U(ibf)]).astype(np.float32)


def voice_mat(sam, r2, xv, wh):   # thu tu cua t3_scores: sam, r221, ecapa, whisper
    return np.hstack([U(sam), U(r2), U(xv), U(wh)]).astype(np.float32)


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
        fkey = np.array([f"{s}|{k}" for s, k in zip(E["f_src"], E["f_spk"])]); vkey = np.array([f"{s}|{k}" for s, k in zip(E["v_src"], E["v_spk"])])
        fk_i, vk_i = np.nonzero(fk)[0], np.nonzero(keep)[0]
        names, PF, PV = pair_table(fkey[fk_i], E["f_vid"][fk_i], vkey[vk_i], E["v_vid"][vk_i], K, np.random.default_rng(seed))
        self.names, self.PF, self.PV = names, fk_i[PF], vk_i[PV]
        self.Fm = torch.tensor(face_mat(E["f_clip"], E["f_ibf"]))
        self.Vm = torch.tensor(voice_mat(E["v_sam"], E["v_r221"], E["v_xv"], E["v_whv3"].astype(np.float32)))
        log(f"pool: {len(names)} nguoi co cap mat/giong khac video (K={K})")


class V4Set:
    def __init__(self, V4, persons, K=64, seed=SEED):
        ps = set(map(str, persons))
        fi = np.nonzero([str(s) in ps for s in V4["f_spk"]])[0]; vi = np.nonzero([str(s) in ps for s in V4["v_spk"]])[0]
        names, PF, PV = pair_table(V4["f_spk"][fi], V4["f_vid"][fi], V4["v_spk"][vi], V4["v_vid"][vi], K, np.random.default_rng(seed), strict=False)
        self.names, self.PF, self.PV = names, fi[PF], vi[PV]
        g = {str(s): int(x) for s, x in zip(V4["f_spk"], V4["f_gender"])}
        self.gender = np.array([g[str(n)] for n in names])
        self.Fm = torch.tensor(face_mat(V4["f_clip"], V4["f_ibf"]))
        self.Vm = torch.tensor(voice_mat(V4["v_sam"], V4["v_r221"], V4["v_xv"], V4["v_whv3"].astype(np.float32)))


# ------------------------------------------------------------------ projector QV-Whisper
def nce(f, v, tau):
    f, v = F.normalize(f, dim=-1), F.normalize(v, dim=-1)
    lg = f @ v.T / tau; y = torch.arange(len(f), device=f.device)
    return (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2


def train_proj(pool, v4, seed, cfg=CFG, steps=None, dev=DEV):
    C = core(); steps = steps or cfg["steps"]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = C.ProjQ(pool.Fm.shape[1], pool.Vm.shape[1], cfg).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / cfg["warmup"]))
    Fp, Vp, Fv, Vv = pool.Fm.to(dev), pool.Vm.to(dev), v4.Fm.to(dev), v4.Vm.to(dev)
    n, npp = len(pool.names), min(cfg["persons"], len(pool.names)); K = pool.PF.shape[1]
    gi = [x for x in (np.nonzero(v4.gender == g)[0] for g in (0, 1)) if len(x) >= 2]
    t0 = time.time()
    for t in range(steps):
        p = rng.choice(n, npp, replace=False); k = rng.integers(0, K, npp)
        loss = nce(m.f(Fp[pool.PF[p, k]]), m.v(Vp[pool.PV[p, k]]), cfg["tau"])
        if gi:   # 32 nguoi v4 train cung gioi
            ids = gi[rng.integers(len(gi))]; q = rng.choice(ids, min(32, len(ids)), replace=False); k2 = rng.integers(0, v4.PF.shape[1], len(q))
            loss = loss + 0.5 * nce(m.f(Fv[v4.PF[q, k2]]), m.v(Vv[v4.PV[q, k2]]), cfg["tau"])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if t % 500 == 0 or t == steps - 1:
            log(f"  proj seed {seed} step {t}/{steps} loss {loss.item():.4f} [{time.time() - t0:.0f}s]")
    return {k: v.detach().cpu() for k, v in m.eval().cpu().state_dict().items()}


def load_models(sds, dv):
    C = core(); ms = []
    for sd in sds:
        m = C.ProjQ(768 + 1024, dv, CFG); m.load_state_dict(sd); ms.append(m.eval())
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
    s = {"t3": C.t3_scores(M, cl, ibf, sam, r2, xv, vextra=[wh])}
    s["nf"], s["nv"] = C.qf_norms(M, cl, ibf, sam, r2, xv, [wh])
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
    work = Path(a.work); sp = json.loads((work / "split.json").read_text()); E, V4 = load_bundle(a.bundle, work); pool = Pool(E, read_lid(a))
    steps = int(os.environ.get("RFS_STEPS", CFG["steps"])); NT = int(os.environ.get("RFS_NTRIAL", 3000)); dv = pool.Vm.shape[1]
    rng = np.random.default_rng(SEED + 1); gender = sp["gender"]; out = {}
    const = const_params(V4, [p for f in sp["folds"] for p in f]); (work / "fold_proj").mkdir(exist_ok=True)
    rt = {pr: make_trials(V4, sp["reserve"], gender, NT, pr, rng) for pr in ("gender", "no_gender")} if sp["reserve"] else {}
    for k, fold in enumerate(sp["folds"]):
        tr = [p for j, f in enumerate(sp["folds"]) if j != k for p in f]
        pth = work / "fold_proj" / f"fold{k}.pt"
        if pth.exists(): sds = torch.load(pth)
        else: sds = train_set(pool, V4Set(V4, tr), steps); torch.save(sds, pth)
        M = load_models(sds, dv); P = dict(const); P.update(fit_probes(V4, tr)); P["age_b"] = fit_age(V4, tr)
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
    work = Path(a.work); sp = json.loads((work / "split.json").read_text()); E, V4 = load_bundle(a.bundle, work); pool = Pool(E, read_lid(a))
    steps = int(os.environ.get("RFS_STEPS", CFG["steps"]))
    tr = [p for f in sp["folds"] for p in f] + (sp["reserve"] if int(os.environ.get("RFS_RESERVE_IN_FINAL", 1)) else [])
    torch.save(train_set(pool, V4Set(V4, tr), steps), work / "projector_Q.pt")
    (work / "final_persons.json").write_text(json.dumps(tr))


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
    man["system"] = "RFS-01 (FLAG_163): RF-L2-092 recipe re-learned on our clean person pool, learning code written from the RF report: " + man["system"].split(": ", 1)[1]
    man["declared"] = ("RFS-01. External pool (person level; voice never learned for speakers flagged by the team's strict Bengali rule): VoxCeleb2 dev, VoxBlink, MAV-Celeb v1/v2, "
                       "filtered with Experiment/LIDSTRICT-01/lid_strict.json (VoxLingua107 + Whisper-large-v3 votes, bn or as); MAV voice files with a Whisper Bengali vote removed. "
                       "LoRA C1 on the ImageBind-huge audio tower trained on cleaned VoxCeleb2 + MAV clips (centre 3 s). Probes, age map and fusers fitted on FLAG train (70 persons, English). "
                       "Written from the RF-092 report (not RF's training code); LoRA batch, fold split and pool differ from RF. No score normalisation, no cohort, no test statistics. "
                       "Frozen pretrained models: CLIP-L, ImageBind-huge, SimAM-ResNet100 (VoxBlink2), WeSpeaker ResNet221, SpeechBrain ECAPA, Whisper-large-v3 encoder, ViT-age, audeering age-gender.")
    man["files"] = {n: sha(art / n) for n in ("final_params.npz", "lora_C1.pt", "projector_Q.pt")}
    (art / "manifest_final.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    for n in ("fuse_report.json", "split.json"):
        if (work / n).exists(): shutil.copy(work / n, art / n)
    selfcheck(art, a)


def selfcheck(art, a):
    """FinalScorer nap artifact va cham duoc; muc 0 (cham nua = cham ca); khop trial_streams + bo tron voi cung tham so."""
    C = core(); sc = C.FinalScorer(art); E, V4 = load_bundle(a.bundle, a.work); rng = np.random.default_rng(0)
    n = 40; fi, vi = rng.integers(0, len(V4["f_spk"]), n), rng.integers(0, len(V4["v_spk"]), n); f32 = lambda k: V4[k].astype(np.float32)
    x = dict(xf=f32("f_xf")[fi], ibf=f32("f_ibf")[fi], clip=f32("f_clip")[fi], age_f=V4["f_age"][fi], xv=f32("v_xv")[vi], lora=f32("v_lora")[vi],
             r221=f32("v_r221")[vi], sam=f32("v_sam")[vi], whv3=f32("v_whv3")[vi], age_v=V4["v_age"][vi])
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
    fn = dict(split=stage_split, lora=stage_lora, oof=stage_oof, proj=stage_proj, fuse=stage_fuse, release=stage_release)
    for c in (["split", "oof", "proj", "fuse", "release"] if a.cmd == "all" else [a.cmd]):
        log(f"== {c} =="); fn[c](a)


if __name__ == "__main__":
    main()
