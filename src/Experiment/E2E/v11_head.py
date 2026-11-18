r"""v11's T3 head for the end-to-end pipeline (FLAG_50; docs/PLAN_E2E_2026-10-10.md).

  train   = BIGB-01/build_v11.py head(): InfoNCE, PK 512 x 2, ~1 200 steps, 3 nets (seeds 0 / 100 / 200), on all 70 v4 + 151
            MAV-Celeb v1/v2 persons + the Bengali-filtered VoxCeleb2 identities (VOX_LID), T3 encoders (FaRL16 + CLIP-L /
            SimAM100 p + ft). The trained head is EXPORTED (net weights, standardisation + PCA, CCA, z constants) to
            <out>/v11_t3head.pt, so that the Evaluation run scores with the SAME head as the dev run (attach the dev output).
            The export is checked against the in-memory scorer (max |diff| < 1e-5 on 3 000 train pairs).
  score   waits for <feats>/t3_ready.json (written by the notebook once the T3 arrays of the test files exist; it lists the test
          files' split tags), then writes
          <out>/t3_streams_<split>.npz = per test file [anP512_deep, anP512_cca, t3_relg], each pair on its own (level-0
          self-test on half of every file). t3_relg = BIGB-01's: anchors every v4 person + every v1/v2 person, voice-gender
          mask from the trial's own w2v2 gender head, z constants from v4 train pairs.
    VOX_LID=... python v11_head.py train <out>
    VOX_LID=... python v11_head.py score <out> <feats dir> [<split tag> ...]    (tags like gender_Bangla; default: t3_ready.json)
    VOX_LID=... python v11_head.py both <out> <feats dir>                         (the notebook's background job)
"""
import json, os, sys, time, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
P, K = 512, 2
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def lab():
    spec = importlib.util.spec_from_file_location("allneg01run", EXP / "ALLNEG-01/run.py")
    A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)          # puts SAFE-04, VOX2-01, T3ENC-01 on sys.path
    assert A.LID, "v11's head is the Bengali-filtered one: set VOX_LID"
    import torch
    import rel_sweep5 as R
    import run as T                                                               # T3ENC-01/run.py
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "4")))
    return A, R, T


class Head:
    """deep0 / cca of an exported L0 member (= R.L0.streams for those two streams)."""
    def __init__(self, ck, R):
        import torch
        self.R, self.units, self.c = R, [], ck["cca"]
        cfg = dict(R.REC, **ck.get("cfg_over", {}))                              # head size of a variant (HEADSWEEP-01)
        for u in ck["units"]:
            net = R.Model(u["d_f"], u["d_v"], u["n_cls"], cfg); net.load_state_dict(u["state"]); net.to(R.DEVICE).eval()
            self.units.append((net, u["prep"], u["sd"]))
        self.zc = ck["zc"]

    def deep(self, Xf, Xv):
        import torch
        out = []
        for net, p, (m, s) in self.units:
            f = (Xf - p["mf"]) / p["sf"]; f = f if p["Pf"] is None else f @ p["Pf"]
            v = (Xv - p["mv"]) / p["sv"]; v = v if p["Pv"] is None else v @ p["Pv"]
            with torch.no_grad():
                a = net.ef(torch.tensor(f.astype(np.float32), device=self.R.DEVICE))
                b = net.ev(torch.tensor(v.astype(np.float32), device=self.R.DEVICE))
                r = net.score(a, b).float().cpu().numpy()
            out.append((r - m) / s)
        return np.mean(out, 0)

    def cca(self, Xf, Xv):
        c = self.c
        tx = l2n((((Xf - c["mx"]) / c["sx"]) @ c["Px"]) @ c["Wx"]); ty = l2n((((Xv - c["my"]) / c["sy"]) @ c["Py"]) @ c["Wy"])
        return (np.sum(tx * ty, 1) - self.zc[0]) / self.zc[1]


def export(sc, cfg_over=None):
    units = []
    for (net, prep), sd in zip(sc.mem.units, sc.sd):
        st = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if k != "log_tau"}   # a learned temperature is not used to score
        units.append(dict(state=st, d_f=net.bf[1].in_features, d_v=net.bv[1].in_features, n_cls=int(net.W.shape[0]),
                          prep=dict(mf=prep.mf, sf=prep.sf, mv=prep.mv, sv=prep.sv, Pf=prep.Pf, Pv=prep.Pv), sd=(sd.m, sd.s)))
    c = sc.mem.cca
    return dict(units=units, cca=dict(mx=c.mx, sx=c.sx, Px=c.Px, Wx=c.Wx, my=c.my, sy=c.sy, Py=c.Py, Wy=c.Wy), zc=(sc.zc.m, sc.zc.s),
                cfg_over=dict(cfg_over or {}))


def train(out):
    import torch
    A, R, T = lab()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    f = out / "v11_t3head.pt"
    if f.exists():
        R.log(f"{f.name} exists: kept (the dev run's head is reused)"); return
    t0 = time.time()
    F4, V4v, FE, VE = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    allp = dict(s=0, v4tr=np.arange(len(R.V4S)), ext_pool=list(R.PERSONS), v4_train=sorted(set(R.V4S)))
    TF, TV, TS, info = R.train_rows(allp, ext=True, arrays=(F4, V4v, FE, VE))
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    TF, TV, TS = np.concatenate([TF, F2]), np.concatenate([TV, V2]), np.concatenate([TS, S2])
    bpe = max(1, len(TS) // (P * K)); ep = max(1, int(round(R.STEPS / bpe)))
    R.log(f"v11 T3 head (P512): {len(set(TS))} identities ({len(set(S2))} VoxCeleb2, {len(T.dropped_speakers())} dropped by the LID "
          f"filter), {len(TS)} rows, {ep * bpe} steps (P {P} K {K}), {R.N_NETS} nets, device {R.DEVICE}")
    units = [R.train_pk(TF, TV, TS, seed=0 + 100 * i, epochs=ep, loss_fn=A.make_loss(0.0), P=P, K=K) for i in range(R.N_NETS)]
    sc = R.L0(R.Member(units, TF, TV), allp["v4tr"], X4=dict(R.V4, vgg=F4, btc=V4v))
    ck = export(sc)
    ck["info"] = dict(recipe=f"InfoNCE PK {P}x{K}", identities=len(set(TS)), vox_ids=len(set(S2)), dropped=sorted(T.dropped_speakers()),
                      rows=len(TS), steps=ep * bpe, nets=R.N_NETS, device=str(R.DEVICE), torch=torch.__version__)
    rr = np.random.RandomState(7); i, j = rr.randint(len(F4), size=3000), rr.randint(len(F4), size=3000)
    z = np.zeros(3000, np.float32)
    S_ = sc.streams(dict(vgg=F4[i], btc=V4v[j], ibf=np.ones((3000, R.V4["ibf"].shape[1]), np.float32),
                         iba=np.ones((3000, R.V4["iba"].shape[1]), np.float32), agef=z, agev=z))
    H = Head(ck, R)
    d = float(max(np.abs(H.deep(F4[i], V4v[j]) - S_["deep0"]).max(), np.abs(H.cca(F4[i], V4v[j]) - S_["cca"]).max()))
    assert d < 1e-5, d
    ck["info"]["export_check_max_abs_diff"] = d
    torch.save(ck, f)
    R.log(f"-> {f} (export check max |diff| {d:.1e}) [{time.time() - t0:.0f}s]")


def score(out, feats, splits, wait_h=8.0):
    import torch
    out, feats = Path(out), Path(feats)
    ready = feats / "t3_ready.json"
    t0 = time.time()
    while not ready.exists():
        assert time.time() - t0 < wait_h * 3600, f"{ready} never appeared"
        time.sleep(20)
    splits = list(splits) or json.loads(ready.read_text())["splits"]        # default: the test files the notebook lists
    A, R, T = lab()
    ck = torch.load(out / "v11_t3head.pt", map_location="cpu", weights_only=False)
    H = Head(ck, R)
    F4, V4v, FE, VE = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    allp = dict(s=0, v4tr=np.arange(len(R.V4S)), ext_pool=list(R.PERSONS), v4_train=sorted(set(R.V4S)))
    norel = os.environ.get("NOREL") == "1"   # NOREL-01 (5/11, BTC Rules & FAQ): no train-identity anchors at test time; t3_relg = 0, never fused
    if not norel:
        R.FACE["t3f"], R.VOICE["t3v"] = (F4, FE), (V4v, VE)                   # t3_relg (= BIGB-01 / T3ENC-01 build_v10)
        gf, gv = R.id_rows(allp, "t3f", "face"), R.id_rows(allp, "t3v", "voice")
        rel = R.RelMulti([R.Side([a for a, _ in gf], [b for _, b in gf], colnorm=True)],
                         [R.Side([a for a, _ in gv], [b for _, b in gv], colnorm=True)])
        amale = np.array([R.GMAP.get(p) == "m" for p in rel.ids])
        keep = lambda vm: (amale[None, :] == vm[:, None])
        rr = np.random.RandomState(0); n = len(R.V4S)
        rf, rv = rr.randint(n, size=R.N_REF), rr.randint(n, size=R.N_REF)
        zg = R.Std(rel.score([F4[rf]], [V4v[rv]], R.V4S[rf], R.V4S[rv], keep=keep(R.VGEN4[rv])))
    else:
        R.log("NOREL=1: t3_relg not computed (column = 0)")
    res, checks = {}, {}
    for tag in splits:
        f = np.concatenate([l2n(np.load(feats / f"face_{n_}_{tag}.npy").astype(np.float32)) for n_ in ("farl16", "clipL")], 1)
        v = np.concatenate([l2n(np.load(feats / f"voice_{n_}_{tag}.npy").astype(np.float32)) for n_ in ("sam100p", "sam100f")], 1)
        vg = R.pmale_voice(np.load(feats / f"voice_w2v2aggen_{tag}.npy")) > 0.5
        fn = lambda rows: np.stack([H.deep(f[rows], v[rows]), H.cca(f[rows], v[rows]),
                                    np.zeros(len(rows)) if norel else zg(rel.score([f[rows]], [v[rows]], keep=keep(vg[rows])))], 1)
        Z = fn(np.arange(len(f)))
        half = np.sort(np.random.RandomState(1).choice(len(f), len(f) // 2, replace=False))
        d = float(np.abs(fn(half) - Z[half]).max())
        assert d < 1e-5 and np.isfinite(Z).all(), (tag, d)
        res[tag] = Z; checks[tag] = dict(rows=len(Z), level0_half_file_max_abs_diff=d)
        R.log(f"{tag}: v11 T3 streams on {len(Z)} trials | level-0 self-test {d:.1e}")
    np.savez(out / "t3_streams.npz", **res, info=json.dumps(dict(head=ck["info"], checks=checks)))
    R.log(f"-> {out / 't3_streams.npz'} [{time.time() - t0:.0f}s incl. waiting]")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif sys.argv[1] == "both":                     # the notebook's background job: train (or keep), then score when ready
        train(sys.argv[2]); score(sys.argv[2], sys.argv[3], sys.argv[4:])
