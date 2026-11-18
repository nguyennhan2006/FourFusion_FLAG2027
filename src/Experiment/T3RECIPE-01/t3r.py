r"""T3RECIPE-01 trainer: Track 3's projector recipe (SUB19B, ref_track3/t3train.py), rebuilt on our rows.

Reconstructed from t3train.py (verbatim where the code is there) and from the docstrings / names of the three functions it
imports from the unseen track2/model.py (TODO A5):
  DualLinearProjector      per modality LayerNorm -> Dropout -> Linear -> L2 ("LayerNorm->dropout->linear->L2 per modality";
                           attribute names normalization / linear as used by t3train's trunc_k code)
  cross_modal_supervised_contrastive_loss   = t3train.supcon_logits on the cosine matrix (the bil_rank branch of con())
  batch_identity_prototype_alignment_loss   per identity of the batch: 1 - cos(mean face embedding, mean voice embedding)
Loss of one step (t3train.train_projector, SUB19B config): shared AAM head over the main identities on both modalities +
w_con SupCon + w_proto prototype alignment, + w_sg SupCon inside each gender (main batch), + per auxiliary source its OWN AAM
head on aux_ids identities x per_id rows with the same three terms (weight 1.0). Gender-balanced main batches (half male,
half female identities). Constant LR AdamW, grad-norm clip 5. Steps = epochs x main rows // (ids_per_batch x per_id).
"""
import copy, math, os, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SUB19B = dict(dim=512, face_dropout=.85, voice_dropout=.55, margin=.1, scale=30., w_con=.5, temp=.2, w_proto=.6, w_sg=.5,
              epochs=40, ids_per_batch=32, per_id=2, lr=3e-4, wd=1e-4, clip=5., aux_ids=64, aux_weight=1.0)
# Track 3's report of 11/10 (ref_track3/T3_REPORT_2026-10-11.md): their best SUB21a = SUB19B with dropout face .70 / voice .40
# (CB -0.55) and 32 identities per auxiliary step (64 = SUB19b gave no gain)
SUB21A = dict(SUB19B, face_dropout=.70, voice_dropout=.40, aux_ids=32)


_CAP = None                                                              # QFEMA-01: list -> Stem collects the pre-normalisation norms


def _n(z):
    if _CAP is not None:
        _CAP.append(z.norm(dim=1))
    return F.normalize(z, dim=1)


class Stem(nn.Module):
    """LayerNorm -> Dropout -> map -> L2. kind 'linear' (default) = the original module, unchanged (same parameters, same init
    order). PROJ-01 (17/10): Track 3's projector families (ref_track3/cv16/t3proj.py), re-implemented:
      resmlp  linear + zero-initialised GELU residual (256)      gated  linear + sigmoid-gated GELU residual
      orthhard  semi-orthogonal map (QR every forward) + bias     orthres  orthhard + zero-initialised low-rank residual"""
    def __init__(self, d_in, dim, p, kind="linear", rank=64):
        super().__init__()
        if kind.startswith(("alf:", "gate:")):                       # ALF-01 (31/10): attention / gate over appended tap blocks
            self._init_blocks(d_in, dim, p, kind); return
        self.kind = kind
        self.normalization, self.dropout = nn.LayerNorm(d_in), nn.Dropout(p)
        if kind in ("linear", "resmlp", "gated"):
            self.linear = nn.Linear(d_in, dim)
        if kind in ("resmlp", "gated"):
            self.h, self.o = nn.Linear(d_in, 256), nn.Linear(256, dim)
            nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
        if kind == "gated":
            self.g = nn.Linear(d_in, dim); nn.init.constant_(self.g.bias, -2.0)
        if kind in ("orthhard", "orthres"):
            self.Wfree = nn.Parameter(torch.randn(dim, d_in) / d_in ** .5)
            self.bias = nn.Parameter(torch.zeros(dim))
            if kind == "orthres":
                self.rd, self.ru = nn.Linear(d_in, rank, bias=False), nn.Linear(rank, dim, bias=False)
                nn.init.zeros_(self.ru.weight)
        if kind not in ("linear", "resmlp", "gated", "orthhard", "orthres"):
            raise ValueError(kind)

    def _init_blocks(self, d_in, dim, p, kind):
        """ALF-01 (Experiment/ALF-01/README.md, 31/10): x = [base d0 | n blocks of width w]. base: LayerNorm -> Dropout; blocks:
        LayerNorm -> Linear 256 + block embedding -> 'alf': one learned query, multi-head attention (h heads) over the blocks /
        'gate': input-dependent softmax gate (Linear 256 -> 1) over the blocks; pooled 256 -> Dropout; [base | pooled] -> Linear -> L2.
        kind = 'alf:d0:n:w[:h]' or 'gate:d0:n:w'."""
        f = kind.split(":")
        self.kind, self.d0, self.nb, self.w = f[0], int(f[1]), int(f[2]), int(f[3])
        assert self.d0 + self.nb * self.w == d_in, (kind, d_in)
        self.normalization, self.dropout = nn.LayerNorm(self.d0), nn.Dropout(p)
        self.tnorm, self.tproj = nn.LayerNorm(self.w), nn.Linear(self.w, 256)
        self.pos = nn.Parameter(torch.zeros(self.nb, 256))
        if self.kind == "alf":
            self.q = nn.Parameter(torch.randn(1, 1, 256) * .02)
            self.mha = nn.MultiheadAttention(256, int(f[4]) if len(f) > 4 else 4, batch_first=True)
        else:
            self.g = nn.Linear(256, 1)
        self.tdrop = nn.Dropout(p)
        self.linear = nn.Linear(self.d0 + 256, dim)

    def _blocks(self, x):
        b = self.dropout(self.normalization(x[:, :self.d0]))
        T = self.tproj(self.tnorm(x[:, self.d0:].reshape(len(x), self.nb, self.w))) + self.pos
        if self.kind == "alf":
            z = self.mha(self.q.expand(len(x), -1, -1), T, T, need_weights=False)[0][:, 0]
        else:
            z = (torch.softmax(self.g(torch.tanh(T)).squeeze(-1), 1).unsqueeze(-1) * T).sum(1)
        return _n(self.linear(torch.cat([b, self.tdrop(z)], 1)))

    def ortho_W(self):
        W = self.Wfree
        if W.shape[0] >= W.shape[1]:
            Q, R = torch.linalg.qr(W); return Q * torch.sign(torch.diagonal(R))[None, :]
        Q, R = torch.linalg.qr(W.T); return (Q * torch.sign(torch.diagonal(R))[None, :]).T

    def forward(self, x):
        if self.kind == "linear":
            return _n(self.linear(self.dropout(self.normalization(x))))
        if self.kind in ("alf", "gate"):
            return self._blocks(x)
        x = self.dropout(self.normalization(x))
        if self.kind == "resmlp":
            z = self.linear(x) + self.o(F.gelu(self.h(x)))
        elif self.kind == "gated":
            z = self.linear(x) + torch.sigmoid(self.g(x)) * self.o(F.gelu(self.h(x)))
        else:
            z = x @ self.ortho_W().T + self.bias
            if self.kind == "orthres":
                z = z + self.ru(self.rd(x))
        return _n(z)


class DualLinearProjector(nn.Module):
    def __init__(self, d_f, d_v, dim, pf, pv, kind="linear", rank=64, kind_face=None, kind_voice=None):
        super().__init__()                                             # kind_face / kind_voice: ALF-01 per-side stems (default: kind)
        self.face, self.voice = Stem(d_f, dim, pf, kind_face or kind, rank), Stem(d_v, dim, pv, kind_voice or kind, rank)

    def forward(self, f, v):
        return self.face(f), self.voice(v)


def supcon(fe, ve, lab, temp, keep=None, beta=0.0):
    """Cross-modal supervised contrastive loss, both directions (t3train.supcon_logits on the cosine matrix).
    keep (HARDNEG-AGE-01): bool matrix of the pairs allowed in the denominators (positives always); rows left without a
    negative are not averaged. None = the original loss.
    beta (LOSSCUR-01): hardness-weighted negatives (Robinson et al. 2021): weight of a negative ∝ exp(beta * cos), mean 1 over
    the row's negatives, detached. 0 = the original loss."""
    S = F.normalize(fe, dim=1) @ F.normalize(ve, dim=1).T
    pos = lab[:, None].eq(lab[None, :]).to(S.dtype)
    if keep is not None:
        keep = keep | pos.bool()

    def direction(L, K):
        if beta:
            neg = pos.eq(0) if K is None else (K & pos.eq(0))
            bc = (beta * L).masked_fill(~neg, -1e4)
            n = neg.sum(1, keepdim=True).clamp(min=1).to(L.dtype)
            lw = (bc - torch.logsumexp(bc, dim=1, keepdim=True) + n.log()).detach()
            L = L / temp + lw * neg.to(L.dtype)
        else:
            L = L / temp
        if K is not None:
            L = L.masked_fill(~K, -1e4)
        L = L - L.max(dim=1, keepdim=True).values.detach()
        lp = L - torch.logsumexp(L, dim=1, keepdim=True)
        r = -((pos * lp).sum(1) / pos.sum(1))
        if K is None:
            return r.mean()
        has = (K & ~pos.bool()).any(1)
        return r[has].mean() if bool(has.any()) else r.sum() * 0
    return 0.5 * (direction(S, keep) + direction(S.T, None if keep is None else keep.T))


def proto_align(fe, ve, lab):
    """Per identity in the batch: 1 - cos(normalised mean face embedding, normalised mean voice embedding)."""
    u, inv = torch.unique(lab, return_inverse=True)
    M = torch.zeros(len(u), len(lab), device=fe.device, dtype=fe.dtype)
    M[inv, torch.arange(len(lab), device=fe.device)] = 1.0
    pf, pv = F.normalize(M @ fe, dim=1), F.normalize(M @ ve, dim=1)
    return (1 - (pf * pv).sum(1)).mean()


def aam_logits(emb, W, labels, margin, scale, cur=None, sub=1):
    """cur (LOSSCUR-01): CurricularFace state {"t": EMA of the target cosine}; hard negatives (cos > cos(theta_y + m)) become
    cos * (t + cos) (Huang et al. 2020). None = plain AAM.
    sub (CCPROP-01): W holds `sub` consecutive sub-centres per identity; an identity's cosine is its closest sub-centre
    (sub-center ArcFace, Deng et al. 2020). 1 = one centre per identity."""
    cos = F.normalize(emb, dim=1) @ F.normalize(W, dim=1).T
    if sub > 1:
        cos = cos.view(len(cos), -1, sub).max(-1).values
    theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
    onehot = F.one_hot(labels, cos.shape[1]).bool()
    tgt = torch.cos(theta + margin)
    if cur is None:
        return scale * torch.where(onehot, tgt, cos)
    with torch.no_grad():
        cur["t"] = .99 * cur["t"] + .01 * cos[onehot].mean()
    hard = cos > tgt[onehot][:, None]
    return scale * torch.where(onehot, tgt, torch.where(hard, cos * (cur["t"] + cos), cos))


class Source:
    """Rows of one source grouped by identity: sampling of `per_id` distinct rows (with replacement only if an identity has
    fewer rows) for a set of identities, vectorised."""
    def __init__(self, Fx, Vx, S, device, male=None):
        o = np.argsort(S, kind="mergesort")
        self.ids, start, count = np.unique(S[o], return_index=True, return_counts=True)
        self.order, self.start, self.count = o, start, count
        self.F = torch.from_numpy(np.ascontiguousarray(Fx[o], np.float32)).to(device)
        self.V = torch.from_numpy(np.ascontiguousarray(Vx[o], np.float32)).to(device)
        self.male = None if male is None else np.array([male.get(s) == "m" for s in self.ids])
        self.n_rows = len(S)

    def rows(self, ids, per_id, rng):
        c, s0 = self.count[ids], self.start[ids]
        r1 = np.floor(rng.random(len(ids)) * c).astype(np.int64)
        out = [r1]
        if per_id >= 2:
            r2 = np.floor(rng.random(len(ids)) * np.maximum(c - 1, 1)).astype(np.int64)
            r2 = np.where(c > 1, r2 + (r2 >= r1), r1)                     # distinct second row when the identity has two
            out.append(r2)
        for _ in range(per_id - 2):
            out.append(np.floor(rng.random(len(ids)) * c).astype(np.int64))
        idx = np.stack([s0 + r for r in out], 1).reshape(-1)               # identity-major: id0 row a, id0 row b, id1 ...
        lab = np.repeat(ids, per_id)
        return torch.from_numpy(idx), torch.from_numpy(lab)


_S1 = {}                                                                   # stage-1 models of ADAPT-01 (one fold per process)


@torch.no_grad()
def _neighbours(main, groups, k2):
    """SGW-01 (16/10): per gender group, each identity's k2 most similar identities of the SAME group in the input space
    (cos of the identities' mean face rows + cos of their mean voice rows; fixed, computed once per training)."""
    n = len(main.ids)
    lab = torch.repeat_interleave(torch.arange(n, device=main.F.device), torch.as_tensor(main.count, device=main.F.device))
    mean = lambda X: F.normalize(torch.zeros(n, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, lab, F.normalize(X, dim=1)), dim=1)
    Mf, Mv = mean(main.F), mean(main.V)
    out = {}
    for name, g in groups.items():
        gi = torch.as_tensor(g, device=main.F.device)
        S = Mf[gi] @ Mf[gi].T + Mv[gi] @ Mv[gi].T
        S.fill_diagonal_(-1e9)
        out[name] = S.topk(min(k2, len(g) - 1), dim=1).indices.cpu().numpy()        # positions inside g
    return out


def _hard_pick(nb, g, half, rng):
    """one seed identity of group g + (half - 1) of its nearest same-group identities, drawn from its k2 neighbours (semi-hard)."""
    s = int(rng.integers(len(g)))
    c = rng.choice(nb[s], min(half - 1, len(nb[s])), replace=False)
    return g[np.r_[s, c]]


def _age_bins(main, g, nb):
    """HARDNEG-AGE-01: identities of group g that have a face age (main.age, train persons only), sorted by age, and the
    edges of nb equal-count bins (positions)."""
    g = g[np.isfinite(main.age[g])]
    g = g[np.argsort(main.age[g], kind="mergesort")]
    return g, np.linspace(0, len(g), nb + 1).round().astype(int)


def _age_pick(b, half, rng):
    """half identities of one random age bin; a bin with fewer than half grows into its nearest-age neighbours (a window of
    half positions centred on the bin), so the batch keeps the control's size."""
    g, e = b
    k = int(rng.integers(len(e) - 1)); lo, hi = e[k], e[k + 1]
    if hi - lo >= half:
        return rng.choice(g[lo:hi], half, replace=False)
    lo = int(np.clip((lo + hi - half) // 2, 0, max(0, len(g) - half)))
    return g[lo:lo + half]


def _subset_ids(main, which):
    """Main identities of a stage-2 set: main = all; flag = v4 + MAV (no 'vox2:' / 'vb:' prefix); v4 = v4 persons (idNNN)."""
    ids = main.ids.astype(str)
    ext = np.char.startswith(ids, "vox2:") | np.char.startswith(ids, "vb:")
    keep = {"main": np.ones(len(ids), bool), "flag": ~ext,
            "v4": np.array([re.fullmatch(r"id\d{3}", s) is not None for s in ids])}[which]
    return np.flatnonzero(keep)


def train(main, aux, d_f, d_v, cfg, seed, device, log=None):
    """main: Source with genders (male map); aux: list of Source. Returns the projector (cpu, eval) and the step count.
    ADAPT-01: cfg keys s2_* add a stage 2 = the stage-1 model (cached: same rows, seed and stage-1 cfg train once per process)
    trained on, the main identities of s2_set only ('flag' default), no auxiliary heads, lr x s2_lr, s2_epochs epochs of
    those rows. No s2_* key = the original single stage, uncached."""
    s2 = {k: v for k, v in cfg.items() if k.startswith("s2_")}
    if not s2:
        return _train(main, aux, d_f, d_v, cfg, seed, device, log)[:2]
    c1 = {k: v for k, v in cfg.items() if not k.startswith("s2_")}
    key = (main.n_rows, len(main.ids), tuple((len(a.ids), a.n_rows, getattr(a, "w", 1.0)) for a in aux), d_f, d_v, seed,
           tuple(sorted(c1.items())))
    if key not in _S1:
        _S1[key] = _train(main, aux, d_f, d_v, c1, seed, device, log)
    model, steps, W = _S1[key]
    if not s2.get("s2_epochs"):
        return model, steps
    return _stage2(copy.deepcopy(model), W.clone(), main, dict(SUB19B, **c1), s2, seed, device, log, steps)


def _main_loss(model, W, main, pick, cfg, rng, device, loss_on):
    idx, lab = main.rows(pick, cfg["per_id"], rng)
    idx, lab = idx.to(device), lab.to(device)
    fe, ve = model(main.F[idx], main.V[idx])
    loss = loss_on(fe, ve, lab, W)
    st = getattr(loss_on, "st", None)                                      # LOSSCUR-01 schedule, None = constant recipe
    tau, beta = (cfg["temp"], 0.0) if st is None else (st["tau"], st["beta"])
    if cfg["w_sg"]:                                                        # same-gender contrast inside the main batch
        gm = torch.from_numpy(main.male[lab.cpu().numpy()]).to(device)
        for sel in (gm, ~gm):
            if sel.sum() > 2:
                loss = loss + cfg["w_sg"] * supcon(fe[sel], ve[sel], lab[sel], tau, beta=beta)
    if cfg.get("w_age"):                                                   # HARDNEG-AGE-01: same gender and |face age gap| <= age_tau
        l_ = lab.cpu().numpy(); a, m = main.age[l_], main.male[l_]         # unknown age (NaN) never compares <= tau
        keep = (m[:, None] == m[None, :]) & (np.abs(a[:, None] - a[None, :]) <= cfg.get("age_tau", 5.))
        loss = loss + cfg["w_age"] * supcon(fe, ve, lab, tau, keep=torch.from_numpy(keep).to(device), beta=beta)
    if cfg.get("fn_k"):                                                    # FACENEG-01: faces of Bengali speakers as negatives
        B = main.fn_bank
        j = torch.from_numpy(rng.choice(len(B), min(int(cfg["fn_k"]), len(B)), replace=False)).to(B.device)
        loss = loss + faceneg(fe, ve, lab, model.face(B[j]), tau, cfg.get("fn_lam", 1.0), cfg.get("fn_mode", "N1"))
    return loss


def faceneg(fe, ve, lab, fn, tau, lam, mode="N1"):
    """FACENEG-01 (Experiment/FACENEG-01/README.md): N1 = InfoNCE voice -> face over the batch faces plus the K bank faces fn
    (weight lam in the denominator; positives = the faces of the voice's identity); N2 adds face -> face (positives = the other
    faces of the identity, the face itself left out, the bank faces as extra negatives)."""
    fe, ve, fn = F.normalize(fe, dim=1), F.normalize(ve, dim=1), F.normalize(fn, dim=1)
    same = lab[:, None] == lab[None, :]

    def nce(a, pos, skip=None):
        s = a @ fe.T / tau
        if skip is not None:
            s = s.masked_fill(skip, float("-inf"))
        lse = torch.logsumexp(torch.cat([s, a @ fn.T / tau + math.log(lam)], 1), 1)
        ok = pos.any(1)
        return (lse - torch.logsumexp(s.masked_fill(~pos, float("-inf")), 1))[ok].mean()

    l = nce(ve, same)
    if mode == "N2":
        eye = torch.eye(len(lab), dtype=torch.bool, device=lab.device)
        l = l + nce(fe, same & ~eye, skip=eye)
    return l


def _loss_fn(cfg, st=None, sub=1):
    """st (LOSSCUR-01): dict {m, tau, beta, cur} updated every step by _lc_sched; None = the constant recipe.
    sub: sub-centres per identity of the head this loss is used with (CCPROP-01; 1 = plain AAM)."""
    def loss_on(fe, ve, lab, head):
        if st is None:
            m, tau, beta, cf, cv = cfg["margin"], cfg["temp"], 0.0, None, None
        else:
            m, tau, beta = st["m"], st["tau"], st["beta"]
            cf, cv = (None, None) if st["cur"] is None else \
                (st["cur"].setdefault((id(head), 0), {"t": 0.0}), st["cur"].setdefault((id(head), 1), {"t": 0.0}))
        l = .5 * (F.cross_entropy(aam_logits(fe, head, lab, m, cfg["scale"], cf, sub), lab)
                  + F.cross_entropy(aam_logits(ve, head, lab, m, cfg["scale"], cv, sub), lab))
        return l + cfg["w_con"] * supcon(fe, ve, lab, tau, beta=beta) + cfg["w_proto"] * proto_align(fe, ve, lab)
    loss_on.st = st
    return loss_on


def _utt_loss(model, src, idx, ve, cfg):
    """UTTSTAB-01: 1 - cos(student, teacher) over the rows of idx that have a teacher; the teacher is detached and dropout-free.
    utt_teacher "video": student = the rows' voice embeddings ve, teacher = mean embedding of the other utterances of the same
    video (src.utt_peers: up to k peer rows per row, -1 padded); "crop": student = the embedding of the row's 3 s crop
    (src.Vc, src.Vc_ok = a crop exists), teacher = the row's own full-clip embedding."""
    video = cfg["utt_teacher"] == "video"
    if video:
        P = src.utt_peers[idx]
        has = (P >= 0).any(1)
    else:
        has = src.Vc_ok[idx]
    if not bool(has.any()):
        return ve.sum() * 0
    if video:
        P = P[has]
        r, c = (P >= 0).nonzero(as_tuple=True)                             # only real peers are projected
        student, rows = ve[has], src.V[P[r, c]]
    else:
        student, rows = model.voice(src.Vc[idx[has]]), src.V[idx[has]]
    with torch.no_grad():
        model.voice.eval()
        T = model.voice(rows)
        model.voice.train()
        if video:
            T = torch.zeros(len(P), T.shape[1], device=T.device, dtype=T.dtype).index_add_(0, r, T)
        T = F.normalize(T, dim=1)
    return (1 - (F.normalize(student, dim=1) * T).sum(1)).mean()


def _lc_sched(cfg, st, p):
    """LOSSCUR-01 (Experiment/LOSSCUR-01/README.md): loss hardness at training progress p; r ramps 0 -> 1 from lc_p1 to lc_p2.
    lc_m_hi: margin -> lc_m_hi with r; lc_ft_m: margin from lc_p2 on (large-margin fine-tuning; lr cut in _train);
    lc_beta: hardness weight beta = lc_beta * r; lc_temp_end: temperature -> lc_temp_end linearly from lc_p1 to the end."""
    p1, p2 = cfg.get("lc_p1", .4), cfg.get("lc_p2", .8)
    r = min(1., max(0., (p - p1) / (p2 - p1)))
    m = cfg["margin"]
    if cfg.get("lc_m_hi") is not None:
        m += (cfg["lc_m_hi"] - m) * r
    if cfg.get("lc_ft_m") is not None and p >= p2:
        m = cfg["lc_ft_m"]
    st["m"], st["beta"] = m, cfg.get("lc_beta", 0.) * r
    if cfg.get("lc_temp_end") is not None:
        st["tau"] = cfg["temp"] + (cfg["lc_temp_end"] - cfg["temp"]) * min(1., max(0., (p - p1) / (1 - p1)))


def _stage2(model, W, main, cfg, s2, seed, device, log, steps1):
    rng = np.random.default_rng(seed + 7); torch.manual_seed(seed + 7)
    model.to(device).train()
    W = nn.Parameter(W.to(device))
    params = list(model.parameters()) + [W]
    opt = torch.optim.AdamW(params, lr=cfg["lr"] * s2.get("s2_lr", 1.0), weight_decay=cfg["wd"])
    sub = _subset_ids(main, s2.get("s2_set", "flag"))
    male_ids, female_ids = sub[main.male[sub]], sub[~main.male[sub]]
    assert len(male_ids) > 1 and len(female_ids) > 1, (s2, len(male_ids), len(female_ids))
    half = cfg["ids_per_batch"] // 2
    steps = int(s2["s2_epochs"] * max(1, int(main.count[sub].sum()) // (cfg["ids_per_batch"] * cfg["per_id"])))
    if cfg.get("max_steps"):
        steps = min(steps, int(cfg["max_steps"]))
    loss_on = _loss_fn(cfg)
    for step in range(steps):
        pick = np.concatenate([rng.choice(male_ids, min(half, len(male_ids)), replace=False),
                               rng.choice(female_ids, min(half, len(female_ids)), replace=False)])
        loss = _main_loss(model, W, main, pick, cfg, rng, device, loss_on)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["clip"]); opt.step()
    if log:
        log(f"    stage 2 {s2}: {len(sub)} identities, {steps} steps")
    model.to("cpu").eval()
    model._ema = None
    return model, steps1 + steps


def _train(main, aux, d_f, d_v, cfg, seed, device, log=None):
    cfg = dict(SUB19B, **cfg)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = DualLinearProjector(d_f, d_v, cfg["dim"], cfg["face_dropout"], cfg["voice_dropout"],
                                cfg.get("proj", "linear"), cfg.get("proj_rank", 64),                     # PROJ-01: projector family
                                kind_face=cfg.get("proj_face"), kind_voice=cfg.get("proj_voice")).to(device)   # ALF-01
    W = nn.Parameter((torch.randn(len(main.ids), cfg["dim"]) * .01).to(device))
    k_aux = int(cfg.get("aux_subcenters", 1))                               # CCPROP-01: sub-centres of the auxiliary heads
    heads = [nn.Parameter((torch.randn(len(a.ids) * k_aux, cfg["dim"]) * .01).to(device)) for a in aux]
    params = list(model.parameters()) + [W] + heads
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    male_ids, female_ids = np.flatnonzero(main.male), np.flatnonzero(~main.male)
    if cfg.get("main_gender"):                                             # CCPROP-01 GENDHEAD: main batches of one gender only
        male_ids, female_ids = (male_ids, female_ids[:0]) if cfg["main_gender"] == "m" else (male_ids[:0], female_ids)
    half = cfg["ids_per_batch"] // 2
    steps = cfg["epochs"] * max(1, main.n_rows // (cfg["ids_per_batch"] * cfg["per_id"]))
    if cfg.get("max_steps"):                                               # local smoke tests only
        steps = min(steps, int(cfg["max_steps"]))

    lc = {k: v for k, v in cfg.items() if k.startswith("lc_")}             # LOSSCUR-01: no lc_* key = the constant recipe
    st = dict(m=cfg["margin"], tau=cfg["temp"], beta=0.0, cur={} if cfg.get("lc_cur") else None) if lc else None
    if lc and log:
        log(f"    LOSSCUR-01 schedule {lc}")
    loss_on = _loss_fn(cfg, st)
    loss_aux = _loss_fn(cfg, st, sub=k_aux) if k_aux > 1 else loss_on
    hard = float(cfg.get("hard_frac", 0) or 0)                             # SGW-01: share of steps with neighbour batches
    if hard:
        NB = _neighbours(main, {"m": male_ids, "f": female_ids}, int(cfg.get("hard_k2", 2 * (half - 1))))
    ab = int(cfg.get("age_bins", 0) or 0)                                  # HARDNEG-AGE-01: age_frac of the steps from face-age bins
    if ab or cfg.get("w_age"):
        assert getattr(main, "age", None) is not None, "age_bins / w_age need main.age (HARDNEG-AGE-01 run.py)"
    if ab:
        AB = [_age_bins(main, g, ab) for g in (male_ids, female_ids)]
        if log:
            log(f"    age bins {ab}: identities with a face age m {len(AB[0][0])} / f {len(AB[1][0])}, edges {[list(b[1]) for b in AB]}")
    ema_d = float(os.environ.get("T3R_EMA", 0) or 0)                       # QFEMA-01: EMA of the head weights, no RNG, no effect on training
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if ema_d else None
    model.train()
    for step in range(steps):
        if st is not None:
            _lc_sched(cfg, st, step / steps)
            if cfg.get("lc_ft_lr") and step == int(cfg.get("lc_p2", .8) * steps):
                for g in opt.param_groups:
                    g["lr"] = cfg["lr"] * cfg["lc_ft_lr"]
        if hard and rng.random() < hard:                                   # no extra draw at all when hard_frac = 0
            pick = np.concatenate([_hard_pick(NB["m"], male_ids, half, rng), _hard_pick(NB["f"], female_ids, half, rng)])
        elif ab and rng.random() < cfg["age_frac"]:                        # no extra draw when age_bins = 0
            pick = np.concatenate([_age_pick(b, half, rng) for b in AB])
        else:
            pick = np.concatenate([rng.choice(male_ids, min(half, len(male_ids)), replace=False),
                                   rng.choice(female_ids, min(half, len(female_ids)), replace=False)])
        loss = _main_loss(model, W, main, pick, cfg, rng, device, loss_on)
        for a, h in zip(aux, heads):
            ids = rng.choice(len(a.ids), min(cfg["aux_ids"], len(a.ids)), replace=False)
            ia, la = a.rows(ids, cfg["per_id"], rng)
            ia, la = ia.to(device), la.to(device)
            fa, va = model(a.F[ia], a.V[ia])
            loss = loss + cfg["aux_weight"] * getattr(a, "w", 1.0) * loss_aux(fa, va, la, h)  # a.w: per-source weight (AUXP-02), default 1
            if cfg.get("utt_w"):                                           # UTTSTAB-01: utterance-stability distillation, aux rows only
                loss = loss + cfg["utt_w"] * _utt_loss(model, a, ia, va, cfg)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["clip"]); opt.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(ema_d).add_(v.detach(), alpha=1 - ema_d) if v.is_floating_point() else ema[k].copy_(v)
        if log and (step + 1) % max(1, steps // 4) == 0:
            log(f"    step {step + 1}/{steps} loss {float(loss):.3f}")
    model.to("cpu").eval()
    model._ema = {k: v.cpu() for k, v in ema.items()} if ema is not None else None
    return model, steps, W.detach().to("cpu")


@torch.no_grad()
def embed(model, Xf, Xv, device, chunk=8192):
    model.to(device)
    fe = torch.cat([model.face(torch.from_numpy(np.ascontiguousarray(Xf[i:i + chunk], np.float32)).to(device)) for i in range(0, len(Xf), chunk)])
    ve = torch.cat([model.voice(torch.from_numpy(np.ascontiguousarray(Xv[i:i + chunk], np.float32)).to(device)) for i in range(0, len(Xv), chunk)])
    model.to("cpu")
    return fe, ve


@torch.no_grad()
def pair_cos(model, Xf, Xv, device):
    fe, ve = embed(model, Xf, Xv, device)
    return (fe * ve).sum(1).double().cpu().numpy()


@torch.no_grad()
def pair_stats(model, Xf, Xv, device, ema=False, chunk=8192):
    """QFEMA-01: (cos, log|face|, log|voice|) before L2 normalisation; ema=True scores with the EMA weights (None if none)."""
    global _CAP
    if ema:
        if getattr(model, "_ema", None) is None:
            return None
        model = copy.deepcopy(model); model.load_state_dict(model._ema)
    model.to(device)
    out = []
    try:
        for side, X in ((model.face, Xf), (model.voice, Xv)):
            _CAP = []
            e = torch.cat([side(torch.from_numpy(np.ascontiguousarray(X[i:i + chunk], np.float32)).to(device)) for i in range(0, len(X), chunk)])
            out.append((e, torch.cat(_CAP).clamp(min=1e-12).log().double().cpu().numpy()))
    finally:
        _CAP = None
        model.to("cpu")
    return (out[0][0] * out[1][0]).sum(1).double().cpu().numpy(), out[0][1], out[1][1]
