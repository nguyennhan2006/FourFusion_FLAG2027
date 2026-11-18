"""Relational face-voice bridge (SAFE-04): no network, no test statistic.

A face is described by its similarity to the face centroids of K TRAIN identities, a voice by its similarity to the voice
centroids of the SAME identities; score = correlation of the two profiles over identities ("does this face look like
the people whose voices this voice sounds like"). Everything is computed from the sample itself and train centroids, so a
pair's score never depends on other test samples (level 0).

Scale: z-scored with 20 000 random TRAIN pairs whose own identities are dropped from both profiles (leave-own-out), so
the reference pairs look like pairs of new persons (a train face sits on its own centroid otherwise).
"""
import numpy as np

l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


class Relational:
    def __init__(self, ids, Fc, Vc, mode="z", T=0.05):
        """ids: identity names (K); Fc / Vc: (K, d) mean of the L2-normalised train embeddings of each identity."""
        self.ids = np.asarray(ids)
        self.cf, self.cv = Fc.mean(0), Vc.mean(0)
        self.Cf, self.Cv = l2n(Fc - self.cf), l2n(Vc - self.cv)
        self.mode, self.T = mode, T
        self.col = {p: i for i, p in enumerate(self.ids)}

    @classmethod
    def from_rows(cls, groups, **kw):
        """groups: list of (identity, face rows (n, df), voice rows (m, dv))."""
        ids = [g[0] for g in groups]
        Fc = np.stack([l2n(np.asarray(g[1], np.float32)).mean(0) for g in groups])
        Vc = np.stack([l2n(np.asarray(g[2], np.float32)).mean(0) for g in groups])
        return cls(ids, Fc, Vc, **kw)

    def profiles(self, Xf, Xv):
        return l2n(l2n(np.asarray(Xf, np.float32)) - self.cf) @ self.Cf.T, l2n(l2n(np.asarray(Xv, np.float32)) - self.cv) @ self.Cv.T

    def _score(self, Pf, Pv, mask=None):
        if mask is None:
            mask = np.ones_like(Pf, dtype=bool)
        n = mask.sum(1, keepdims=True)
        if self.mode == "z":
            def z(P):
                m = np.where(mask, P, 0).sum(1, keepdims=True) / n
                s = np.sqrt(np.where(mask, (P - m) ** 2, 0).sum(1, keepdims=True) / n) + 1e-9
                return np.where(mask, (P - m) / s, 0)
            return (z(Pf) * z(Pv)).sum(1) / n[:, 0]
        if self.mode == "kr":                                     # kernel regression both ways: softmax weights
            def sm(P):
                E = np.where(mask, np.exp((P - P.max(1, keepdims=True)) / self.T), 0)
                return E / E.sum(1, keepdims=True)
            return 0.5 * ((sm(Pf) * Pv).sum(1) + (sm(Pv) * Pf).sum(1))
        raise ValueError(self.mode)

    def score(self, Xf, Xv, chunk=4096):
        out = []
        for i in range(0, len(Xf), chunk):
            out.append(self._score(*self.profiles(Xf[i:i + chunk], Xv[i:i + chunk])))
        return np.concatenate(out)

    def ref_score(self, Xf, Xv, pf, pv, chunk=4096):
        """Train reference pairs: the face's and the voice's own identities are removed from both profiles."""
        out = []
        for i in range(0, len(Xf), chunk):
            Pf, Pv = self.profiles(Xf[i:i + chunk], Xv[i:i + chunk])
            mask = np.ones_like(Pf, dtype=bool)
            for j, (a, b) in enumerate(zip(pf[i:i + chunk], pv[i:i + chunk])):
                for p in (a, b):
                    if p in self.col:
                        mask[j, self.col[p]] = False
            out.append(self._score(Pf, Pv, mask))
        return np.concatenate(out)


class Side:
    """One modality / one encoder: identity centroids (+ optional per-identity column statistics from TRAIN samples of
    the other identities: removes hub centroids that are close to everybody)."""
    def __init__(self, ids, rows_per_id, colnorm=False, max_rows=40, seed=0):
        self.ids = list(ids)
        C = np.stack([l2n(np.asarray(r, np.float32)).mean(0) for r in rows_per_id])
        self.c = C.mean(0); self.C = l2n(C - self.c)
        self.mu = self.sd = None
        if colnorm:
            rng = np.random.RandomState(seed)
            X, who = [], []
            for i, r in enumerate(rows_per_id):
                r = np.asarray(r, np.float32)
                take = r if len(r) <= max_rows else r[rng.choice(len(r), max_rows, replace=False)]
                X.append(take); who += [i] * len(take)
            P = self.raw(np.concatenate(X)); who = np.asarray(who)
            other = who[:, None] != np.arange(len(self.ids))[None, :]
            n = other.sum(0)
            self.mu = np.where(other, P, 0).sum(0) / n
            self.sd = np.sqrt(np.where(other, (P - self.mu) ** 2, 0).sum(0) / n) + 1e-6

    def raw(self, X):
        return l2n(l2n(np.asarray(X, np.float32)) - self.c) @ self.C.T

    def __call__(self, X):
        P = self.raw(X)
        return P if self.mu is None else (P - self.mu) / self.sd


def zrow(P, mask):
    n = mask.sum(1, keepdims=True)
    m = np.where(mask, P, 0).sum(1, keepdims=True) / n
    s = np.sqrt(np.where(mask, (P - m) ** 2, 0).sum(1, keepdims=True) / n) + 1e-9
    return np.where(mask, (P - m) / s, 0)


class RelMulti:
    """Profiles of several face encoders averaged (after per-sample z), same for voices; score = correlation."""
    def __init__(self, face_sides, voice_sides, post=None):
        self.F, self.V, self.post = face_sides, voice_sides, post
        self.ids = face_sides[0].ids
        assert all(s.ids == self.ids for s in face_sides + voice_sides)
        self.col = {p: i for i, p in enumerate(self.ids)}

    def _score(self, Xfs, Xvs, mask):
        Pf = zrow(np.mean([zrow(s(X), mask) for s, X in zip(self.F, Xfs)], 0), mask)
        Pv = zrow(np.mean([zrow(s(X), mask) for s, X in zip(self.V, Xvs)], 0), mask)
        if self.post == "rank":                                   # Spearman: ranks inside the kept identities
            big = np.where(mask, 0, -1e9)
            Pf = zrow(np.argsort(np.argsort(Pf + big, 1), 1).astype(np.float32), mask)
            Pv = zrow(np.argsort(np.argsort(Pv + big, 1), 1).astype(np.float32), mask)
        elif self.post and self.post.startswith("top"):           # keep each profile's k most similar identities
            k = int(self.post[3:])
            for P in (Pf, Pv):
                thr = np.sort(np.where(mask, P, -1e9), 1)[:, -k][:, None]
                P[P < thr] = 0
            return (Pf * Pv).sum(1) / k
        return (Pf * Pv).sum(1) / mask.sum(1)

    def score(self, Xfs, Xvs, pf=None, pv=None, chunk=4096, keep=None):
        """Xfs / Xvs: one array per face / voice side (row-aligned trials). pf / pv (train reference pairs only): the
        identities to leave out of both profiles. keep: optional (n, K) bool, anchors each trial may use (e.g. the
        anchors of the gender predicted from the trial's own voice)."""
        out = []
        n = len(Xfs[0])
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            m = np.ones((len(Xfs[0][sl]), len(self.ids)), dtype=bool) if keep is None else keep[sl].copy()
            if pf is not None:
                for j, (a, b) in enumerate(zip(pf[sl], pv[sl])):
                    for p in (a, b):
                        if p in self.col:
                            m[j, self.col[p]] = False
            out.append(self._score([X[sl] for X in Xfs], [X[sl] for X in Xvs], m))
        return np.concatenate(out)


class CrossSide:
    """Cross-modal profile through a joint space (ImageBind): a sample of modality A against the identity centroids of
    modality B (each modality centred by its own train centroid mean). Centroids average many samples, so the per-sample
    noise of the joint space (ERR-03) is averaged out on the anchor side."""
    def __init__(self, ids, rows_a, rows_b):
        self.ids = list(ids)
        Ca = np.stack([l2n(np.asarray(r, np.float32)).mean(0) for r in rows_a])
        Cb = np.stack([l2n(np.asarray(r, np.float32)).mean(0) for r in rows_b])
        self.ca, self.cb = Ca.mean(0), Cb.mean(0)
        self.C = l2n(Cb - self.cb)

    def __call__(self, X):
        return l2n(l2n(np.asarray(X, np.float32)) - self.ca) @ self.C.T
