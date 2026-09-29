"""EXT-02 — bilingual validation with TRUE labels on external MAV-Celeb sources (docs/PLAN_V4.md §1 A2/A3).

Every MAV-Celeb v1/v2/v3 identity speaks English AND a second language, so a FLAG-like problem with real labels
can be built from identities the bridge never saw:

    external identities -> 5 identity-disjoint folds
    held-out identities of the fold:
        heard   file : trials (face, English voice)          -> does the recipe sacrifice English?
        unheard file : trials (face, Urdu/Hindi/German voice) -> does the bridge transfer across languages?
    each file: 50% targets; the face of a positive trial comes from a DIFFERENT video than its voice
    (no same-session shortcut); per-file mean centring + CCA w=0.25 exactly as for the dev submission.

Arms (recipe EXP-007; the identities of the fold never enter any training set):
    v4         v4 train only                                     (control; identical in every fold)
    mix_en     v4 + English rows of the training identities      (more people, same language)
    mix_all    v4 + all rows of the training identities          (= FLAG_08 MIX; the unheard language of
                                                                   the held-out people is then SEEN in training,
                                                                   so its "unheard" number is optimistic vs Bangla)
    pt_en>ft   pretrain on English external rows, fine-tune on v4
    pt_all>ft  pretrain on all external rows, fine-tune on v4
The v4 control and the pretraining sets are the same for every arm of a fold, so arm differences are paired.
"""
from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd

import flag_v2 as V
from flag_lib import eer_from_scores

REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)      # EXP-007
FT = dict(epochs=20, lr=3e-4)                                        # fine-tune schedule (design choice, not tuned)
ARMS = ("v4", "mix_en", "mix_all", "pt_en>ft", "pt_all>ft")


def id_folds(ids, n=5, seed=0):
    ids = np.array(sorted(ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    return [set(p) for p in np.array_split(ids, n)]


def persons(D, src):
    """Person key per voice row and per face row. ext_<src>_speakers.csv merges ids of one person (v1 lists
    Imran Khan twice); without it every id is its own person."""
    key = lambda raw: D.get("key_of", {}).get(raw, f"{src}:{raw}")
    return D["vm"].spk.map(key).values, D["fm"].spk.map(key).values


def make_trials(D, vper, fper, held, lang, n_pos, n_neg, seed, gender_of=None, same_gender=False):
    """(face_idx, voice_idx, label) over the held-out PERSONS `held`, voices in language `lang`.
    Positive: the voice's person, face frame from another video. Negative: a face of another held-out person
    (same gender if same_gender, which needs gender_of from the dataset's own metadata)."""
    vm, fm = D["vm"], D["fm"]
    rng = np.random.RandomState(seed)
    vi = np.where(np.isin(vper, list(held)) & (vm.lang == lang).values)[0]
    faces = {s: np.where(fper == s)[0] for s in held}
    fvid = fm.video.values
    others = {s: [t for t in held if t != s and (not same_gender or gender_of[t] == gender_of[s])] for s in held}
    pos, neg, tries = [], [], 0
    while (len(pos) < n_pos or len(neg) < n_neg) and tries < 50 * (n_pos + n_neg):
        tries += 1
        v = rng.choice(vi); s = vper[v]
        if len(pos) < n_pos:
            cand = faces[s][fvid[faces[s]] != vm.video.values[v]]
            if len(cand):
                pos.append((rng.choice(cand), v))
        if len(neg) < n_neg and others[s]:
            t = others[s][rng.randint(len(others[s]))]
            neg.append((rng.choice(faces[t]), v))
    P, N = np.array(pos), np.array(neg)
    return np.r_[P[:, 0], N[:, 0]], np.r_[P[:, 1], N[:, 1]], np.r_[np.ones(len(P)), np.zeros(len(N))]


def _cat(*rows):
    return tuple(np.concatenate([r[i] for r in rows]) for i in range(3))


def train_arm(store, arm, v4rows, ext_en, ext_all, seeds):
    """-> list of (net, prep, mu_f, mu_v) and the raw rows the final model was trained on (for CCA)."""
    main = {"v4": v4rows, "mix_en": _cat(v4rows, ext_en), "mix_all": _cat(v4rows, ext_all)}.get(arm, v4rows)
    pre = {"pt_en>ft": ext_en, "pt_all>ft": ext_all}.get(arm)
    models = []
    for sd in seeds:
        if pre is None:
            net, prep = V.train_one(store, REC, None, sd, rows=main)
        else:
            both = _cat(pre, main)
            prep = V.Prep(both[0], both[1], REC["pca_f"], REC["pca_v"])
            pnet, _ = V.train_one(store, REC, None, sd, rows=pre, prep=prep)
            net, _ = V.train_one(store, dict(REC, **FT), None, sd, rows=main, prep=prep, init=pnet.state_dict())
        models.append((net, prep, prep.f(main[0]).mean(0), prep.v(main[1]).mean(0)))
    return models, main, _cca(main)


_CCA = {}


def _cca(main):
    """RidgeCCA on the final training rows, fitted once per distinct row set (both PT arms reuse the v4 fit;
    V.cca_proj would refit a 4096-d SVD for every file)."""
    key = (main[0].shape, float(main[0][:, :8].sum()), float(main[1][:, :8].sum()))
    if key not in _CCA:
        _CCA[key] = V.RidgeCCA(**V.CCA_CFG).fit(main[0], main[1])
    return _CCA[key]


def score_file(models, main, cca, Af, Av, cca_w=0.25):
    """Same numbers as V.cca_proj + V.fuse in dev_scores: per-file centring, CCA k=4 at weight cca_w."""
    embs = [V.embed(net, prep, Af, Av, mf, mv) for net, prep, mf, mv in models]
    ii = np.arange(len(Af))
    proj = (cca.transform_x(V.centre(Af, main[0].mean(0))), cca.transform_y(V.centre(Av, main[1].mean(0))))
    return V.fuse(embs, ii, ii, proj, cca_w)


def run(store, src, arms=ARMS, n_folds=5, n_models=3, n_trials=1500, exclude=(), cap=150, out_csv=None, log=print):
    """EER per (fold, arm, cell, language) on held-out external PERSONS. Resumable through out_csv.
    A `gender` cell (same-gender negatives) is added when the source ships gender metadata."""
    D = store.ext_source(src)
    langs = sorted(D["vm"].lang.unique())
    assert "english" in langs and len(langs) >= 2, f"{src}: need English + another language, got {langs}"
    vper, fper = persons(D, src)
    raw_of = pd.Series(D["vm"].spk.values, index=vper)
    excl = set(exclude)
    people = sorted(p for p in set(vper) if p not in excl and not any(f"{src}:{r}" in excl for r in raw_of[[p]]))
    gender_of = D.get("gen_of") or None
    if gender_of is not None:
        assert all(p in gender_of for p in people), f"{src}: gender missing for some persons"
    done = pd.read_csv(out_csv) if out_csv is not None and os.path.exists(out_csv) else pd.DataFrame()
    v4rows = (store.Xf, store.Xv, store.spk)
    seeds = [1 + 100 * i for i in range(n_models)]
    cells = [("no_gender", False)] + ([("gender", True)] if gender_of is not None else [])
    log(f"{src}: {len(people)} persons ({len(set(D['vm'].spk))} ids), languages {langs}, cells {[c for c, _ in cells]}")
    v4_models = None
    rows = [] if done.empty else done.to_dict("records")
    for f, held in enumerate(id_folds(people, n_folds)):
        train_raw = sorted(set(raw_of[[p for p in people if p not in held]]))
        ext_en = V.pair_ext(D, src, cap, np.random.RandomState(f), exclude, ids=train_raw, langs=["english"])
        ext_all = V.pair_ext(D, src, cap, np.random.RandomState(f), exclude, ids=train_raw)
        assert not set(np.unique(ext_all[2])) & set(held), "held-out person leaked into training rows"
        trials = {(lang, c): make_trials(D, vper, fper, sorted(held), lang, n_trials, n_trials, seed=1000 * f + k,
                                          gender_of=gender_of, same_gender=sg)
                  for k, lang in enumerate(langs) for c, sg in cells}
        for arm in arms:
            if not done.empty and ((done.fold == f) & (done.arm == arm)).any():
                continue
            t0 = time.time()
            if arm == "v4":
                v4_models = v4_models or train_arm(store, "v4", v4rows, ext_en, ext_all, seeds)
                models, main, cca = v4_models
            else:
                models, main, cca = train_arm(store, arm, v4rows, ext_en, ext_all, seeds)
            for (lang, c), (fi, vj, lab) in trials.items():
                s = score_file(models, main, cca, D["Fa"][fi], D["V"][vj])
                rows.append(dict(src=src, fold=f, arm=arm, cell=c, lang=lang,
                                 role="heard" if lang == "english" else "unheard",
                                 eer=round(eer_from_scores(s, lab), 3), n=len(lab),
                                 n_held=len(held), n_ext_rows=len(main[2]) - len(v4rows[2]) if arm.startswith("mix") else 0))
            if out_csv is not None:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
            log(f"{src} fold {f} {arm:10s} " + " ".join(f"{r['cell'][:2]}/{r['lang'][:3]} {r['eer']:.2f}"
                                                       for r in rows if r["fold"] == f and r["arm"] == arm)
                + f"  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def summarise(R, guard=0.5, gain=1.0):
    """Mean EER per arm/role and the paired per-fold delta vs the v4 control, with the PLAN_V4 A2 verdict."""
    out = []
    for (src, cell), g in R.groupby(["src", "cell"]):
        base = g[g.arm == "v4"].set_index(["fold", "role"]).eer
        for arm, a in g.groupby("arm", sort=False):
            rec = dict(src=src, cell=cell, arm=arm)
            for role in ("heard", "unheard"):
                x = a[a.role == role].set_index("fold").eer
                d = x - base.xs(role, level="role").reindex(x.index)
                rec[f"{role}"] = round(x.mean(), 2)
                rec[f"d_{role}"] = round(d.mean(), 2)
                rec[f"sd_{role}"] = round(d.std(), 2) if len(d) > 1 else np.nan
                rec[f"folds_better_{role}"] = f"{int((d < 0).sum())}/{len(d)}"
            if arm == "v4":
                rec["verdict"] = "control"
            elif rec["d_unheard"] <= -gain and rec["d_heard"] <= guard:
                rec["verdict"] = "transfer"
            elif rec["d_unheard"] <= -gain:
                rec["verdict"] = "trade-off"
            elif rec["d_unheard"] > 0 and rec["d_heard"] > 0:
                rec["verdict"] = "drop"
            else:
                rec["verdict"] = "no clear effect"
            out.append(rec)
    return pd.DataFrame(out)
