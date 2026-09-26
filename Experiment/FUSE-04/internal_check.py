"""Internal (labelled, English, speaker-disjoint) cross-modal EER for each voice encoder, same EXP-007 recipe.
Also: unimodal voice EER and a linear gender probe, to test 'stronger speaker encoder -> worse face-voice bridge'."""
import os, sys, numpy as np, pandas as pd, torch
K = r"D:/Sinh viên CNhan/FLAG_2027_FourFusion/kaggle"
sys.path.insert(0, K); torch.set_num_threads(8)
os.environ["FLAG_FEATS_EXTRA"] = K + "/NB4/feats_v3"
import flag_v2 as V
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
store = V.FeatureStore(K + "/../kaggle_upload", feats_dir=K + "/output/feats_v2")
REC = dict(head="mlp", drop=0.3, emb=128, epochs=40)
y = np.array([store.gmap[s] == "m" for s in store.spk])
rows = []
for v in ["given", "ecapa192", "rdn6vox", "rdn6multi"]:
    X = store.voice(v, "train"); Z = V.l2n((X - X.mean(0)) / (X.std(0) + 1e-6))
    gacc = cross_val_score(LogisticRegression(C=0.1, max_iter=3000), Z, y, groups=store.spk, cv=GroupKFold(5)).mean()
    r = dict(voice=v, **{f"uni_{k}": x for k, x in V.unimodal_eer(X, store.spk, store.gmap).items()},
             gender_probe=round(gacc, 3), **V.evaluate(store, dict(REC, voice=v), seeds=(1, 2, 3), n_models=1))
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv("internal_check.csv", index=False)
