import numpy as np, time, run as R
Xf, Xv, spk, _, txt, gmap = R.load_train()
for name, cfg in [("FAME1 (linear+aam+drop.9+emb192)", dict(R.BASE, epochs=8)),
                  ("ours  (mlp+infonce+drop.3+emb128)", dict(R.BASE, epochs=8, head="mlp", drop=0.3, loss="infonce", emb=128))]:
    t0 = time.time()
    ng, g = R.evaluate(cfg, Xf, Xv, spk, gmap, seeds=(1,), n_models=1, tag=name[:8])
    print(f"{name:36s} int_ng {ng:6.2f}  int_g {g:6.2f}  ({round(time.time()-t0)}s)", flush=True)
