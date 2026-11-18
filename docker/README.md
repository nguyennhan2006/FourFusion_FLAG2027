# Environment specification of the 13.54 chain

The submitted system ran as Kaggle notebooks (GPU: NVIDIA Tesla T4, 1 or 2 per run; CPU runs for the small steps) plus a few CPU
scripts on the team machine. Docker is **not required** to use this repository; this folder pins the software so that a run on
Kaggle, on a Linux machine or in a container uses the same versions. Status: the specification is complete for the packages the
notebooks install; the Kaggle base image is pinned by digest; a full lock file of the image (`pip freeze`) does not exist yet
(see "Not yet available").

## 1. Kaggle base image (pinned by digest)

| flavor | image | source of the digest |
|---|---|---|
| GPU | `gcr.io/kaggle-gpu-images/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01` (Kaggle tag `v173`; the runs report it as `gcr.io/kaggle-private-byod/python@sha256:de153e6d…`) | `docker_image` in the kernel metadata of the R11 runs (2026-11-05) and of the five pulled runs 155s / 164 (early November) |
| CPU | `gcr.io/kaggle-images/python@sha256:0717e81990be591f8c5f98f033d29ab23e9ffde057e5b3a6bd4371f93969e962` | kernel metadata of the CPU R11 runs |

**Scope of this digest and of the pins: the November runs only** (F / R families, Evaluation, FVBIND, the t160 / t162 feature
runs). Versions printed by those runs (logs and notebook outputs kept in `stages/*/*/original/`): **GPU image Python 3.13.16 +
`torch 2.11.0+cu130`, CPU image Python 3.13.18 + `torch 2.11.0+cpu`** (the CPU runs t163b, t184, t219, e112f0sx, e112f1sx all
print 3.13.18; the difference follows the CPU vs GPU image, not one odd run), `transformers 5.20.2`. The September / October runs
used **older images**: `torch 2.11.0+cu128` (collection c37 / c46 / c53, l42, f40, f41) and `torch 2.10.0+cu128` (f04, f14). The
per-stage table is `docker/ENV_PER_STAGE.md` (machine-readable: `configs/env_per_stage.json`). Unverified: the Evaluation and FVBIND
runs of 11–16/11 used the same digest (their `docker_image` was not read; `kaggle kernels pull -m`, `tools/PULL_MISSING_RUNS.md` §3).

## 2. Packages the notebooks install on top of the image

| file | use |
|---|---|
| `constraints-kaggle-v173.txt` | pip constraints = the versions pinned in `Dockerfile`; `FF_PIP=pinned` makes the runner set `PIP_CONSTRAINT` to it for every stage, so the notebooks' own unpinned `pip install -U "speechbrain>=1.1.1" …` cells install exactly these versions |
| `requirements-kaggle-pinned.txt` | the same set as an installable list (+ the three `--no-deps` git / wheel installs and the CLIP clone of `Dockerfile` layer 4) for a non-Docker machine built on the same base image |
| `requirements-dp-local.txt` | the CPU machine of the DP branch and the local assembly steps (Python 3.14.0, `torch 2.12.1+cpu`, …; `pip freeze` of 2026-11-18, layer map S8 E3) |
| `Dockerfile` | verbatim copy of `release/docker/Dockerfile` of the team repository (SHA-256 `ad84e6fea5f55541c087d85fe5de806049d2f93fe0a3a5a1530842fbbbe66eb9`, identical to the team package `docker/Dockerfile`); reference only |
| `dockerignore` | verbatim copy of `release/docker/dockerignore` (rename to `.dockerignore` at the build context root to use it) |

Pins inside the notebooks themselves: openai CLIP `@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1`, wespeaker
`@9fecd6cb4f47475d01761d87c826298dff4ef18c`, `insightface==2.0`. **Not pinned by the original notebooks** (the newest release was
installed at run time): `speechbrain>=1.1.1`, `onnxruntime-gpu`, ImageBind from `main` (the Dockerfile and `FF_PIP=pinned` use
`53680b02d7e37b19b124fa37bae4b6c98c38f5be`, the `main` of 2026-11-05; unverified that it equals the HEAD at each run date),
`ruamel.yaml<0.19`, AudioCLIP (`git clone --depth 1`).

Runner policy (`FF_PIP`, `runner/ffcore.py`):

- `original` (default): the notebooks' pip cells run unchanged (Internet needed), exactly like the original runs.
- `pinned`: `PIP_CONSTRAINT=docker/constraints-kaggle-v173.txt` + the unpinned ImageBind git URL rewritten at run time to the commit above (logged in the stage's `_DONE.json`).
  The pins describe the November (v173) image family only: the runner warns for every stage of another family
  (`configs/env_per_stage.json`), and `FF_PIP_SCOPE=v173` keeps `original` for those stages.
- `skip`: pip is replaced by a no-op shim (offline machine or container that already holds the packages).

Pretrained models are files, not packages: `python tools/fetch_models.py --dest <FF_DATA>/06_pretrained` downloads every one at a
pinned revision and checks its SHA-256 (`SOURCES.json`); the runner seeds `FLAG_CACHE` / `HF_HUB_CACHE` from that folder.

## 3. Docker (optional, reference only)

`Dockerfile` builds `fourfusion:gpu` / `fourfusion:cpu` from the digests above with the pins of section 2:

```bash
docker build -f docker/Dockerfile -t fourfusion:gpu .
docker build -f docker/Dockerfile -t fourfusion:cpu --build-arg FLAVOR=cpu \
  --build-arg BASE=gcr.io/kaggle-images/python@sha256:0717e81990be591f8c5f98f033d29ab23e9ffde057e5b3a6bd4371f93969e962 .
```

It is copied unchanged from the older U′ package, so its last line `CMD ["python", "/opt/fourfusion/docker/ff.py", "plan"]` names a
runner that is not part of this repository; start containers with an explicit command instead, for example
`docker run --gpus all -v <FF_DATA>:/data:ro -v <work>:/work fourfusion:gpu python /opt/fourfusion/runner/ffrun.py run --mode infer --data /data --work /work --pip skip`.
History: the CPU image was built and ran the notebook 112f0sx on 2026-11-07 (U′ package); the GPU image has never been built or run
(the team machine has no GPU).

## 4. Not yet available

- `requirements.lock.txt` (`pip freeze` inside the built image; the Dockerfile writes it to `/opt/fourfusion/docker/requirements.lock.txt`). Nobody has built the image for this repository; the closest existing lock is inside the CPU image of 7/11 (`docker run --rm fourfusion:cpu cat /opt/fourfusion/docker/requirements.lock.txt`). The stage `env_docker` (`tools/stages/00_env/env_docker`) writes `pip_freeze.txt` of whatever environment runs it.
- `pip freeze` of the Kaggle runs themselves (the notebooks never printed it): only the versions of section 1 are known.
