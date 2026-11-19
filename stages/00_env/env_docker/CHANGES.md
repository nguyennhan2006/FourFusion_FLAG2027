# env_docker: changes with respect to the original

Origin: `new` (DESIGN 2.3 rank 7). There is no original run: the stage is new code. Its reference input is the Dockerfile of the
team repository, `release/docker/Dockerfile`, copied **unchanged** to `original/Dockerfile` and to `docker/Dockerfile`
(SHA-256 `ad84e6fea5f55541c087d85fe5de806049d2f93fe0a3a5a1530842fbbbe66eb9`, identical to `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/docker/Dockerfile`).

| # | where | old -> new | reason | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | whole stage | DESIGN 4.0 "build the image" -> "check the environment against the pins; print the docker build commands" | advisor's decision 18/11: Docker is not required, no new Docker work; the environment specification (base image digest + pip pins) is kept | none (no stage of the chain reads its output) | `pin` |
| 2 | `docker/Dockerfile` last line `CMD ["python", "/opt/fourfusion/docker/ff.py", "plan"]` | kept verbatim | the Dockerfile is a reference copy; `docker/ff.py` belonged to the older U′ package and is not part of this repository. Start containers with an explicit command (`docker/README.md` section 3) | none | none |
| 3 | `docker/constraints-kaggle-v173.txt`, `docker/requirements-kaggle-pinned.txt` | new files | the Dockerfile pins written as a pip constraints file (used by `FF_PIP=pinned`) and as an installable list | none | `pin` |
| 4 | `stage.ipy` cell 3, `original_runs` | "python 3.13.16 (GPU runs; one CPU run 3.13.18)" -> "GPU image 3.13.16, CPU image 3.13.18 (November runs; older images in Sept / Oct)", torch 2.11.0+cu128 / 2.10.0+cu128 added for the Sept / Oct runs | review 18/11: every kept CPU log prints 3.13.18 and the collection / early feature runs used older images (docker/ENV_PER_STAGE.md, configs/env_per_stage.json) | none (report text) | `pin` |

The stage writes `pip_freeze.txt` (every installed distribution, read with `importlib.metadata` so that it also works when pip is
replaced by the no-op shim of `FF_PIP=skip`) and `env_report.json`.
