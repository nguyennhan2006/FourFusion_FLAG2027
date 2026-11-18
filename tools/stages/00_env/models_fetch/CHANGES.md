# models_fetch: changes with respect to the original

Origin: `new` (DESIGN 2.3 rank 7). The reference code is `cmd_fetch_models` of `release/docker/ff.py` (team repository; file copied
unchanged to `original/ff.py`, SHA-256 `c706b0b729b3b77961515d6e888fe9a8a567ea64dd220a821ad81bc64a06d8d2`, lines 286–306), which
downloads ImageBind-huge from the pinned Hugging Face copy and keeps it only when its SHA-256 is Meta's. The stage runs
`tools/fetch_models.py` (new) and streams its output line by line.

| # | where | old -> new | reason | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | model list | ImageBind only -> the full table of DESIGN 9.2 / S8 E10 (25 entries in four groups: core, dp, lid, wiring) | offline runs and a pinned record of every public model | none: same files the notebooks download (SHA-256 checked where measured) | `pin` |
| 2 | HF downloads | the notebooks call `from_pretrained` / `snapshot_download` with the default revision `main` (ViT-age, audeering, AdaFace, CLIP-L in places, SpeechBrain) -> `tools/fetch_models.py` downloads `snapshot_download(revision=<commit>)` into `06_pretrained/cache/huggingface/hub` and then writes `models--<repo>/refs/main = <commit>`, so that the notebooks' unchanged `main` lookups resolve to the pinned commit when the runner points `HF_HUB_CACHE` at that cache. Pinned only with `FF_HF_OFFLINE=1` (HF_HUB_OFFLINE): online, huggingface_hub re-resolves `main` on the Hub and may fetch a newer commit | `main` can move; the commits are the ones the team caches hold (E: package `pretrained/cache/huggingface/hub/*/refs/main`, team machine cache) | none when the pinned commit is used (offline); online runs may differ if a repository changed after the original run | `pin` |
| 3 | WeSpeaker ModelScope zips | the notebooks fetch them through the ModelScope tree API at run time -> the same API, then unpacked like `wespeaker_dir()` into `cache/flag/wespeaker/<stem>/` | lets `FLAG_CACHE` skip the download | none (zip SHA-256 checked: values measured on `D:/ff_models/`, 2026-11-18) | `pin` |
| 4 | ImageBind URL | Meta's `dl.fbaipublicfiles.com` (dead since 2026-11-12) -> HF `PKU-Alignment/AnyRewardModel@40c7e2a` | dead link; same SHA-256 `d6f6c22b…2157` | none | `url` |
| 5 | smoke mode | — -> `FF_FETCH_DRY=1` lists the table and writes `PLAN.txt` + a placeholder `SOURCES.json` | smoke test without network | none | — |
