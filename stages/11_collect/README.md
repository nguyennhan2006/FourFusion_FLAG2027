# 11_collect: auxiliary identities collected from public Hugging Face sources

These stages fetched the auxiliary speakers (VoxCeleb2, VoxBlink-clean) that the face adapters, the identity heads and the
pair-matching family learn from. **For the 13.54 submission the collected outputs are the raw input**: they are kept under
`FF_DATA/01_collected/` and none of these stages runs by default. The runner skips them because collect stages run only when forced
(`configs/modes.json`: `collect_kind_runs_only_when_forced`) and every collected dataset is read from `FF_DATA` (class `collected` is
in `always_saved_classes`). The code is here so that the provenance of every collected file is open and a re-collection is possible,
not because the chain needs it.

Why "always use the collected data":

- **Sampling is not deterministic.** FLAG_37 / FLAG_53 skip broken tar streams and failed range reads and fill in with spare clips;
  the VoxBlink mirror is an unofficial crawl that keeps changing. A re-collection gives other rows than the arrays every later
  stage was trained on.
- **The 13.54 variant depends on the cache layout.** The audio cache keys clips by the bare video id; that is exactly why the
  ORIGINAL LoRA C1 of the pair-matching family sees only 210 MAV-Celeb clips. A "fixed" cache would change the variant.
- **Access changed after the runs.** `ProgramComputer/voxceleb` is now gated (auto-approved terms): a re-collection needs
  `HF_TOKEN` of an account that accepted them. Meta's ImageBind link is dead (FLAG_37 is patched to the pinned HF mirror).

| stage | original run | code source (exactness) | inputs → outputs | compute (original) | private |
|---|---|---|---|---|---|
| `c37_vox2` | unknown slug (forked notebook, 06/10) | executed copy | HF → `flag2027-vox2-keys` (+ `feats_vox2/`) | T4, ~2 h | yes |
| `c53_vox2b` | `nguynlcnhtho/flag-2027-nb-53-scaleid` | repo file = pushed bytes (md5); part 0 only | lab-data + HF → `flag2027-feats-vox2b-p` (+ listings) | T4×2, 8.1 h | yes |
| `c46_voxblink` | unknown slug (2 sessions, 09-10/10) | executed copy (session 2 only; session-1 code presumed identical) | HF → `feats_vb/` | T4, session 2 measured 5.1 h + session 1 estimated ~10.5 h | yes |
| `c45_frames_vox2` | `onnhqunhh/flag-2027-nb-45-facelora-farl` | md5-verified reconstruction of the pushed bytes; frames part only | raw-v4 (train) + lab-data + HF → `vox_frames_png.zip` | whole run T4 ~70 min; frames part needs CPU only | yes |
| `c_frames_index` | none (packed by hand 12/10) | new | c46 + c53 outputs → `flag2027-frames-index-p` | seconds | yes |
| `c63_frames` | `onnhqunhh/flag-2027-nb-63-frames` | repo file = pushed bytes (md5) | frames-index + HF → `frames/` | CPU, 5.6 h | yes |
| `c87_frames_bridge` | `onnhqunhh/flag-2027-nb-87-bridge` | repo file = pushed bytes (md5) | c45 + c63 → `onnhqunhh/flag2027-lora-frames-p` | CPU | yes |
| `c63v_audio_cache` | `duongphihoang/flag-2027-nb-63v-audio-cache` | builder output written 7 min before the push (pushed copy lost) | frames-index + lab-vb + lab-data + feats-vox2b + HF → `duongphihoang/flag2027-audio-cache-p` | CPU, 7.4 h | yes |

Data flow: `c37` → `flag2027-vox2-keys` → (`f41`, `l42`, then `b_lab_data` with `f40`, `f10`, `f12`, `f14`) → lab-data → `c45`, `c53`,
`c63v`; `c46` → (`b_lab_vb`, `l_lid_strict`, `l_dqual_g3`); `c46` + `c53` (kernel output) → `c_frames_index` → `c63`, `c63v`;
`c45` + `c63` (kernel outputs) → `c87`.

Each stage folder: `stage.json` (metadata, inputs / outputs, patches), `stage.ipy` (the original code cells, patched, separated by
`# <<<FF-CELL>>>`), `CHANGES.md` (every change with its reason), `original/` (the exact notebook + `SOURCE.json` with md5 / sha256 and
the Kaggle push record). Code changes are limited to: a dead URL (ImageBind), an optional Hugging Face revision parameter
(`FF_HF_REVS`, unset = the original `main`), the token header for the now-gated repo (no `HF_TOKEN` = the original anonymous
request), removal of dev-set reads and of experiment / training code outside the chain, replacing uploads by the runner's export, an
optional resume folder for the three time-limited stages (`VB_RESUME_FROM`, `FRAMES_RESUME_FROM`, `AUDIO_RESUME_FROM`; unset = the
original behaviour) and one inserted completeness cell in `c46`.

## Re-collecting (optional, never needed for 13.54)

1. **Force the whole closure in one run.** The runner does not invalidate collected artifacts downstream of a forced stage: forcing
   `c37`, `c53` or `c46` alone would mix the new rows with the old frames and audio cache (frame names are FLAG_37 row numbers; the
   cache and frames are keyed by speaker / video / utterance). Use a **fresh `FF_WORK`** and **`FF_USE_SAVED=collected`** (never
   `all`, which would serve the old lab-data / lab-vb / LID tables built from the old rows):

   ```
   FF_MODE=train FF_USE_SAVED=collected FF_WORK=<new folder> \
   FF_FORCE=c37_vox2,c53_vox2b,c46_voxblink,c45_frames_vox2,c_frames_index,c63_frames,c87_frames_bridge,c63v_audio_cache \
   python runner/ffrun.py run
   ```

   `b_lab_data`, `b_lab_vb`, `f41_t3enc_vox2`, `l42_vox2_lid` and everything after them are intermediate and are rebuilt anyway in this
   mode, so this is a full mode (a) rebuild on new auxiliary data. `c87` and `c_frames_index` cannot be forced alone: they read kernel
   outputs of `c45` / `c63` / `c53` that are kept nowhere (only `c53`'s, if `FF_DATA/01_collected/voxceleb2_flag53/feats_vox2b/` also
   holds the two listing parquets: see `c53_vox2b/stage.json` `skip_with_requires`).
2. **Access.** Set `HF_TOKEN` (an account that accepted the `ProgramComputer/voxceleb` terms) and, once the commits are recorded,
   `FF_HF_REVS='{"gaunernst/voxceleb2-dev-wds": "<commit>", "ProgramComputer/voxceleb": "<commit>", ...}'` (the VoxBlink mirror needs
   its 8 part repos listed one by one).
3. **Time limits.** `c46` (VoxBlink) does not fit in one 12 h Kaggle session (the original needed 2 sessions); `c63` / `c63v` finished
   in 5.6 h / 7.4 h but stop at 11 h / 10.5 h on a slower machine. Each of the three now fails a fatal post-check instead of being
   accepted with a partial output. Outside Kaggle set the limit in the caller's environment (the runner passes it through):
   `VB_TIME_LIMIT_H=48`, `FRAMES_TIME_LIMIT_H=48`, `AUDIO_TIME_LIMIT_H=48`. Two sessions (Kaggle, or after a failure):
   1. session 1 stops at the limit; the stage is reported `verify_failed` (`ff_completeness.json` / the manifest's `complete` flag);
   2. move its output out of `FF_WORK/art` (this also removes its `_DONE.json`, so no later stage can take the partial output), e.g.
      `mv $FF_WORK/art/notebooks/ff/c46_voxblink $FF_WORK/c46_session1`;
   3. session 2: same command with `VB_RESUME_FROM=$FF_WORK/c46_session1` (`FRAMES_RESUME_FROM` with
      `art/notebooks/onnhqunhh/flag-2027-nb-63-frames`, `AUDIO_RESUME_FROM` with `art/notebooks/duongphihoang/flag-2027-nb-63v-audio-cache`);
      finished shards / chunks are copied and skipped. Unset the variable afterwards: a stale value would resume a later
      re-collection from old data.
4. **Smoke.** `c37` and `c46` smoke only in the notebooks' own local mode (`FLAG_LOCAL=1` skips their `!pip` cell): install the two
   pip lines listed in `stage.json` `smoke_prerequisite` first (no Docker needed). The other stages smoke on plain Kaggle CPU.
   Never export `FLAG_IN` / `FLAG_OUT` / `FLAG_LOCAL` / `FLAG_SMOKE` in the caller's shell: `FLAG_LOCAL` / `FLAG_SMOKE` are pinned to
   `"0"` in `stage.json` `env`, `FLAG_IN` / `FLAG_OUT` would redirect the notebooks away from the runner's paths.

## Licences and privacy

| source | licence | handling |
|---|---|---|
| VoxCeleb2 (`ProgramComputer/voxceleb`, `gaunernst/voxceleb2-dev-wds`) | CC BY 4.0 (video copyright stays with the owners); the gated mirror's auto-approved terms were not recorded | collected keys, features, PNG frames and audio clips kept in the private `FF_DATA` share only |
| VoxBlink-clean (`adollahamini1998/VoxBlink-Video-clean-8-{0..7}`, annotations `ScriptsForVoxBlinkResources`) | CC BY-NC-SA 4.0, no redistribution | everything derived from it private |
| `adollahamini1998/VoxBlinkProgressVideos` | not recorded | treated as VoxBlink: private |
| models used by the collect notebooks | per `stage.json` `models` (ImageBind weights and audeering CC BY-NC-SA 4.0; other cards not re-checked offline) | frozen, public checkpoints; not redistributed |

The face-frame dataset (`face_frames_png`) as published also holds `feats_facelora3_b/facelora_ckpt.pt`, an unused leftover of
FLAG_65-b with no producing stage that the 13.54 chain never reads (private: trained on VoxBlink / VoxCeleb2 faces). `c45`'s kernel
output contains copies of the organisers' train list and train meta file (no dev data).

References of the form "design notes", "team run log", "team dataset log", "push log line N (kaggle/kapi_runs.jsonl)" and gap ids
such as G22 / G27 in the metadata point to the team's internal provenance records, which are not shipped; the facts they support are
restated where they are used.
