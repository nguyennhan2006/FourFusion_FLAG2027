print("[cell] l42 compare: this run vs the original FLAG_42 output (new cell, report only)", flush=True)
# New cell (FourFusion reproduction repo). VoxLingua107 on GPU and the order in which tar files stream are not bit-reproducible, and a
# tar that fails after the retries is skipped (its clips stay NaN). The 13.54 chain therefore uses the STORED output of the original
# run (FF_DATA/02_intermediate/feats_vox2lid = Experiment/T3ENC-01/feats_vox2lid); this cell only reports how close a rerun is.
import hashlib, json
_ref = {"clips": 38345, "clips_scored": 38345, "clips_flagged": 54, "speakers": 4877, "speakers_dropped": 25,
        "speakers_dropped_sha256": "35d31f681975b96950d6af4ad04e56c4281732dc504b74f2be693e205127822d",
        "lid_summary_sha256": "4d05cb34fc0fa66951750bb8f37d1f01b18f32a3e26847c7f42caad8e3341085",
        "voice_lid_probs_sha256": "2bd938fc64872bd0352de4a5b400264208c023e48a4b02f6a7715ffb48960f49",
        "voice_meta_sha256": "3db779d3e58e1bdda05ea460335b808c5e17b66d67aab32dc6adeb4fb84cdfa4"}
_s = json.loads((OUT / "lid_summary.json").read_text())
_got = {k: _s.get(k) for k in ("clips", "clips_scored", "clips_flagged", "speakers", "speakers_dropped")}
_got["speakers_dropped_sha256"] = hashlib.sha256("\n".join(sorted(_s["speakers_dropped_ids"])).encode()).hexdigest()
for _n, _k in (("lid_summary.json", "lid_summary_sha256"), ("voice_lid_probs.npy", "voice_lid_probs_sha256"), ("voice_meta.csv", "voice_meta_sha256")):
    _got[_k] = hashlib.sha256((OUT / _n).read_bytes()).hexdigest()
_cmp = {k: {"original": v, "this_run": _got.get(k), "same": _got.get(k) == v} for k, v in _ref.items()}
(WORKDIR / "ff_l42_compare.json").write_text(json.dumps(_cmp, indent=1))
for k, v in _cmp.items():
    print(f"  {k:26s} {'same' if v['same'] else 'DIFFERENT'}  original {v['original']}  this run {v['this_run']}", flush=True)
