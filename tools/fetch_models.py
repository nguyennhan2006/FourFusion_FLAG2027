#!/usr/bin/env python3
"""Download every public pretrained model of the FourFusion 13.54 chain at a pinned revision and check its SHA-256.

    python tools/fetch_models.py --dest <FF_DATA>/06_pretrained                 # core + dp groups (what the 13.54 numbers need)
    python tools/fetch_models.py --dest D --group all                           # + LID filters (13_lid) + wiring-only models
    python tools/fetch_models.py --dest D --only imagebind_huge,farl_ep16       # some models
    python tools/fetch_models.py --dest D --dry                                 # print the table, download nothing
    python tools/fetch_models.py --dest D --verify-only                         # hash what is there, download nothing
    python tools/fetch_models.py --list

Layout of D (= the 'pretrained/' folder of the team package, so the runner can seed caches from it: runner/ffcore.py models_env):
    imagebind_huge.pth                      direct file
    cache/flag/<file>                       the notebooks' FLAG_CACHE (fetch(url, CACHE / name, size)); WeSpeaker zips are also
    cache/flag/wespeaker/<zip stem>/        unpacked here exactly like kaggle/vendor_t3enc/t3enc_cell.py wespeaker_dir() does
    cache/huggingface/hub/                  HF hub cache (snapshot_download at the pinned revision + refs/main = that commit)
                                            -> HF_HUB_CACHE; the pin holds offline (FF_HF_OFFLINE=1 -> HF_HUB_OFFLINE=1)
    cache/clip/ViT-B-16.pt                  openai CLIP download cache; the runner copies it to ~/.cache/clip when missing
    rfs_models/                             copies under the names the RFS / RFX notebooks use for their *_CKPT variables
    insightface/models/buffalo_l/           insightface model zoo; the runner copies it to ~/.insightface/models when missing
    SOURCES.json                            what was fetched: URL / repo@revision, SHA-256, size, license, used by, time

Derived from release/docker/ff.py cmd_fetch_models (ImageBind only) in the team repository; the table follows DESIGN 9.2 and the
layer map S8 E10. SHA-256 values were measured on the team machine / team package on 2026-11-18. A model whose SHA-256 is not
known is downloaded and its hash recorded in SOURCES.json (status 'recorded', not 'verified').
HF_TOKEN (optional) is read from the environment and never printed. Standard library + huggingface_hub (for the hf entries)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

WESPK_TREE = "https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree"   # kaggle/vendor_t3enc/t3enc_cell.py:16
IB_HF = "https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth"

# group: core = enters the 13.54 numbers through F1s' / RFX / RFX'; dp = the DP branch (English); lid = Bengali-voice filtering only
# (13_lid); wiring = only so that verbatim notebooks run (not in the numbers)
MODELS = [
    dict(name="imagebind_huge", group="core", kind="url", url=IB_HF, dest="imagebind_huge.pth",
         sha256="d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157", size=4803584173,
         also_as=[], license="CC BY-NC-SA 4.0 (ImageBind)", code="facebookresearch/ImageBind@53680b02d7e37b19b124fa37bae4b6c98c38f5be",
         used_by="XALL_ib stream (e50cc), RFS f_ibf / v_ibv, base of LoRA C1 and FVBIND B2",
         note="Meta's URL dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth is dead since 2026-11-12; this HF copy (commit 40c7e2a) has the same SHA-256"),
    dict(name="farl_ep16", group="core", kind="url",
         url="https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth",
         dest="cache/flag/FaRL-Base-Patch16-LAIONFace20M-ep16.pth", sha256="67b4d6bcad0cd4bdaa302741e05e92eb82ecb334ef92446555b3abd3e3cd3862",
         size=652165542, license="FaRL (MIT code; LAION-Face weights, research use)", used_by="face farl16 of the T3 / CA heads, anchor of FVBIND B2",
         note="epoch 16 (FARL16_URL of 47rs / 50cc / 65x); kaggle/flag_models.py downloads epoch 64 for the old FLAG_10 features (group wiring)"),
    dict(name="clip_vit_b16_openai", group="core", kind="url",
         url="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
         dest="cache/clip/ViT-B-16.pt", sha256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f", size=350837078,
         license="MIT (openai/CLIP)", code="openai/CLIP@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1", used_by="backbone FaRL loads its weights into"),
    dict(name="clip_vit_large_patch14", group="core", kind="hf", repo="openai/clip-vit-large-patch14", revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
         allow=["config.json", "merges.txt", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json"],
         sha256={"model.safetensors": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
                 "config.json": "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a"},
         license="MIT (openai CLIP)", used_by="base of the face LoRA adapters C5s / B3s, RFS f_clip"),
    dict(name="simam_resnet100_vb2", group="core", kind="modelscope", zip="voxblink2_samresnet100.zip", size=186890339,
         sha256="8acb906f51366e07d6fdba0498f7464b58d466ccf5ffc60aa272e38bb856462c", dest="cache/flag/wespeaker/voxblink2_samresnet100",
         license="CC BY-NC-SA (VoxBlink2-trained WeSpeaker model)", used_by="voice sam100p (T3 / CA heads)"),
    dict(name="simam_resnet100_vb2_ft", group="core", kind="modelscope", zip="voxblink2_samresnet100_ft.zip", size=186890839,
         sha256="ad0873d380acaa7f4256ff37d40217ee31e4955b26a45064a13a14998cc89d16", dest="cache/flag/wespeaker/voxblink2_samresnet100_ft",
         license="CC BY-NC-SA (VoxBlink2-trained WeSpeaker model)", used_by="voice sam100f (T3 / CA heads)"),
    dict(name="simam_resnet100_ft_rfs", group="core", kind="url",
         url="https://huggingface.co/MCplayer/voxblink2_samresnet100_ft/resolve/main/models/voxblink2_samresnet100_ft/avg_model.pt",
         dest="cache/flag/avg_model.pt", sha256="5aeee438ca23c0ca6e341bab6c6bf7f465497e1dc323bb1bc1074d6a0c778b11", size=201318407,
         also_as=["rfs_models/voxblink2_samresnet100_ft.pt"], license="CC BY-NC-SA (VoxBlink2)", used_by="RFS v_sam (162a/b/c, 163i)",
         note="revision 'main' (never pinned by a commit); the SHA-256 pins the file"),
    dict(name="wespeaker_resnet293_lm", group="core", kind="modelscope", zip="voxceleb_resnet293_LM.zip", size=124116126,
         sha256="6a96760ea9b100f5721bc645988af98d95c48075cea0293e44b1e428260b9d53", dest="cache/flag/wespeaker/voxceleb_resnet293_LM",
         license="Apache-2.0 (WeSpeaker) / VoxCeleb", used_by="voice rn293 of CAT4 (CA head), v_rl block of RFX / RFX' (via e47rs)"),
    dict(name="wespeaker_ecapa1024_lm", group="core", kind="modelscope", zip="voxceleb_ECAPA1024_LM.zip", size=67665905,
         sha256="ea14d0f55b1c101ee14ea7734a05ae9929b5d30edd3735a9b572cb5e9c453c62", dest="cache/flag/wespeaker/voxceleb_ECAPA1024_LM",
         license="Apache-2.0 (WeSpeaker) / VoxCeleb", used_by="voice ecapa1024 of CAT4, v_rl block"),
    dict(name="wespeaker_resnet221_lm", group="core", kind="url",
         url="https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet221-LM/resolve/main/voxceleb_resnet221_LM.pt",
         dest="cache/flag/voxceleb_resnet221_LM.pt", sha256="47d76239f1e865b273e31e30f691959061ff6565ed6c4e7be9639a65d9662eb5", size=114410390,
         also_as=["rfs_models/resnet221_lm.pt"], license="Apache-2.0 (WeSpeaker)", used_by="RFS v_r221", note="revision 'main'; the SHA-256 pins the file"),
    dict(name="ecapa_btc_space", group="core", kind="hf", repo="yangwang825/ecapa-tdnn-vox2", revision="c605f57148d4e910d4241eec714d7bc948220adb",
         allow=["classifier.ckpt", "embedding_model.ckpt", "hyperparams.yaml", "label_encoder.txt"],
         sha256={"embedding_model.ckpt": "a9c25a7b62e68b1ad75831cda9b4c5e5e21672580c9a11cb369db7e1f2250bbf"},
         license="unknown (model card not read offline; not redistributed)", used_by="organisers-space ECAPA: XALL (FLAG_08 rows), RFS xv, DP (MAV rows)",
         note="revision = the team machine cache (refs/main); the Kaggle runs used 'main' at run time; the embedding SHA-256 equals the team package copy rfs_models/sb_btc/embedding_model.ckpt"),
    dict(name="vgg_face_dag", group="core", kind="url", url="https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth",
         dest="cache/flag/vgg_face_dag.pth", sha256="f070b241e3faf17a08c78f8700d334413230bac5de7fbcfd01b3cf05ce10de1a", size=580015466,
         also_as=["rfs_models/vgg_face.pt"], license="VGG-Face (non-commercial research)", used_by="FLAG_08 rows (organisers-space faces), RFS xf, DP (MAV rows)"),
    dict(name="whisper_large_v3", group="core", kind="hf", repo="openai/whisper-large-v3", revision="06f233fe06e710322aca913c1bc4249a0d71fce1",
         allow=["config.json", "preprocessor_config.json", "model.safetensors", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json", "special_tokens_map.json"],
         sha256={}, license="Apache-2.0", used_by="RFS v_whv3 (160a/b/c, 163i); language ID of FLAG_150 (13_lid)",
         note="file SHA-256 not measured (not in the team package): recorded at fetch time"),
    dict(name="vit_age", group="core", kind="hf", repo="nateraw/vit-age-classifier", revision="4b5acc345d9692c3ff4f45d250531afd3a3da0ef",
         allow=["config.json", "model.safetensors", "preprocessor_config.json"],
         sha256={"model.safetensors": "44813db5205ae64124c5fe19244453d866e9516aa18afca454cc285cfec67262"}, license="unknown (model card not read offline; not redistributed)",
         used_by="face age (fuser conditions, age-match stream)"),
    dict(name="audeering_age_gender", group="core", kind="hf", repo="audeering/wav2vec2-large-robust-24-ft-age-gender",
         revision="bad2737f529168bc3d3fcd97edfaba9583c7d36d", allow=["config.json", "model.safetensors", "preprocessor_config.json", "vocab.json"],
         sha256={"model.safetensors": "5868cc9548a1dba701654ae4e0599104ba734eb0b3f0a03384f9180d2dfb06a5"}, license="CC BY-NC-SA 4.0",
         used_by="voice age / gender"),
    dict(name="spkrec_ecapa_voxceleb", group="dp", kind="hf", repo="speechbrain/spkrec-ecapa-voxceleb", revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
         allow=["classifier.ckpt", "embedding_model.ckpt", "hyperparams.yaml", "label_encoder.txt", "mean_var_norm_emb.ckpt", "config.json"],
         sha256={"embedding_model.ckpt": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"}, license="Apache-2.0",
         used_by="DP voice ECAPA 192 + pooled-statistics 6144; also loaded by e50cc"),
    dict(name="adaface_ir101_webface12m", group="dp", kind="hf", repo="minchul/cvlface_adaface_ir101_webface12m",
         revision="54f602a0737bd1ee4a4e7e9fd089a485f397fefd", allow=None,
         sha256={"model.safetensors": "2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56",
                 "pretrained_model/model.pt": "e312d79222d28027f146ed495e182e48fe0dddf404cdbbdabcccdbdd07cc3758"},
         license="MIT (CVLface) / WebFace12M terms", used_by="DP face AdaFace (512)",
         note="the code used snapshot_download without a revision; this revision = the team machine snapshot"),
    dict(name="insightface_buffalo_l", group="dp", kind="zip_url", url="https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip",
         dest="insightface/models/buffalo_l",
         sha256={"w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43", "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"},
         license="insightface (non-commercial research)", used_by="DP face ArcFace w600k_r50 (ResNet-50, WebFace600K) + SCRFD det_10g",
         note="URL of the insightface 2.0 model zoo (insightface/utils/storage.py); only the two files above are used"),
    dict(name="audioclip_full", group="wiring", kind="url", url="https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt",
         dest="cache/flag/AudioCLIP-Full-Training.pt", sha256="2441d35b353352c8b1bbfb8f7c687f46314c3d2909e940eaf763b8c17f632c44", size=537302068,
         license="MIT (AudioCLIP)", used_by="only so that e50cc runs verbatim (old English member, not in the 13.54 zip)"),
    dict(name="clip_vit_base_patch16", group="wiring", kind="hf", repo="openai/clip-vit-base-patch16", revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
         allow=["config.json", "merges.txt", "preprocessor_config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
         sha256={"pytorch_model.bin": "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0"}, license="MIT", used_by="e50cc verbatim (unverified: whether it enters the numbers)"),
    dict(name="clip_vit_base_patch32", group="wiring", kind="hf", repo="openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
         allow=["config.json", "merges.txt", "preprocessor_config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
         sha256={"pytorch_model.bin": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"}, license="MIT", used_by="e50cc verbatim"),
    dict(name="farl_ep64", group="wiring", kind="url",
         url="https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
         dest="cache/flag/FaRL-Base-Patch16-LAIONFace20M-ep64.pth", sha256=None, license="FaRL", used_by="old FLAG_10 'farl' arrays (wiring; only rel streams)"),
    dict(name="siglip2_base", group="wiring", kind="hf", repo="google/siglip2-base-patch16-224", revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
         allow=["config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"],
         sha256={}, license="Apache-2.0", used_by="old FLAG_10 'siglip2' arrays (wiring)", note="revision = the team machine cache; the FLAG_10 run used 'main'"),
    dict(name="voxlingua107", group="lid", kind="hf", repo="speechbrain/lang-id-voxlingua107-ecapa", revision="0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9",
         allow=["classifier.ckpt", "embedding_model.ckpt", "hyperparams.yaml", "label_encoder.txt"],
         sha256={"embedding_model.ckpt": "ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a"}, license="Apache-2.0",
         used_by="Bengali-voice filtering of FLAG_42 / 46 / 53 (13_lid)",
         note="the FLAG_42 run resolved 'main' to this commit (Kaggle's __huggingface_repos__.json in its output, per WP-B); same as the team machine cache"),
    dict(name="mms_lid_4017", group="lid", kind="hf", repo="facebook/mms-lid-4017", revision=None, allow=None, sha256={},
         license="CC BY-NC 4.0", used_by="MMS-LID audit of the MAV voice files (FLAG_152, report input)", note="revision not recorded by the run: resolved at fetch time"),
]
GROUPS = {"core": ["core"], "dp": ["dp"], "lid": ["lid"], "wiring": ["wiring"], "default": ["core", "dp"], "all": ["core", "dp", "lid", "wiring"]}


def sha256(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dst, size=None, sha=None):
    dst = Path(dst)
    if dst.is_file() and (size is None or dst.stat().st_size == size) and (sha is None or sha256(dst) == sha):
        print(f"  present  {dst}", flush=True)
        return "present"
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    print(f"  download {url}\n        -> {dst}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "fourfusion-fetch-models/1.0"})
    tok = os.environ.get("HF_TOKEN")
    if tok and "huggingface.co" in url:
        req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=180) as r, open(part, "wb") as f:
        n, total, mark = 0, int(r.headers.get("Content-Length") or 0), 0
        for b in iter(lambda: r.read(1 << 22), b""):
            f.write(b)
            n += len(b)
            if n - mark >= 1 << 30:
                mark = n
                print(f"    {n / 2**30:.1f} / {total / 2**30:.1f} GB", flush=True)
    if size is not None and part.stat().st_size != size:
        got = part.stat().st_size
        part.unlink()
        raise RuntimeError(f"{url}: {got} bytes, expected {size}")
    if sha is not None:
        got = sha256(part)
        if got != sha:
            part.unlink()
            raise RuntimeError(f"{url}: sha256 {got} != {sha}: file discarded")
    part.replace(dst)
    return "downloaded"


def unpack_wespeaker(zp, d):
    """Same as kaggle/vendor_t3enc/t3enc_cell.py wespeaker_dir(): members without __MACOSX, flattened; a single <model>.pt /
    <model>.yaml renamed to avg_model.pt / config.yaml."""
    with zipfile.ZipFile(zp) as zf:
        for mem in zf.namelist():
            if not mem.startswith("__MACOSX") and not mem.endswith("/"):
                (d / Path(mem).name).write_bytes(zf.read(mem))
    pts = sorted(p for p in d.glob("*.pt"))
    ymls = sorted(d.glob("*.yaml"))
    if not (d / "avg_model.pt").exists() and len(pts) == 1:
        pts[0].rename(d / "avg_model.pt")
    if not (d / "config.yaml").exists() and len(ymls) == 1:
        ymls[0].rename(d / "config.yaml")
    assert (d / "avg_model.pt").exists() and (d / "config.yaml").exists(), f"{zp.name}: unexpected content"


def fetch_one(m, dest, verify_only=False):
    rec = {k: m.get(k) for k in ("name", "group", "kind", "url", "repo", "revision", "zip", "license", "used_by", "note", "code")}
    rec["files"] = {}
    kind = m["kind"]
    if kind == "url":
        t = dest / m["dest"]
        if not verify_only:
            rec["status"] = download(m["url"], t, m.get("size"), m.get("sha256"))
        if t.is_file():
            h = sha256(t)
            rec["files"][m["dest"]] = {"sha256": h, "bytes": t.stat().st_size}
            rec["verified"] = (h == m["sha256"]) if m.get("sha256") else None
            for alt in m.get("also_as", []):
                a = dest / alt
                if not a.is_file():
                    a.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(t, a)
                    except OSError:
                        shutil.copy2(t, a)
                rec["files"][alt] = {"sha256": h, "copy_of": m["dest"]}
    elif kind == "modelscope":
        d = dest / m["dest"]
        z = d / m["zip"]
        if not verify_only and not z.is_file():
            import json as _j
            with urllib.request.urlopen(urllib.request.Request(WESPK_TREE, headers={"User-Agent": "fourfusion-fetch-models/1.0"}), timeout=120) as r:
                tree = _j.loads(r.read().decode("utf-8"))["Data"]
            url = [it["Url"] for it in tree if it.get("Key") == m["zip"]][0]
            rec["url"] = url
            rec["status"] = download(url, z, m.get("size"), m.get("sha256"))
        if z.is_file():
            h = sha256(z)
            rec["files"][f"{m['dest']}/{m['zip']}"] = {"sha256": h, "bytes": z.stat().st_size}
            rec["verified"] = h == m["sha256"]
            if not verify_only and not ((d / "avg_model.pt").exists() and (d / "config.yaml").exists()):
                unpack_wespeaker(z, d)
            for f in ("avg_model.pt", "config.yaml"):
                if (d / f).exists():
                    rec["files"][f"{m['dest']}/{f}"] = {"sha256": sha256(d / f)}
    elif kind == "zip_url":
        d = dest / m["dest"]
        z = d.parent / (Path(m["url"]).name)
        if not verify_only and not all((d / f).is_file() for f in m["sha256"]):
            rec["status"] = download(m["url"], z)
            d.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                for mem in zf.namelist():
                    if not mem.endswith("/"):
                        (d / Path(mem).name).write_bytes(zf.read(mem))
        ok = True
        for f, h in m["sha256"].items():
            p = d / f
            got = sha256(p) if p.is_file() else None
            rec["files"][f"{m['dest']}/{f}"] = {"sha256": got}
            ok &= got == h
        rec["verified"] = ok
    elif kind == "hf":
        cache = dest / "cache" / "huggingface" / "hub"
        snap = None
        if not verify_only:
            from huggingface_hub import snapshot_download  # noqa: E402
            snap = snapshot_download(m["repo"], revision=m.get("revision"), cache_dir=str(cache), allow_patterns=m.get("allow"),
                                     token=os.environ.get("HF_TOKEN") or None)
        else:
            base = cache / ("models--" + m["repo"].replace("/", "--")) / "snapshots"
            if base.is_dir():
                cand = [base / m["revision"]] if m.get("revision") else sorted(base.iterdir())
                snap = next((c for c in cand if c.is_dir()), None)
        if snap:
            snap = Path(snap)
            rec["resolved_revision"] = snap.name
            if m.get("revision") and len(m["revision"]) == 40 and not verify_only:
                # snapshot_download(revision=<commit>) writes no refs/<name> when the revision IS a commit hash, so the notebooks'
                # unchanged from_pretrained(repo) (revision 'main') could not resolve this snapshot offline. Point refs/main at the
                # pinned commit: with HF_HUB_OFFLINE=1 (runner: FF_HF_OFFLINE=1) every 'main' lookup then loads the pinned files.
                refs = cache / ("models--" + m["repo"].replace("/", "--")) / "refs"
                refs.mkdir(parents=True, exist_ok=True)
                (refs / "main").write_text(m["revision"], encoding="utf-8")
                rec["refs_main"] = m["revision"]
            ok = True
            for f, h in (m.get("sha256") or {}).items():
                p = snap / f
                got = sha256(p) if p.is_file() else None
                rec["files"][f] = {"sha256": got}
                ok &= got == h
            if not m.get("sha256"):
                for p in sorted(snap.rglob("*")):
                    if p.is_file() and p.stat().st_size > 1 << 20:
                        rec["files"][p.relative_to(snap).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
            rec["verified"] = ok if m.get("sha256") else None
    else:
        raise ValueError(kind)
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", help="target folder (FF_DATA/06_pretrained)")
    ap.add_argument("--group", default="default", choices=sorted(GROUPS))
    ap.add_argument("--only", help="comma-separated model names (see --list)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    sel = [m for m in MODELS if m["group"] in GROUPS[a.group]]
    if a.only:
        names = set(a.only.split(","))
        sel = [m for m in MODELS if m["name"] in names]
        bad = names - {m["name"] for m in sel}
        if bad:
            sys.exit(f"unknown model(s) {sorted(bad)}")
    if a.list or a.dry:
        for m in (MODELS if a.list else sel):
            src = m.get("url") or (f"{m['repo']}@{m.get('revision') or 'main'}" if m.get("repo") else f"ModelScope {m.get('zip')}")
            sha = m.get("sha256")
            shs = sha[:16] + "..." if isinstance(sha, str) else (", ".join(f"{k}={v[:12]}..." for k, v in sha.items()) if sha else "not known (recorded at fetch)")
            print(f"{m['name']:28s} {m['group']:7s} {m['kind']:10s} {src[:110]}\n{'':48s}sha256 {shs}\n{'':48s}used by: {m['used_by']}")
        return 0
    if not a.dest:
        sys.exit("--dest is required")
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    sp = dest / "SOURCES.json"
    sources = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    failed = []
    for m in sel:
        print(f"[{m['name']}] {m['group']} {m['kind']}", flush=True)
        try:
            rec = fetch_one(m, dest, verify_only=a.verify_only)
            sources[m["name"]] = rec
            print(f"  verified: {rec.get('verified')}", flush=True)
            if rec.get("verified") is False:
                failed.append(m["name"])
        except Exception as ex:
            print(f"  FAILED: {type(ex).__name__}: {ex}", flush=True)
            failed.append(m["name"])
        sp.write_text(json.dumps(sources, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(sel) - len(failed)}/{len(sel)} ok; SOURCES.json -> {sp}" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
