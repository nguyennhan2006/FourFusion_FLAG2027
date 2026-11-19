# Licences

## 1. This repository

The code written for this repository (runner, tools, new stages, documentation) is released by team FourFusion for the
reproduction of its FLAG 2027 submission. The verbatim notebooks and scripts under `stages/*/*/original/`, `stages/**/tree/` and
`src/` are the team's own code as it ran. Third-party code fetched at run time (ImageBind, CLIP, WeSpeaker, AudioCLIP, SpeechBrain,
insightface, Hugging Face libraries) keeps its own licence and is not copied here, except where a stage embeds a module verbatim as
the original run did (the stage's `original/` folder says so).

## 2. Data (never redistributed by this repository)

| data | terms | how this repository handles it |
|---|---|---|
| MAV-Celeb v4 (organisers' train, dev and Evaluation sets, meta file) | FLAG 2027 challenge terms | never copied into the repository; the Drive data folder is shared only with the organisers |
| MAV-Celeb v1 / v2 | terms of the public MAV-Celeb release | same |
| VoxCeleb2 (via Hugging Face mirrors) | VoxCeleb terms (research use) | collected copies stay in the private Drive folder |
| VoxBlink (via the Hugging Face mirror `adollahamini1998/VoxBlink-Video-clean-8-*`) | CC BY-NC-SA, no redistribution of the dataset | media, features and weights trained on it are private (Drive folder shared with the organisers' e-mail only); none is in this repository |

AVSpeech and LRS3 are not used by the 13.54 chain.

## 3. Pretrained models

Fetched at a pinned revision with a sha256 check by `tools/fetch_models.py` into `FF_DATA/06_pretrained/` (never committed). The
licence column is copied from `tools/fetch_models.py`; "unknown" means the model card was not read offline, so users must read it
before use. Several models are for non-commercial research only (ImageBind, insightface `buffalo_l`, VGG-Face, the VoxBlink2-trained
SimAM-ResNet100, the audeering age / gender model).

<!-- BEGIN GENERATED: models -->
| model | group | source (pinned) | licence (as recorded in tools/fetch_models.py) | used by |
|---|---|---|---|---|
| `imagebind_huge` | core | https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind | CC BY-NC-SA 4.0 (ImageBind) | XALL_ib stream (e50cc), RFS f_ibf / v_ibv, base of LoRA C1 and FVBIND B2 |
| `farl_ep16` | core | https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16 | FaRL (MIT code; LAION-Face weights, research use) | face farl16 of the T3 / CA heads, anchor of FVBIND B2 |
| `clip_vit_b16_openai` | core | https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416 | MIT (openai/CLIP) | backbone FaRL loads its weights into |
| `clip_vit_large_patch14` | core | openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41 | MIT (openai CLIP) | base of the face LoRA adapters C5s / B3s, RFS f_clip |
| `simam_resnet100_vb2` | core | voxblink2_samresnet100.zip | CC BY-NC-SA (VoxBlink2-trained WeSpeaker model) | voice sam100p (T3 / CA heads) |
| `simam_resnet100_vb2_ft` | core | voxblink2_samresnet100_ft.zip | CC BY-NC-SA (VoxBlink2-trained WeSpeaker model) | voice sam100f (T3 / CA heads) |
| `simam_resnet100_ft_rfs` | core | https://huggingface.co/MCplayer/voxblink2_samresnet100_ft/resolve/main/models/voxblink2_samresnet100_ft/avg_mo | CC BY-NC-SA (VoxBlink2) | RFS v_sam (162a/b/c, 163i) |
| `wespeaker_resnet293_lm` | core | voxceleb_resnet293_LM.zip | Apache-2.0 (WeSpeaker) / VoxCeleb | voice rn293 of CAT4 (CA head), v_rl block of RFX / RFX' (via e47rs) |
| `wespeaker_ecapa1024_lm` | core | voxceleb_ECAPA1024_LM.zip | Apache-2.0 (WeSpeaker) / VoxCeleb | voice ecapa1024 of CAT4, v_rl block |
| `wespeaker_resnet221_lm` | core | https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet221-LM/resolve/main/voxceleb_resnet221_LM.pt | Apache-2.0 (WeSpeaker) | RFS v_r221 |
| `ecapa_btc_space` | core | yangwang825/ecapa-tdnn-vox2@c605f57148d4e910d4241eec714d7bc948220adb | unknown (model card not read offline; not redistributed) | organisers-space ECAPA: XALL (FLAG_08 rows), RFS xv, DP (MAV rows) |
| `vgg_face_dag` | core | https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth | VGG-Face (non-commercial research) | FLAG_08 rows (organisers-space faces), RFS xf, DP (MAV rows) |
| `whisper_large_v3` | core | openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1 | Apache-2.0 | RFS v_whv3 (160a/b/c, 163i); language ID of FLAG_150 (13_lid) |
| `vit_age` | core | nateraw/vit-age-classifier@4b5acc345d9692c3ff4f45d250531afd3a3da0ef | unknown (model card not read offline; not redistributed) | face age (fuser conditions, age-match stream) |
| `audeering_age_gender` | core | audeering/wav2vec2-large-robust-24-ft-age-gender@bad2737f529168bc3d3fcd97edfaba9583c7d36d | CC BY-NC-SA 4.0 | voice age / gender |
| `spkrec_ecapa_voxceleb` | dp | speechbrain/spkrec-ecapa-voxceleb@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286 | Apache-2.0 | DP voice ECAPA 192 + pooled-statistics 6144; also loaded by e50cc |
| `adaface_ir101_webface12m` | dp | minchul/cvlface_adaface_ir101_webface12m@54f602a0737bd1ee4a4e7e9fd089a485f397fefd | MIT (CVLface) / WebFace12M terms | DP face AdaFace (512) |
| `insightface_buffalo_l` | dp | https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip | insightface (non-commercial research) | DP face ArcFace w600k_r50 (ResNet-50, WebFace600K) + SCRFD det_10g |
| `audioclip_full` | wiring | https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt | MIT (AudioCLIP) | only so that e50cc runs verbatim (old English member, not in the 13.54 zip) |
| `clip_vit_base_patch16` | wiring | openai/clip-vit-base-patch16@57c216476eefef5ab752ec549e440a49ae4ae5f3 | MIT | e50cc verbatim (unverified: whether it enters the numbers) |
| `clip_vit_base_patch32` | wiring | openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 | MIT | e50cc verbatim |
| `farl_ep64` | wiring | https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64 | FaRL | old FLAG_10 'farl' arrays (wiring; only rel streams) |
| `siglip2_base` | wiring | google/siglip2-base-patch16-224@75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 | Apache-2.0 | old FLAG_10 'siglip2' arrays (wiring) |
| `voxlingua107` | lid | speechbrain/lang-id-voxlingua107-ecapa@0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9 | Apache-2.0 | Bengali-voice filtering of FLAG_42 / 46 / 53 (13_lid) |
| `mms_lid_4017` | lid | facebook/mms-lid-4017 | CC BY-NC 4.0 | MMS-LID audit of the MAV voice files (FLAG_152, report input) |
<!-- END GENERATED: models -->

Runtime downloads that the verbatim code does not pin by revision (the runner seeds the caches from `FF_DATA/06_pretrained`
when `FF_MODELS` is set, which pins them in practice): the ImageBind code (`pip install git+…/ImageBind`, HEAD; pinned to commit
`53680b02d7e37b19b124fa37bae4b6c98c38f5be` only with `FF_PIP=pinned`), `speechbrain>=1.1.1`, the organisers-space ECAPA
`yangwang825/ecapa-tdnn-vox2`, VoxLingua107, MMS-LID-4017 and Whisper in the LID stages (`docs/VERIFICATION.md` §4).
