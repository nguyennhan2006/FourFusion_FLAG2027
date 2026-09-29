"""Age-gender feature extraction — the auxiliary streams of the FAME 2026 winning system.

Their system concatenates, per modality, a general identity embedding with an age-gender embedding
before the single linear mapping layer:

    face  : VGGFace 4096-d   ++  ViT age-gender 768-d    -> linear -> 192
    voice : ECAPA  6144-d    ++  ECAPA age-gender 1536-d -> linear -> 192

Their age-gender ECAPA (6.5M params, 1536-d) is not public, so we use the closest public stand-ins:

    voice : audeering/wav2vec2-large-robust-24-ft-age-gender  -> 1024-d pooled hidden state
            (fine-tuned on aGender, Common Voice, TIMIT, VoxCeleb2)
            LICENCE: CC BY-NC-SA 4.0, non-commercial. Fine for a research challenge, but it MUST be
            declared in the system description.
    face  : nateraw/vit-age-classifier -> 768-d CLS embedding (ViT-base, exactly their dimension)

Our own EDA already showed these attributes are strongly present and currently used implicitly:
a linear gender probe reaches 93% on face and 92% on voice (EDA-4). Making them an explicit,
separate stream is what lets the identity part of the embedding stop carrying them.

Used by the Kaggle extraction notebook; kept in its own file so the notebook stays readable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------- voice
class _Head(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))


def build_voice_age_gender(device):
    """Returns (model, processor). model(x) -> (pooled_1024, age_logit, gender_probs)."""
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = _Head(config, 1)
            self.gender = _Head(config, 3)
            self.post_init()        # not init_weights(): transformers >= 5 sets all_tied_weights_keys here

        def forward(self, input_values):
            h = self.wav2vec2(input_values)[0].mean(dim=1)       # mean-pool over time -> 1024-d
            return h, self.age(h), torch.softmax(self.gender(h), dim=1)

    name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    proc = Wav2Vec2Processor.from_pretrained(name)
    model = AgeGenderModel.from_pretrained(name).to(device).eval()
    return model, proc


@torch.no_grad()
def voice_age_gender_embed(model, proc, paths, root, device, bs=8, max_sec=8.0):
    """Mean-pooled 1024-d hidden state plus the age/gender predictions, in `paths` order."""
    import soundfile as sf
    from pathlib import Path

    emb, age, gen = [], [], []
    for s in range(0, len(paths), bs):
        sigs = []
        for p in paths[s:s + bs]:
            w, sr = sf.read(str(Path(root) / p), dtype="float32", always_2d=True)
            assert sr == 16000, sr
            sigs.append(w[: int(max_sec * sr), 0])               # cap length: this model is heavy
        L = max(len(w) for w in sigs)
        x = np.zeros((len(sigs), L), dtype=np.float32)
        for i, w in enumerate(sigs):
            x[i, : len(w)] = w
        x = proc(list(x), sampling_rate=16000, return_tensors="pt", padding=True).input_values
        h, a, g = model(x.to(device))
        emb.append(h.cpu().numpy()); age.append(a.cpu().numpy()); gen.append(g.cpu().numpy())
        if s % (bs * 25) == 0:
            print(f"   voice-ag {s}/{len(paths)}", flush=True)
    return np.concatenate(emb), np.concatenate(age), np.concatenate(gen)


# --------------------------------------------------------------------- face
def build_face_age(device, name="nateraw/vit-age-classifier"):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
    return model, proc


@torch.no_grad()
def face_age_embed(model, proc, paths, root, device, bs=32):
    """768-d CLS embedding from the ViT age classifier, in `paths` order."""
    from PIL import Image
    from pathlib import Path

    out = []
    for s in range(0, len(paths), bs):
        imgs = [Image.open(Path(root) / p).convert("RGB") for p in paths[s:s + bs]]
        x = proc(images=imgs, return_tensors="pt").pixel_values.to(device)
        h = model.base_model(x).last_hidden_state[:, 0]           # CLS token
        out.append(h.cpu().numpy())
        if s % (bs * 25) == 0:
            print(f"   face-ag {s}/{len(paths)}", flush=True)
    return np.concatenate(out)


# --------------------------------------------------------------------- check
def gender_agreement(gen_probs, gender_labels):
    """Sanity check against our own labels: the model's male/female call should mostly agree.

    gen_probs columns are (female, male, child): config.json id2label. The model card's prose says
    "child, female, or male", but the config and our v4 check (f -> column 0, m -> column 1) agree.
    """
    pred = np.where(gen_probs[:, 1] >= gen_probs[:, 0], "m", "f")
    return float((pred == np.asarray(gender_labels)).mean())
