# FLAG 2027 — EXP-000c Pipeline

> Canonical baseline: **EXP-000c — Source-Faithful FOP**
>
> Current validated CodaBench baseline: **Overall EER = 42.17%**
>
> Important scoring note: current live CodaBench behaves as **distance-oriented**:
>
> `submission_score = squared_L2_distance`  
> **Lower score = more likely same speaker**

---

## 1. End-to-end pipeline

```mermaid
flowchart TD

    A["FLAG 2027 / MAV-Celeb v4"] --> B["Training data<br/>English only<br/>70 identities<br/>6,485 aligned samples"]

    B --> C1["Face features<br/>4096-D"]
    B --> C2["Voice features<br/>192-D"]

    C1 --> D1["Face branch<br/>Linear 4096→128<br/>BatchNorm<br/>ReLU<br/>Dropout 0.5<br/>L2 Normalize"]
    C2 --> D2["Voice branch<br/>Linear 192→128<br/>BatchNorm<br/>ReLU<br/>Dropout 0.5<br/>L2 Normalize"]

    D1 --> E1["Face embedding u<br/>128-D"]
    D2 --> E2["Voice embedding v<br/>128-D"]

    E1 --> F["Gated Fusion"]
    E2 --> F

    F --> G["Fused embedding l<br/>128-D"]

    G --> H1["Identity Classifier<br/>Linear 128→C"]
    G --> H2["Orthogonal Projection Loss"]

    H1 --> I1["Cross-Entropy Loss"]
    H2 --> I2["OPL"]

    I1 --> J["Total Loss<br/>L = CE + α × OPL"]
    I2 --> J

    J --> K["Optimize with Adam<br/>lr=1e-5<br/>weight_decay=0.01"]

    K --> L["Internal speaker-disjoint validation<br/>56 train speakers<br/>14 unseen validation speakers"]

    L --> M["Sweep α<br/>0, 0.1, 0.5, 1, 2, 5"]
    M --> N["Evaluate unseen-speaker EER<br/>after each epoch"]
    N --> O["Select best α + epoch"]

    O --> P["Retrain fresh model<br/>on all 70 speakers"]

    P --> Q["Final FOP checkpoint"]

    Q --> R1["Dev no_gender / English"]
    Q --> R2["Dev no_gender / Bangla"]
    Q --> R3["Dev gender / English"]
    Q --> R4["Dev gender / Bangla"]

    R1 --> S["Inference"]
    R2 --> S
    R3 --> S
    R4 --> S

    S --> T["face_trans = tanh(face embedding)<br/>voice_trans = tanh(voice embedding)"]

    T --> U["Squared Euclidean distance<br/>d = Σ(face_trans - voice_trans)^2"]

    U --> V["Write score files<br/><pair_id> <distance>"]

    V --> W["submission.zip"]

    W --> X["CodaBench"]
    X --> Y["4 EER cells"]
    Y --> Z["Overall = mean(4 EERs)"]
```

---

## 2. FOP model detail

```mermaid
flowchart LR

    A1["Face feature<br/>4096-D"]
    A2["Voice feature<br/>192-D"]

    A1 --> B1["Linear 4096→128"]
    B1 --> C1["BatchNorm"]
    C1 --> D1["ReLU"]
    D1 --> E1["Dropout 0.5"]
    E1 --> F1["L2 Normalize"]
    F1 --> G1["Face embedding u<br/>128-D"]

    A2 --> B2["Linear 192→128"]
    B2 --> C2["BatchNorm"]
    C2 --> D2["ReLU"]
    D2 --> E2["Dropout 0.5"]
    E2 --> F2["L2 Normalize"]
    F2 --> G2["Voice embedding v<br/>128-D"]

    G1 --> H["Concatenate [u,v]<br/>256-D"]
    G2 --> H

    H --> I["Attention network<br/>Linear 256→128<br/>BN → ReLU<br/>Linear 128→128"]
    I --> J["Sigmoid gate k"]

    G1 --> K1["tanh(u)"]
    G2 --> K2["tanh(v)"]

    J --> L["Element-wise gated fusion"]
    K1 --> L
    K2 --> L

    L --> M["l = k ⊙ tanh(u)<br/>+ (1-k) ⊙ tanh(v)"]

    M --> N1["Identity Classifier"]
    M --> N2["OPL"]

    N1 --> O1["Cross Entropy"]
    N2 --> O2["Orthogonality constraint"]
```

---

## 3. Gated fusion formula

Given:

- `u` = face embedding, shape `128`
- `v` = voice embedding, shape `128`

Concatenate:

```text
[u, v] ∈ R^256
```

Attention:

```text
k = sigmoid(F_att([u, v]))
```

where:

```text
F_att:
Linear(256 → 128)
→ BatchNorm1d
→ ReLU
→ Linear(128 → 128)
```

Transform each modality:

```text
u_t = tanh(u)
v_t = tanh(v)
```

Fuse:

```text
l = k ⊙ u_t + (1 - k) ⊙ v_t
```

where `⊙` is element-wise multiplication.

Interpretation:

```text
k_j ≈ 1  → dimension j relies more on face
k_j ≈ 0  → dimension j relies more on voice
```

---

## 4. Training objective

```mermaid
flowchart LR

    A["Fused embedding l"] --> B1["Identity classifier"]
    A --> B2["Pairwise cosine geometry"]

    B1 --> C1["Cross-Entropy Loss"]
    B2 --> C2["Orthogonal Projection Loss"]

    C1 --> D["Total loss"]
    C2 --> D

    D --> E["L = L_CE + α L_OPL"]
```

### Cross-Entropy

Purpose:

> Make the fused embedding discriminative for speaker identity.

```text
L_CE = classification loss over training identities
```

### Orthogonal Projection Loss

For same-identity embeddings:

```text
cosine → 1
```

For different-identity embeddings:

```text
|cosine| → 0
```

Formula used by the FOP implementation:

```text
L_OPL =
(1 - mean_positive_cosine)
+ 0.7 × mean_absolute_negative_cosine
```

Total:

```text
L_total = L_CE + α × L_OPL
```

Alpha candidates:

```text
α ∈ {0.0, 0.1, 0.5, 1.0, 2.0, 5.0}
```

---

## 5. Speaker-disjoint model selection

FLAG evaluates **unseen identities**, so random row splitting is not appropriate.

```mermaid
flowchart TD

    A["70 official training speakers"] --> B["Speaker-level split"]

    B --> C["56 speakers<br/>model fitting"]
    B --> D["14 speakers<br/>unseen validation"]

    C --> E["Train FOP"]
    E --> F["Epoch 1"]
    E --> G["Epoch 2"]
    E --> H["..."]
    E --> I["Epoch 50"]

    F --> J["Validation EER"]
    G --> J
    H --> J
    I --> J

    D --> J

    J --> K["Best epoch for this α"]

    K --> L["Repeat for every α"]

    L --> M["Select best α + epoch"]

    M --> N["Fresh model"]
    N --> O["Retrain on all 70 speakers"]
```

Important:

```text
train loss ≠ model-selection metric
```

The primary internal selection signal is:

```text
unseen-speaker verification EER
```

---

## 6. Internal validation pair construction

For each held-out sample:

### Positive pair

```text
face_i + voice_i
label = same speaker
```

### Negative pair

```text
face_i + voice_j
speaker(i) != speaker(j)
label = different speaker
```

The validation set is balanced:

```text
50% positive
50% negative
```

---

## 7. Inference pipeline

```mermaid
flowchart LR

    A["Face feature 4096-D"] --> B["Face FOP branch"]
    B --> C["128-D normalized face embedding"]
    C --> D["tanh"]
    D --> E["face_trans"]

    F["Voice feature 192-D"] --> G["Voice FOP branch"]
    G --> H["128-D normalized voice embedding"]
    H --> I["tanh"]
    I --> J["voice_trans"]

    E --> K["Squared L2 distance"]
    J --> K

    K --> L["d = Σ(face_trans - voice_trans)^2"]
```

The classifier is **not used** at test time.

The final task is not:

```text
predict one of the 70 training identities
```

It is:

```text
does this face match this voice?
```

---

## 8. Local score vs CodaBench score

This distinction is critical.

### Local ROC / AUC / EER

`sklearn` convention:

```text
higher score = positive / same speaker
```

So locally:

```text
similarity = - squared_L2_distance
```

### Current live CodaBench

Empirically validated:

```text
submission_score = + squared_L2_distance
```

and:

```text
LOWER score = more likely same speaker
```

Same checkpoint:

| Submitted value | Overall EER |
|---|---:|
| `- squared_L2` | 57.83% |
| `+ squared_L2` | **42.17%** |

Therefore:

```text
LOCAL:
-d²

CODABENCH:
+d²
```

Do **not** confuse these two.

---

## 9. Four official evaluation cells

```mermaid
flowchart TD

    A["Final English-trained FOP model"]

    A --> B1["no_gender / English<br/>heard"]
    A --> B2["no_gender / Bangla<br/>unheard"]
    A --> B3["gender / English<br/>heard"]
    A --> B4["gender / Bangla<br/>unheard"]

    B1 --> C1["EER"]
    B2 --> C2["EER"]
    B3 --> C3["EER"]
    B4 --> C4["EER"]

    C1 --> D["Overall"]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E["Overall = (EER1+EER2+EER3+EER4)/4"]
```

Definitions:

### no_gender

Standard protocol.

Negative speaker may belong to a different gender.

### gender

Gender-constrained protocol.

Negative speaker is restricted to the same gender as the positive speaker.

Purpose:

> Reduce the usefulness of gender as an identity shortcut.

### English / heard

English is available during training.

### Bangla / unheard

Bangla is not used to train the baseline.

This evaluates language generalization.

---

## 10. Submission pipeline

```mermaid
flowchart TD

    A["Dev pair TXT"]

    A --> B["Read pair_id for row i"]

    C["Face CSV row i"] --> D["FOP inference"]
    E["Voice CSV row i"] --> D

    D --> F["face_trans / voice_trans"]

    F --> G["squared L2 distance"]

    B --> H["Write result"]
    G --> H

    H --> I["pair_id distance"]

    I --> J["4 score files"]

    J --> K["submission_EXP000c_VALIDATED_DISTANCE.zip"]

    K --> L["CodaBench"]
```

Expected archive:

```text
submission.zip
├── no_gender/
│   ├── sub_score_v4_English_heard.txt
│   └── sub_score_v4_Bangla_unheard.txt
└── gender/
    ├── sub_score_v4_English_heard.txt
    └── sub_score_v4_Bangla_unheard.txt
```

Each line:

```text
<pair_id> <raw_positive_squared_L2_distance>
```

No header.

---

## 11. Current validated baseline

| Protocol | EXP-000c EER | Organizer FOP reference |
|---|---:|---:|
| no_gender / English | **36.31** | 32.54 |
| no_gender / Bangla | **38.98** | 38.12 |
| gender / English | **43.38** | 32.99 |
| gender / Bangla | **50.00** | 44.01 |
| **Overall** | **42.17** | **36.92** |

Current gap:

```text
42.17 - 36.92 = 5.25 EER points
```

Main observed weakness:

```text
gender-constrained protocols
```

especially:

```text
gender / Bangla = 50.00 EER
```

This does not yet prove that gender shortcutting is the only cause.

---

## 12. Compact pipeline

```mermaid
flowchart LR

    A["Face 4096"] --> B["Face FOP"]
    C["Voice 192"] --> D["Voice FOP"]

    B --> E["128-D"]
    D --> F["128-D"]

    E --> G["Gated Fusion"]
    F --> G

    G --> H["CE + α·OPL"]

    H --> I["Speaker-disjoint validation"]
    I --> J["Select α + epoch"]
    J --> K["Retrain 70 speakers"]

    K --> L["face_trans"]
    K --> M["voice_trans"]

    L --> N["Squared L2"]
    M --> N

    N --> O["4 FLAG protocols"]
    O --> P["CodaBench EER"]
```

---

## 13. One-line summary

```text
English face/voice features
→ modality projections
→ gated multimodal FOP training with CE + OPL
→ speaker-disjoint alpha/epoch selection
→ retrain on all 70 identities
→ compare face_trans and voice_trans using squared L2
→ submit raw distance to 4 FLAG protocol cells
→ minimize mean EER
```
