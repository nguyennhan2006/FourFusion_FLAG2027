# FLAG 2027 — FourFusion  

> **Lịch sử (27/09). Đề xuất về identity-set training (SetProto) đã được kiểm chứng với cùng ngân sách và **không có tác dụng** (74); face sequence / cross-lingual robustness: xem OPEN_DIRECTIONS.** Trạng thái hiện hành (29/09, nhật ký tới mục 75): cấu hình Evaluation là **SET-13 (CB 21.01) qua `BEST/v6_eval`**; xem [DECISIONS.md](DECISIONS.md) và [OPEN_DIRECTIONS.md](OPEN_DIRECTIONS.md).
## Research Directions V6: Embedding Generalization, Identity-Set Training, Face Sequence Processing, and Cross-Lingual Robustness

**Ngày lập:** 2026-09-27  
**Mục tiêu:** gom toàn bộ các hướng kỹ thuật còn hợp lý sau các thí nghiệm hiện tại của FourFusion, kèm hướng dẫn triển khai, tiêu chí đánh giá, stop-rule và các lưu ý để tránh lặp lại các hướng đã thất bại.

---

# 0. Trạng thái bài toán hiện tại

## 0.1. Best system hiện tại

Theo experiment log mới nhất:

- **FUSE-03:** 28.69 EER.
- **SET-02:** **21.68 EER**, hiện là bước nhảy lớn nhất.
- SET-02 đạt:
  - `no_gender / English`: **12.50**
  - `no_gender / Bangla`: **21.34**
  - `gender / English`: **28.31**
  - `gender / Bangla`: **24.59**

Kết luận lớn nhất sau SET-02:

> **Phần lớn lỗi trước đây không chỉ đến từ bridge yếu, mà còn đến từ variance rất lớn giữa các sample của cùng một người.**  
> Nếu có thể gom nhiều quan sát cùng identity trước khi chấm, EER giảm rất mạnh.

ORA-01 / SET-01 cho thấy:

- trung bình nhiều ảnh thật của cùng người giúp khoảng 3–5 EER;
- trung bình nhiều giọng thật của cùng người cũng giúp tương tự;
- trung bình cả hai phía có thể giúp 7–12 EER;
- clustering không giám sát bằng encoder nhận dạng mạnh lấy được phần lớn gain oracle.

Điều này dẫn tới một thay đổi quan trọng:

```text
trước đây:
row-level embedding
        ↓
cross-modal bridge
        ↓
score

nên nghiên cứu tiếp:
identity-set / prototype
        ↓
cross-modal bridge
        ↓
identity-level score
```

---

# 0.2. Các kết luận đã đủ mạnh để coi là constraint

## Không tiếp tục theo hướng “encoder recognition càng mạnh càng tốt”

### Face
- VGGFace fc7:
  - face↔face yếu hơn ArcFace
  - nhưng cross-modal tốt hơn.
- ArcFace:
  - face↔face cực mạnh
  - cross-modal rất tệ.

### Voice
- ReDimNet2:
  - speaker verification tốt hơn BTC ECAPA
  - nhưng cross-modal không tốt hơn.

**Kết luận:**

> Recognition embedding tối ưu cho unimodal identity không nhất thiết chứa những latent factors mà modality kia có thể dự đoán.

Do đó:

- ArcFace / ReDimNet rất phù hợp cho **clustering / teacher / quality estimation**.
- VGGFace / BTC ECAPA phù hợp hơn cho **cross-modal bridge**.

---

# 0.3. Các hướng đã đóng hoặc hạ ưu tiên

Không mở lại nếu không có cơ chế mới rõ ràng:

- full CORAL / partial CORAL;
- AS-norm;
- explicit gender fusion;
- gender GRL;
- same-gender negatives kiểu cũ;
- blind WavLM/XLS-R pooling;
- ArcFace replacement;
- ReDimNet replacement chỉ vì unimodal EER tốt;
- AAM / shared center như nguồn gain chính;
- dropout 0.9;
- linear head kiểu FAME26;
- graph score smoothing;
- quality router nhiều tham số fit trên dev;
- MSE alignment đơn thuần trên L2-normalized embedding;
- Dual-LoRA nếu vẫn giữ pipeline frozen-feature.

---

# 1. Triết lý nghiên cứu mới

Bốn câu hỏi chính:

1. **Embedding có đang học đúng đơn vị test không?**  
   Test tốt nhất hiện nay hoạt động ở cấp identity-set, trong khi train chủ yếu row-level.

2. **Loss hiện tại có false negative cùng identity không?**  
   Nếu diagonal-only InfoNCE được dùng, nhiều sample cùng speaker trong batch có thể đang bị coi là negatives.

3. **Embedding có generalize sang identity mới không?**  
   70 speaker v4 quá ít để học một mapping identity-specific phức tạp.

4. **Cross-lingual generalization có được học trực tiếp không?**  
   v1/v2 có song ngữ thật, vì vậy có thể tạo supervision En↔Urdu/Hindi thay vì chỉ hy vọng invariance tự xuất hiện.

---

# 2. EMB-01 — Audit positive construction và false negatives

## 2.1. Câu hỏi

Loss hiện tại có xem các sample cùng identity ngoài diagonal là negative không?

Ví dụ batch:

```text
face_A1
face_A2
face_B1
face_B2

voice_A1
voice_A2
voice_B1
voice_B2
```

Nếu target chỉ là ma trận identity:

```text
diag = positive
off-diag = negative
```

thì:

```text
face_A1 ↔ voice_A2
```

bị coi là negative dù cùng speaker A.

Đây là false negative.

---

## 2.2. EDA bắt buộc

Ghi lại cho mỗi batch:

```text
batch_size
num_unique_id
same_id_offdiag_pairs
false_negative_ratio
```

Ví dụ:

```python
same = spk_face[:, None] == spk_voice[None, :]
diag = torch.eye(len(spk_face), dtype=torch.bool, device=same.device)

false_neg = same & (~diag)

ratio = false_neg.sum() / ((~diag).sum())
```

Báo cáo:

- mean false-negative ratio;
- p50 / p90;
- theo dedup=True / dedup=False.

---

## 2.3. Stop rule

Nếu:

```text
false-negative ratio ≈ 0
```

thì không cần đổi loss vì lý do này.

Nếu:

```text
false-negative ratio > 5–10%
```

thì EMB-01 trở thành ưu tiên cao.

---

# 3. EMB-02 — Multi-positive InfoNCE

## 3.1. Ý tưởng

Thay vì mỗi anchor chỉ có 1 positive:

\[
P(i)=\{j:id_j=id_i\}
\]

Mọi sample cùng identity đều là positive.

Một dạng loss:

\[
L_i=
-\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\log
\frac{\exp(s_{ip}/\tau)}
{\sum_j \exp(s_{ij}/\tau)}
\]

Làm symmetric:

```text
face → voice
voice → face
```

---

## 3.2. Code skeleton

```python
def multi_positive_nce(zf, zv, ids_f, ids_v, tau=0.07):
    zf = F.normalize(zf, dim=1)
    zv = F.normalize(zv, dim=1)

    logits = zf @ zv.T / tau
    pos = ids_f[:, None].eq(ids_v[None, :])

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    pos_count = pos.sum(dim=1).clamp_min(1)
    loss_f2v = -(log_prob * pos).sum(dim=1) / pos_count

    logits_t = logits.T
    pos_t = pos.T
    log_prob_t = logits_t - torch.logsumexp(logits_t, dim=1, keepdim=True)
    pos_count_t = pos_t.sum(dim=1).clamp_min(1)
    loss_v2f = -(log_prob_t * pos_t).sum(dim=1) / pos_count_t

    return 0.5 * (loss_f2v.mean() + loss_v2f.mean())
```

---

## 3.3. Lưu ý

Không dùng duplicate rows làm “nhiều positive giả” nếu chúng thực chất cùng một clip được copy nguyên.

Nên phân biệt:

```text
same identity
same voice clip
same frame
```

để biết gain đến từ identity supervision thật hay chỉ duplicate weighting.

---

# 4. EMB-03 — Pairwise sigmoid / SigLIP-style cross-modal loss

## 4.1. Vì sao khác InfoNCE

InfoNCE cạnh tranh softmax toàn batch.

Sigmoid loss xem từng pair độc lập:

\[
y_{ij}=
\begin{cases}
+1 & id_i=id_j\\
-1 & id_i\neq id_j
\end{cases}
\]

\[
L=-\frac1{N}
\sum_{ij}
\log\sigma(y_{ij}(s_{ij}/\tau+b))
\]

Ưu điểm:

- dễ hỗ trợ multi-positive;
- ít phụ thuộc batch size hơn;
- không ép toàn bộ positive cạnh tranh với nhau qua softmax denominator.

---

## 4.2. Code skeleton

```python
def pairwise_sigmoid_loss(zf, zv, ids_f, ids_v, scale=10.0, bias=0.0):
    zf = F.normalize(zf, dim=1)
    zv = F.normalize(zv, dim=1)

    sim = zf @ zv.T
    y = ids_f[:, None].eq(ids_v[None, :]).float() * 2 - 1

    logits = scale * sim + bias

    return F.softplus(-y * logits).mean()
```

---

## 4.3. Ablation tối thiểu

```text
A: diagonal InfoNCE
B: multi-positive InfoNCE
C: pairwise sigmoid
```

Giữ nguyên:

- architecture;
- optimizer;
- seed;
- epoch;
- batch sampling.

---

# 5. EMB-04 — Identity-positive resampling

## 5.1. Ý tưởng

Thay vì positive cố định theo row:

```text
face_A1 ↔ voice_A1
```

sample độc lập trong cùng identity:

```text
face_A1 ↔ voice_A3
face_A2 ↔ voice_A7
face_A9 ↔ voice_A2
```

Tức:

\[
f\sim F_i,\quad
v\sim V_i
\]

---

## 5.2. Vì sao hợp FLAG

SES-01 cho thấy feature hiện tại gần như không giữ usable same-session information.

Do đó positive không cần gắn với đúng clip ban đầu.

Điều cần học là:

```text
identity consistency
```

không phải:

```text
instance synchronization
```

---

## 5.3. Implementation

Dataloader nên index theo speaker:

```python
speaker_to_faces = {...}
speaker_to_voices = {...}

for speaker in sampled_speakers:
    f = random.choice(speaker_to_faces[speaker])
    v = random.choice(speaker_to_voices[speaker])
```

Có thể mỗi speaker lấy:

```text
2 faces
2 voices
```

để tạo nhiều positives.

---

# 6. EMB-05 — Stochastic Identity-Set / Prototype Training

Đây là hướng quan trọng nhất.

---

## 6.1. Ý tưởng

Mỗi batch sample nhiều quan sát cùng người:

```text
speaker i
 ├─ K face
 └─ M voice
```

Projection:

\[
z^f_{ik}=h_f(f_{ik})
\]

\[
z^v_{im}=h_v(v_{im})
\]

Prototype:

\[
p_i^f=\operatorname{norm}\left(
\frac1K\sum_k z^f_{ik}
\right)
\]

\[
p_i^v=\operatorname{norm}\left(
\frac1M\sum_m z^v_{im}
\right)
\]

Train:

\[
L=
InfoNCE(p_i^f,p_i^v)
\]

hoặc multi-positive sigmoid.

---

## 6.2. Vì sao không giống AAM

AAM:

```text
class i → learned parameter center_i
```

SetProto:

```text
class i → prototype tính từ samples hiện tại
```

Identity mới ở test vẫn áp dụng được.

Đây là điểm cực kỳ quan trọng cho unseen identities.

---

## 6.3. Random set size

Không dùng cố định K.

Khuyến nghị:

```text
K_face  ∈ {2,4,8}
K_voice ∈ {2,4}
```

random mỗi iteration.

Mục tiêu:

> embedding hoạt động tốt khi số sample trong cluster thay đổi.

---

## 6.4. Sampling

Batch:

```text
N speaker × K_face × K_voice
```

Ví dụ:

```text
16 speaker
4 face / speaker
2 voice / speaker
```

Projection trước rồi mới average.

Không average raw VGG trước projection ở experiment đầu.

---

## 6.5. Stop rule

Nếu true-label validation:

```text
set-level train
```

không thắng row-level train ≥ 1 EER hoặc không giảm variance theo speaker, dừng.

---

# 7. EMB-06 — Set consistency regularization

## 7.1. Ý tưởng

Cùng một identity, lấy hai subset khác nhau:

```text
S1 = random 4 samples
S2 = random 4 samples
```

Prototype:

\[
p_i^{(1)},p_i^{(2)}
\]

Consistency:

\[
L_{cons}
=
1-\cos(p_i^{(1)},p_i^{(2)})
\]

Tổng:

\[
L=
L_{cross}
+
\lambda L_{cons}
\]

---

## 7.2. Mục tiêu

Giảm sensitivity với:

- frame cụ thể;
- utterance cụ thể;
- session noise;
- crop noise.

---

## 7.3. Lưu ý

Không để consistency quá mạnh.

Nếu λ lớn, model có thể collapse mọi sample cùng người và mất information hữu ích.

Sweep nhỏ:

```text
λ ∈ {0.05, 0.1, 0.2}
```

---

# 8. EMB-07 — Cross-language positive curriculum

Đây là hướng quan trọng nhất khi v1/v2 hoàn chỉnh.

---

## 8.1. Dữ liệu

v1:

```text
English + Urdu
```

v2:

```text
English + Hindi
```

---

## 8.2. Tạo positive cố tình cross-language

Ví dụ:

```text
face English  ↔ voice Urdu
face Urdu     ↔ voice English
```

v2:

```text
face English  ↔ voice Hindi
face Hindi    ↔ voice English
```

Không chỉ random mix.

---

## 8.3. Curriculum

### Stage 1

```text
same-language positives
```

để bridge ổn định.

### Stage 2

mix:

```text
50% same-language
50% cross-language
```

### Stage 3

có thể tăng:

```text
70% cross-language
```

nếu validation tốt.

---

## 8.4. Metric quan trọng

Validation bilingual:

```text
heard EER
unheard EER
language gap = unheard - heard
```

Một model tốt phải:

```text
unheard giảm
heard không giảm quá guardrail
```

---

# 9. EMB-08 — Domain-balanced training

Khi có:

```text
v4-English
v1-English
v1-Urdu
v2-English
v2-Hindi
```

không sample hoàn toàn theo số rows.

Nếu không, domain lớn sẽ thống trị gradient.

---

## 9.1. Balanced domain sampler

Mỗi batch:

```text
25% v4-En
20% v1-En
20% v1-Urdu
17.5% v2-En
17.5% v2-Hindi
```

hoặc balance theo source trước, language sau.

---

## 9.2. Lưu ý

Không balance quá cứng nếu một domain có ít identity.

Ưu tiên balance theo:

```text
speaker
```

hơn theo row count.

---

# 10. EMB-09 — GroupDRO

## 10.1. Ý tưởng

Tối ưu worst-domain:

\[
\min_\theta \max_d L_d
\]

Domain:

```text
v4-En
v1-En
v1-Urdu
v2-En
v2-Hindi
```

---

## 10.2. Simplified implementation

Maintain domain weights:

```python
q[d] *= torch.exp(eta * loss_d.detach())
q /= q.sum()
loss = sum(q[d] * loss_d for d in domains)
```

---

## 10.3. Khi nào dùng

Chỉ sau khi baseline domain-balanced sampler đã chạy.

Nếu balance đơn giản đủ tốt, GroupDRO có thể không cần.

---

# 11. EMB-10 — Fishr / gradient invariance

## 11.1. Khác CORAL

CORAL:

```text
align feature covariance
```

Fishr:

```text
align gradient variance across domains
```

Không đụng trực tiếp covariance test.

---

## 11.2. Khi nào dùng

Chỉ nếu:

```text
GroupDRO có signal
```

và gradient giữa domains khác nhau rõ.

EDA trước:

```text
cosine(grad_v4, grad_v1)
cosine(grad_v4, grad_v2)
```

Nếu gradient gần cùng hướng:

```text
Fishr likely unnecessary
```

---

# 12. EMB-11 — Teacher-guided relational distillation

EDA-002 cho biết:

- ArcFace mạnh cho face neighborhood;
- ReDimNet mạnh cho voice neighborhood;
- nhưng không dùng chúng làm bridge trực tiếp.

Giải pháp:

> giữ geometry hữu ích của teacher, nhưng student vẫn dùng VGG/BTC.

---

## 12.1. Face teacher

Teacher:

```text
ArcFace
```

Student:

```text
VGG → cross-modal projection
```

Teacher similarity:

\[
T_{ij}^f=
\cos(a_i,a_j)
\]

Student similarity:

\[
S_{ij}^f=
\cos(z_i^f,z_j^f)
\]

Loss:

\[
L_{rel}^f=
\|S^f-T^f\|
\]

Tương tự voice bằng ReDimNet.

---

## 12.2. Chỉ distill local neighborhood

Không ép toàn bộ matrix.

Ví dụ:

```text
top-k = 3 hoặc 5
```

Mask teacher pairs ngoài local neighborhood.

Lý do:

> teacher geometry toàn cục có thể quá identity-specialized.

---

## 12.3. Tổng loss

\[
L=
L_{cross}
+
\lambda_f L_{rel}^f
+
\lambda_v L_{rel}^v
\]

Sweep nhỏ:

```text
λ ∈ {0.02, 0.05, 0.1}
```

---

# 13. EMB-12 — Teacher-guided quality weighting

Đây là version đơn giản hơn relational KD.

---

## 13.1. Face quality từ ArcFace

Identity centroid:

\[
c_i=\frac1N\sum_j a_{ij}
\]

Sample quality:

\[
q_{ij}=\cos(a_{ij},c_i)
\]

Dùng q để weight VGG student features:

\[
p_i^f=
\frac{\sum_j q_{ij}h_f(f_{ij})}
{\sum_j q_{ij}}
\]

---

## 13.2. Voice quality từ ReDimNet

Teacher:

```text
ReDimNet
```

Bridge feature:

```text
BTC-192
```

Weight:

```text
cos(ReDim sample, ReDim speaker centroid)
```

---

## 13.3. Lưu ý

Teacher chỉ được dùng để:

```text
quality / grouping / neighborhood
```

không concatenate vào bridge embedding trong experiment đầu.

Nếu concatenate, ta lại quay về problem:

```text
recognition feature ≠ cross-modal feature
```

---

# 14. EMB-13 — Learned set attention

Sau teacher-weight baseline.

---

## 14.1. Scalar attention

\[
\alpha_j=
softmax(g(z_j))
\]

\[
p=\sum_j \alpha_j z_j
\]

Network:

```text
256 → 64 → 1
```

rất nhỏ.

---

## 14.2. Train loss

Không dùng face classification.

Train bằng:

```text
cross-modal loss
```

để attention học:

> sample nào hữu ích cho face↔voice bridge.

---

## 14.3. Baselines

So:

```text
mean pooling
teacher quality
learned scalar attention
```

Không nhảy thẳng tới Transformer.

---

# 15. EMB-14 — Component-wise attention

Scalar attention:

```text
một weight / sample
```

Component-wise:

```text
một weight / dimension / sample
```

\[
p_d=
\sum_j
\alpha_{jd} z_{jd}
\]

Chỉ chạy nếu scalar attention có gain.

---

# 16. EMB-15 — Alignment + Uniformity regularization

Contrastive embedding tốt cần:

```text
alignment:
positive gần nhau

uniformity:
toàn embedding không dồn thành hubs
```

---

## 16.1. Alignment

\[
L_{align}
=
E\|z_f-z_v\|^2
\]

Không dùng riêng vì tương đương cosine-positive khi L2-normalized.

---

## 16.2. Uniformity

\[
L_{unif}
=
\log E_{i\ne j}
e^{-2\|z_i-z_j\|^2}
\]

Khuyến nghị tính trên:

```text
identity prototypes
```

không phải instance rows.

---

## 16.3. Lý do

ERR-01 có hiện tượng:

```text
voice hub
```

một voice bị match với nhiều face.

Uniformity có thể giảm clustering quá mức.

---

# 17. EMB-16 — VICReg-style regularization

Không dùng VICReg thay toàn bộ contrastive loss.

Dùng như regularizer.

---

## 17.1. Variance

Mỗi dimension cần đủ variance:

\[
L_{var}
=
\frac1D\sum_d
\max(0,\gamma-\sigma_d)
\]

---

## 17.2. Covariance

Giảm redundancy:

\[
L_{cov}
=
\sum_{i\ne j} C_{ij}^2
\]

---

## 17.3. Tổng

\[
L=
L_{cross}
+
\lambda_v L_{var}
+
\lambda_c L_{cov}
\]

Thử:

```text
A: cross only
B: + variance
C: + variance + covariance
```

Không làm large sweep.

---

# 18. EMB-17 — Effective-rank regularization / monitoring

Trước khi regularize, đo:

\[
r_{eff}
=
\exp(H(p))
\]

với:

\[
p_i=\lambda_i/\sum_j\lambda_j
\]

λ là eigenvalues covariance embedding.

Nếu effective rank quá thấp:

```text
128-d embedding
r_eff ~ 5–10
```

thì embedding đang collapse vào vài hướng.

Nếu:

```text
model tốt hơn
→ effective rank tăng
```

mới cân nhắc explicit rank regularization.

---

# 19. EMB-18 — Hubness-aware diagnostics

ERR-01 cho thấy voice hub.

Đo:

\[
N_k(x)=
\#\{\text{query mà x nằm trong top-k}\}
\]

Metrics:

```text
max N10
mean N10
Gini(N10)
top-5% hub mass
mutual-kNN rate
```

So:

```text
003c
007
SET-02
new embedding
```

---

## 19.1. Stop rule

Không thêm hubness loss nếu:

```text
hubness metric không correlate với EER
```

---

# 20. EMB-19 — Multi-Similarity Loss

Không phải ưu tiên đầu.

Dùng khi multi-positive loss vẫn chưa đủ.

Ý tưởng:

- mining informative positives;
- mining informative negatives;
- weight theo similarity hardness.

Điểm mạnh:

```text
không coi mọi negative như nhau
```

Rủi ro:

- dễ overfit 70 speaker;
- hyperparameter sensitivity.

---

# 21. EMB-20 — Ranked List Loss

Khác contrastive loss:

```text
không bắt toàn bộ positive collapse vào một điểm
```

Giữ cùng class trong một hypersphere.

Có thể hợp với:

```text
same speaker
different language
different session
```

vì variation thật sự tồn tại.

Chỉ thử nếu set/prototype training cho thấy:

```text
collapse positive quá mạnh
```

---

# 22. EMB-21 — Circle Loss

Circle Loss adaptive weighting:

```text
easy pairs → weight thấp
hard pairs → weight cao
```

Có thể dùng như pairwise objective.

Không ưu tiên hơn:

```text
multi-positive InfoNCE
sigmoid
SetProto
```

---

# 23. EMB-22 — SWA

Log cho thấy:

- 40 epoch tốt;
- 100 / 200 epoch tệ;
- seed variance lớn.

SWA có thể giúp tìm flat minimum.

---

## 23.1. Implementation

Sau warmup:

```text
epoch 20 → 40
```

average checkpoint mỗi:

```text
2 epoch
```

PyTorch:

```python
from torch.optim.swa_utils import AveragedModel

swa_model = AveragedModel(model)

if epoch >= swa_start:
    swa_model.update_parameters(model)
```

MLP không có BatchNorm thì rất đơn giản.

---

# 24. EMB-23 — SWAD

SWAD:

```text
không average toàn bộ late checkpoints
```

mà chọn vùng validation loss ổn định.

Phù hợp domain generalization.

Chỉ chạy sau SWA.

---

# 25. EMB-24 — SAM

SAM tìm flat minima bằng adversarial weight perturbation.

Chi phí:

```text
~2× backward
```

T4×2 vẫn chạy được nhưng không phải ưu tiên.

Chỉ thử nếu:

```text
SWA/SWAD có signal
```

---

# 26. FACE-SEQ-01 — Face sequence temporal statistics

Không thay encoder.

Per-frame VGG:

\[
f_1,\ldots,f_T
\]

Tạo:

\[
[\mu(f), \sigma(f), \mu(|f_{t+1}-f_t|)]
\]

---

## 26.1. Ablation

```text
mean
mean + std
mean + std + delta
```

---

## 26.2. Shuffle control

```text
original temporal order
vs
random shuffle
```

Nếu giống nhau:

```text
order không có ích
```

→ chuyển sang set aggregation.

---

# 27. FACE-SEQ-02 — Dynamic Image

Chuỗi RGB:

```text
frame1
frame2
...
frameT
```

→ dynamic image.

Sau đó:

```text
dynamic image → VGGFace → motion feature
```

Fusion:

```text
static VGG
+
dynamic VGG
```

---

## 27.1. Baselines

```text
static only
dynamic only
static + dynamic
```

---

# 28. FACE-SEQ-03 — Landmark / mouth-motion embedding

Extract landmarks per frame.

Features:

```text
mouth width
mouth height
lip opening
jaw displacement
yaw
pitch
velocity
acceleration
```

Normalize:

```text
translation
scale
head pose
```

Sequence:

```text
T × D
```

→ small TCN / GRU → 32–64-d motion vector.

---

## 28.1. Mục tiêu

Kiểm xem:

```text
articulatory dynamics
```

có phải shared cue với voice không.

---

## 28.2. Stop rule

Nếu:

```text
positive vs same-gender negative
```

không separable trên true-label external validation, dừng.

---

# 29. FACE-SEQ-04 — NAN/QAN-style learned frame weighting

Input:

```text
VGG per-frame
```

Attention:

```text
256 → 64 → 1
```

Train bằng cross-modal loss.

Không train bằng face-ID classification.

---

# 30. FACE-SEQ-05 — Set Transformer

Chỉ chạy nếu:

```text
shuffle ≈ original
```

tức order không quan trọng nhưng interaction giữa frames có ích.

Config nhỏ:

```text
1 block
d=128
2–4 heads
T ≤ 16
```

---

# 31. FACE-SEQ-06 — REAN / GRU

Chỉ chạy nếu:

```text
original > shuffled
```

rõ ràng.

Config:

```text
input 256
hidden 128
1 layer
```

Không dùng large video transformer.

---

# 32. FACE-SEQ-07 — Voice-conditioned face attention

Đây là hướng cross-modal nhất.

Voice:

\[
q=W_v z_v
\]

Face frame:

\[
k_t=W_k f_t
\]

Attention:

\[
\alpha_t=
softmax(q^\top k_t/\sqrt d)
\]

Aggregate:

\[
z_f=
\sum_t \alpha_t W_f f_t
\]

Ý nghĩa:

> voice quyết định frame nào của khuôn mặt hữu ích cho matching.

Chỉ chạy khi multi-frame baseline đã cho gain.

---

# 33. SET-05 — Train cho đúng inference SET-02

SET-02 dùng:

```text
cluster face
cluster voice
block-average score
```

Train nên mô phỏng điều này.

---

## 33.1. Synthetic cluster batches

Trong train:

```text
true identity = cluster
```

Tạo centroid / block score ngay trong batch.

Loss:

```text
cluster-level cross-modal loss
```

thay vì row-level.

---

## 33.2. Mixed training

Tổng:

\[
L=
L_{row}
+
\lambda L_{cluster}
\]

Thử:

```text
λ = 0.25, 0.5
```

Không bỏ row loss ngay.

---

# 34. SET-06 — Language-dependent voice clustering

SET-04 cho thấy threshold voice:

```text
0.8 tốt English
```

nhưng Bangla có thể overmerge.

Do đó threshold clustering không nên universal.

Nên hiệu chỉnh bằng true-label:

```text
English threshold
Urdu threshold
Hindi threshold
```

sau đó suy ra cách chọn threshold cho unseen Bangla bằng statistics không nhãn.

Cẩn thận:

> không tune Bangla threshold bằng pseudo-label trực tiếp nếu điều đó làm tăng transductive complexity.

---

# 35. SET-07 — Clustering by role

Theo EDA-002:

```text
face clustering:
ArcFace

voice clustering:
ReDimNet / ECAPA depending language

bridge:
VGG + BTC
```

Đây phải là default design pattern:

```text
best recognizer ≠ best bridge
```

Không dùng một embedding cho mọi role.

---

# 36. EXT-03 — External bilingual validation

Validation thật:

```text
v1:
En heard
Urdu unheard

v2:
En heard
Hindi unheard
```

Split speaker-disjoint.

Cố định face pool.

---

## 36.1. A0 baseline

```text
v4 only
```

đánh trên held-out v1/v2.

---

## 36.2. Compare

```text
MIX
PT → FT
cross-language curriculum
SetProto
GroupDRO
```

---

# 37. EXT-04 — MIX vs PT→FT

Giữ cùng compute budget.

Ví dụ:

```text
MIX: 40 epoch
PT: 20
FT: 20
```

hoặc cùng total optimizer steps.

Không để PT→FT thắng chỉ vì nhiều update hơn.

---

# 38. EXT-05 — Source compatibility analysis

Không chỉ hỏi:

```text
external data có giúp không?
```

mà hỏi:

```text
nguồn nào giúp?
```

Báo cáo theo:

```text
v1 only
v2 only
v1+v2
v3 only
v1+v2+v3
```

Cùng training recipe.

---

# 39. EDA bắt buộc cho embedding generalization

## EDA-E1 — False negative ratio

Đã mô tả ở EMB-01.

---

## EDA-E2 — Prototype curve

Test:

```text
K = 1,2,4,8,16
```

cho face / voice riêng.

Plot:

```text
EER vs K
```

Nếu curve giảm mạnh:

```text
sample noise lớn
```

---

## EDA-E3 — Intra-speaker variance

\[
\sigma_i^2=
E_j\|z_{ij}-\bar z_i\|^2
\]

Report mean / p90.

Model tốt nên:

```text
giảm within-speaker variance
```

nhưng không collapse between-speaker.

---

## EDA-E4 — Inter-speaker margin

\[
m=
E_{i\ne j}\|\bar z_i-\bar z_j\|
\]

Quan sát:

```text
within ↓
between giữ hoặc ↑
```

---

## EDA-E5 — Effective rank

Theo section 18.

---

## EDA-E6 — Alignment / uniformity

Report:

```text
positive cosine
negative cosine
uniformity
```

---

## EDA-E7 — Hubness

Theo section 19.

---

## EDA-E8 — Cross-modal neighborhood consistency

Face prototype:

```text
top-k voice neighbors
```

Voice prototype:

```text
top-k face neighbors
```

Đo:

```text
mutual top-k rate
```

---

## EDA-E9 — Teacher/student neighborhood overlap

So:

```text
ArcFace neighborhood
vs
student face neighborhood
```

và:

```text
ReDim neighborhood
vs
student voice neighborhood
```

---

## EDA-E10 — Domain gradient cosine

Per domain:

\[
g_d=\nabla_\theta L_d
\]

Đo:

\[
\cos(g_i,g_j)
\]

Nếu âm:

```text
domains conflict
```

→ GroupDRO / Fishr đáng thử.

---

## EDA-E11 — CCA spectrum

Fit CCA speaker-disjoint.

Plot:

```text
canonical correlation vs dimension
```

Bootstrap identities.

Câu hỏi:

> shared cross-modal subspace thực sự có bao nhiêu chiều ổn định?

---

## EDA-E12 — Canonical direction stability

Bootstrap speaker identities.

Với mỗi bootstrap:

```text
fit CCA
```

So principal angles giữa shared subspaces.

Nếu top-4 directions ổn định nhưng direction 20+ không ổn định:

```text
bridge nên low-rank
```

---

# 40. Training protocol chuẩn cho EMB experiments

## 40.1. Split

Luôn speaker-disjoint.

Tối thiểu:

```text
3 split
```

Không chọn config từ một seed.

---

## 40.2. Selection

Ưu tiên:

1. true-label held-out v4;
2. true-label external bilingual;
3. SEL-01b chỉ sau đó.

Không dùng pseudo-label để train.

---

## 40.3. Same compute budget

Khi so architecture/loss:

```text
same epochs
same optimizer steps
same batch identities
same seeds
```

---

## 40.4. Same preprocessing

Không refit PCA/scaler trên validation identities.

Mọi:

```text
Scaler
PCA
CCA
normalization stats
```

fit train fold only.

---

# 41. Default model recipe đề xuất

## Face

```text
VGGFace fc7 4096
→ standardize
→ PCA 256
→ MLP 512 → 128
→ L2
```

## Voice

```text
BTC ECAPA-192
→ standardize
→ MLP 512 → 128
→ L2
```

## Regularization

```text
dropout 0.3
```

Không dropout 0.9.

---

# 42. Dataloader cho identity-set

Pseudo-code:

```python
class IdentitySetDataset:
    def __getitem__(self, spk):
        faces = sample_k(face_by_spk[spk], k_face)
        voices = sample_k(voice_by_spk[spk], k_voice)

        return {
            "speaker": spk,
            "faces": faces,
            "voices": voices,
        }
```

Collate:

```text
B identities
×
K samples
```

Không flatten identity metadata mất dấu.

---

# 43. Loss recipe đề xuất đầu tiên

\[
L=
L_{proto-sigmoid}
+
0.1L_{set-consistency}
\]

Không thêm teacher / VICReg ngay.

Sau khi baseline ổn:

\[
L=
L_{proto-sigmoid}
+
0.1L_{cons}
+
0.05L_{teacher}
\]

Mỗi experiment chỉ thêm **một cơ chế**.

---

# 44. Expected experiment tree

```text
EMB-01 audit false negatives
        |
        +-- false neg low --> giữ loss
        |
        +-- false neg high
                |
                v
        EMB-02 multi-positive
                |
                v
        EMB-03 sigmoid
                |
                v
        EMB-04 identity resampling
                |
                v
        EMB-05 SetProto
                |
          +-----+------+
          |            |
          v            v
       EMB-06       EMB-07
    consistency    cross-lang
          |            |
          +-----+------+
                |
                v
             SWA
                |
                v
        teacher / uniformity
```

---

# 45. Priority order theo information gain

## P0 — chạy trước

1. **EMB-01 false-negative audit**
2. **EMB-02 multi-positive InfoNCE**
3. **EMB-03 pairwise sigmoid**
4. **EMB-04 identity-positive resampling**
5. **EMB-05 stochastic SetProto**

Lý do:

- không cần extract mới;
- rẻ;
- đánh đúng mismatch row-level vs identity-level;
- có thể thay đổi bản chất generalization.

---

## P1 — khi v1/v2 hoàn chỉnh

6. **EMB-07 cross-language positives**
7. **EXT bilingual validation**
8. **domain-balanced sampling**
9. **GroupDRO**

---

## P2 — embedding geometry

10. **teacher quality weighting**
11. **relational KD**
12. **SWA/SWAD**
13. **uniformity**
14. **VICReg variance**

---

## P3 — face sequence

15. temporal statistics
16. dynamic image
17. learned frame attention
18. mouth/landmark motion
19. Set Transformer / GRU
20. voice-conditioned attention

---

# 46. Stop rules

## Loss

Dừng nếu:

```text
gain < 0.5 EER trên true-label validation
```

và không có improvement rõ ở:

```text
within-speaker variance
hubness
cross-language gap
```

---

## Set training

Dừng nếu:

```text
row-level == set-level
```

trên ≥3 split.

---

## Teacher KD

Dừng nếu:

```text
student cross-modal tốt hơn không đáng kể
```

hoặc:

```text
teacher overlap tăng
nhưng EER xấu đi
```

→ student đang copy recognition geometry không hữu ích.

---

## Domain method

Dừng nếu:

```text
unheard tốt
heard giảm mạnh
```

→ trade-off, không phải generalization.

---

# 47. Các lỗi dễ mắc

## 47.1. Dùng pseudo-label để train

Không làm.

SEL-01b chỉ:

```text
selection / analysis
```

không:

```text
training target
```

---

## 47.2. Refit preprocessing trên validation

Không fit:

```text
PCA
Scaler
CCA
```

trên held-out identities.

---

## 47.3. Winner's curse

Không chạy 100 config rồi chọn best dev.

Giới hạn:

```text
≤5 candidates / hypothesis
```

---

## 47.4. Nhầm recognition quality với cross-modal quality

Luôn báo cả:

```text
unimodal EER
cross-modal EER
```

---

## 47.5. Quên identity-level bootstrap

Bootstrap:

```text
speaker / pseudo-identity
```

không bootstrap row độc lập.

---

## 47.6. Overfit gender

Cell `no_gender` dễ hưởng shortcut gender.

Bất kỳ model nào chỉ thắng `no_gender` nhưng không `gender` cần kiểm:

```text
gender probe
same-gender negatives
```

---

# 48. References kỹ thuật

## Contrastive / metric learning

- Supervised Contrastive Learning — Khosla et al., NeurIPS 2020  
  https://arxiv.org/abs/2004.11362

- Sigmoid Loss for Language Image Pre-Training (SigLIP) — Zhai et al., ICCV 2023  
  https://arxiv.org/abs/2303.15343

- Prototypical Networks for Few-shot Learning — Snell et al., NeurIPS 2017  
  https://arxiv.org/abs/1703.05175

- Multi-Similarity Loss — Wang et al., CVPR 2019  
  https://arxiv.org/abs/1904.06627

- Ranked List Loss — Wang et al., CVPR 2019  
  https://arxiv.org/abs/1903.03238

- Circle Loss — Sun et al., CVPR 2020  
  https://arxiv.org/abs/2002.10857

---

## Embedding geometry / regularization

- Understanding Contrastive Representation Learning through Alignment and Uniformity — Wang & Isola, ICML 2020  
  https://arxiv.org/abs/2005.10242

- VICReg — Bardes et al., ICLR 2022  
  https://arxiv.org/abs/2105.04906

- Similarity-Preserving Knowledge Distillation — Tung & Mori, ICCV 2019  
  https://arxiv.org/abs/1907.09682

- Relational Knowledge Distillation — Park et al., CVPR 2019  
  https://arxiv.org/abs/1904.05068

---

## Domain generalization

- GroupDRO / Distributionally Robust Neural Networks — Sagawa et al.  
  https://arxiv.org/abs/1911.08731

- SWAD — Cha et al., NeurIPS 2021  
  https://arxiv.org/abs/2102.08604

- Fishr — Rame et al., ICML 2022  
  https://arxiv.org/abs/2109.02934

- SAM — Foret et al., ICLR 2021  
  https://arxiv.org/abs/2010.01412

---

## Face set / sequence aggregation

- Neural Aggregation Network (NAN) — Yang et al., CVPR 2017  
  https://arxiv.org/abs/1603.05474

- Quality Aware Network (QAN) — Liu et al., CVPR 2017  
  https://arxiv.org/abs/1611.02197

- Feature Aggregation Network / component-wise aggregation  
  https://arxiv.org/abs/1902.07327

- Recurrent Embedding Aggregation Network (REAN)  
  https://arxiv.org/abs/1904.12019

- Set Transformer — Lee et al., ICML 2019  
  https://arxiv.org/abs/1810.00825

---

## Face–voice / dynamic face

- Seeing Voices and Hearing Faces — Nagrani et al., CVPR 2018  
  https://arxiv.org/abs/1804.00326

---

## Hubness

- Cross Modal Retrieval with Querybank Normalisation — Bogolin et al., CVPR 2022  
  https://arxiv.org/abs/2112.12777

---

# 49. Đề xuất file / notebook tiếp theo

```text
Experiment/
├── EMB-01_false_negative_audit/
├── EMB-02_multipos/
├── EMB-03_sigmoid/
├── EMB-04_id_resample/
├── EMB-05_setproto/
├── EMB-06_set_consistency/
├── EMB-07_crosslang/
├── EMB-08_teacher_quality/
├── EMB-09_rkd/
├── EMB-10_flat_minima/
├── FACE-SEQ-01_stats/
├── FACE-SEQ-02_dynamic/
├── FACE-SEQ-03_landmark/
└── FACE-SEQ-04_attention/
```

Kaggle:

```text
kaggle/
├── FLAG_09_embedding.ipynb
├── FLAG_10_setproto.ipynb
├── FLAG_11_crosslang.ipynb
└── FLAG_12_face_sequence.ipynb
```

---

# 50. Một baseline duy nhất nên build ngay

Nếu chỉ build một notebook mới:

## `FLAG_09_embedding.ipynb`

Nó nên chạy:

```text
A. current InfoNCE
B. multi-positive InfoNCE
C. pairwise sigmoid
D. identity-positive resampling
E. stochastic SetProto
F. SetProto + consistency
```

Cùng:

```text
3 speaker splits
same optimizer
same compute budget
same architecture
```

Output:

```text
internal ng/g
prototype EER K=1/2/4/8
within-speaker variance
effective rank
hubness
EXT bilingual heard/unheard nếu feature sẵn
```

**Không nộp CodaBench từ notebook này.**

Chỉ khi một hướng thắng rõ trên true-label validation mới đưa sang pipeline SET-02 / selector.

---

# 51. Kết luận

Hướng có cơ sở mạnh nhất hiện tại không phải là:

```text
model lớn hơn
encoder mới hơn
loss phức tạp hơn
```

mà là:

```text
1. sửa positive construction
2. học ở cấp identity-set
3. randomize các positive trong cùng người
4. học cross-language trực tiếp
5. giảm sample-level noise
6. dùng recognizer mạnh làm teacher / clustering
7. regularize geometry để tránh hubness / collapse
8. chọn flat minima để generalize tốt hơn
9. chỉ dùng sequence model khi EDA chứng minh temporal order có signal
```

Đặc biệt, SET-02 cho thấy inference đã chuyển sang identity-level.  
Do đó bước nghiên cứu tự nhiên tiếp theo là:

> **làm cho training objective cũng trở thành identity-level.**

Đây là trục nghiên cứu chính nên ưu tiên trong vòng tiếp theo.
