# FLAG 2027 — FourFusion

> **Lịch sử (27/09). Danh sách hướng rộng; phần lớn đã được thử và đóng, xem bảng "Đã làm" trong OPEN_DIRECTIONS §4. Đặc biệt: SetProto, SWA, chỉnh ImageBind (adapter / CCA / LoRA), cầu nối lớn hơn đều đã đóng (70, 74, 75).** Trạng thái hiện hành (29/09, nhật ký tới mục 75): cấu hình Evaluation là **SET-13 (CB 21.01) qua `BEST/v6_eval`**; xem [DECISIONS.md](DECISIONS.md) và [OPEN_DIRECTIONS.md](OPEN_DIRECTIONS.md).
## PLAN_V6_FULL — Toàn bộ hướng có khả năng tăng điểm

**Ngày:** 2026-09-27  
**Mục tiêu:** gom tất cả hướng còn có cơ sở kỹ thuật và có khả năng cải thiện điểm FLAG 2027, dựa trên experiment log hiện tại và các kiểm tra đã chạy. Kế hoạch này không giới hạn ở một vài thí nghiệm “an toàn”; mọi hướng còn có khả năng tạo gain đều được ghi vào, nhưng được xếp theo **expected gain / information gain / chi phí / rủi ro overfit** để triển khai có thứ tự.

---

# 0. Trạng thái hiện tại và kết luận nền

## 0.1. Best hiện tại

Các mốc quan trọng:

- FUSE-03: **28.69**
- SET-02: **21.68**
  - ng/En: **12.50**
  - ng/Bn: **21.34**
  - g/En: **28.31**
  - g/Bn: **24.59**

SET-02 là bước nhảy lớn nhất đến hiện tại.

---

## 0.2. Điều đã được experiment log chứng minh

### A. Nhận dạng unimodal mạnh hơn không đồng nghĩa cross-modal tốt hơn

- ArcFace mạnh hơn VGG rõ rệt ở face↔face nhưng cross-modal tệ hơn.
- ReDimNet mạnh hơn BTC ECAPA ở speaker recognition nhưng không cải thiện bridge face↔voice.
- VGG/BTC giữ các cue mềm dùng được cho bridge tốt hơn các embedding nhận dạng quá chuyên biệt.

### B. Gộp nhiều quan sát cùng người cực kỳ quan trọng

ORA-01 / SET-01 / SET-02 cho thấy:

- trung bình nhiều ảnh cùng identity giúp;
- trung bình nhiều voice cùng identity giúp;
- dùng cả hai phía có thể giảm rất mạnh EER;
- clustering bằng recognizer mạnh lấy được gần hết gain oracle.

### C. Embedding nên được chọn theo vai trò

- **Clustering face:** ArcFace.
- **Clustering voice:** ReDimNet / ECAPA tùy ngôn ngữ.
- **Bridge:** VGGFace + BTC ECAPA.
- Không ép một embedding làm tất cả nhiệm vụ.

### D. External identities có thể dạy bridge tốt hơn

EXT-02 trên v3 cho thấy:

- v4-only không generalize tốt sang identity/source khác;
- thêm external identities có thể cải thiện cả heard và unseen language trong cùng source;
- MIX đang có tín hiệu mạnh hơn PT→FT trên v3 validation;
- external source phù hợp với target domain là biến số lớn.

### E. Current loss đã multi-positive

Pipeline hiện tại đã dùng mọi pair cùng identity trong batch làm positive.

Vì vậy trọng tâm không phải “sửa diagonal InfoNCE”, mà là:

- sampler theo identity;
- prototype/set-level objective;
- domain/language balancing;
- cross-language supervision;
- geometry/generalization regularization.

### F. Phần "gộp" gần bão hoà trên English; lỗi còn lại chủ yếu ở mức cả người

- SET-01: cụm tự ước lượng chỉ kém cụm theo danh tính thật **~0.9 EER** (gender 23.68 vs 22.75, no_gender 14.80 vs 13.87).
- Cụm mặt ArcFace gần sạch (ARI 0.98–0.995) → phía mặt gần như không còn dư địa gộp.
- SET-03: ngưỡng giọng **lỏng hơn** (cụm lẫn người nhiều hơn, 38% mẫu ở tv 0.8) lại **tốt hơn** → số mẫu trong cụm quan trọng hơn độ sạch.
- Log (45)/(46): g/En trên dev không cải thiện do cầu nối ánh xạ sai ở mức cả người (σ_p²), không phải do nhiễu từng mẫu.
- Dư địa gộp còn lại chủ yếu ở **cụm giọng Bangla** (ECAPA tách ~30 người thành 39–41 cụm) → SET-05.

---

## 0.3. Đo trần trước khi đầu tư (bắt buộc, CPU vài phút)

Trên v4 giữ ra 30 người (nhãn thật, 5 split) và v1/v2 Urdu/Hindi giữ ra:

| Phép đo | So sánh | Trần của |
|---|---|---|
| CEIL-1 | cụm ước lượng vs cụm theo danh tính thật (cùng cầu nối, cùng b) | A1 + A2 (gộp / ngưỡng) |
| CEIL-2 | cụm thật + Sinkhorn vs cụm thật + điểm thô | A3 (gán toàn cục), tách khỏi lỗi gom cụm |
| CEIL-3 | cụm thật + cầu nối hiện tại, EER mức người | phần còn lại chỉ training/dữ liệu (Track B/C/D) giải được |

Luật dùng:

- Trần CEIL-1 < 0.5 EER ở một ngôn ngữ → **không** làm A1/A2 cho ngôn ngữ đó.
- Trần CEIL-2 ≤ 0.4 (vùng hòa) → đóng A3.
- Báo cả ba trần theo từng protocol (ng / g) và ngôn ngữ (En / Urdu / Hindi).

---

# 1. TRACK A — Cải thiện SET inference

Track rẻ nhất (không train, CPU). Nhưng theo §0.2.F, phần **gộp** (A1/A2) có trần thấp trên English; phần **gán toàn cục** (A3) là hướng test-time duy nhất nhắm vào lỗi mức người. Làm theo kết quả §0.3.

---

# A1. Robust cluster aggregation

## Mục tiêu

SET-02 hiện dùng trung bình trong cluster.

Vấn đề:

- cluster có thể lẫn người;
- cluster voice ở Bangla có thể overmerge;
- một vài sample outlier có thể kéo centroid sai;
- SET-03 cho thấy cluster vẫn hữu ích dù chứa sample lẫn identity.

Do đó cần robust estimator thay mean thường.

> ⚠️ **Trần thấp trên English** (§0.2.F: ≤ ~0.9 EER tổng, phía mặt ≈ 0). Trim / medoid / top-k còn **giảm số mẫu** trong cụm, trong khi SET-03 cho thấy số mẫu quan trọng hơn độ sạch → có thể hại. Chỉ chạy khi CEIL-1 ≥ 0.5, và ưu tiên phía giọng Bangla/Urdu/Hindi.
>
> **Gộp trong không gian embedding cầu nối** (sau projection, như SET-02: `agg(u, cluster)` trong [../Experiment/SET-05/part_a.py](../Experiment/SET-05/part_a.py)), **không** gộp trên VGG fc7 / BTC thô. Như vậy chỉ trọng số thay đổi, mọi thứ khác giữ nguyên. Trong các công thức dưới, \(f_i^{VGG}, v_i^{BTC}\) hiểu là embedding cầu nối của mẫu i.
>
> Tập biến thể tối thiểu (≤ 5 ứng viên, §47.3 cũ): `mean` · `trim 20%` · `teacher-weight softmax β=10` · `leave-one-out weight` · `geometric median`. Medoid và hybrid chỉ khi một trong số này thắng.

---

## A1.1. Arithmetic mean — baseline

\[
c = \frac{1}{N}\sum_i x_i
\]

Giữ làm baseline.

---

## A1.2. Teacher-weighted mean

### Face

Teacher:

```text
ArcFace
```

Bridge feature:

```text
VGGFace fc7
```

Với sample i trong cluster:

\[
q_i = \cos(a_i, \bar a)
\]

trong đó \(a_i\) là ArcFace embedding.

Weight:

\[
w_i = \exp(\beta q_i)
\]

hoặc đơn giản:

\[
w_i = \max(q_i,0)
\]

Bridge centroid:

\[
c_{VGG}
=
\frac{\sum_i w_i f_i^{VGG}}
{\sum_i w_i}
\]

### Voice

Teacher:

```text
ReDimNet hoặc ECAPA recognizer
```

Bridge:

```text
BTC ECAPA-192
```

Tương tự:

\[
c_{BTC}
=
\frac{\sum_i w_i v_i^{BTC}}
{\sum_iw_i}
\]

### Biến thể nên chạy

```text
mean
linear-weight
softmax-weight β=5
softmax-weight β=10
```

Không cần fit β trên dev; dùng grid nhỏ cố định.

---

## A1.3. Leave-one-out quality

Để tránh centroid bị outlier tự kéo:

\[
c_{-i}
=
\frac{1}{N-1}
\sum_{j\ne i} t_j
\]

\[
q_i=\cos(t_i,c_{-i})
\]

Cách này đáng ưu tiên hơn quality dùng centroid đầy đủ.

---

## A1.4. Trimmed mean

Tính quality \(q_i\), bỏ sample thấp nhất:

```text
trim 10%
trim 20%
trim 30%
```

Sau đó mean bridge feature còn lại.

Đây là cách rất rẻ và có thể chống mixed cluster tốt.

---

## A1.5. Top-central-k aggregation

Nếu cluster lớn:

```text
giữ top 50%
hoặc top min(8,N)
```

theo teacher-centroid similarity.

Useful khi cluster có tail nhiễu.

---

## A1.6. Geometric median

Thay mean bằng:

\[
c = \arg\min_z \sum_i \|x_i-z\|
\]

Ưu điểm:

- robust với outlier;
- không cần training;
- phù hợp cluster lẫn nhẹ.

Có thể dùng Weiszfeld algorithm.

---

## A1.7. Medoid

Chọn sample thật:

\[
i^* = \arg\max_i \sum_j \cos(x_i,x_j)
\]

Sau đó dùng bridge feature của sample medoid.

Hữu ích nếu mean tạo một vector “không thật” trong vùng embedding cong.

---

## A1.8. Hybrid medoid + mean

```text
lọc bằng teacher medoid
→ giữ sample gần medoid
→ mean bridge features
```

Đây có thể là robust default tốt.

---

## Evaluation

So trên true-label held-out:

```text
mean
weighted mean
trimmed mean
geometric median
medoid
hybrid
```

Báo:

- row-level EER;
- cluster-level EER;
- g / ng riêng;
- English / Urdu / Hindi / Bangla riêng;
- mixed-cluster robustness.

---

# A2. Better clustering threshold by language/domain

Voice clustering threshold không thể dùng universal.

Cần hiệu chỉnh trên true-label external:

```text
English
Urdu
Hindi
```

Rồi suy cho Bangla. Cách duy nhất có bằng chứng hiện nay là **chuyển ngưỡng giữa hai ngôn ngữ không-English**: hiệu chỉnh trên Urdu → đo trên Hindi và ngược lại ([../Experiment/SET-05/part_b.py](../Experiment/SET-05/part_b.py), B1-ii). Nếu ngưỡng chuyển giữa Urdu↔Hindi giữ được gần trần oracle của ngôn ngữ đích thì dùng ngưỡng không-English đó cho Bangla. "Suy từ statistics của score distribution" chưa có phương pháp cụ thể → không làm cho đến khi có đề xuất kiểm được.

---

## A2.1. Threshold calibration

Cho threshold grid:

```text
0.60
0.65
0.70
0.75
0.80
0.85
```

Đo:

```text
ARI
#clusters
overmerge rate
oversplit rate
SET EER
```

Quan trọng nhất là **SET EER**, không phải ARI đơn thuần.

---

> A2.2–A2.4 (ngưỡng thích nghi, mutual-kNN, HDBSCAN) thêm nút tinh chỉnh mà SET-05 chưa cho thấy nhu cầu → **mở lại khi có tín hiệu**: chỉ khi A2.1 + ReDimNet vẫn over/under-merge rõ ở Urdu/Hindi.

## A2.2. Adaptive threshold theo density

Thay fixed threshold:

\[
t_i = \mu_i + \alpha \sigma_i
\]

hoặc local-neighborhood quantile.

Mục tiêu:

- domain voice “compact” hơn → threshold cao;
- domain khó / language shift → threshold thấp hơn.

---

## A2.3. Mutual-kNN clustering

Chỉ nối edge nếu:

```text
i ∈ kNN(j)
và
j ∈ kNN(i)
```

Giảm overmerge do hub.

Test k:

```text
3, 5, 8
```

---

## A2.4. HDBSCAN / agglomerative with distance threshold

Dùng teacher recognizer embeddings.

So:

```text
threshold graph
agglomerative
HDBSCAN
```

Chọn bằng true-label validation ngoài.

---

# A3. Cluster assignment / one-to-one constraints

Sau khi có face clusters và voice clusters, hiện score matrix vẫn được xử lý tương đối local.

Có thể thêm global assignment.

> **Ưu tiên cao nhất trong Track A**: ràng buộc "mỗi người một giọng trong file" là cách test-time duy nhất có thể sửa lỗi **ánh xạ mức người** (§0.2.F).
>
> **Tiền lệ cần nhớ:** CSLS (ERR-01) và GRAPH-01 cũng chuẩn hoá hàng/cột trên ma trận score và **mất tác dụng khi vào fusion**. Khác biệt ở đây: làm trên ma trận **cụm × cụm** và có ràng buộc khối lượng. Đo CEIL-2 trước.
>
> **Chỉ dùng dạng mềm** (A3.3 + A3.4, sửa score bằng rank). Hungarian cứng biến điểm thành nhị phân → phá thứ hạng mà EER cần; chỉ giữ làm chẩn đoán (tỉ lệ cặp gán đúng).
>
> Đây là giả định transductive mạnh nhất tới giờ → khai báo trong system description; luôn giữ bản không-Sinkhorn làm dự phòng.

---

## A3.1. Hungarian matching

Score matrix:

\[
S \in \mathbb R^{N_f \times N_v}
\]

Nếu số cluster gần bằng nhau, solve:

\[
\min_{\pi}
\sum_i S_{i,\pi(i)}
\]

với lower = same.

Dùng như baseline global assignment.

---

## A3.2. Balanced Sinkhorn

Convert similarity:

\[
K_{ij} = \exp(-S_{ij}/\tau)
\]

Sinkhorn normalize rows/cols tạo doubly stochastic transport matrix.

Có thể dùng transport probability làm correction cho score.

---

## A3.3. Unbalanced Sinkhorn

Cực kỳ quan trọng khi:

```text
N_face_clusters != N_voice_clusters
```

Không ép mass bằng nhau tuyệt đối.

Objective:

\[
\min_P
\langle P,C\rangle
+
\epsilon H(P)
+
\tau_f KL(P1 || a)
+
\tau_v KL(P^T1 || b)
\]

---

## A3.4. Dustbin / unmatched cluster

Thêm một row/column “no-match”.

Cho phép:

```text
face cluster không có voice counterpart
voice cluster không có face counterpart
```

Không bắt buộc match sai.

---

## A3.5. Cluster-size-aware mass

Mass:

\[
a_i \propto |C_i|
\]

hoặc quality-weighted:

\[
a_i \propto |C_i|\bar q_i
\]

So với uniform mass.

---

## A3.6. Score correction từ transport

Các cách:

```text
raw score
raw - λ log P
raw / P
rank fusion(raw, transport rank)
```

Ưu tiên rank-based để tránh calibration mismatch.

---

# A4. Hubness-aware assignment

ERR-01 cho thấy voice hub.

Có thể dùng:

- mutual nearest neighbor;
- inverse-degree weighting;
- querybank-like correction;
- cluster-level CSLS;
- Sinkhorn naturally discouraging many-to-one hubs.

Đo:

```text
N10
Gini degree
top-5% hub mass
```

Nếu assignment giảm hubness và EER giảm cùng chiều → giữ.

---

# A5. Cluster confidence

Tạo confidence:

```text
cluster size
mean intra-cluster cosine
teacher ARI proxy
distance to nearest competing cluster
```

Dùng để:

- quyết định trim ratio;
- chọn strict/loose threshold;
- điều chỉnh Sinkhorn mass;
- fallback về row score nếu cluster confidence thấp.

> ⚠️ Dễ quay lại thành "quality router nhiều tham số" đã bị đóng (PLAN_V4: ≤ 2 tham số fit trên dev). Giới hạn: **một** biến confidence, **một** luật khai báo trước, hiệu chỉnh trên nhãn thật (v4 giữ ra / Urdu / Hindi), không trên dev. Chỉ làm sau khi A1/A3 đã có gain.

---

# 2. TRACK B — External-data bridge learning

Đây là track có khả năng tạo gain lớn nhất ở representation level.

---

# B1. Hoàn tất v1 / v2 extraction

Nguồn:

- v1: English + Urdu
- v2: English + Hindi

Metadata bắt buộc:

```text
source
speaker
language
gender nếu có thật
video
duration
```

Identity trùng tên phải merge.

Kiểm tra duplicate với v4 train/dev bằng recognizer teacher, **và giữa v1 với v2** (người nổi tiếng có thể xuất hiện ở cả hai; trùng giữa hai nguồn sẽ làm rò rỉ ở protocol B2.3/B2.4).

---

# B2. Validation suite thật

Không dùng một validation duy nhất.

Cần 4 protocol:

---

## B2.1. Within-source En→Urdu

Train external speaker identities với English.

Held-out identities:

```text
face pool cố định
voice English = heard
voice Urdu = unseen language
```

Đo:

```text
EER heard
EER unseen
language gap
```

---

## B2.2. Within-source En→Hindi

Tương tự v2.

---

## B2.3. Cross-non-English Urdu→Hindi

Train:

```text
v4 + v1 Urdu
```

Validation:

```text
v2 Hindi held-out identities
```

Mục tiêu:

> model đã thấy một non-English language có transfer sang non-English language khác không?

---

## B2.4. Cross-non-English Hindi→Urdu

Chiều ngược.

---

# B3. MIX training

Baseline external chính.

Train:

```text
v4
+
v1
+
v2
```

Không để source lớn áp đảo.

---

## B3.1. Speaker-balanced sampler

Sample theo identity trước:

```text
source
→ speaker
→ rows
```

Không sample trực tiếp theo row.

Ví dụ mỗi batch:

```text
1/3 v4 identities
1/3 v1 identities
1/3 v2 identities
```

---

## B3.2. Language-balanced sampler

Trong mỗi source:

```text
50% English
50% Urdu/Hindi
```

hoặc balance theo available speakers.

---

## B3.3. Source-balanced + language-balanced

Đây nên là default MIX recipe.

---

# B4. PT→FT

So với MIX trong cùng compute budget.

Ví dụ:

```text
MIX: 40 epochs

PT→FT:
20 epoch external
20 epoch v4
```

hoặc cùng total optimizer steps.

Ablations:

```text
PT(v1) → FT(v4)
PT(v2) → FT(v4)
PT(v1+v2) → FT(v4)
```

---

# B5. Interleaved curriculum

Khác MIX và PT→FT.

> Ưu tiên thấp: là điểm giữa MIX và PT→FT, thêm lịch trọng số cần chọn. Chỉ làm nếu B3 và B4 chênh nhau rõ (> vùng hòa 0.4) và không bên nào thắng ở cả heard lẫn unseen.

Schedule:

```text
epoch 1–10:
external-heavy

epoch 11–30:
balanced

epoch 31–40:
v4-heavy
```

Ví dụ source weights:

```text
0.2/0.4/0.4
→
0.34/0.33/0.33
→
0.6/0.2/0.2
```

Mục tiêu:

- học shared bridge từ nhiều identities;
- cuối cùng adapt lại target v4.

---

# B6. Cross-language oversampling

Loss hiện tại đã multi-positive, nhưng sampling quyết định bao nhiêu cross-language positives xuất hiện.

Tăng xác suất batch chứa cùng identity ở hai ngôn ngữ.

Ví dụ mỗi external speaker:

```text
1 face En
1 face Urdu/Hindi
1 voice En
1 voice Urdu/Hindi
```

để multi-positive loss tạo trực tiếp:

```text
En-face ↔ Urdu-voice
Urdu-face ↔ En-voice
```

---

# B7. Domain-balanced GroupDRO

Sau baseline MIX.

Groups:

```text
v4-En
v1-En
v1-Urdu
v2-En
v2-Hindi
```

Update group weights:

\[
q_d \leftarrow q_d \exp(\eta L_d)
\]

\[
q \leftarrow q/\sum q
\]

Final:

\[
L=\sum_d q_d L_d
\]

Mục tiêu:

> không để model hy sinh domain khó để tối ưu average loss.

---

# B8. Fishr / gradient-invariance training

Nếu gradient giữa source/domain conflict mạnh.

Đo trước:

\[
\cos(g_i,g_j)
\]

Nếu v4 gradient đối nghịch v1/v2 rõ ràng, Fishr có thể giúp.

Implementation chỉ sau khi GroupDRO baseline có.

---

# B9. Source-aware adapters nhỏ

Không thay backbone.

Bridge:

```text
shared trunk
+
tiny source adapter
```

Train external source-specific adapter, nhưng inference target v4 dùng shared hoặc mixture.

Có thể dùng:

```text
Linear 128→128 bottleneck 16
```

Mục tiêu:

- shared part giữ cross-source factors;
- adapter hấp thụ source-specific nuisance.

> ⚠️ **Chưa xác định lúc test:** dev Bangla (và dữ liệu Evaluation phase) không thuộc nguồn nào đã train → không biết dùng adapter nào. Nếu dùng "shared" thì B9 chỉ là regularizer gián tiếp; "mixture" cần trọng số → thêm tham số. **Mở lại khi có tín hiệu**, và phải định nghĩa trước cách chọn ở inference.

---

# B10. Domain-conditioned normalization

Dùng source/language-specific normalization **chỉ trong training external**, không fit dev.

Ví dụ:

```text
LayerNorm / affine adapter per source
```

Shared projection sau đó.

Có thể giảm domain shift mà không làm CORAL kiểu test-time.

> ⚠️ Cùng vấn đề với B9: normalization của nguồn nào áp cho Bangla? Nếu phải fit thống kê trên dev thì quay lại CORAL/centering test-time (per-file mean centering đã có, bậc 2 đã thất bại ở EXP-004). **Mở lại khi có tín hiệu.**

---

# 3. TRACK C — SetProto / identity-level training

Đây là hướng representation trực tiếp nhất sau SET-02.

> Kỳ vọng **nhỏ nếu chỉ train trên v4**: 40–56 người train → loss prototype chỉ có ~40–56 điểm dữ liệu/epoch, và SetProto không thêm thông tin mới về người; trần là EER mức người của cầu nối (CEIL-3). Giá trị thật khi chạy trên **v4 + v1/v2** (Track B). Luôn báo EER mức người (L3), không chỉ EER từng mẫu.

---

# C1. P×K identity sampler

Mỗi batch:

```text
P identities
K_face samples / identity
K_voice samples / identity
```

Ví dụ:

```text
P=16
K_face=4
K_voice=2
```

Random K qua epoch:

```text
K_face ∈ {2,4,8}
K_voice ∈ {2,4}
```

---

# C2. Row loss + prototype loss

Không thay current multi-positive loss.

Tổng:

\[
L = L_{row} + \lambda L_{proto}
\]

Prototype:

\[
p_i^f =
norm\left(\frac1K\sum_k h_f(f_{ik})\right)
\]

\[
p_i^v =
norm\left(\frac1M\sum_m h_v(v_{im})\right)
\]

Prototype loss dùng current multi-positive cross-modal objective hoặc InfoNCE ở cấp identity.

Ablation:

```text
λ = 0
0.25
0.5
1.0
```

---

# C3. Teacher-weighted prototype training

Prototype không mean đều.

Face:

```text
ArcFace quality
→ weight VGG projected features
```

Voice:

```text
ReDim/ECAPA quality
→ weight BTC projected features
```

Train bridge trực tiếp với robust identity prototype.

---

# C4. Trimmed prototype training

Trong train, giả lập mixed/noisy clusters:

```text
teacher quality
→ bỏ bottom 10–20%
→ prototype
```

Giúp train objective gần SET inference hơn.

---

# C5. Cluster-noise augmentation

Cố tình inject một tỷ lệ sample sai identity vào training set prototype:

```text
0%
5%
10%
20%
```

Sau đó dùng robust aggregation.

Mục tiêu:

> bridge/prototype chịu được imperfect clustering giống SET-02.

Đây là augmentation có cơ chế phù hợp hơn dropdim/noise cũ.

---

# C6. Subset consistency

Cùng speaker:

```text
subset A
subset B
```

Prototype:

\[
p_A,p_B
\]

Loss:

\[
L_{cons}=1-\cos(p_A,p_B)
\]

Tổng:

\[
L=L_{row}+\lambda_pL_{proto}+\lambda_cL_{cons}
\]

Sweep nhỏ:

```text
λc = 0.05, 0.1
```

---

# C7. Prototype variance penalty

Đo within-speaker dispersion:

\[
V_i = E_j\|z_{ij}-p_i\|^2
\]

Regularize nhẹ:

\[
L_{var-id}
=
\frac1B\sum_i V_i
\]

Chỉ dùng weight nhỏ để tránh collapse.

---

# C8. Prototype margin

Giữ prototypes khác identity đủ xa:

\[
L_{margin}
=
\max(0,m-\|p_i-p_j\|)
\]

Có thể dùng same-gender hard negatives ở prototype level.

Khác với same-gender negative experiment cũ vì bây giờ đơn vị là identity prototype, ít sample noise hơn.

> **Mở lại khi có tín hiệu.** Hard negative cùng giới: EXP-003 B chỉ nhỉnh ở gender (int_g 32.7 vs 33.2) nhưng no_gender tệ hẳn (30.5 vs 24.4); NB-2 P4 tệ hơn cả ở int_g. "Ít nhiễu hơn ở prototype" là giả thuyết chưa kiểm; chỉ thử sau khi C2 thắng.

---

# 4. TRACK D — Low-rank/shared-subspace bridge

CCA đã chứng minh low-dimensional linear shared space rất mạnh.

Cần khai thác bài học này sâu hơn.

---

# D1. Bootstrap CCA spectrum

Mỗi speaker bootstrap:

1. split speaker;
2. fit preprocessing train-only;
3. fit CCA;
4. lưu canonical correlations;
5. lưu canonical basis.

Report:

```text
ρ1 ... ρ32
mean ± sd
```

---

# D2. Subspace stability

Tính principal angles giữa canonical subspaces của các bootstrap.

Nếu:

```text
top 8 stable
top 16 moderately stable
top 32 unstable
```

thì shared bridge có effective rank thấp.

---

# D3. Low-rank MLP bottleneck

Nếu D1/D2 xác nhận:

```text
shared rank ≈ k
```

thử embedding:

```text
8
16
32
64
```

thay vì 128.

Điều này có thể giảm overfit trên 56 identities.

> ⚠️ Đã kiểm một phần: EXP-003 sweep emb 32–128; EXP-006 (28 cfg kiến trúc, chênh ~0.5) chọn emb128. Chỉ đáng làm lại nếu D1/D2 chỉ ra rank ổn định **< 32** (ngoài vùng đã sweep), hoặc khi train trên v4 + v1/v2 (số người khác hẳn).

---

# D4. Factorized linear bridge

Thay full MLP bằng:

\[
W = UV^T
\]

rank r nhỏ.

Ví dụ:

```text
4096 → 256 PCA
→ U 256×r
→ shared r
```

Voice tương tự.

r:

```text
8,16,32
```

---

# D5. Residual CCA initialization

Initialize first shared layer từ CCA directions:

```text
CCA projection
+
small trainable residual
```

\[
z = W_{CCA}x + \alpha \Delta W x
\]

α nhỏ ban đầu.

Mục tiêu:

- bắt đầu từ solution đã generalize tốt;
- chỉ học correction nhỏ.

> Ưu tiên thấp: CCA đã đóng góp qua fusion (w = 0.25, EXP-007: bỏ CCA +1.3 EER). D5 chỉ đáng làm nếu thắng được chính fusion deep + CCA đó.

---

# D6. Orthogonal residual bridge

Tách:

```text
shared CCA subspace
+
residual learned subspace
```

và regularize residual orthogonal với shared.

Mục tiêu:

> không phá shared directions ổn định trong khi vẫn thêm capacity.

---

# 5. TRACK E — Generalization / optimization

> **Mở lại khi có tín hiệu (cả track).** Pipeline đã ensemble 10 model (EXP-007) và trung bình nhiều seed → phần lợi của SWA/EMA (giảm phương sai theo seed, minimum phẳng hơn) phần lớn đã có. Nếu thử: so với **ensemble cùng chi phí inference**, không so với một model đơn. SAM (2× backward) chỉ khi SWA có tín hiệu.

---

# E1. SWA

Average weights cuối training.

Dùng trên best SetProto/MIX model.

Ví dụ:

```text
epoch 25, 30, 35, 40
```

weight average.

Chi phí inference không tăng.

---

# E2. SWAD

Chọn vùng checkpoint ổn định theo validation speaker-disjoint rồi average.

Có thể phù hợp khi epoch dài làm overfit.

---

# E3. SAM

Chỉ trên best model.

Mục tiêu:

> tìm flatter minimum.

Chi phí khoảng 2× backward.

Có thể hữu ích nếu seed variance vẫn lớn.

---

# E4. EMA

Exponential moving average weights:

\[
\theta_{ema}
=
\beta\theta_{ema}
+
(1-\beta)\theta
\]

β:

```text
0.99
0.995
0.999
```

Rất rẻ.

---

# E5. Weight decay / norm control around best recipe

Không sweep lớn.

Chỉ:

```text
wd / 3
wd
3 × wd
```

trên best SetProto/external model.

Mục tiêu generalization, không architecture hunting.

---

# 6. TRACK F — Embedding geometry diagnostics + regularization

Các metric này được đo cho mọi model mới.

> F1–F5 là **cột báo cáo** (rẻ, giữ). F6–F8 (regularizer) → **mở lại khi có tín hiệu**: các cách xử lý hub trước đây (CSLS ở ERR-01, GRAPH-01) mất tác dụng ở mức fusion; hub ở mức cụm để A3 xử lý. Chỉ thêm regularizer khi F-metric **tương quan với EER** qua các model đã có.

---

# F1. Effective rank

Covariance eigenvalues:

\[
p_i=\lambda_i/\sum_j\lambda_j
\]

\[
r_{eff}=\exp(-\sum_i p_i\log p_i)
\]

Báo riêng:

```text
face
voice
shared prototype
```

---

# F2. Alignment

Positive similarity:

\[
A=E[\cos(z_f,z_v)|same]
\]

---

# F3. Uniformity

\[
U=
\log
E_{i\neq j}
e^{-2\|z_i-z_j\|^2}
\]

Đo trên identity prototypes.

---

# F4. Hubness

Metrics:

```text
max N10
Gini(N10)
top-5% hub mass
mutual-kNN rate
```

---

# F5. Intra/inter identity ratio

\[
R=
\frac{\text{within-speaker variance}}
{\text{between-speaker variance}}
\]

Model tốt kỳ vọng R giảm.

---

# F6. VICReg variance regularizer

Nếu effective rank thấp:

\[
L_{var}
=
\frac1D
\sum_d
\max(0,\gamma-\sigma_d)
\]

Thêm nhẹ vào best model.

---

# F7. Covariance decorrelation

Nếu dimensions redundant mạnh:

\[
L_{cov}
=
\sum_{i\ne j}C_{ij}^2
\]

Dùng weight nhỏ.

---

# F8. Prototype uniformity regularizer

Nếu hubness cao:

\[
L =
L_{main}
+
\lambda U(p)
\]

Prototype-level, không instance-level.

---

# 7. TRACK G — Recognition-teacher assisted bridge

Dùng recognizer mạnh để hỗ trợ bridge mà không thay bridge feature.

---

# G1. Quality teacher

Đã mô tả ở A1/C3.

Đây là hướng teacher ưu tiên nhất.

---

# G2. Local relational distillation

Teacher:

```text
ArcFace / ReDimNet
```

Student:

```text
VGG/BTC projected space
```

Không copy vector.

Chỉ preserve local neighbor ranking.

Teacher top-k:

```text
k=3
k=5
```

Loss:

\[
L_{rel}
=
\sum_{(i,j)\in N_k}
|
s^{student}_{ij}
-
s^{teacher}_{ij}
|
\]

Weight nhỏ:

```text
0.02
0.05
0.1
```

> **Mở lại khi có tín hiệu.** Rủi ro kéo cầu nối về hình học ArcFace/ReDimNet — đúng loại hình học đã thất bại ở vai trò cầu nối (ArcFace + bất kỳ: 33–37 ở mức người, EDA-002).

---

# G3. Triplet mining bằng teacher

> **Mở lại khi có tín hiệu.** Hard negative cùng giới không cho lợi ích tổng (EXP-003 B: int_g 32.7 vs 33.2 nhưng int_ng 30.5 vs 24.4; NB-2 P4 tệ hơn ở int_g).

Teacher chọn hard negatives:

```text
same gender
similar appearance
similar voice
```

Nhưng student vẫn train bằng VGG/BTC.

Đây là cách dùng recognition geometry cho sampling thay vì representation.

---

# G4. Teacher agreement sampling

Ưu tiên identities/sample pairs mà:

```text
ArcFace face cluster confidence cao
AND
ReDim voice cluster confidence cao
```

để xây clean prototype batches.

Sau đó curriculum dần thêm sample khó.

### G4-EDA — Kiểm nhãn sai trong v4 train (rẻ, làm trước)

Biến thể không cần train của G4: với mỗi speaker v4 train, tính cos ArcFace của từng mặt tới centroid leave-one-out của speaker đó (tương tự cho giọng bằng ReDimNet). Báo:

```text
tỉ lệ mẫu có cos < ngưỡng khác-người (hiệu chỉnh trên train)
số speaker bị ảnh hưởng
ví dụ ảnh (xem tận mắt)
```

Nếu có nhãn sai đáng kể (> 2–3% hàng), thử train lại bỏ các hàng đó (một biến, 5 split paired). Nếu ≈ 0 → đóng G4.

---

# 8. TRACK H — Alternative pair losses

Current loss đã multi-positive, nhưng vẫn có thể thử các objective thật sự khác sau khi SetProto baseline có.

> **Mở lại khi có tín hiệu (cả track).** Đổi loss trên dữ liệu này đã phẳng: AAM ≈ InfoNCE (30.66 vs 30.53, EXP-009), 28 cfg kiến trúc chênh ~0.5 (EXP-006). MS / RLL / Circle thêm hyperparameter trên 40–56 người. Nếu thử, chỉ H1 (sigmoid) và gộp vào cùng một run với C2.

---

# H1. Pairwise sigmoid loss

Mỗi pair:

\[
y_{ij}\in\{-1,+1\}
\]

\[
L=
\log(1+\exp(-y_{ij}(a s_{ij}+b)))
\]

Ablation nhỏ:

```text
current loss
sigmoid
```

Không sweep lớn.

---

# H2. Circle loss

Adaptive weighting hard pairs.

Chạy ở prototype level trước.

---

# H3. Multi-Similarity Loss

Mining/weighting informative pairs.

Dùng nếu hard-negative structure quan trọng.

---

# H4. Ranked List Loss

Giữ class trong hypersphere thay vì collapse.

Có thể hợp multilingual speaker variation.

---

# H5. Proxy-free supervised metric learning on prototypes

Không learned class centers.

Dùng only batch prototypes, tránh overfit 56 fixed identities.

---

# 9. TRACK I — Language/source invariance

---

# I1. Cross-language prototype alignment

External speaker có hai language.

Tạo:

\[
p_{i,En}
,\quad
p_{i,Urdu}
\]

Voice-side consistency:

\[
L_{lang}
=
1-\cos(p_{i,En}^v,p_{i,Urdu}^v)
\]

Tương tự Hindi.

Dùng nhỏ:

```text
λ=0.05,0.1
```

---

# I2. Shared identity prototype across languages

Face/voice/language observations cùng identity cùng kéo về prototype tính từ samples, không learned center.

---

# I3. Language-specific nuisance subspace removal learned on train

Train linear language classifier trên external voice embeddings.

Lấy language directions:

```text
W_lang
```

Project out một phần:

\[
z' = z - \alpha P_{lang}z
\]

α:

```text
0.25
0.5
```

Khác CORAL vì:

- train-only;
- remove explicit language direction;
- không dùng dev covariance.

> **Rẻ và kiểm được ngay, không cần train lại:** học hướng ngôn ngữ trên Urdu (vs English cùng người), chiếu bỏ trên Hindi và ngược lại, đo EER unseen trên B2.3/B2.4. Câu hỏi: hướng "English ↔ không-English" có **chung** giữa các ngôn ngữ không? Nếu có → đáng áp cho Bangla; nếu không → đóng I3. Lưu ý: chiếu bỏ quá mạnh có thể xoá cả identity (bài học EXP-004), nên chỉ α ≤ 0.5 và chỉ vài hướng.

---

# I4. Gradient reversal revisited only on external bilingual data

GRL cũ thất bại trên v4-only English vì không có real language diversity.

External bilingual data thay đổi điều kiện.

Train:

```text
identity bridge objective
+
language adversary
```

trên v1/v2.

Đây là một experiment mới về cơ chế, không phải lặp lại exact old setup.

> **Mở lại khi có tín hiệu.** DANN ngôn ngữ (NB-2 P5: 25.33, tệ hơn đối chứng ~0.8) và GRL giới tính (P4) đều tệ hơn; với 2–3 ngôn ngữ, adversary học được rất ít hướng. Chỉ thử nếu I3 cho thấy hướng ngôn ngữ chung giữa Urdu và Hindi (tức có thứ để adversary khử).

---

# I5. Domain-specific nuisance heads

Predict:

```text
source
language
```

từ embedding.

Dùng probe để đo invariance.

Nếu model tốt hơn nhưng source/language predictability giảm trong khi identity prototype EER giảm, đó là tín hiệu tích cực.

---

# 10. TRACK J — Fusion ở level mới

> Làm **cuối cùng**, sau khi các thành phần ổn định. Danh sách ứng viên cố định trước khi xem kết quả (như FUSE-01), ≤ 5 tổ hợp / cell, chọn cùng tổ hợp cho các cell cùng ngôn ngữ để tránh winner's curse. J3 chịu cùng giới hạn như A5.

---

# J1. Cluster-level score fusion

Thay fusion row-level:

```text
system A cluster score
system B cluster score
```

sau robust aggregation.

Có thể các hệ complementarity thay đổi sau SET.

---

# J2. Rank fusion after robust SET

Các hệ nên thử:

```text
007
010B
003c
r2_mix
new SetProto
external MIX
```

Rank fusion per cell / per protocol.

---

# J3. Recognition-confidence-weighted fusion

Không fit dev parameter.

Ví dụ:

```text
cluster confidence cao:
ưu tiên SET score

cluster confidence thấp:
mix row-level score nhiều hơn
```

Rule predeclared theo confidence bins từ true-label validation.

---

# J4. Assignment-aware fusion

Fusion raw bridge score với:

```text
Sinkhorn transport rank
cluster confidence
```

bằng rank averaging.

---

# 11. TRACK K — Calibration / score robustness

> EER tính trong từng file nên **bất biến với mọi biến đổi đơn điệu** của score một hệ. K3/K4 chỉ ảnh hưởng khi fusion, mà fusion đã dùng rank (K2) → K3/K4 **mở lại khi có tín hiệu** (chỉ nếu quay lại z-fusion có trọng số). K1, K2 giữ nguyên như hiện tại.

---

# K1. Per-file mean centering

Giữ vì đã có lợi.

---

# K2. Robust rank transform

Rank within file trước fusion.

SET/FUSE experiments đã cho thấy rank fusion ổn định hơn z-fusion ở nhiều cell.

---

# K3. Quantile normalization

Map score ranks sang Gaussian quantiles:

\[
z_i = \Phi^{-1}\left(\frac{rank_i-0.5}{N}\right)
\]

Có thể ổn định score scale giữa systems.

---

# K4. Median/MAD scaling

Thay mean/std:

\[
z=
\frac{s-\mathrm{median}(s)}
{1.4826\,MAD(s)}
\]

Rẻ, robust với score tails.

---

# 12. TRACK L — Better validation and experiment statistics

---

# L1. 5 paired speaker-disjoint splits

Mọi training experiment mới dùng ít nhất:

```text
5 splits
```

nếu chi phí cho phép.

Báo:

```text
Δ per split
mean Δ
median Δ
std Δ
wins/5
```

---

# L2. Identity bootstrap

Bootstrap theo identity, không theo rows.

> Lý do L1 là bắt buộc: với 30 người giữ ra, EER **giữa các split** dao động rất mạnh — [../Experiment/SET-05/part_a.csv](../Experiment/SET-05/part_a.csv): cùng cấu hình `ecapa192`, no_gender 17.1 / 24.6 / 34.2 ở 3 split, và riêng split 3 gộp cụm không giúp gì. Trung bình 3 split không phân giải được 0.5 EER; chỉ Δ **paired theo split** mới phân giải được.

---

# L3. Prototype-level metric

Báo song song:

```text
row EER
identity/prototype EER
```

Vì inference hiện là identity-level.

---

# L4. External bilingual generalization matrix

Bảng chuẩn:

| Train | Test heard | Test unseen language |
|---|---|---|
| v4 | v1 En | v1 Urdu |
| v4+v1 En | v1 En | v1 Urdu |
| v4+v1 all | v2 En | v2 Hindi |
| v4+v2 all | v1 En | v1 Urdu |
| v4+v1+v2 | held-out source/lang | held-out source/lang |

---

# 13. Priority roadmap

Thời gian: progress phase đến **10/11** (~6 tuần từ 27/9), evaluation 11–18/11 (15 lượt). Không thể chạy 34 mục → roadmap dưới đây chỉ giữ những gì có bằng chứng hoặc rẻ; phần còn lại mang trạng thái **mở lại khi có tín hiệu** (ghi ngay tại từng mục).

## P0 — làm ngay (theo thứ tự)

| # | Việc | Loại | Chi phí | Điều kiện đi tiếp |
|---|---|---|---|---|
| 1 | **Hoàn tất SET-05** (gom cụm giọng ReDimNet, ngưỡng Urdu/Hindi) | test-time | đang chạy | — |
| 2 | **§0.3 đo trần CEIL-1/2/3** | chẩn đoán | CPU vài phút | quyết định A1/A2/A3 |
| 3 | **A3 Sinkhorn mềm** (không cân bằng + dustbin, sửa score bằng rank) | test-time | CPU < 1 giờ | CEIL-2 > 0.4 |
| 4 | **A1 gộp chống nhiễu** (≤ 5 biến thể, phía giọng trước) | test-time | CPU < 1 giờ | CEIL-1 ≥ 0.5 ở ngôn ngữ đó |
| 5 | **B1 hoàn tất v1/v2** + kiểm trùng (v4, và v1↔v2) | dữ liệu | đang trích | — |
| 6 | **B3 MIX speaker-balanced + language-balanced**, đánh giá bằng bảng L4; **B4 PT→FT** cùng budget | training | Kaggle GPU | — |
| 7 | **C2 L_row + λ·L_proto**, trên v4 và trên v4 + v1/v2 | training | Kaggle GPU | báo EER mức người |

## P1 — EDA rẻ, chạy song song

| Việc | Câu hỏi |
|---|---|
| **G4-EDA** nhãn sai trong v4 train | có hàng train sai người không? |
| **I3** chiếu bỏ hướng ngôn ngữ, Urdu → Hindi | hướng "English ↔ không-English" có chung giữa các ngôn ngữ không? |
| **D1/D2** phổ CCA + độ ổn định | cầu nối có rank ổn định < 32 không? |
| **F1–F5** | cột báo cáo cho mọi model mới |

## P2 — khi P0 có tín hiệu

- B6 cross-language oversampling, B7 GroupDRO (sau MIX baseline)
- C3/C4/C5 prototype có trọng số / trim / cluster-noise (sau C2 thắng)
- A5 confidence (≤ 1 biến, sau A1/A3 có gain)
- J fusion ở mức cụm (cuối cùng)

## Mở lại khi có tín hiệu (không lên lịch)

A2.2–A2.4 · B5 · B8 · B9 · B10 · C6–C8 · D3–D6 · Track E · F6–F8 · G2 · G3 · Track H · I4 · K3/K4.
Lý do từng mục ghi ngay tại mục đó.

---

# 14. Experiment naming đề xuất

Tránh trùng SET-05/SET-06 hiện tại. Mã EMB-xx ở đây **thay thế** cách đánh số trong [FLAG2027_RESEARCH_DIRECTIONS_V6.md](FLAG2027_RESEARCH_DIRECTIONS_V6.md) (ở đó EMB-01 = audit false negative, đã có sẵn trong loss → bỏ). Chỉ tạo thư mục khi thí nghiệm thực sự bắt đầu.

```text
Experiment/
├── CEIL-01_headroom/            # §0.3, P0
│
├── SET-07_robust_pool/          # A1, P0 (nếu CEIL-1 ≥ 0.5)
├── SET-08_assignment/           # A3: Sinkhorn mềm + dustbin (Hungarian chỉ làm chẩn đoán), P0
├── SET-09_cluster_fusion/       # J, P2
│
├── EXT-03_bilingual_matrix/     # L4 + B2, P0
├── EXT-04_mix_balanced/         # B3, P0
├── EXT-05_ptft/                 # B4, P0
├── EXT-06_groupdro/             # B7, P2
│
├── EMB-01_setproto/             # C2, P0
├── EMB-02_weighted_proto/       # C3/C4, P2
├── EMB-03_cluster_noise/        # C5, P2
│
├── EDA-003_embedding_geometry/  # F1–F5 + D1/D2, P1
├── EDA-004_train_label_noise/   # G4-EDA, P1
└── EDA-005_lang_subspace/       # I3, P1
```

Các mục "mở lại khi có tín hiệu" (low-rank, CCA-init, SWA, geometry reg, relational teacher, sigmoid/metric losses, curriculum, adapters…) chưa có mã; cấp mã khi mở lại.

---

# 15. Notebook plan

```text
kaggle/
├── FLAG_09_ext_v1v2.ipynb
├── FLAG_10_setproto.ipynb
├── FLAG_11_robust_set.ipynb
├── FLAG_12_assignment.ipynb
├── FLAG_13_lowrank_bridge.ipynb
└── FLAG_14_generalization.ipynb
```

---

# 16. Minimal implementation details

## 16.1. SetProto training

```python
for batch in loader:
    faces = batch["faces"]      # [P,Kf,Df]
    voices = batch["voices"]    # [P,Kv,Dv]

    zf = face_head(faces)
    zv = voice_head(voices)

    row_loss = multipos_crossmodal_loss(
        zf.reshape(-1, zf.size(-1)),
        zv.reshape(-1, zv.size(-1)),
        face_ids,
        voice_ids,
    )

    pf = F.normalize(zf.mean(1), dim=-1)
    pv = F.normalize(zv.mean(1), dim=-1)

    proto_loss = crossmodal_identity_loss(pf, pv)

    loss = row_loss + lambda_proto * proto_loss
```

---

## 16.2. Weighted prototype

```python
qf = arcface_quality(face_teacher_emb)       # [P,Kf]
qv = redim_quality(voice_teacher_emb)         # [P,Kv]

wf = torch.softmax(beta * qf, dim=1)
wv = torch.softmax(beta * qv, dim=1)

pf = (wf[..., None] * zf).sum(1)
pv = (wv[..., None] * zv).sum(1)
```

---

## 16.3. Robust test centroid

```python
def weighted_centroid(bridge_x, teacher_x, beta=10.0):
    t = F.normalize(teacher_x, dim=1)
    c = F.normalize(t.mean(0, keepdim=True), dim=1)
    q = (t @ c.T).squeeze(1)
    w = torch.softmax(beta * q, dim=0)
    return (w[:, None] * bridge_x).sum(0)
```

---

## 16.4. Trimmed centroid

```python
def trimmed_centroid(bridge_x, teacher_x, keep_ratio=0.8):
    t = F.normalize(teacher_x, dim=1)
    c = F.normalize(t.mean(0, keepdim=True), dim=1)
    q = (t @ c.T).squeeze(1)

    k = max(1, int(len(q) * keep_ratio))
    idx = torch.topk(q, k=k).indices

    return bridge_x[idx].mean(0)
```

---

# 17. Reporting template cho mọi experiment

```text
Experiment:
Parent:
Hypothesis:

Train sources:
Validation:
Seeds/splits:

Row EER:
Prototype EER:

ng/En:
ng/Bn:
g/En:
g/Bn:

paired Δ:
mean Δ:
std Δ:
wins / splits:

effective rank:
within-speaker variance:
between-speaker distance:
hubness:
language gap:

Runtime:
VRAM:
Notes:
Decision:
```

---

# 18. Success criteria theo loại experiment

Vùng hòa chung: **|mean paired Δ| ≤ 0.4 EER → không promote, không nộp** (giữ luật PLAN_V4).

## SET / deterministic inference

Promote nếu cả ba:

```text
mean paired Δ ≤ −0.5 EER trên v4 giữ ra (5 split)
≥ 4/5 split thắng
cùng chiều trên ít nhất một validation không-English (Urdu hoặc Hindi)
```

và không cell nào (ng / g) tệ hơn > 0.5.

---

## Training representation

Promote nếu:

```text
≥ 4/5 split thắng
mean paired Δ ≤ −0.5 EER (EER mức người, L3)
```

và ít nhất một validation external độc lập cùng chiều.

---

## External-data recipe

Promote nếu:

```text
unseen language: mean Δ ≤ −0.5, ≥ 4/5 fold
heard: không tệ hơn +0.5
```

và kết quả **giữ trên protocol chéo ngôn ngữ** (B2.3/B2.4: train Urdu → test Hindi và ngược lại), không chỉ trên đúng ngôn ngữ đã train.

---

## Geometry method

Promote nếu:

```text
thỏa tiêu chí "Training representation"
và
geometry metric thay đổi theo cơ chế kỳ vọng
```

Ví dụ:

```text
hubness giảm
effective rank hợp lý hơn
within/inter ratio giảm
```

Metric hình học đổi mà EER không đổi → không promote.

---

# 19. Các hướng có khả năng tạo bước nhảy lớn nhất

Xếp theo **trần tiềm năng có bằng chứng**, không theo độ hấp dẫn của ý tưởng. Trần của Track A phải xác nhận bằng §0.3.

### 1. External v1/v2 + balanced identity training (Track B)

Đòn bẩy duy nhất đã có bằng chứng giảm lỗi ở mức người (σ_p²): EXT-02 v3 −6.65 ở cả heard lẫn unseen. Lỗi còn lại sau SET-02 chủ yếu ở mức người (§0.2.F). Rủi ro: tương thích quần thể (v3 hại v4 ở EXT-01) → v1/v2 là phép thử thật.

### 2. Global assignment (A3)

Hướng test-time duy nhất có thể sửa lỗi ánh xạ mức người. Chưa có bằng chứng; tiền lệ CSLS / GRAPH-01 là âm tính ở mức fusion → trần phải đo (CEIL-2).

### 3. SetProto / identity-level training (Track C)

Khớp objective với cách chấm lúc test. Trên v4 đơn thuần kỳ vọng nhỏ (40–56 người); giá trị thật khi kết hợp với #1.

### 4. Cross-language supervision (B6, I3)

Target có Bangla unseen. Kiểm bằng protocol chéo Urdu ↔ Hindi.

### 5. Robust aggregation + ngưỡng theo ngôn ngữ (A1, A2)

Trên English trần ≤ ~0.9 EER (SET-01), phía mặt ≈ 0. Dư địa còn lại ở cụm giọng Bangla — phần lớn đã do SET-05 (ReDimNet + ngưỡng không-English) xử lý.

### 6. Low-rank bridge (Track D)

Emb 32–128 đã sweep; chỉ còn ý nghĩa nếu D1/D2 chỉ ra rank ổn định < 32 hoặc khi số người tăng nhờ #1.

### 7. Flat-minima / geometry regularization / loss khác (Track E, F6–F8, H)

Phần lớn đã được ensemble 10 model hấp thụ, hoặc đã phẳng trên dữ liệu này. Không lên lịch.

---

# 20. Kết luận kế hoạch

Roadmap nghiên cứu từ đây không tập trung vào thay encoder.

Ba tầng cần tối ưu đồng thời là:

```text
Tầng 1 — Identity discovery
ArcFace / ReDimNet / ECAPA
        ↓
cluster đúng hơn
quality tốt hơn

Tầng 2 — Identity representation
VGG / BTC
        ↓
robust aggregation
SetProto
external identities
low-rank bridge
cross-language training

Tầng 3 — Global matching
cluster score
        ↓
Hungarian / Sinkhorn / unbalanced OT
        ↓
rank fusion
```

Mọi hướng trong tài liệu này đều có ít nhất một cơ chế hợp lý có thể tạo gain:

- giảm sample noise;
- giảm mixed-cluster error;
- tăng số identity học bridge;
- tăng cross-language invariance;
- giảm overfit capacity;
- giảm hubness;
- khai thác one-to-one structure;
- hoặc cải thiện optimization/generalization.

Nhưng chúng **không có cùng trọng lượng bằng chứng**. Sau SET-02, lỗi còn lại chủ yếu nằm ở **tầng 2 mức người** (cầu nối ánh xạ sai cả người), không ở tầng 1 (gộp gần bão hoà trên English). Vì vậy:

- đo trần trước (§0.3), rồi mới đầu tư vào tầng 1 / tầng 3;
- đòn bẩy chính cho tầng 2 là **thêm người cùng quần thể** (v1/v2), SetProto đi kèm;
- tầng 3 (Sinkhorn) là cách test-time duy nhất nhắm vào lỗi mức người — đáng thử sớm vì rẻ;
- các hướng "mở lại khi có tín hiệu" được giữ trong tài liệu nhưng không lên lịch.

Triển khai theo §13; mọi quyết định promote theo §18 (paired 5 split, vùng hòa 0.4).
