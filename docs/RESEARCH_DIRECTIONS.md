# Hướng nghiên cứu tiếp theo — FLAG 2027

Lập 2026-09-25, sau khi đạt **30.90** và đọc phương pháp của 2 đội đầu FAME 2026.

> **⚠️ Tài liệu lịch sử — kế hoạch hiện hành là [PLAN_V4.md](PLAN_V4.md) (2026-09-26, mốc 28.69).**
> Chẩn đoán §2 ("trần feature / 192-d là nút cổ chai") và lộ trình §5 **đã bị bằng chứng bác một phần**:
> encoder unimodal mạnh hơn không làm ghép chéo tốt hơn; nút thắt là cầu nối face↔voice.
>
> | Hướng | Trạng thái | Kết quả |
> |---|---|---|
> | H1 AAM bổ sung InfoNCE | ❌ Đóng | AAM ≈ InfoNCE (EXP-009, NB-2 P3) |
> | H2a ECAPA 6144-d | ◐ Dùng trong fusion | Thắng ng/En, thua Bangla; thành phần English của FUSE-01 |
> | H2a ReDimNet / CAM++ / ERes2Net | ❌ Đóng | ReDimNet2: speaker EER 3.17 (BTC 4.77) nhưng cross-modal phẳng, dev tệ hơn 4.5–6.7 |
> | H2b WavLM / XLS-R | ❌ Đóng | Pooling L4–9 kém mọi cell, hại fusion English |
> | H3 ArcFace / AdaFace / MagFace | ❌ Đóng | ArcFace **đã align**: face↔face 1.7 EER, cross-modal 33.92 (VGG 26.16) — không phải lỗi alignment |
> | H4 dữ liệu cặp ngoài | ◐ Đang chạy (PLAN_V4 A) | Được phép nếu khai báo; MAV-Celeb v3 (châu Âu) hại mọi cell; v1 Urdu / v2 Hindi đang chạy |
> | H5 ImageBind + LoRA | ☐ Chưa làm | Không ưu tiên trong PLAN_V4 |
> | H6 age-gender | ◐ Một phần | Giới tường minh không có lợi (GEN-01); tuổi/F0 là nhánh phụ của PLAN_V4 B |
> | H7 backend cổ điển | ◐ | CCA w=0.25 giữ; PLDA/OT chưa làm |
> | H8 transductive | ◐ Giới hạn cứng | Chỉ chọn model + fit ≤ 2 tham số (PLAN_V4 §3); không train trên pseudo-label |
> | H9 validation cho Bangla | ✅ Giải xong | SEL-01b → `Experiment/SEL-01_selector/pseudo_eval.py` |
>
> Ghi chú §5.1 bên dưới: encoder BTC **không** phải `speechbrain/spkrec-ecapa-voxceleb`; đã xác minh là
> `yangwang825/ecapa-tdnn-vox2` (cos 1.0000), và 6144-d không phải "layer trước" của feature BTC.

---

## 1. Định vị: chúng ta thua ở đâu

Leaderboard FLAG 2027 (dev), per-cell:

| Hạng | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| 1 | 26.64 | 21.23 | 24.32 | 24.24 | 36.78 |
| 2 | 29.15 | 23.02 | 26.17 | 27.49 | 39.92 |
| 3 | 29.39 | 23.02 | 26.60 | 34.42 | **33.51** |
| **4 — chúng ta** | **30.90** | 25.60 | 29.02 | 30.96 | 38.01 |
| Δ so với hạng 1 | −4.26 | **−4.37** | **−4.70** | **−6.72** | −1.23 |

Đọc bảng này kỹ hơn con số tổng:

- **Khoảng cách lớn nhất KHÔNG nằm ở language transfer.** `g/Bn` của ta chỉ thua hạng 1 là 1.23 và còn **hơn hạng 2**. Ta thua nặng nhất ở `g/En` (−6.72) và hai cell `ng/*` (−4.4) — tức **khả năng phân biệt identity cơ bản**, không phải khả năng chịu đổi ngôn ngữ.
- Hạng 3 có `g/Bn` = 33.51, tốt nhất bảng, nhưng `g/En` = 34.42 (kém ta). Xác nhận có **đánh đổi English ↔ Bangla giữa các approach** — ghép per-cell là chiến lược đúng và nên duy trì.
- Suy ra ưu tiên: **nâng chất lượng biểu diễn identity** trước, tinh chỉnh language transfer sau.

## 2. Chẩn đoán: chúng ta đã chạm trần của feature được cấp

Bằng chứng nội bộ:

| Quan sát | Nguồn |
|---|---|
| CCA tuyến tính (không train) đạt 35.79; 7 thí nghiệm deep sau đó chỉ kéo thêm ~5 điểm | EXP-000d → 007 |
| Sweep 28 cfg kiến trúc trên GPU: top-12 chỉ chênh nhau **0.5 điểm** | EXP-006 |
| Augmentation: `none` thắng cả 13 biến thể | EXP-006 |
| Partial CORAL: xấu đi đơn điệu theo α | EXP-006 |
| PCA 90% của voice chỉ cần 85/192 chiều; effective dim 53 | EDA-2 |

Bằng chứng bên ngoài — **cả 2 đội đầu đều không dùng feature được cấp theo cách chúng ta dùng**:

| Đội | Face | Voice | Điểm mấu chốt |
|---|---|---|---|
| Hạng 1 FAME26 (23.99) | VGGFace 4096 (như ta) | **ECAPA-TDNN 6144-d** — lấy layer *trước* output, không phải 192-d đã pooled | + đặc trưng age-gender riêng (ECAPA 1536-d + ViT 768-d), **freeze toàn bộ backbone, chỉ train lớp mapping**, dropout **0.9**, **AAM loss**, train trên **VoxCeleb2** |
| Hạng 2 FAME26 (24.73) | \multicolumn{2}{c}{**ImageBind + LoRA**} | fine-tune chỉ trên tiếng Ả Rập (VoxBlink) vẫn tổng quát tốt sang English/German |

> Voice 192-d mà ban tổ chức cấp là **bản đã pooled và nén** của một biểu diễn 6144 chiều. Đây nhiều khả năng là nút cổ chai lớn nhất của chúng ta.

**Chúng ta đã có sẵn raw wav/jpg** trong `Input/train_set.zip` và `Input/dev_set.zip` (2.4 GB) — chưa hề dùng. Hướng re-extract khả thi ngay trên Kaggle GPU.

## 3. Các hướng, xếp theo ROI

### H1 — AAM-softmax **bổ sung** InfoNCE (không thay thế) ⭐

⚠️ **Sửa so với bản đầu của tài liệu này.** Không nên bỏ InfoNCE để lấy AAM thuần. Mục tiêu của FLAG là **cross-modal verification**: AAM tối ưu góc giữa embedding và *class center*, không tối ưu trực tiếp góc giữa `z_face` và `z_voice`. Một hệ AAM thuần có thể phân biệt speaker tốt ở **từng** modality mà hai modality vẫn không nằm cạnh nhau — đúng thứ chỉ số EER của ta đo.

Thiết kế nên dùng — **shared class centers**:

```text
z_f = f_face(x_f),  z_v = f_voice(x_v)          # hai nhánh, cùng số chiều
W ∈ R^{n_spk × d}                                # MỘT ma trận center dùng chung cho cả 2 modality
L = L_AAM(z_f, y; W) + L_AAM(z_v, y; W) + λ · L_align(z_f, z_v)
```

Chia sẻ `W` đã tạo alignment **gián tiếp** (cả hai modality bị kéo về cùng một center của speaker đó), nên `λ` không cần lớn — nhưng vẫn cần `L_align` (InfoNCE hoặc cosine) vì "cùng gần một center" không đảm bảo "gần nhau" khi center chỉ có 70.

- **Kỹ thuật**: sweep `margin ∈ {0.2, 0.3, 0.4}`, `scale ∈ {16, 32, 64}`, `λ ∈ {0.3, 1, 3}`. Biến thể: CosFace, AdaCos, **Sub-center ArcFace**, Circle loss.
- Theo quy tắc một-biến: chạy `InfoNCE` (hiện tại) → `InfoNCE + AAM` → `AAM + λ·align` để tách đóng góp.
- **Chi phí**: ~1 ngày, không cần dữ liệu mới.
- **Từ khóa**: `AAM-softmax`, `additive angular margin`, `ArcFace loss speaker verification`, `sub-center ArcFace`, `AdaCos`, `shared classifier cross-modal`, `class center alignment`

### H2 — Re-extract voice features ⭐⭐⭐ **làm đầu tiên**

Hai nhánh, nên làm **cả hai** rồi so:

**H2a — cùng họ, nhiều chiều hơn (bắt chước hạng 1)**
Lấy ECAPA-TDNN nhưng ở layer trước output (6144-d) thay vì 192-d.
- Model: `speechbrain/spkrec-ecapa-voxceleb`, WeSpeaker, NeMo `titanet_large`
- Mới hơn ECAPA: **ReDimNet**, **CAM++**, **ERes2Net**, **NeXt-TDNN** — đều mạnh hơn ECAPA trên VoxCeleb

**H2b — encoder đa ngôn ngữ (đánh thẳng vào gap Bangla)**
EDA-5 cho thấy shift nằm gần như hoàn toàn ở voice (language probe 88%, centroid lệch 2.6 sd). Một encoder được pretrain đa ngôn ngữ **có Bengali** sẽ giảm shift từ gốc, thay vì sửa ở phía sau (mọi cách sửa phía sau đều đã thất bại: CORAL, partial CORAL, AS-norm).
- **WavLM-Large** (+ x-vector head), **wav2vec2-XLS-R-300m/1B**, **MMS-1B** (1000+ ngôn ngữ, có Bengali), Whisper encoder
- Cách dùng: frozen feature extractor + attentive stat pooling, hoặc fine-tune nhẹ với LoRA

- **Chi phí**: GPU, vài giờ. Raw wav đã có sẵn.
- **Từ khóa**: `ECAPA-TDNN embedding layer`, `ReDimNet speaker verification`, `CAM++ speaker`, `ERes2Net`, `WavLM speaker verification`, `XLS-R`, `MMS speech`, `language-agnostic speaker embedding`, `cross-lingual speaker verification`, `attentive statistics pooling`

### H3 — Re-extract face features ⭐⭐

VGGFace fc7 là kiến trúc 2015. Các encoder nhận dạng mặt hiện đại mạnh hơn nhiều.
- **ArcFace / AdaFace / MagFace** trên IResNet-100, train MS1MV3 hoặc WebFace42M (`insightface`)
- Foundation: **CLIP ViT-L**, **DINOv2**, **FaRL** (face representation learning)
- EDA-2 cho thấy face 4096-d có effective dim chỉ 74 và cần 526 chiều cho 90% phương sai → nhiều chiều nhưng thưa thông tin.
- **Từ khóa**: `ArcFace`, `AdaFace`, `insightface model zoo`, `MagFace`, `WebFace42M`, `DINOv2 face`, `FaRL face representation`

### H4 — Dữ liệu ngoài: VoxCeleb2 ⭐⭐⭐ đòn bẩy lớn nhất, cần xác minh luật

Vấn đề gốc của ta: **chỉ 70 speaker**. EDA-3 đã cho thấy CCA k>32 là overfit; sweep kiến trúc bão hòa ở 0.5 điểm — đều là triệu chứng thiếu dữ liệu, không phải thiếu kiến trúc.

VoxCeleb2 có **6112 speaker** kèm video → cặp face-voice tự nhiên, gấp 87 lần. Hạng 1 FAME26 train trên đó với 3 biến thể (full / loại German / loại English) để chọn theo protocol. Hạng 2 dùng VoxBlink.

- ⚠️ **Phải xác nhận với ban tổ chức trước.** Evaluation plan chỉ ghi *"A pretrained encoder for faces or voices is allowed"*, **không nói rõ** về train trên dataset ngoài. Hạng 1 FAME 2026 đã làm và được chấp nhận, nhưng đó là mùa trước. Rủi ro là **bị loại**, nên cần hỏi trước khi đầu tư.

Phân định mức rủi ro theo đúng văn bản hiện có:

| Việc | Tình trạng theo rule |
|---|---|
| Dùng ECAPA pretrained (trên VoxCeleb) rồi **freeze**, trích feature từ raw wav của FLAG | ✅ Có cơ sở rõ — "pretrained encoder is allowed" |
| Dùng ArcFace/AdaFace pretrained trích feature từ jpg của FLAG | ✅ Như trên |
| Fine-tune encoder pretrained **chỉ trên FLAG train** | 🟡 Hợp lý, nên xác nhận nếu muốn chắc |
| Tải VoxCeleb2 rồi **train mapping/model trên cặp face–voice của VoxCeleb2** | 🔴 **Chưa được rule xác nhận** |
| Fine-tune backbone bằng VoxCeleb2/VoxBlink | 🔴 **Phải hỏi trước** |
- ⚠️ MAV-Celeb nhiều khả năng **giao với VoxCeleb** (cùng là người nổi tiếng). Phải loại trừ identity trùng, nếu không là leak.
- Dataset khác: **VoxBlink / VoxBlink2**, **AVSpeech**, **CN-Celeb**, **MSR-VTT**
- **Từ khóa**: `VoxCeleb2`, `VoxBlink`, `AVSpeech`, `audio-visual speaker dataset`, `cross-modal pretraining face voice`

### H5 — Foundation model + adapter (ImageBind + LoRA) ⭐⭐

Đúng công thức của hạng 2. ImageBind đã có audio và image trong **một không gian chung** — nghĩa là phần khó (alignment) đã được pretrain sẵn trên dữ liệu khổng lồ; ta chỉ cần adapter nhỏ.

- **Kỹ thuật**: LoRA (rank 8–32) trên các khối attention, hoặc chỉ train linear probe. PEFT.
- Lựa chọn khác: LanguageBind, AudioCLIP, CLAP + CLIP ghép qua lớp mapping
- Phù hợp với dữ liệu ít: số tham số huấn luyện nhỏ.
- **Từ khóa**: `ImageBind`, `LoRA`, `PEFT`, `parameter-efficient fine-tuning`, `multimodal foundation model`, `AudioCLIP`, `CLAP`, `LanguageBind`

### H6 — Đặc trưng phụ trợ: age / gender ⭐

Hạng 1 nối thêm nhánh age-gender tường minh. Ta đã đo: gender probe đạt **93% (face) / 92% (voice)** (EDA-4) — tín hiệu rất mạnh và hiện đang bị dùng ngầm, không kiểm soát.

Làm tường minh sẽ giúp: ở 2 cell `no_gender` được phép khai thác gender; ở 2 cell `gender` thì tách nó ra để phần identity còn lại sạch hơn.

- **Kỹ thuật**: multi-task (identity + age + gender), hoặc concat embedding phụ như hạng 1, hoặc **tách không gian**: một phần chiều cho thuộc tính, phần còn lại cho identity.
- **Từ khóa**: `speaker profiling`, `age gender estimation from speech`, `soft biometrics`, `multi-task speaker embedding`, `attribute disentanglement`

### H7 — Backend ML cổ điển ⭐

Ta đã có CCA đóng góp thật (1.3 điểm, w≈0.25 — EXP-007). Còn chưa thử:
- **PLDA** — backend chuẩn của speaker verification, mạnh hơn cosine khi có nhiễu phiên/kênh; biến thể cross-modal PLDA
- **Deep CCA (DCCA)**, KCCA, PLS
- **Optimal Transport** để căn chỉnh phân phối giữa 2 modality hoặc giữa 2 ngôn ngữ (mềm hơn CORAL — mà CORAL đã hỏng vì quá cứng)
- Procrustes alignment (dùng nhiều trong cross-lingual word embedding — đúng bài toán tương tự)
- **Từ khóa**: `PLDA backend`, `deep CCA`, `optimal transport domain adaptation`, `Procrustes alignment`, `cross-lingual embedding mapping`, `unsupervised bilingual lexicon induction`

### H8 — Test-time / transductive trên dev ⭐

Dev có **5288 cặp không nhãn** mà ta mới chỉ dùng để tính mean. Có thể dùng nhiều hơn:
- **Pseudo-labeling**: chấm dev bằng model hiện tại, lấy các cặp tự tin nhất làm positive giả, fine-tune tiếp
- **Test-time adaptation**: TENT (entropy minimization), BN-statistics adaptation
- ⚠️ Rủi ro: overfit dev. Và đã có tiền lệ đau — AS-norm cũng là kỹ thuật dùng thống kê test-time và nó **hại** trên dev thật.
- **Từ khóa**: `test-time adaptation`, `TENT entropy minimization`, `pseudo-labeling`, `transductive learning`, `self-training domain adaptation`

### H9 — Sửa lỗ hổng validation (vấn đề riêng của ta) ⭐⭐

Đã xác lập qua 3 thí nghiệm liên tiếp: **mọi cải tiến chọn bằng internal (chỉ English) đều mua điểm English và trả giá ở Bangla** (003b → 003c → 007). Internal hiện tại về nguyên tắc không chọn được model tốt cho Bangla.

Cách sửa:
- Tạo internal val **có language shift thật**: lấy speaker đa ngôn ngữ từ VoxCeleb/CN-Celeb → train một ngôn ngữ, val ngôn ngữ khác
- Hoặc: proxy không cần nhãn trên dev Bangla (độ tách rời của phân phối score, entropy, độ ổn định giữa các seed)
- Hoặc: chấp nhận dùng CodaBench làm val cho Bangla (còn ~95 lượt) nhưng **ghi rõ** đó là tuning trên dev, và giữ một recipe "sạch" để đối chiếu nếu có test phase riêng
- **Từ khóa**: `unsupervised model selection`, `proxy metric domain adaptation`, `validation without labels`

## 4. Trả lời trực tiếp các câu hỏi

**Finetune, train từ đầu, hay adapter?**
→ **Không train backbone từ đầu** — 70 speaker là quá ít, ta đã thấy overfit ở mọi mức capacity. Chọn theo thứ tự:
1. **Frozen backbone + train lớp mapping nhỏ** — đúng công thức hạng 1 (dropout 0.9!). An toàn nhất với dữ liệu ít.
2. **Adapter / LoRA** trên foundation model — công thức hạng 2. Khi muốn backbone thích nghi mà không overfit.
3. **Fine-tune toàn phần** chỉ khi đã có VoxCeleb2 (đủ dữ liệu).

**Kết hợp DL và ML cổ điển thế nào?**
→ Đã chứng minh có lợi trong chính dự án này: deep ensemble + CCA tuyến tính với **trọng số tường minh w≈0.25** (EXP-007, đóng góp 1.3 điểm). Mở rộng: thêm PLDA làm thành viên thứ ba, mỗi thành viên một trọng số riêng, tối ưu trọng số trên internal. ML cổ điển đóng vai trò regularizer — nó không overfit 70 speaker theo cách mạng neural overfit.

**Kỹ thuật sâu của NLP dùng được gì?**
→ Bài toán này gần với **cross-lingual word embedding alignment** một cách đáng ngạc nhiên: hai không gian khác nhau, cần ánh xạ, không có nhãn ở miền đích. Các kỹ thuật chuyển được: Procrustes alignment, adversarial mapping (MUSE), optimal transport, iterative refinement/self-learning (Artetxe). Ngoài ra: contrastive learning (InfoNCE — đã dùng), hard negative mining, temperature scaling, memory bank kiểu MoCo cho batch hiệu dụng lớn hơn, knowledge distillation từ ensemble về một model.

## 5. Lộ trình đề xuất

> **Cập nhật 2026-09-25 sau khi chạy H2a và H1.** Hai hướng đầu đã có kết quả thật, và cả hai đều
> không như kỳ vọng ban đầu — bảng dưới đã được viết lại theo bằng chứng.
>
> * **H2a (ECAPA 6144-d)**: internal tốt hơn 1.31 nhưng **CodaBench tệ hơn 0.81** (31.71 vs 30.90).
>   Chỉ `ng/En` cải thiện (23.81, tốt nhất mọi lượt nộp); 3 cell kia tệ hơn và **gap ngôn ngữ tăng gấp đôi**.
>   → Feature giàu hơn cũng mang nhiều thông tin phi-identity hơn. **Cross-lingual cần bất biến, không cần giàu.**
> * **H1 (công thức hạng 1)**: 18 cấu hình, không cái nào thắng cấu hình hiện tại. AAM ≈ InfoNCE;
>   dropout 0.9 mới là thứ phá (37.82). Lợi thế của họ nằm ở **VoxCeleb2 + age-gender**, không ở loss.
> * **H3 (ArcFace)**: +10 EER — nghi lỗi alignment, chưa kết luận được.

| Thứ tự | Việc | Chi phí | Kỳ vọng |
|---|---|---|---|
| **1** | **H9 — sửa validation để nhìn thấy Bangla.** ❌ Nhánh "proxy từ phân phối score" đã thử và **chết** (EXP-012: tương quan trong-cell ≈ 0; lưỡng đỉnh có thể do giới tính/ngôn ngữ chứ không phải identity). Còn 2 nhánh: **(a) proxy từ embedding** trên dev không nhãn — mutual-kNN, tương ứng cụm giữa 2 modality; **(b) validation có language shift thật** từ speaker đa ngôn ngữ ngoài | 1–2 ngày | internal đã dẫn sai 4/4 lần ở mọi quyết định đụng Bangla |
| **2** | **H2b — WavLM / XLS-R / MMS** (đa ngôn ngữ, có Bengali) | GPU vài giờ | nhắm đúng tính **bất biến** thay vì độ giàu |
| 3 | **H6 — nhánh age-gender** (code sẵn ở `kaggle/agegender.py`) | 1 ngày | thành phần thật sự của hạng 1 mà ta chưa thử |
| 4 | **H3' — ArcFace có face alignment** (`app.get()`) | GPU vài giờ | chạy lại cho đúng trước khi kết luận về face |
| 4 | **H2b — WavLM/XLS-R** cho Bangla | GPU vài giờ | nhắm riêng 2 cell Bangla |
| 5 | **H6 — nhánh age-gender** | 1 ngày | 0.5–1.5 điểm |
| 6 | **H9 — sửa validation** | 1–2 ngày | không ra điểm trực tiếp nhưng chặn việc đi sai hướng tiếp |
| 7 | **H4 — VoxCeleb2** *(sau khi hỏi BTC)* | nhiều ngày | đòn bẩy lớn nhất |
| 8 | **H5 — ImageBind+LoRA** | vài ngày | đường độc lập, có thể ghép per-cell |

### 5.1 Phép thử quyết định — chạy trước mọi thứ khác

Mục đích: **xác nhận hoặc bác bỏ giả thuyết "192-d voice là nút cổ chai"** với ít biến số nhất.

```text
A (đối chứng):  pipeline hiện tại + mapping hiện tại + loss hiện tại + voice 192-d (BTC cấp)   → int_g 30.42, CB 30.90
B (phép thử):   pipeline hiện tại + mapping hiện tại + loss hiện tại + voice 6144-d (tự trích)
```

Chỉ đổi **một** thứ. Nếu B kéo EER xuống rõ rệt → giả thuyết được xác nhận, dồn GPU cho H2b/H3. Nếu B ngang A → giả thuyết sai, feature không phải nút thắt, quay lại H1/H6.

Phép thử này còn có một **kiểm chứng phụ miễn phí**: cùng lúc trích luôn ECAPA **192-d** và so với CSV của BTC. Nếu trùng khớp cao → xác nhận BTC dùng đúng ECAPA đó và 6144-d thật sự là "layer trước"; nếu khác → BTC dùng encoder khác và ta vừa biết thêm một điều quan trọng.

### 5.2 Kỷ luật thí nghiệm

Không đổi nhiều thứ trong một run (`feature mới + AAM + kiến trúc mới + augmentation`) — có tăng 3 điểm cũng không biết đến từ đâu. Quy ước một-biến đã ghi ở [../README.md](../README.md) §5.

Ghép per-cell luôn chạy ở cuối mỗi vòng — đã cho 33.35 và 30.58 mà không tốn gì.

## 6. Việc cần làm ngay, không phải nghiên cứu

**Câu 2 đã được Evaluation Plan trả lời — không cần chờ.** Bản kế hoạch ghi rõ dữ liệu được phát *"Alongside the audios (.wav) and images (.jpg)"* và *"provided alongside pre-extracted features"*, tức raw media là dữ liệu chính thức song song, feature CSV chỉ là tiện ích; cộng với *"A pretrained encoder for faces or voices is allowed."* → **H2/H3 khởi động ngay, không đứng chờ email.**

**Vẫn nên hỏi BTC 1 câu** (chỉ ảnh hưởng H4):
> Có được train/fine-tune hệ thống trên các cặp face–voice của dataset ngoài (VoxCeleb2, VoxBlink) không, hay "pretrained encoder is allowed" chỉ có nghĩa là dùng encoder đã pretrain ở trạng thái đóng băng?

Gửi email để có bằng chứng bằng văn bản, nhưng **không chặn** H1/H2/H3 trong lúc chờ.

## 7. Nguồn

- [FLAG 2027 Challenge Evaluation Plan](https://arxiv.org/html/2609.17913)
- [FAME 2026 Challenge Evaluation Plan](https://arxiv.org/abs/2508.04592)
- [Linking Faces and Voices Across Languages: Insights from the FAME 2026 Challenge](https://arxiv.org/html/2512.20376) — bảng xếp hạng, baseline FOP 41.57
- [Hạng 1 FAME 2026 — 23.99% EER](https://arxiv.org/abs/2512.04814) — ECAPA 6144-d + age-gender + AAM + VoxCeleb2, freeze backbone
- [Hạng 2 FAME 2026 — 24.73% EER](https://arxiv.org/abs/2512.02759) — ImageBind + LoRA, fine-tune trên tiếng Ả Rập
- [FAME 2024 Evaluation Plan](https://arxiv.org/abs/2404.09342)
- [Exploring Robust Face-Voice Matching in Multilingual Environments](https://arxiv.org/pdf/2407.19875)
- [Trang challenge](https://mavceleb.github.io/dataset/competition.html)
