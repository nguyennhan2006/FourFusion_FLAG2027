# Kế hoạch giai đoạn 2 — re-extract feature + mô hình mới trên Kaggle

Lập 2026-09-25. Nền: [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) (hướng H1–H9), EDA mới [EDA-001](../Experiment/EDA-001_flag_questions/).
Mốc cần vượt: **30.90** (top 4), ghép per-cell kỳ vọng 30.58. Hạng 1: 26.64.

---

## 0. Tóm tắt quyết định

1. **Tách trích feature khỏi huấn luyện.** Một notebook trích feature chạy **một lần**, ghi `.npy` và publish thành Kaggle Dataset. Mọi notebook huấn luyện sau đó chỉ đọc khoảng 300 MB `.npy`, không đụng lại wav/jpg, nên không thể OOM vì audio dài.
2. **Mỗi lần chỉ đổi một biến.** Theo thứ tự: feature → head/regularization → loss → nhánh gender → nhánh language → fusion.
3. **Adversarial gender/language chỉ là ablation cuối**, chỉ ở nhánh Gender và chỉ giữ nếu CodaBench xác nhận. Lý do ở §2.
4. **Hướng mới từ EDA: khớp độ dài utterance.** Bangla ngắn hơn English khá rõ (§1, Q3). Có raw wav nên có thể crop audio train theo phân phối độ dài của Bangla ngay lúc trích feature. Việc này rẻ và chưa ai trong nhóm thử.

---

## 1. Trả lời 7 câu EDA

Dev **không có nhãn identity**. Vì vậy Q4, Q5, Q7 được ước lượng bằng pseudo-identity:
- Gom cụm face và voice bằng agglomerative clustering. Ngưỡng được hiệu chỉnh trên train (có nhãn): ARI face 0.82, voice 0.96.
- Nối mỗi cụm voice với cụm face mà nó bị ghép cặp nhiều hơn hẳn mức ngẫu nhiên.
- Kết quả phủ 3209/4864 hàng dev. Script: [pseudo.py](../Experiment/EDA-001_flag_questions/pseudo.py), [dur.py](../Experiment/EDA-001_flag_questions/dur.py).

**Q1. Số lượng theo split**

| Split | Identity | Nam/Nữ | Utterance En | Utterance Bn |
|---|---|---|---:|---:|
| train | 70 | 40 / 30 | 6485 | **0** |
| dev (4 file) | ~30 theo Evaluation Plan (100 − 70); gom cụm ra 37 cụm face | Bangla ~65% nam (EDA-000) | 1990 | 2874 |
| test | chưa phát hành | – | – | – |

> ⚠️ **Train không có một câu Bangla nào.** Mọi phương pháp cần nhãn ngôn ngữ (ví dụ GRL language) chỉ có thể học từ **audio dev không nhãn**, tức là adaptation có dùng dữ liệu test (transductive). Xem §2.

**Q2. Utterance theo identity (train):** min 2 (id006, id052), median 92, max 226. Dev không có nhãn nên không đếm được.

**Q3. Độ dài audio (giây)** — toàn bộ 16 kHz, mono, 16-bit:

| | n | p5 | median | p95 | p99 | max | tổng giờ |
|---|---:|---:|---:|---:|---:|---:|---:|
| train En | 6485 | 2.2 | 5.2 | 19.6 | 34.2 | 60.0 | 13.2 |
| dev En | 1990 | 2.3 | 5.8–6.1 | 19.6–21.7 | 37.6–41.4 | **81.6** | 4.5 |
| dev Bn | 2874 | 2.2 | **4.7** | **12.4–12.7** | 18.6–19.0 | 43.9 | 4.6 |

→ Bangla ngắn hơn: median thấp hơn 20%, đuôi p95 ngắn bằng khoảng một nửa. Embedding của câu ngắn thường nhiễu hơn, nên **một phần "language shift" có thể là "duration shift"**.
→ Tác động tới Kaggle: file dài nhất 81.6 s. Nếu padding batch theo thứ tự file, một batch 16 câu có thể phình lên 16 × 81 s, và đây đúng là nguồn OOM của lần chạy trước.

**Q4 + Q5. Cặp positive/negative (ước lượng):**

| File | Tỉ lệ positive | Negative cùng giới | Negative khác giới | m-m | f-f |
|---|---:|---:|---:|---:|---:|
| no_gender/En | 0.48 | 172 | 165 | 119 | 53 |
| no_gender/Bn | 0.48 | 235 | 245 | 157 | 78 |
| gender/En | 0.48 | 321 | 12* | 203 | 118 |
| gender/Bn | 0.52 | 460 | 22* | 318 | 142 |

\* Khoảng 4% negative "khác giới" trong protocol gender gần như chắc chắn là lỗi của probe giới tính, không phải dữ liệu.
→ Mỗi file cân bằng khoảng 50/50. Ở `no_gender`, negative chia đôi cùng giới/khác giới, nên gender shortcut có giá trị cho khoảng một nửa số negative. Ở `gender`, shortcut này hoàn toàn vô dụng.

**Q6. Cosine unimodal trên train** (feature BTC, có centering):

| | Cùng speaker | Khác speaker, cùng giới | Khác speaker, khác giới |
|---|---:|---:|---:|
| face | 0.49 ± 0.27 | 0.02 ± 0.09 | −0.04 ± 0.06 |
| voice | 0.55 ± 0.19 | 0.02 ± 0.12 | −0.04 ± 0.09 |

→ Ở mức unimodal, giới tính chỉ dịch cosine khoảng 0.06. Hai modality đều phân biệt identity tốt; phần khó là **cầu nối cross-modal**.

**Q7. Voice cùng speaker, En↔Bn (quan trọng nhất)** — 10 pseudo-speaker có đủ cả hai ngôn ngữ:

| Cặp voice | cos trung bình ± sd |
|---|---:|
| Cùng speaker En–En | 0.51 ± 0.24 |
| Cùng speaker Bn–Bn | 0.50 ± 0.20 |
| **Cùng speaker En–Bn** | **0.35 ± 0.20** |
| Khác speaker cùng giới (mọi cặp ngôn ngữ) | −0.02 … 0.01 ± 0.11 |

→ Đổi ngôn ngữ làm cosine của cùng một speaker giảm khoảng 30%, nhưng vẫn cách xa phân phối của người khác (d′ khoảng 2.2 so với 2.9). **Identity trong voice sống sót qua đổi ngôn ngữ.** Chỗ hỏng là ánh xạ cross-modal: ánh xạ này chỉ được học trên voice English.
Caveat: con số En–En và Bn–Bn bị thổi phồng, vì gom cụm chọn sẵn những voice giống nhau. Con số En–Bn nối qua face nên ít thiên lệch hơn. Khoảng cách thật có thể nhỏ hơn 0.16.

---

## 2. Đánh giá đề xuất GRL gender + language

| Thành phần | Đánh giá | Quyết định |
|---|---|---|
| Shared-center AAM + alignment (Shared Embedding, XM-ALIGN) | Khớp H1. XM-ALIGN chính là "shared classifier + MSE align" | **Làm**, phase P2 |
| Bỏ age/gender auxiliary của đội hạng 1 | Đúng với cell `gender/*`, nhưng cell `no_gender/*` hưởng lợi từ giới tính | Nhánh Standard **không** loại gender; chỉ nhánh Gender mới loại |
| GRL gender | EDA-000 #4: xóa tuyến tính hướng gender **không giúp**; same-gender negative (B) chỉ được 0.5 ở `int_g` và mất 6 ở `int_ng`. Chỉ có 70 speaker nên adversary dễ học thuộc speaker → giới tính | Ablation P4, chỉ ở nhánh Gender, λ nhỏ có warm-up |
| GRL language | Train **không có Bangla**. Domain duy nhất có nhãn ngôn ngữ là **audio dev không nhãn** (DANN transductive). Q7 cho thấy identity đã bền với đổi ngôn ngữ; phần shift chủ yếu là lệch mean (centering đã xử lý) cộng độ dài câu | Ablation P5, **sau** duration-matching. Ghi rõ là dùng dữ liệu dev để adaptation |
| Orthogonality / MCS (Max Class Separation) | Rẻ, có code | Biến thể loss trong P2 |
| PAEFF / Fuse-after-Align (learned pair scorer) | Có nguy cơ overfit 70 speaker. Bằng chứng nội bộ: head sâu chưa bao giờ thắng | Để sau cùng, chỉ nếu P2 bão hòa |
| ImageBind + LoRA | Là đường độc lập (H5), tốn vài ngày | P7, tùy chọn |

---

## 3. Môi trường Kaggle và cách chống OOM / mất chất lượng

Giới hạn (Kaggle, 2026): GPU **T4 × 2** (khoảng 15 GB mỗi card) hoặc P100, RAM khoảng 29 GB, `/kaggle/working` khoảng 20 GB, phiên tối đa 12 giờ, **khoảng 30 giờ GPU mỗi tuần**. NB-1 cần bật Internet, các notebook còn lại tắt.

| Rủi ro | Biện pháp |
|---|---|
| Padding tới câu 81 s → OOM | **Sắp xếp theo độ dài + gom batch theo tổng số giây** (≤ 240 s audio mỗi batch, tối đa 32 câu). Câu > 60 s chạy riêng với bs = 1. Nếu vẫn OOM thì tự chia đôi batch, `empty_cache` rồi thử lại, không crash |
| Padding làm sai lệch embedding | Truyền `wav_lens` tương đối cho ECAPA (ASP có mask). Kiểm tra: trích 50 file với bs = 1 so với batch, cosine phải > 0.999 |
| fp16 làm lệch feature | ECAPA/ArcFace giữ **fp32** (tổng chỉ khoảng 22 giờ audio, T4 xử lý vài phút). WavLM/XLS-R dùng fp16 kèm kiểm tra cosine so với fp32 trên 50 file |
| WavLM attention O(T²) trên câu dài | Cửa sổ 20 s, bước 10 s, mean-pool theo độ dài cửa sổ |
| ArcFace trên ảnh 224 chưa align (resize thẳng về 112 làm giảm chất lượng) | Chạy detector SCRFD của `buffalo_l` → `norm_crop` 5 điểm. Không phát hiện được mặt thì center-crop 0.7 rồi resize. **In tỉ lệ detect** |
| `onnxruntime-gpu` lệch CUDA, âm thầm chạy CPU | In provider thực tế. CPU vẫn chấp nhận được (11k ảnh) |
| Phiên chết giữa chừng | Mỗi (encoder, split) ghi một file `.npy`; nếu file đã tồn tại thì bỏ qua → chạy lại là resume |
| Hết đĩa | Không giải nén vào `/kaggle/working` nếu Kaggle đã tự giải nén. Feature fp32 tổng khoảng 0.5 GB |
| RAM khi huấn luyện | 6485 × 6144 fp32 = 160 MB, đưa hết lên GPU. PCA/CCA của 6144-d chạy qua SVD trên ma trận đã PCA (đã sửa trong `RidgeCCA`) |
| Thứ tự hàng lệch CSV của BTC | Index lấy từ `*.txt` như FLAG_03; assert số hàng và kiểm tra cosine giữa ECAPA-192 tự trích và CSV của BTC |

---

## 4. Bộ notebook

| NB | Việc | GPU | Internet | Output |
|---|---|---|---|---|
| **NB-1 `FLAG_04_extract`** (viết lại từ FLAG_03) | Trích: ECAPA-192, **ECAPA-6144**, ECAPA-6144 **crop theo độ dài Bangla** (3 crop/wav train), ArcFace-512 (có align), tùy chọn WavLM-Large / XLS-R-300m (pool trung bình có trọng số theo layer, lưu 25 layer × 1024 ở fp16) | khoảng 1–1.5 giờ | ON | Dataset `flag2027-feats-v2` |
| **NB-2 `FLAG_05_ablate`** | Chạy các phase P1–P5 bên dưới, chọn bằng internal 3 split × 3 seed, ghi `ablation.csv` | khoảng 2–4 giờ | OFF | csv + `.pt` |
| **NB-3 `FLAG_06_submit`** | Train lại cấu hình thắng trên đủ 70 speaker, ensemble n = 10, centering, fusion với CCA w, ghép per-cell, preflight dấu score, kiểm tra zip | khoảng 30 phút | OFF | `submission_*.zip` |

NB-2 và NB-3 attach cả `flag2027-feats-v2` lẫn `flag2027-features` (CSV của BTC, cho run đối chứng).
Có thể chạy **song song** hai phiên: NB-2 P1 (không cần feature mới) trong lúc NB-1 đang trích.

---

## 5. Các phase thí nghiệm

Số in đậm là cấu hình mang sang phase kế tiếp. Chọn bằng `proxy = (int_ng + int_g) / 2`, tối thiểu 3 split.

| Phase | Câu hỏi | Các run (một biến mỗi lần) | Cổng quyết định |
|---|---|---|---|
| **P0** EDA sau khi trích | Feature mới có đúng không? | cos(ECAPA-192 tự trích, CSV BTC); lặp lại Q7 trên 6144-d; tỉ lệ detect face | cos > 0.95 → 6144 đúng là "layer trước" |
| **P1** head | Công thức đội hạng 1 FAME 2026 có hợp với 70 speaker? (dùng feature cũ) | MLP drop .3 (đối chứng: int 21.94 / 30.53) → linear → linear + drop .9 → emb 192 | giữ nếu proxy giảm ≥ 0.5 |
| **P2** feature | Nút thắt nằm ở feature nào? | A: cũ/cũ · B: face cũ + **v6144** · C: face cũ + v192 tự trích · D: **ArcFace** + v cũ · E: ArcFace + v6144 · F: + WavLM | **Nộp 1 lượt CodaBench cho run tốt nhất** (lượt đầu tiên kiểm tra Bangla) |
| **P3** loss | AAM có giúp khi đi cùng InfoNCE? | InfoNCE → + AAM W dùng chung (m ∈ {.2,.3}, s ∈ {16,32}) → + λ align (MSE hoặc cos) → + orthogonality | giữ nếu proxy giảm |
| **P4** nhánh Gender | Loại gender cho cell `gender/*` | InfoNCE_sg → + GRL gender (λ ∈ {.05,.2}, warm-up) | **chỉ tính trên int_g**; giữ GRL chỉ khi thắng InfoNCE_sg |
| **P5** Bangla | Duration hay language? | train với crop khớp độ dài → + DANN language trên voice dev (transductive) | internal không đo được Bangla → **quyết định bằng CodaBench** (2 lượt) |
| **P6** fusion | | CCA trên feature mới, sweep w ∈ {0, .15, .25, .35}; ghép per-cell 4 cell | nộp |
| P7 (tùy chọn) | Đường độc lập | ImageBind + LoRA | ghép per-cell nếu thắng một cell |

Ngân sách: khoảng **8–10 giờ GPU** cho P0–P6. **Khoảng 6–8 lượt CodaBench** (còn khoảng 95).

### Quy tắc đọc kết quả (đã rút từ các EXP trước)
- Internal chỉ có English. Mọi thay đổi ảnh hưởng Bangla (P5, score-norm) → **chỉ tin CodaBench**.
- Không dùng AS-norm; score = −cosine (lower = same, xem [DATA.md](DATA.md)).
- Xếp hạng từ 1 seed không có giá trị (EXP-005).

---

## 6. Đọc paper theo phase

| Phase | Đọc |
|---|---|
| P1, P3 | Shared Multi-modal Embedding (hạng 1 FAME 2026), XM-ALIGN, Max Class Separation |
| P4 | Disentangled Representation Learning (TMM 2021), Seeing Voices & Hearing Faces (lý do có protocol gender) |
| P5 | Cross-Modal Speaker Verification — Multilingual Perspective (CVPRW 2021), Robust Face-Voice Matching in Multilingual Environments (FAME 2024) |
| P7 | ImageBind + LoRA (hạng 2 FAME 2026) |

## 7. Rủi ro còn lại

- **Dùng audio dev để adaptation (DANN)** là transductive. Luật không cấm rõ ràng, nhưng nên ghi vào system description. Luôn giữ một bản "sạch" để đối chiếu.
- Pseudo-identity trong EDA-001 dựa trên gom cụm: chỉ dùng để định hướng, **không bao giờ dùng làm nhãn huấn luyện**.
- 6144-d cộng head tuyến tính trên 70 speaker có thể overfit hơn 192-d. P1 có mặt chính là để tách hiệu ứng này ra.
