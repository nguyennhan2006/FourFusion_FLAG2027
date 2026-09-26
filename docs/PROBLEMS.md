# Các problem cần giải — để tìm đúng paper

Lập 2026-09-25, sau 13 lượt nộp CodaBench và NB-2 v2. Mục đích: mỗi problem có **phát biểu, bằng chứng, câu hỏi nghiên cứu, từ khóa** và **tiêu chí lọc paper**. Nhờ vậy việc tìm tài liệu nhắm đúng chỗ đang mất điểm, không tìm lan man.
Nguồn số liệu: [../Experiment/EXPERIMENT_LOG.md](../Experiment/EXPERIMENT_LOG.md), [PLAN_KAGGLE.md](PLAN_KAGGLE.md), [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md).


> **Cập nhật 2026-09-26 — trạng thái từng problem** (kế hoạch hiện hành: [PLAN_V4.md](PLAN_V4.md); chi tiết: [EXPERIMENT_LOG.md](../Experiment/EXPERIMENT_LOG.md)).
> Mốc hiện tại **28.69** (23.61 / 26.88 / 29.12 / 35.15). Các mục §0 và §3 bên dưới là trạng thái lúc lập (30.13).
>
> | Problem | Trạng thái | Kết quả |
> |---|---|---|
> | PB-1 chọn model cho Bangla | ✅ **Giải xong** | Selector pseudo-identity SEL-01b (ArcFace + ECAPA-192): Spearman 0.95–0.996 trên 9 hệ, sai số ≤ 0.9 EER/cell qua 3 lượt nộp. Dùng để chọn model và fit ≤ 2 tham số; không train trên pseudo-label |
> | PB-2 `g/En` | ◐ Cải thiện một phần | 30.96 → 29.12 nhờ rank fusion 007 + 010B. Còn cách hạng 1 4.07. ATTR (tuổi/pitch) chưa làm |
> | PB-3 feature giàu vs bất biến | ◐ | Hai encoder 192-d có hồ sơ ngôn ngữ ngược nhau (BTC-192 hợp English, speechbrain ECAPA hợp Bangla) → ghép theo cell. **Encoder mạnh hơn không giúp** (ReDimNet2, ArcFace): nút thắt là cầu nối mặt–giọng |
> | PB-4 duration | ❌ Giả thuyết bị bác | `r2_mix` thắng ở câu Bangla **dài**, thua ở câu ngắn; gate FUSE-03 cho 003c trọng số *tăng* theo độ dài (0.24 → 0.44) → lợi ích chỉ là đổi trọng số, không phải cơ chế độ dài |
> | PB-5 chỉ 70 identity | ◐ Đang chạy (EXT-01) | Dữ liệu ngoài được phép nếu khai báo. Encoder BTC tái tạo chính xác. v3 (châu Âu) **hại mọi cell** → ánh xạ phụ thuộc quần thể; v1 (Urdu) / v2 (Hindi) đang chạy lại |
> | PB-6 re-ranking đồ thị | ❌ Đóng | Làm mượt trên ma trận đầy đủ (không dùng danh sách trial): English tệ hơn, ng/Bn +0.5 |

---

## 0. Đang đứng ở đâu

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| Hạng 1 | 26.64 | 21.23 | 24.32 | 24.24 | 36.78 |
| **Ta — ghép per-cell (EXP-011)** | **30.13** (hạng 4) | 23.81 | 27.88 | 30.96 | 37.87 |
| Khoảng cách | −3.49 | −2.58 | −3.56 | **−6.72** | −1.09 |
| Nguồn của cell | | ECAPA 6144 (010-B) | 003c | 007 | 003c |

Điểm đáng chú ý: **ba trong bốn cell tốt nhất của ta đến từ ba hệ khác nhau.** Chưa có hệ nào tốt đồng thời cả English lẫn Bangla.

---

## 1. Những gì đã xác lập — không cần tìm thêm

| Đã chứng minh | Bằng chứng |
|---|---|
| MLP 2 nhánh + InfoNCE + dropout thấp là đúng cho 70 speaker; công thức FAME26 (linear, dropout .9, AAM) **không** chuyển giao được | EXP-009 (18 cfg), NB-2 P1 |
| Loss không phải đòn bẩy: AAM ≈ InfoNCE (±0.3) | EXP-009, NB-2 P3 |
| Mọi thao tác trên **thống kê bậc 2** lúc test đều phá identity: CORAL, partial CORAL, AS-norm | EXP-004 (41.36), 006c, 003b→003c |
| Chỉ căn **mean** theo từng file là an toàn và có lợi | EXP-003c |
| Mọi cách **loại gender tường minh** đều hại, kể cả ở cell gender: same-gender negatives, GRL gender, project-out hướng gender | EDA-4, EXP-003, NB-2 P4 (int_g 28.89 → 29.21 / 29.27) |
| **Face encoder càng chuyên identity thì ghép face–voice càng kém:** ArcFace (EER mặt–mặt 1.7, tốt hơn VGG 4 lần) cho cross-modal 33.9 so với VGG 26.2, dù đã align đúng | NB-2 P0/P2, 010-D |
| Feature voice giàu hơn (ECAPA 6144) thắng English nhưng thua Bangla | 010-B (31.71), sub_base (31.95) |
| Proxy EER tính **chỉ từ phân phối score** không dự đoán được EER trong cùng một cell | EXP-012 (tương quan ≈ 0) |
| Dev: mỗi file đúng 50% target; English và Bangla là **cùng một tập người**; voice cùng speaker En↔Bn cos 0.35 so với En↔En 0.51, còn người khác ≈ 0 | CB, EDA-000 #9, EDA-001 |
| Train **không có một câu Bangla nào**; Bangla ngắn hơn (median 4.7 s so với ~6 s, p95 12.5 s so với ~20 s) | EDA-001 |

→ **Không tìm paper về:** kiến trúc fusion mới, loss margin mới, CORAL/whitening, AS-norm, adversarial gender, face recognition SOTA, speaker encoder SOTA (ReDimNet2: speaker EER 3.17 nhưng ghép chéo không tốt hơn).

---

## 2. Các problem, xếp theo mức ưu tiên

### PB-1 — Chọn model cho Bangla khi không có dữ liệu Bangla có nhãn ⭐⭐⭐ (gốc rễ)

**Phát biểu.** Validation nội bộ chỉ có English. Cả **5/5** lần nó chọn một thay đổi có ảnh hưởng tới Bangla, nó đều chọn sai: 003b, 007, 010-B, AS-norm, sub_base. Đây là lý do mỗi quyết định cho Bangla phải tốn một lượt CodaBench, và là lý do chưa có hệ nào tốt cả hai ngôn ngữ.

**Câu hỏi nghiên cứu.** Làm sao ước lượng EER trên miền đích (Bangla) khi chỉ có dữ liệu miền đích *không nhãn*?

**Đầu mối đang có trong tay.** Dev English và Bangla là cùng một tập người, và khuôn mặt thì gần như không đổi giữa hai ngôn ngữ. EDA-001 đã dựng được **pseudo-identity** bằng cách nối cụm voice với cụm face (ARI train 0.82 / 0.96). Từ đó có thể tạo **pseudo-trial Bangla** để chọn model. Cách này khác EXP-012 ở chỗ nó dùng embedding và cấu trúc cặp, không chỉ dùng phân phối score. Rủi ro là vòng lặp: pseudo-nhãn được dựng từ chính feature đang được đánh giá.

**Từ khóa:** `unsupervised model selection domain adaptation`, `deep embedded validation`, `importance weighted cross validation covariate shift`, `reverse validation`, `validation without target labels`, `accuracy estimation under distribution shift`, `pseudo-label based model selection`.
**Paper chắc chắn tồn tại:** You et al., *Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation* (ICML 2019, "DEV"); Sugiyama et al., *Covariate Shift Adaptation by Importance Weighted Cross Validation* (JMLR 2007).
**Lọc paper:** phải làm được với **~30 identity không nhãn** và trên embedding có sẵn (không cần train lại backbone). Ưu tiên phương pháp có kiểm định kiểu "tương quan giữa proxy và metric thật trên nhiều model", đúng như cách ta đã kiểm định EXP-012.

---

### PB-2 — `gender/English`: ghép identity thuần khi không còn gender ⭐⭐⭐ (khoảng cách lớn nhất)

**Phát biểu.** Cell `g/En` không có language shift và không cho dùng gender shortcut, tức là nó đo **chất lượng ghép identity thuần**. Ta thua hạng 1 tới **6.72 điểm** ở đây. Không thay đổi nào của ta nhúc nhích được cell này ngoài fusion CCA của EXP-007.

**Nghịch lý phải giải thích.** Loại gender khỏi embedding thì hại, nhưng face encoder chuyên identity (ArcFace) cũng hại. Giả thuyết: **mối liên hệ face↔voice đi qua các thuộc tính mềm** như tuổi, cân nặng/BMI, hình dạng mặt–hàm (liên quan tới vocal tract), chứ không đi qua identity trừu tượng. Hạng 1 FAME26 dùng thêm nhánh **age-gender**, đúng là cung cấp thêm các thuộc tính đó. Vậy cần **nhiều thuộc tính hơn chứ không phải ít hơn**, kể cả thuộc tính ngoài giới.

**Câu hỏi nghiên cứu.** (a) Những thuộc tính nào của mặt dự đoán được giọng, sau khi đã cố định giới? (b) Làm sao đưa chúng vào embedding mà không overfit với 70 speaker?

**Từ khóa:** `face voice association cues age gender ethnicity`, `what makes faces and voices match`, `soft biometrics cross-modal`, `speaker profiling age height BMI from speech`, `facial attributes from voice`, `attribute-aware cross-modal matching`, `hard negative mining same gender curriculum`.
**Paper chắc chắn tồn tại:** Nagrani et al., *Seeing Voices and Hearing Faces* (CVPR 2018); Nagrani et al., *Learnable PINs* (ECCV 2018); Kim et al., *On Learning Associations of Faces and Voices* (ACCV 2018); Oh et al., *Speech2Face* (CVPR 2019); hạng 1 FAME26 (arXiv 2512.04814).
**Lọc paper:** ưu tiên paper **định lượng được** các thuộc tính trung gian (ví dụ: độ chính xác matching khi cố định cả giới lẫn tuổi), và phương pháp dùng **encoder thuộc tính pretrained đóng băng**. Encoder đóng băng là hợp lệ theo luật và không cần thêm cặp face–voice.

---

### PB-3 — Feature giàu thì thắng English, feature bất biến thì giữ Bangla ⭐⭐

**Phát biểu.** 6144-d mang thêm thông tin phi-identity (ngôn ngữ, kênh, phiên). Nó giúp phân biệt trong cùng miền nhưng làm model nhạy với đổi miền: gap ngôn ngữ tăng từ +3.4 lên +7.1. Train không có Bangla, nên đây là **domain generalization** (không có dữ liệu đích lúc train), không phải domain adaptation.

**Câu hỏi nghiên cứu.** (a) Làm sao lấy phần identity của feature giàu mà vứt phần nhạy ngôn ngữ, khi chỉ có một ngôn ngữ để train? (b) Có encoder voice nào được thiết kế bất biến ngôn ngữ không? (Vòng r2 đang đo WavLM.)

**Từ khóa:** `cross-lingual speaker verification`, `language-invariant speaker embedding`, `language mismatch speaker recognition`, `domain generalization single source`, `MixStyle`, `feature statistics perturbation`, `instance normalization domain generalization`, `information bottleneck speaker embedding`, `multilingual SSL speaker verification WavLM XLS-R`.
**Paper chắc chắn tồn tại:** Zhou et al., *Domain Generalization with MixStyle* (ICLR 2021); Zhou et al., *Domain Generalization: A Survey* (TPAMI 2022); các paper dùng bộ CN-Celeb (cross-genre / cross-language speaker verification).
**Lọc paper:** ưu tiên phương pháp **single-source**, áp được trên embedding cố định (augment hoặc perturb ở feature space). Loại các phương pháp cần dữ liệu đích có nhãn.

---

### PB-4 — Câu Bangla ngắn hơn: duration mismatch ⭐⭐ (chờ kết quả r2)

**Phát biểu.** Một phần của "language shift" có thể chỉ là "duration shift". `r2_ecapa192_mix` (train trên crop có độ dài giống Bangla) sẽ trả lời trực tiếp. Internal short-crop cho thấy crop giúp nhẹ (29.68 so với 29.86).

**Câu hỏi nghiên cứu.** Embedding speaker bền với câu ngắn thế nào, và có nên hiệu chỉnh score theo độ dài không?

**Từ khóa:** `short utterance speaker verification`, `duration mismatch speaker verification`, `quality measure function score calibration duration`, `duration-aware score normalization`, `variable-length training speaker embedding`.
**Lọc paper:** chỉ cần phương pháp áp được **lúc train mapping** hoặc **lúc chấm score**. Hiệu chỉnh theo duration là metadata đã biết, khác hẳn AS-norm.
**Điều kiện:** chỉ đầu tư nếu `r2_ecapa192_mix` thắng `r2_ecapa192` ở ≥ 1 cell Bangla.

---

### PB-5 — Ghép cross-modal với chỉ 70 identity ⭐⭐

**Phát biểu.** Mọi triệu chứng đều là thiếu dữ liệu: CCA k > 32 overfit, sweep 28 cfg chỉ chênh 0.5 điểm, head càng mạnh càng tệ. Hạng 1 và hạng 2 FAME26 đều dùng dữ liệu cặp face–voice bên ngoài (VoxCeleb2, VoxBlink).

**Câu hỏi nghiên cứu.** (a) Luật có cho train trên cặp face–voice ngoài không? (**Chưa hỏi BTC.**) (b) Nếu không: có thể dùng **foundation model đã căn chỉnh audio–image sẵn** (ImageBind), chỉ cần adapter rất nhỏ, được không?

**Từ khóa:** `ImageBind audio image`, `LoRA few-shot cross-modal`, `parameter-efficient fine-tuning small data`, `VoxCeleb2 face voice pretraining`, `cross-modal metric learning few identities`.
**Paper chắc chắn tồn tại:** Girdhar et al., *ImageBind* (CVPR 2023); hạng 2 FAME26 (arXiv 2512.02759, ImageBind + LoRA).
**Lọc paper:** số tham số train được phải nhỏ (< ~1M), hoặc zero-shot. Ghi rõ phương pháp có cần dữ liệu cặp ngoài hay không.
**Việc không phải nghiên cứu:** gửi email hỏi BTC về dữ liệu ngoài (xem RESEARCH_DIRECTIONS §6).

---

### PB-6 — Tận dụng cấu trúc của tập dev lúc test (transductive) ⭐

**Phát biểu.** Mỗi file dev chứa khoảng 30 người, với rất nhiều trial lặp lại cùng khuôn mặt và cùng giọng. Hiện ta chấm từng cặp độc lập. Tận dụng láng giềng (face A giống face B, nên giọng của A cũng nên hợp với B) là **re-ranking**, khác cơ chế với AS-norm (chuẩn hóa thống kê cohort, đã thất bại).

**Câu hỏi nghiên cứu.** Re-ranking dựa trên láng giềng tương hỗ (reciprocal neighbours) có cải thiện verification cross-modal không, khi gallery nhỏ và mỗi file 50% target?

**Từ khóa:** `k-reciprocal re-ranking`, `query expansion verification`, `transductive face verification`, `graph-based score refinement`, `cross-modal re-ranking`, `cluster-level verification`.
**Paper chắc chắn tồn tại:** Zhong et al., *Re-ranking Person Re-identification with k-reciprocal Encoding* (CVPR 2017).
**Lọc paper:** chỉ nhận phương pháp không cần nhãn. Rủi ro giống AS-norm: **chỉ đánh giá được bằng CodaBench**, và phải ghi rõ trong system description là có dùng dữ liệu test (transductive).

---

## 3. Thứ tự đề xuất

| # | Problem | Vì sao | Chi phí thử |
|---|---|---|---|
| 1 | **PB-1** validation cho Bangla | Không có nó, mọi hướng khác phải dò bằng CodaBench | Thấp: đã có pseudo-identity từ EDA-001, kiểm định bằng 13 lượt nộp sẵn có |
| 2 | **PB-2** thuộc tính mềm cho `g/En` | Khoảng cách lớn nhất (−6.72); nếu chỉ riêng `g/En` đạt mức hạng 1 thì Overall ≈ 28.5 | Trung bình: trích thêm feature thuộc tính (tuổi, giới) bằng encoder đóng băng |
| 3 | PB-4 duration | Đang chờ r2, gần như miễn phí | Đã chạy |
| 4 | PB-3 bất biến ngôn ngữ | Nhắm đúng 2 cell Bangla | Trung bình |
| 5 | PB-6 re-ranking | Rẻ nhưng rủi ro, và chỉ đo được bằng CB | Thấp |
| 6 | PB-5 dữ liệu ngoài / ImageBind | Đòn bẩy lớn nhất nhưng chờ luật, tốn nhiều ngày | Cao |

## 4. Khi gửi lại kế hoạch

Với mỗi paper bạn chọn, ghi giúp mình 4 dòng để mình map vào codebase nhanh:
1. **Problem** nó giải (PB-x) và **cơ chế** trong một câu.
2. **Đầu vào cần:** embedding có sẵn / raw audio–ảnh / dữ liệu ngoài / nhãn thuộc tính.
3. **Số trong paper** và trên dataset nào (có identity-disjoint không, có cross-lingual không).
4. **Code** có public không.
