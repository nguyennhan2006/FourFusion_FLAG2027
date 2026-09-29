# Hướng đi còn mở — để mọi người cùng thử

Cập nhật **2026-09-29**, đồng bộ với nhật ký tới mục (75) và [DECISIONS.md](DECISIONS.md). Mỗi người **nhận một hướng** (ghi tên vào cột "Người nhận" ở §3), làm theo luật chung ở §1, rồi báo kết quả theo mẫu ở §5.
Mọi số liệu dẫn ở đây có trong [../Experiment/EXPERIMENT_LOG.md](../Experiment/EXPERIMENT_LOG.md). Bản 27/09 của tài liệu này (S1–S5, P1–P2, X1–X4) đã được làm gần hết; kết quả nằm ở §4.

## 0. Tình hình trong năm dòng

1. **Cấu hình Evaluation là SET-13** qua [../BEST/v6_eval/](../BEST/v6_eval/README.md): CB **21.01** (11.31 / 20.91 / 23.22 / 28.61), 15 mạng mỗi thành phần. Toàn bộ quy trình đã chạy thử từ ảnh / giọng gốc.
2. **Cell tụt xa nhất giờ là g/Bn (28.61)**, rồi g/En (23.22). Dev không phân xử được chênh lệch cỡ ±1 (61), nên mọi quyết định dựa vào validation nhãn thật.
3. **Gom cụm đã hết dư địa:** cụm ước lượng cách cụm theo danh tính thật 0.1–2 EER (73). Nó cũng an toàn ở mọi dạng file đã thử (STRESS-01).
4. **ImageBind zero-shot là tín hiệu mạnh nhất; mọi cách "chỉnh" nó với dữ liệu hiện có đều thua** (adapter tuyến tính, CCA; LoRA hoãn).
5. **Thêm người train vẫn giúp cầu nối, nhưng ImageBind che phần lớn cái lợi đó.** Ở bản production chỉ còn English g được lợi rõ (74). SetProto và SWA không có tác dụng (74, 75).

## 1. Luật chung khi thử (bắt buộc)

**Luật cuộc thi** (BTC trả lời, nhật ký 63):
- được dùng encoder pretrained (kể cả ImageBind), dữ liệu cặp ngoài và gom cụm lúc test;
- người và giọng ở Evaluation không trùng train / dev;
- phải khai báo mọi thứ trong system description;
- không train trên tiếng Bengali.

**Không bao giờ dùng danh sách trial**, kể cả để chấm điểm lẫn để chọn hoặc trộn hệ. Phép thử nhanh: ghép lại các trial khác theo cách khác mà điểm của một cặp thay đổi, tức là đang dùng danh sách trial. Chi tiết ở [PLAN_MODELS.md](PLAN_MODELS.md) §0b.

**Giao thức validation chung** (để kết quả của mọi người so được với nhau):

| | |
|---|---|
| Người v4 | `V.speaker_split(store.spk, s, n_val=30)`, s = 1…5: 40 người train, 30 giữ ra |
| Người v1/v2 | 5 fold theo người (id gộp theo tên, bỏ 2 người trùng v4); fold giữ ra không bao giờ được train; **mặt của cặp dương lấy khác video với giọng** |
| File đánh giá | dựng như dev: mỗi ảnh / giọng một lần, 50% cặp đúng, ng và g (g cùng giới) |
| **Thước đo chính** | **EER mức người và mức cụm của bản production** (s007′ = s007 + ImageBind + tuổi, chấm như SET-13). Mức mẫu và thành phần đứng riêng chỉ là phụ (A7 trong DECISIONS) |
| So sánh | hiệu **ghép cặp** với đối chứng, cùng split / fold, **cùng số bước tối ưu** |
| Cổng | tốt hơn ≥ 1 EER (hoặc ≥ 0.5 cho thay đổi không fit tham số) ở ≥ 4/5 split, ô khác không tệ hơn quá 0.5 |
| Khung code | [../Experiment/EXTSCALE-01/run.py](../Experiment/EXTSCALE-01/run.py): có sẵn bảng đặc trưng v4 + v1/v2, fold người, file đánh giá, chấm `member` / `prod` ở ba mức. Có thể import như module (xem [../Experiment/SWA-01/run.py](../Experiment/SWA-01/run.py)) |

**Kỷ luật.**
- Một biến mỗi lần.
- Liệt kê các nhánh (arm) **trước** khi xem kết quả.
- Ghi vào EXPERIMENT_LOG ngay khi có số.
- Chỉ nộp CodaBench khi đã qua cổng, **tối đa 1 lượt cho mỗi ứng viên**, và chỉ nộp từ tài khoản của đội (nguyennhan2006).

## 2. Hạ tầng có sẵn

| Cần gì | Ở đâu |
|---|---|
| Feature BTC (VGG fc7 4096, ECAPA 192) | `kaggle_upload/`, đọc qua `flag_v2.FeatureStore` |
| ArcFace, ECAPA-192 / 6144 (speechbrain), WavLM, XLS-R | `kaggle/output/feats_v2/` |
| **Mọi đặc trưng của pipeline Evaluation, kể cả v1/v2 ArcFace + ECAPA** | `kaggle/output/feats_eval/`: output của [FLAG_12_features](../kaggle/FLAG_12_features.ipynb) (khớp `feats_v2` / `feats_models` tuyệt đối) |
| ViT-age, FaRL, SigLIP2, AdaFace, ImageBind, w2v2 tuổi–giới | `kaggle/output/feats_models/` (FLAG_10) |
| Lớp khác của encoder BTC, TTA | `kaggle/output/feats_layers/` (FLAG_11) |
| MAV-Celeb v1 / v2 trong không gian feature BTC | `kaggle/output/feats_ext_full/` (bản đầy đủ; mảng của FLAG_10 / FLAG_12 theo đúng thứ tự dòng của nó) |
| Pipeline Evaluation (SET-13, `noclus`, INDEP-01, chẩn đoán cụm) | [../BEST/v6_eval/pipeline.py](../BEST/v6_eval/pipeline.py); 45 mạng production đã cache trong `_members/` |
| Notebook Kaggle mới | **code đầy đủ trong cell** ([../kaggle/build_eval_notebooks.py](../kaggle/build_eval_notebooks.py)); chạy thử bằng `kaggle/run_notebook_local.py --jupyter` với transformers 5.0.0 trước khi giao |

## 3. Các hướng còn mở

| ID | Hướng | Cell đích | Chi phí | Bằng chứng hiện có | Người nhận |
|---|---|---|---|---|---|
| **O1** | **s007 train thêm v1/v2, chỉ cho các cell English** | g/En | CPU vài giờ | English g ở bản production: cụm +2.33 [5/5] (EXT-VAL70), v4 g −4.55 [5/5] (SCALE-ID). English ng ≈ 0 (74) | |
| **O2** | **g/Bn: tìm tín hiệu còn lại cho ngôn ngữ chưa nghe, cùng giới** | g/Bn | CPU | Cell tệ nhất (28.61). Chưa có hướng nào đã kiểm chứng. Urdu (gần Bangla nhất) không được lợi từ v1/v2 hay SetProto (74) | |
| **O3** | **Speech2Face (P2 cũ)**: giọng → VGG fc7 | g, ng | tuỳ checkpoint | Chưa thử: cần tìm checkpoint công khai | |
| **O4** | **VoxCeleb (X1)**, chỉ khi có lý do mới | tất cả | nhiều ngày | Chưa đạt cổng: ở bản production, bước 100 → 161 người chỉ còn đạt ở v4 ng (74) | |

### O1 — v1/v2 chỉ cho các cell English

- **Giả thuyết.** Người v1/v2 dạy cầu nối thêm những gì ImageBind chưa có ở English g. Ở ng và ở Urdu thì không.
- **Làm.**
  - Thành phần `s007x`: công thức EXP-007 trên 70 người v4 + toàn bộ v1/v2 (trừ 2 người trùng). Cùng P × K và số bước với s007 (xem `train_arm` trong EXTSCALE-01).
  - Hệ `SET-15`: SET-13, trong đó s007′ của các cell English dùng `s007x`, còn Bangla giữ nguyên.
- **Kiểm chứng trước khi dùng.**
  - Fold người v1/v2 (English giữ ra), mức cụm của bản production, 5/5.
  - Riêng ng/En: SET-07 (cùng ý tưởng, chưa có ImageBind) cho +2.98 trên dev. Nếu ng/En không trung tính thì chỉ dùng cho g/En.

### O2 — g/Bn

- **Tình hình.** g/Bn là cell tệ nhất và nhạy seed nhất (61). Mọi hướng đã thử (dữ liệu ngoài, SetProto, CCA / adapter trên ImageBind, FaRL, lớp khác của BTC, TTA) đều không cho lợi ổn định ở Urdu g, ngôn ngữ đại diện gần nhất.
- **Gợi ý bắt đầu, chưa ai thử:**
  - phân tích lỗi mức người trên file Urdu g mô phỏng: cặp sai nào bị điểm cao, do tuổi, do giọng hay do ImageBind?
  - trọng số tuổi riêng cho g (AGE.25 từng cho g cụm −2.41 [4/5] nhưng Urdu g người +0.53 (62)).
- Phải đăng ký trước, và kiểm trên Urdu / Hindi giữ ra, **không** trên dev Bangla.

### O3 — Speech2Face

- **Giả thuyết.** Voice encoder của Speech2Face được train để dự đoán **VGG-Face fc7 của khuôn mặt từ giọng nói**, tức đúng không gian feature mặt của BTC.
- **Làm.**
  1. Tìm checkpoint công khai. Nhóm gốc (MIT) không công bố weights; các bản tái hiện thường train trên một phần nhỏ dữ liệu.
  2. Tự trích feature; không nhận file feature của đội khác.
  3. Zero-shot trước: EER của cos(S2F(giọng), VGG(mặt)) trên 70 người train. Nếu có tín hiệu, thử như một số hạng giống ImageBind trong s007′.
- **Luật.** Được dùng (63). Ghi chú nếu dữ liệu train của checkpoint có tiếng Bengali.

### O4 — VoxCeleb (X1)

- **Điều kiện mở lại:**
  - SCALE-ID trên bản production cho thấy lợi ổn định ở bước cuối; hiện chỉ còn v4 ng;
  - hoặc có một cách dùng dữ liệu lớn không bị ImageBind che (ví dụ ghép riêng cho g/Bn, nếu O2 tìm ra cơ chế).
- **Nếu làm:** chạy lại [../Experiment/IB-ADAPT/](../Experiment/IB-ADAPT/README.md) với dữ liệu mới làm bước sàng lọc, rồi mới tính tới LoRA cho ImageBind.
- Chi tiết khả thi ở [PLAN_MODELS.md](PLAN_MODELS.md) §4.

## 4. Đã làm — kết quả, đừng làm lại

### Đã dùng trong cấu hình Evaluation

| Hướng (ID cũ) | Kết quả | Nguồn |
|---|---|---|
| ImageBind zero-shot (P1) | −5…−9 trên validation, 10/10 file; trong SET-13 | IB-01, MODEL-01, VAL-70 (58–66) |
| So khớp tuổi tường minh (S2) | g cụm −1.35 [5/5]; trong SET-13 (w = 0.1) | MODEL-01 (62) |
| Bản dự phòng không gom cụm (S5) | `noclus` và INDEP-01 (26.70) trong v6; `noclus` ≥ INDEP-01 trên file mô phỏng | INDEP-01, STRESS-01 (68, 71) |
| 15 mạng mỗi thành phần | cùng kỳ vọng, bớt may rủi theo seed | VAL-70, SWA-01 (69, 75) |

### Đã đóng

| Hướng | Vì sao đóng | Nguồn |
|---|---|---|
| **SetProto** (kể cả kết hợp thêm người) | cùng ngân sách: ≈ 0, tương tác ≈ 0 | EXTSCALE-01 (74) |
| **SWA / trung bình trọng số** | +0.03 EER, độ lệch theo seed không giảm | SWA-01 (75) |
| **Chỉnh ImageBind: adapter tuyến tính, CCA 4/16/64** | tệ hơn zero-shot hoặc lẫn lộn ±2 | IB-ADAPT, IB-PROBE (70, 74) |
| **ImageBind + LoRA (X3)** | hoãn: adapter tuyến tính còn overfit với dữ liệu hiện có | IB-ADAPT (70) |
| Stream ViT-age (S1a), SigLIP2, AdaFace | ViT-age trượt ở g; SigLIP2 không giúp g; AdaFace +6…+12 | MODEL-01 (59, 62) |
| FaRL (S1b) thêm vào SET-13 (SET-14) | không hơn SET-13 ở VAL-70 (−6.70 so với −7.38); FaRL chỉ còn làm dự phòng không ImageBind (SET-11) | VAL-70 (65) |
| Lớp khác của BTC (S3), TTA (S4) | lợi < 0.5 | MODEL-01 (62) |
| Quy tắc `auto` tự chuyển sang chấm từng cặp; ngưỡng giọng tự thích nghi | mất 2–6 EER; không hơn ngưỡng hiệu chỉnh | STRESS-01 (71, 73) |
| Thay VGG bằng encoder nhận dạng mặt (ArcFace, AdaFace, LVFace, TopoFR, MagFace) ở cầu nối | nhận dạng giỏi ≠ ghép chéo giỏi | 010-D, EDA-002 |
| Thay voice bằng encoder nhận giọng mạnh hơn (ReDimNet2, CAM++, ERes2Net) | ghép chéo phẳng, dev tệ hơn | NB-4 / FUSE-04 |
| WavLM / XLS-R lấy trung bình qua các lớp | kém ở mọi cell | NB-2, EDA-002 |
| CORAL (cả bản partial), AS-norm | CORAL xoá danh tính (41.36); AS-norm đoán sai dấu | 004, 006c, 003c |
| Giới tường minh / GRL / negative cùng giới | không có lợi ổn định | NB-2 P4, GEN-01 |
| Head hạng 1 FAME26, MSE align; cầu nối lớn hơn (X4) | dropout 0.9: 37.82; thêm dung lượng thì tệ; SCALE-ID chưa cho thấy thiếu dung lượng | 009, ARCH-01, EXTSCALE-01 |
| Trộn thêm CCA-16 (SET-08) | lợi nhỏ trên VAL-70 (−0.1…−1.8), nhỏ hơn hẳn ImageBind; bị SET-13 thay | SET-08, VAL-70 (61) |
| Đổi embedding gom cụm giọng, Sinkhorn, aggregator học được | chênh ≤ 0.3; không ổn định | SET-05, SET-06, CEIL |
| Chiếu bỏ hướng ngôn ngữ | hoà | LANG-01 |
| Augmentation lúc train, nhiễu Gaussian vào giọng | không giúp | 006b, 006d |
| Làm mượt điểm bằng đồ thị kNN, đoán EER từ phân phối điểm, tìm thông tin cùng phiên ghi | English tệ hơn; tương quan ≈ 0; AUC 0.50 | GRAPH-01, 012, SES-01 |
| MAV-Celeb v3 (châu Âu) làm dữ liệu train | tệ hơn ở cả 4 cell | EXT-01 |
| Mọi cách dùng danh sách trial | đọc đáp án từ cách ra đề | LEAK-01, PLAN_MODELS §0b |

## 5. Mẫu báo kết quả

Thêm một dòng vào bảng và một mục vào "Nhật ký quyết định" trong [../Experiment/EXPERIMENT_LOG.md](../Experiment/EXPERIMENT_LOG.md):

```text
- **YYYY-MM-DD (NN)** — **<ID hướng>: <kết luận một câu>** ([../Experiment/<ID>/](../Experiment/<ID>/)).
  Nhánh (đăng ký trước): control = …, A = …, B = …; cùng số bước: …
  prod, mức cụm / người: v4 ng Δ … [k/5], v4 g Δ … [k/5] | Urdu Δ …, Hindi Δ …, English (v1/v2) Δ …
  (phụ) mức mẫu, member: …
  Cổng: qua / trượt vì … → nộp CB? (dự đoán trước: …)
```

Δ âm là tốt hơn. [k/5] là số split cải thiện. Luôn ghi dự đoán **trước** khi nộp, để biết công cụ dự đoán sai bao nhiêu. Sau mỗi kết quả, cập nhật [DECISIONS.md](DECISIONS.md).
