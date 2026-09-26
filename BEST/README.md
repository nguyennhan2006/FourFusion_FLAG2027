# Bản tốt nhất — EXP-003c · CodaBench Overall **31.34**

> ⚠️ **Bản này (31.34) đã cũ.** Bản tốt nhất hiện tại: **[v3_28.69/](v3_28.69/) — CodaBench 28.69** (rank fusion theo cell từ 4 hệ, chọn bằng selector trên dev). Tài liệu dưới đây giữ lại vì `reproduce.py` vẫn là cách tái tạo hệ thành phần `c003`.


Nộp ngày 2026-09-25. Tái tạo đầy đủ bằng một file: [`reproduce.py`](reproduce.py) (~330 dòng, chỉ cần numpy/pandas/torch/sklearn).

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| FOP baseline của nhóm (EXP-000c) | 42.17 | 36.31 | 38.98 | 43.38 | 50.00 |
| Organizer FOP reference | 36.92 | 32.54 | 38.12 | 32.99 | 44.01 |
| **EXP-003c (bản này)** | **31.34** | **25.60** | **27.88** | 34.01 | **37.87** |
| Tốt nhất từng cell đã đo | 31.29 | 25.60 | 27.88 | 33.81¹ | 37.87 |

¹ `g/En` 33.81 đến từ EXP-003b (cùng model, có AS-norm). Ghép per-cell chỉ hơn 0.05 → chưa đáng một lượt nộp.

**−10.83 so với baseline nhóm, −5.58 so với organizer reference.** Không dùng dữ liệu ngoài, không re-extract feature; chỉ dùng đúng feature CSV mà ban tổ chức cung cấp.

---

## 1. Công thức

```text
face 4096-d ──► standardise ──► PCA 256 ─┐
                                          ├─► MLP 2 nhánh ──► embedding 128-d (L2-norm)
voice 192-d ──► standardise ──────────────┘
                                                    │
                            symmetric InfoNCE (τ=0.07, positive = cùng speaker)
                                                    │
        ┌───────────────────────────────────────────┴──────────────────────────┐
        │  3 seed (1, 11, 21)                            +  ridge CCA k=4      │
        └───────────────────────────────────────────┬──────────────────────────┘
                                                    │
                   fusion = trung bình cosine đã z-score của 4 model
                                                    │
                              score nộp = −(fusion)   ·  lower = same
```

Nhánh MLP: `Dropout(0.5) → Linear(·,512) → BatchNorm → ReLU → Dropout(0.5) → Linear(512,128) → L2-norm`.

| Siêu tham số | Giá trị | Vì sao |
|---|---|---|
| loss | symmetric InfoNCE, τ = 0.07 | thắng CE+OPL (loss của FOP) 4 điểm và thắng CCA 1.7 điểm trên internal (EXP-003) |
| optimizer | AdamW, lr 1e-3, wd 1e-2, cosine schedule | sweep EXP-005 |
| epochs | 15 | chọn theo `int_g`, early-stop trên internal |
| batch | 256, mỗi epoch lấy 1 mẫu / voice-clip | EDA-0: 6485 hàng chỉ có 4029 voice vector khác nhau |
| embedding | 128-d | EDA-3: k > 32 ở CCA là overfit, nhưng deep model chịu được 128 |
| ensemble | 3 seed + CCA k=4 | mỗi phần đóng góp riêng, xem §3 |
| test-time | trừ mean của **chính file dev**, cộng lại mean train | EDA-6: dev lệch train chủ yếu ở centroid |
| score | cosine thô, **không** AS-norm | EXP-003b vs 003c trên CodaBench |

## 2. Chạy lại

```bash
cd BEST
python reproduce.py                    # → submission_best.zip (~3 phút CPU, <1 phút GPU)
python reproduce.py --internal         # thêm internal EER speaker-disjoint 3 split
python reproduce.py --data <thư mục>   # mặc định ../kaggle_upload
```

Cần feature CSV theo layout `train/` + `dev/` (chính là [`../kaggle_upload/`](../kaggle_upload), hoặc giải nén [`../kaggle/flag2027-features.zip`](../kaggle/flag2027-features.zip)). Script tự dò file theo tên nên không phụ thuộc cấu trúc thư mục.

**Đã kiểm chứng:** output của `reproduce.py` có **rank-correlation 1.000000** với zip đã nộp được 31.34 ở cả 4 cell (chạy lại từ thư mục khác: 0.999995 — sai khác ở mức float/BLAS non-determinism, không đổi EER ở 2 chữ số thập phân). EER chỉ phụ thuộc thứ hạng nên kết quả tái tạo là cùng một con số.

Muốn chạy trên GPU và quét siêu tham số rộng hơn: xem [`../kaggle/README.md`](../kaggle/README.md).

## 3. Mỗi thành phần đóng góp bao nhiêu

Internal EER (speaker-disjoint 56/14, trung bình 3 split; `int_g` là metric chọn model):

| Cấu hình | int_g | int_ng | CodaBench |
|---|---:|---:|---:|
| CCA tuyến tính k=32, không train deep | 34.9 | 29.1 | 35.79 |
| CCA tuyến tính k=4 | 35.5 | 24.3 | 34.46 |
| Ghép per-protocol 2 bản CCA trên | – | – | 33.35 |
| InfoNCE 1 seed, không centering | 34.4 | 25.2 | – |
| + centering per-file | 33.2 | 23.9 | – |
| + ensemble 3 seed | 33.2 | 23.9 | – |
| **+ CCA k=4 vào fusion (= bản này)** | **32.4** | **22.5** | **31.34** |
| ⚠️ + AS-norm | 30.0 | 21.8 | 31.96 ❌ |

Dòng cuối là bài học quan trọng nhất của dự án: **AS-norm cải thiện internal 2.4 điểm nhưng làm CodaBench tệ đi ở 3/4 cell.**

## 4. Những gì đã thử và KHÔNG dùng

| Ý tưởng | Kết quả | Vì sao thất bại |
|---|---|---|
| **AS-norm** (adaptive score norm) | CB 31.96 vs 31.34 | Cohort internal là 14 speaker × ~90 mẫu; cohort dev là các trial pair, mỗi face/voice chỉ xuất hiện ~1 lần, 50% là target → thống kê cohort khác hẳn |
| **Full CORAL** dev→train | CB **41.36** (+10) | Mỗi file dev chỉ có ~500 identity nên **covariance của file CHÍNH LÀ cấu trúc speaker**; whitening nó xoá luôn phương sai phân biệt speaker. Gap ngôn ngữ giảm (+0.54/+2.33) nhưng identity mất sạch |
| **Partial CORAL** α = 0.1 | internal 31.97 vs 31.18 | cùng cơ chế, chỉ nhẹ hơn |
| **CE + OPL** (loss gốc của FOP) | internal 36.8–37.7 | thua cả CCA tuyến tính, dù đã sweep lr/emb/dropout/dedup |
| **Same-gender negatives** (B) | int_g 32.7 nhưng int_ng 30.5 | giúp cell gender, phá cell no_gender; sau ensemble thì bản thường thắng cả hai |
| **Project-out hướng gender** tuyến tính | 40.2 → 40.3 | gender không nằm trên một hướng tuyến tính |
| **Dedup theo voice-clip** | int_g 33.02 có dedup vs 32.26 không | giả định sai từ EDA; voice trùng trong batch hoạt động như nhiều face cho cùng một voice, giúp học alignment. *Bản 31.34 vẫn bật dedup — tắt nó là cải tiến chưa nộp* |

## 5. Bài học phương pháp

1. **Internal val tin được cho representation, KHÔNG tin được cho score normalization.** CORAL: internal dự đoán đúng đến 0.3–1.0 điểm. AS-norm: internal sai dấu. Lý do là cohort/phân phối trial khác nhau.
2. **Xếp hạng từ 1 seed là vô nghĩa.** `hid=1024` dẫn đầu ở seed 1, hạng 12/16 khi lấy trung bình 3 split. Mọi kết luận cfg phải ≥ 3 split.
3. **Chọn model theo `int_g`, không theo `int_ng`.** Calibration từ 2 lượt nộp: `int_ng` lạc quan ổn định +4.0…+4.7 so với CodaBench; `int_g` từng khớp gần như hoàn hảo.
4. **Overall = trung bình 4 cell, trong đó 2 cell `no_gender` thưởng gender shortcut.** Chọn model riêng cho từng protocol là hợp lệ và đã cho 33.35 mà không train gì thêm.
5. **Chạy baseline tuyến tính trước.** CCA (closed-form, không train) đạt 35.79 — tốt hơn FOP 42.17. Nếu không làm bước này, cả tuần sẽ đổ vào tune một kiến trúc thua cả phép chiếu tuyến tính.

## 6. Điểm yếu còn lại

`g/Bn` = **37.87**, cao hơn `g/En` 3.86 điểm. EDA-5 định vị chính xác nguyên nhân: dev English và Bangla **là cùng một tập người** (96% face Bangla có face English với cos > 0.6), khác biệt nằm gần như hoàn toàn ở voice (language probe 88%, centroid lệch 2.6 sd). Mọi cách xoá shift lúc test đều thất bại (§4) → hướng còn lại là làm encoder bất biến **lúc train**: augmentation voice (noise / feature-dropout / mixup cùng speaker / random offset). Script đã sẵn sàng tại [`../Experiment/EXP-006_aug/run.py`](../Experiment/EXP-006_aug/run.py) và trong notebook Kaggle mục 2.

Lưu ý khi đánh giá hướng này: internal **chỉ có tiếng Anh**, nên nó chỉ đo được *chi phí* của augmentation chứ không đo được *lợi ích* Bangla. Quy tắc: chọn cái không làm hại internal rồi đo bằng CodaBench.

## 7. Bản đồ tài liệu

| File | Nội dung |
|---|---|
| [`reproduce.py`](reproduce.py) | toàn bộ code của bản 31.34, tự chứa |
| [`submission_best.zip`](submission_best.zip) | file nộp (tái tạo, rank-corr 1.0000 với bản gốc) |
| [`../Experiment/EXPERIMENT_LOG.md`](../Experiment/EXPERIMENT_LOG.md) | mọi lượt nộp + nhật ký quyết định |
| [`../Experiment/EDA-000_raw/NOTES.md`](../Experiment/EDA-000_raw/NOTES.md) | EDA-0→6: integrity, speaker, geometry, cross-modal, gender, language, domain shift |
| [`../docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md) | kế hoạch, các pipeline ứng viên, tiêu chí quyết định |
| [`../docs/DATA.md`](../docs/DATA.md) | format dữ liệu, score orientation, các cạm bẫy |
| [`../kaggle/README.md`](../kaggle/README.md) | chạy sweep GPU trên Kaggle T4×2 |
