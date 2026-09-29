# FLAG 2027 — FourFusion

Workspace nghiên cứu cho **FLAG 2027 (Face-voice Association in multilingual environments)** trên MAV-Celeb v4.
Bài toán: cho một cặp (face, voice) → chấm điểm "có cùng một người không". Metric: **EER**, lấy trung bình 4 protocol.

| Protocol | Ngôn ngữ | Số pair (dev) |
|---|---|---:|
| `no_gender` | English (heard) | 1008 |
| `no_gender` | Bangla (unheard) | 1406 |
| `gender` | English (heard) | 982 |
| `gender` | Bangla (unheard) | 1468 |

`gender` = negative bị ép cùng giới tính với positive → không thể dùng giới tính làm shortcut.

## Đọc gì trước (thứ tự nguồn)

1. **[Experiment/EXPERIMENT_LOG.md](Experiment/EXPERIMENT_LOG.md)**: bằng chứng gốc, tới mục (75), ngày 29/09.
2. **[docs/DECISIONS.md](docs/DECISIONS.md)**: mọi quyết định, bằng chứng, trạng thái, những gì đã bị đảo ngược. Đồng bộ với nhật ký.
3. **[docs/OPEN_DIRECTIONS.md](docs/OPEN_DIRECTIONS.md)**: hướng còn mở, luật validation chung, danh sách đã đóng.
4. **[docs/PLAN_EVAL.md](docs/PLAN_EVAL.md)**: chuẩn bị Evaluation phase (11–18/11).

Các kế hoạch cũ (PLAN_V3 … V6, RESEARCH_*, các báo cáo deep research) là hồ sơ lịch sử; mỗi file có ghi chú ở đầu về những gì đã lỗi thời.

## Trạng thái hiện tại (2026-09-29)

**Cấu hình Evaluation: SET-13 qua [BEST/v6_eval/](BEST/v6_eval/README.md)** (`--mode auto`, 15 mạng mỗi thành phần).
- Chọn bằng validation độc lập: v4 giữ ra, mức cụm −6.8 / −6.5 (5/5); VAL-70, mức người −7.4 (10/10 file).
- Quy trình từ ảnh / giọng gốc → [kaggle/FLAG_12_features](kaggle/FLAG_12_features.ipynb) → pipeline → zip đã chạy thử trọn: rank-corr 1.0000 với bản đã nộp.

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| EXP-000c (baseline ban đầu của nhóm) | 42.17 | 36.31 | 38.98 | 43.38 | 50.00 |
| Organizer FOP reference | 36.92 | 32.54 | 38.12 | 32.99 | 44.01 |
| EXP-011 (ghép per-cell 3 hệ) | 30.13 | 23.81 | 27.88 | 30.96 | 37.87 |
| FUSE-03 (trộn theo hạng) | 28.69 | 23.61 | 26.88 | 29.12 | 35.15 |
| SET-02 (gom cụm lúc test) | 21.68 | 12.50 | 21.34 | 28.31 | 24.59 |
| SET-08 (s007 + CCA-16) | 22.30 | 12.30 | 21.34 | 28.72 | 26.84 |
| SET-07 (s007 + dữ liệu ngoài v1/v2) | 22.20 | 15.48 | 17.21 | 27.49 | 28.61 |
| SET-09 (s007 + stream FaRL) | 22.49 | 14.19 | 20.34 | 28.72 | 26.70 |
| SET-12 (s007 + ImageBind) | 21.45 | 11.64 | 21.05 | 23.01 | 30.11 |
| **SET-13 (s007 + ImageBind + tuổi) — cấu hình Evaluation** | **21.01** | **11.31** | 20.91 | 23.22 | 28.61 |
| [INDEP-01](Experiment/INDEP-01/README.md) (SET-13 chấm từng cặp: không thống kê file, không gom cụm; đo năng lực mô hình) | 26.70 | 19.05 | 26.32 | 27.09 | 34.33 |
| HYB-01 (ghép cell SET-08 / 07 / 07 / 02, chỉ cho bảng progress) | 20.40 | 12.30 | 17.21 | 27.49 | 24.59 |
| HYB-03 (ghép cell SET-13 / 07 / 12 / 02, chỉ cho bảng progress) | 19.03 (dự kiến, chưa nộp) | 11.31 | 17.21 | 23.01 | 24.59 |

**Dev CodaBench không phân xử được chênh lệch cỡ ±1 EER** (chỉ đổi seed đã làm dev dịch 0.65–1.32; nhật ký 61). Bản ghép cell theo CB chỉ dùng cho bảng progress; mọi quyết định cho Evaluation dựa vào validation nhãn thật.

**Bảng xếp hạng 27/09 18:25** (lần chụp gần nhất):

| # | Đội | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---|---:|---:|---:|---:|---:|
| 1 | susupro | 3.22 | 3.77 | 1.99 | 5.09 | 2.04 |
| 2 | Fight4Job | 4.60 | 5.16 | 2.70 | 5.50 | 5.04 |
| 3 | pi-flag | 19.13 | 11.51 | 19.91 | 19.35 | 25.75 |
| **4** | **FourFusion (nguyennhan2006, HYB-01)** | **20.40** | 12.30 | **17.21** | 27.49 | **24.59** |
| 5 | 24521254 | 22.92 | 15.28 | 23.61 | 20.37 | 32.43 |
| 6 | jeetu_sk | 23.48 | 13.89 | 24.18 | 18.13 | 37.74 |
| 24 | mmosc (= baseline FOP của BTC) | 36.92 | 32.54 | 38.12 | 32.99 | 44.01 |

- **Hai đội đầu (3.22, 4.60) thấp hơn trần oracle của ghép mặt–giọng.** Mức đó chỉ đạt được bằng cấu trúc danh sách trial: LEAK-01 cho 2.10 / 2.50 chỉ bằng cách đếm trial nối cụm. Ta không đi theo hướng này.
- **g/En đã từ 27.49 xuống 23.22 nhờ ImageBind (SET-13).** Cell tệ nhất giờ là **g/Bn (28.61)**.

## Những gì đã học (tóm tắt, chi tiết trong DECISIONS)

- **Bốn bước nhảy:**
  - CCA / InfoNCE thay FOP: 42.17 → 31.34;
  - trộn theo hạng: → 28.69;
  - gom cụm lúc test: → 21.68;
  - ImageBind + tuổi: → 21.01. Trên validation độc lập lợi lớn hơn nhiều (−5…−9).
- **Gom cụm an toàn và gần trần.** STRESS-01 (280 file mô phỏng có nhãn): gom cụm tốt hơn ở mọi dạng file đã thử, kể cả Urdu / Hindi, 150 người mỗi file, 1–2 mẫu mỗi người. Chỉ cách cụm theo danh tính thật 0.1–2 EER.
- **Đã đóng:**
  - encoder nhận dạng mạnh cho cầu nối;
  - cầu nối lớn hơn;
  - CORAL / AS-norm;
  - SetProto;
  - SWA;
  - chỉnh ImageBind (adapter, CCA; LoRA hoãn);
  - dùng danh sách trial.
- **Còn mở:**
  - v1/v2 cho các cell English (O1);
  - g/Bn (O2);
  - Speech2Face (O3);
  - VoxCeleb chỉ khi có lý do mới (O4).

  Xem [docs/OPEN_DIRECTIONS.md](docs/OPEN_DIRECTIONS.md).

⚠️ **Phải khai báo trong system description:**
- gom cụm không giám sát ảnh / giọng trong từng file test;
- trừ mean theo file;
- mọi encoder kèm licence (ImageBind CC-BY-NC 4.0, audeering CC-BY-NC-SA 4.0);
- dữ liệu ngoài nếu dùng;
- ở progress phase đã dùng pseudo-label trên dev để chọn hệ (FUSE-01…03).

Không train trên pseudo-label, không sinh điểm từ cấu trúc danh sách trial.

### Đường đi

| EXP | Thay đổi | Overall |
|---|---|---:|
| 000c | FOP source-faithful (CE + α·OPL, gated fusion) | 42.17 |
| 000d | CCA tuyến tính k=32 — **không train deep** | 35.79 |
| 000f | Ghép per-protocol 2 bản CCA | 33.35 |
| 003c | InfoNCE ×3 seed + CCA + centering, bỏ AS-norm | 31.34 |
| 004 | + full CORAL dev→train | 41.36 ❌ |
| 007 | fusion có trọng số (CCA w=0.25), n=10 | 30.90 |
| 010B | re-extract voice ECAPA 6144-d | 31.71 (nhưng ng/En 23.81) |
| 011 | ghép per-cell 3 nguồn | 30.13 |
| FUSE-01 / 02 / 03 | trộn theo hạng từng cell (+ gate 2 tham số ở g/Bn) | 29.62 / 28.93 / **28.69** |
| SET-02 | gom cụm mặt (ArcFace) / giọng (ECAPA-192) trong từng file, trung bình khối | **21.68** |
| SET-12 | + ImageBind zero-shot vào s007 | 21.45 |
| **SET-13** | + so khớp tuổi (ViT-age / w2v2 age) | **21.01** |
| INDEP-01 | SET-13 chấm từng cặp (không phụ thuộc test) | 26.70 |

## Cấu trúc thư mục

```text
FLAG_2027_FourFusion/
├── README.md                     ← file này
├── BEST/
│   ├── v6_eval/                  ← **pipeline Evaluation**: SET-13 / noclus / INDEP-01, chẩn đoán cụm, 45 mạng đã cache (_members/)
│   ├── v5_set13/                 (SET-13, 5 mạng; tài liệu công thức + khai báo)
│   ├── v4_set02/                 (SET-02)
│   └── v3_28.69/                 (FUSE-03)
├── kaggle/                       ← notebook GPU (Kaggle T4); xem kaggle/README.md
│   ├── FLAG_12_features.ipynb    ← **trích mọi đặc trưng cho Evaluation**, code đầy đủ trong cell (build_eval_notebooks.py)
│   ├── run_notebook_local.py     (chạy thử notebook trên máy; --jupyter giả lập Jupyter)
│   ├── flag_lib.py, flag_v2.py   (train / chấm)
│   ├── flag_extract.py, flag_models.py, agegender.py   (trích feature cho các notebook cũ)
│   └── output/                   (feats_v2, feats_models, feats_layers, feats_ext_full, feats_eval: tải về từ Kaggle)
├── kaggle_upload/                ← feature CSV của BTC đã giải nén phẳng (train/ + dev/)
├── docs/
│   ├── DECISIONS.md              ← **quyết định + trạng thái** (đồng bộ nhật ký)
│   ├── OPEN_DIRECTIONS.md        ← **hướng còn mở, luật validation, danh sách đã đóng**
│   ├── PLAN_EVAL.md              ← **chuẩn bị Evaluation**
│   ├── PLAN_MODELS.md            (kế hoạch model 27/09, có bảng kết quả ở đầu)
│   ├── DATA.md                   (dữ liệu, format file, chiều điểm)
│   ├── email_organizers.md       (câu hỏi gửi BTC)
│   └── PLAN_V3…V6, RESEARCH_*, PROBLEMS, PLAN_KAGGLE, deep-research-report_*   (lịch sử)
├── Input/                        ← dữ liệu gốc, KHÔNG sửa, KHÔNG commit
└── Experiment/
    ├── EXPERIMENT_LOG.md         ← sổ ghi toàn bộ thí nghiệm
    ├── SEL-01_selector/          (pseudo_eval.py: chấm dev local; từ 28/09 chỉ để cảnh báo)
    ├── MODEL-01/, VAL-70/        ← hai harness validation nhãn thật (v4 giữ ra; người v1/v2)
    ├── STRESS-01/                ← gom cụm có hỏng khi file Evaluation khác dev không
    ├── EXTSCALE-01/              ← khung validation fold người v1/v2, chấm bản production ở 3 mức
    ├── INDEP-01/, IB-ADAPT/, IB-PROBE/, SWA-01/, …
    └── EXP-000c … FUSE-04, SET-01 … SET-07, …   (các thí nghiệm trước)
```

## Quy ước làm việc

1. **Mỗi thí nghiệm = 1 thư mục** `Experiment/<ID>/` chứa code, `README.md` hoặc `NOTES.md` (thiết kế, kết quả, kết luận), và file kết quả.
2. **Ghi vào `Experiment/EXPERIMENT_LOG.md` ngay khi có số**, rồi cập nhật [docs/DECISIONS.md](docs/DECISIONS.md) nếu quyết định thay đổi.
3. **Chọn hệ bằng validation nhãn thật**:
   - v4 giữ ra (5 split) và người v1/v2 giữ ra theo fold, với cặp dương khác video;
   - so sánh ghép cặp, cùng số bước tối ưu, đăng ký nhánh trước;
   - **thước đo chính là mức người / mức cụm của bản production.**

   Khung sẵn: [Experiment/EXTSCALE-01/run.py](Experiment/EXTSCALE-01/run.py). Dev CodaBench chỉ để phát hiện hỏng nặng; pseudo-label chỉ để cảnh báo.
4. **Chiều điểm**: CodaBench chấm *lower = same* → nộp **−score** (tức +d²). Đã kiểm chứng: −d² cho 57.83, +d² cho 42.17. Ở Evaluation, nếu lượt đầu ra EER > 50 thì đảo dấu ngay. Xem [docs/DATA.md](docs/DATA.md).
5. **Một biến mỗi thí nghiệm**, ghi rõ đối chứng.
6. **Notebook Kaggle mới:** code đầy đủ trong cell; chạy thử bằng `kaggle/run_notebook_local.py --jupyter` với đúng phiên bản transformers của Kaggle trước khi giao.
7. **Chỉ nộp từ tài khoản của đội** (nguyennhan2006); hai người cộng tác là đội độc lập, không dùng chung feature, bài nộp hay checkpoint.
