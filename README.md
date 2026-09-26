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

## Trạng thái hiện tại (2026-09-26)

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| EXP-000c (baseline ban đầu của nhóm) | 42.17 | 36.31 | 38.98 | 43.38 | 50.00 |
| Organizer FOP reference | 36.92 | 32.54 | 38.12 | 32.99 | 44.01 |
| EXP-011 (ghép per-cell 3 hệ) | 30.13 | 23.81 | 27.88 | 30.96 | 37.87 |
| FUSE-01 (rank fusion English) | 29.62 | 23.61 | 27.88 | 29.12 | 37.87 |
| FUSE-02 (+ rank fusion Bangla) | 28.93 | 23.61 | 26.88 | 29.12 | 36.10 |
| **FUSE-03 — bản tốt nhất hiện tại** | **28.69** | **23.61** | **26.88** | **29.12** | **35.15** |

**−13.48 so với baseline nhóm. Hạng 5** (leaderboard 2026-09-26 10:00):

| # | Đội | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---|---:|---:|---:|---:|---:|
| 1 | pi-flag | 19.13 | 11.51 | 19.91 | 19.35 | 25.75 |
| 2 | jeetu_sk | 23.48 | 13.89 | 24.18 | 18.13 | 37.74 |
| 3 | asp0178 | 27.60 | 24.60 | 22.90 | 32.38 | 30.52 |
| 4 | Fight4Job | 28.50 | 23.02 | 25.60 | 27.49 | 37.87 |
| **5** | **FourFusion** | **28.69** | 23.61 | 26.88 | 29.12 | 35.15 |

`g/Bn` 35.15 tốt thứ ba bảng. Khoảng cách lớn nhất nằm ở English (ng/En −12.1, g/En −9.8 so với hạng 1).
Tái tạo + tài liệu: **[BEST/v3_28.69/](BEST/v3_28.69/)** (`python build_best.py`, khớp zip đã nộp rank-corr 1.000000).
Nhật ký chi tiết mọi quyết định: [Experiment/EXPERIMENT_LOG.md](Experiment/EXPERIMENT_LOG.md). Kế hoạch hiện hành: **[docs/PLAN_V4.md](docs/PLAN_V4.md)**.

**Hướng đi từ v4.** Encoder unimodal đã đủ mạnh: ArcFace và ReDimNet2 nhận dạng tốt hơn hẳn nhưng ghép mặt–giọng
không tốt hơn. Nút thắt là **cầu nối cross-modal** học từ 70 người, và cầu nối này phụ thuộc quần thể. Vì vậy chỉ còn
ba hướng: (A) dữ liệu cặp ngoài khớp quần thể (MAV-Celeb v1 Urdu / v2 Hindi), (B) học cầu nối tốt hơn nếu A có tín hiệu,
(C) fusion các hệ bổ sung, chọn bằng selector SEL-01b. Các hướng đã đóng: [PLAN_V4 §2](docs/PLAN_V4.md).

⚠️ **Phải khai báo trong system description:** dev không nhãn được dùng (transductive) để *chọn* model và fit 2 tham số fusion
(pseudo-label SEL-01b); nếu bản cuối dùng EXT-01 thì khai báo MAV-Celeb v1–v3 là dữ liệu train ngoài. Không train trên
pseudo-label, không sinh score từ cấu trúc danh sách trial ([PLAN_V4 §3](docs/PLAN_V4.md)).

### Đường đi

| EXP | Thay đổi | Overall |
|---|---|---:|
| 000c | FOP source-faithful (CE + α·OPL, gated fusion) | 42.17 |
| 000d | CCA tuyến tính k=32 — **không train deep** | 35.79 |
| 000f | Ghép per-protocol 2 bản CCA | 33.35 |
| 003c | InfoNCE ×3 seed + CCA + centering, bỏ AS-norm | 31.34 |
| 004 | + full CORAL dev→train | 41.36 ❌ |
| 007 | fusion có trọng số (CCA w=0.25), n=10 | 30.90 |
| 009 | công thức hạng 1 FAME26 (linear + dropout .9 + AAM) | ❌ không thắng |
| 010B | re-extract voice ECAPA 6144-d | 31.71 (nhưng ng/En 23.81) |
| 011 | ghép per-cell 3 nguồn | 30.13 |
| SEL-01b | **selector pseudo-identity trên dev** (ArcFace + ECAPA), sai số ≤ 0.9 EER/cell so với CB | – (hạ tầng) |
| FUSE-01 | English ← rank(007) + rank(010B) | 29.62 |
| FUSE-02 | Bangla ← rank(003c) + rank(r2_ecapa192_mix) | 28.93 |
| **FUSE-03** | **g/Bn ← gate 2 tham số (003c / r2_mix)** — thực chất đổi trọng số ~0.3/0.7, không phải hiệu ứng độ dài | **28.69** |
| NB-4 | ReDimNet2 voice (vox2 / đa ngôn ngữ) | ❌ nhận giọng tốt hơn, ghép mặt–giọng không tốt hơn |
| GRAPH-01 | làm mượt score bằng đồ thị kNN unimodal | ❌ English tệ hơn |
| EXT-01 | + MAV-Celeb v3 (En/German) | ❌ hại mọi cell; v1/v2 (Nam Á) đang chạy lại |

## Cấu trúc thư mục

```text
FLAG_2027_FourFusion/
├── README.md                     ← file này
├── BEST/
│   ├── v3_28.69/                 ← **bản tốt nhất**: build_best.py (BEST_CONFIG + check_policy), components/, README
│   ├── README.md, reproduce.py   (bản cũ 31.34, tái tạo bằng 1 file ~3 phút CPU)
│   └── submission_best.zip
├── kaggle/                       ← notebook GPU (Kaggle T4); xem kaggle/README.md
│   ├── flag_lib.py, flag_v2.py   (train / chấm, nhúng vào notebook)
│   ├── flag_extract.py           (trích feature: encoder BTC, ECAPA, ArcFace, SSL; load_audio, fetch_zip)
│   ├── build_*_notebook(s).py    (sinh notebook từ lib; FLAG_04…08)
│   └── NB3/ NB4/ NB5/ output/    (kết quả tải về từ Kaggle)
├── kaggle_upload/                ← feature CSV đã giải nén phẳng (train/ + dev/)
├── docs/
│   ├── PLAN_V4.md                ← **kế hoạch hiện hành**: 3 hướng mở, danh sách đóng, luật transductive
│   ├── PROBLEMS.md               (các problem PB-1…6 và trạng thái)
│   ├── PLAN_V3.md, PLAN_KAGGLE.md, RESEARCH_DIRECTIONS.md, RESEARCH_PLAN.md   (lịch sử)
│   └── DATA.md                   ← mô tả dữ liệu, format file, các lưu ý (score orientation...)
├── Input/                        ← dữ liệu gốc, KHÔNG sửa, KHÔNG commit
│   ├── train_set.zip             (1.4 GB: features CSV + ảnh/wav thô, 70 id, English)
│   ├── dev_set.zip               (990 MB: 4 protocol, features CSV + ảnh/wav thô)
│   └── meta_file_train_set.csv   (giới tính 70 speaker train: 40 m / 30 f)
└── Experiment/
    ├── EXPERIMENT_LOG.md         ← sổ ghi toàn bộ thí nghiệm (1 dòng / run, có kết quả CodaBench)
    ├── templates/EXPERIMENT_TEMPLATE.md
    ├── EDA-000_raw/              ← EDA-0→6 + eda_utils.py (hàm dùng chung cho mọi EXP)
    ├── EXP-000c_fop_baseline/    ← FOP baseline 42.17
    ├── EXP-000d_cca_linear/      ← CCA tuyến tính 35.79 / 34.46 / hybrid 33.35
    ├── EXP-002_cca_centering/    ← Pipeline E: centering + AS-norm
    ├── EXP-003_deep/             ← sweep loss A/B/C + ensemble → 31.96 / 31.34
    ├── EXP-004_domain/           ← CORAL (kết quả âm tính, 41.36)
    ├── EXP-005_tune/ … EXP-012_proxy/
    ├── SEL-01_selector/          ← **pseudo_eval.py**: chấm dev local (pseudo-EER, coverage, CI, A−B, fold OOF)
    └── FUSE-01 … FUSE-04/, GEN-01/, GRAPH-01/
```

Chấm một zip trên dev không tốn lượt nộp: `cd Experiment/SEL-01_selector && python pseudo_eval.py A.zip [B.zip]`.

## Quy ước làm việc

1. **Mỗi thí nghiệm = 1 thư mục** `Experiment/EXP-NNN_<slug>/` chứa: notebook/code, `NOTES.md` (từ template), `out/` (submission zip, checkpoint nhỏ, metadata json).
2. **Ghi vào `Experiment/EXPERIMENT_LOG.md` ngay khi có số** — trước khi làm thí nghiệm tiếp theo.
3. **Model selection hai tầng.** Internal speaker-disjoint (56/14, English-only, ≥ 3 split) cho quyết định representation/cfg; **mọi quyết định đụng dev/Bangla dùng `pseudo_eval.py` (SEL-01b)** — internal đã chọn sai 5/5 lần ở Bangla. Báo cáo pseudo-EER 4 cell + coverage + CI bootstrap theo identity; kiểm chéo nhãn `--labels=btc`. CodaBench là xác nhận cuối; score normalization chỉ validate bằng CodaBench (internal sai dấu với AS-norm).
4. **Score orientation**: CodaBench chấm *lower = same speaker* → nộp **+d²** (hoặc `−cosine`). Đã kiểm chứng: −d² cho 57.83, +d² cho 42.17. Local sklearn dùng dấu ngược. Xem [docs/DATA.md](docs/DATA.md).
7. **Submission cuối = ghép per-cell tốt nhất**; mỗi EXP chỉ cần thắng ở ít nhất một cell và phải báo cáo đủ 4 cell, không so overall.
5. Đổi **một biến / thí nghiệm** so với baseline gần nhất, luôn ghi rõ `parent` experiment.
6. `FAST_DEV_RUN=True` chỉ để test code, không bao giờ nộp.
