# Kế hoạch v5 — sau SET-02 (21.68, hạng 2)

> **Lịch sử (26–27/09, sau SET-02). Các việc gom cụm / fusion ở đây đã xong hoặc đóng: gom cụm đã gần trần (73), SetProto đóng (74). Kế hoạch hiện hành: PLAN_EVAL.** Trạng thái hiện hành (29/09, nhật ký tới mục 75): cấu hình Evaluation là **SET-13 (CB 21.01) qua `BEST/v6_eval`**; xem [DECISIONS.md](DECISIONS.md) và [OPEN_DIRECTIONS.md](OPEN_DIRECTIONS.md).

Lập 2026-09-26. Thay [PLAN_V4.md](PLAN_V4.md) (viết trước SET-02; giữ làm lịch sử). Nền kỹ thuật: [report_SET02/report_set02.pdf](report_SET02/report_set02.pdf).
Số liệu: [../Experiment/EXPERIMENT_LOG.md](../Experiment/EXPERIMENT_LOG.md), EDA mới: [../Experiment/EDA-002_roles/](../Experiment/EDA-002_roles/).

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| FUSE-03 | 28.69 | 23.61 | 26.88 | 29.12 | 35.15 |
| **SET-02 (hiện hành)** | **21.68** | **12.50** | **21.34** | 28.31 | **24.59** |

## 0. Hệ thống bây giờ gồm ba khâu

```text
ảnh mặt của file ──► [1] gom cụm mặt ──┐
                                        ├──► [3] cầu nối chấm cụm × cụm ──► điểm trial
đoạn giọng của file ─► [2] gom cụm giọng┘
```

Mô hình nhiễu (báo cáo SET-02, §4): phương sai điểm = σ_p² (cầu nối sai ở mức người) + σ_f²/n_f + σ_v²/n_v.
Khâu [1][2] quyết định n và độ sạch của cụm; khâu [3] quyết định σ_p². **Mỗi khâu có tiêu chí đo riêng và embedding riêng.**

## 1. EDA-002: embedding nào cho khâu nào (nhãn thật, giữ ra 30 người, 3 split)

**Khâu [1] gom cụm mặt — tiêu chí: ARI**

| Embedding | ARI | Số cụm (30 người) | Số cụm trên 4 file dev (~30 người) |
|---|---:|---:|---|
| **ArcFace** | **0.965** | 38 | 29 / 33 / 32 / 29 |
| VGGFace (BTC) | 0.754 | 71 | 46–51 |

**Khâu [2] gom cụm giọng — tiêu chí: ARI, và phải bền với Bangla**

| Embedding | ARI | Số cụm (30 người) | Dev: ng/En · ng/Bn · g/En · g/Bn |
|---|---:|---:|---|
| **ReDimNet2 vox2** | **0.947** | 47 | 38 · 41 · 35 · 34 |
| **ReDimNet2 đa ngôn ngữ** | **0.945** | 43 | 33 · **29** · 34 · **28** |
| ECAPA-192 tự trích (đang dùng) | 0.933 | 54 | 41 · 39 · 40 · 41 |
| ECAPA-192 BTC | 0.928 | 53 | 41 · 38 · 41 · 37 |
| WavLM L4–9 / XLS-R L4–6 / ECAPA-6144 | 0.70 / 0.60 / 0.57 | 68–94 | – |

→ ReDimNet2 đa ngôn ngữ cho **số cụm Bangla sát số người nhất** (28–29 so với ~30; ECAPA hiện tại tách thành 39–41). Đây đúng là điểm yếu SET-04 lộ ra ở Bangla.

**Khâu [3] cầu nối — tiêu chí: EER sau khi gộp theo người (không phải EER từng mẫu)**

| Face + voice | Mẫu ng / g | **Người ng / g** |
|---|---:|---:|
| **VGG + BTC-192** | 31.8 / 39.6 | **23.9 / 32.1** |
| **VGG + ECAPA-6144** | 30.0 / 38.5 | **24.4 / 32.1** |
| VGG + ECAPA-192 tự trích | 32.9 / 40.7 | 25.7 / 35.0 |
| VGG + WavLM | 30.3 / 38.1 | 26.3 / 34.5 |
| VGG + ReDimNet2 đa ngôn ngữ | 31.3 / 38.5 | 26.9 / 34.8 |
| ArcFace + bất kỳ | 37.6–39.0 / 42.4–44.2 | 33.3–36.7 / 38.0–40.6 |

→ Cầu nối vẫn là **VGG + voice của BTC / 6144**. ArcFace và ReDimNet vẫn kém *ở vai trò cầu nối*, kể cả ở mức người.

**Kết luận EDA:** encoder "nhận dạng giỏi" (ArcFace, ReDimNet) và encoder "ghép chéo giỏi" (VGG, ECAPA-BTC) là **hai nhóm khác nhau**. Trước SET-02, ta chỉ có vai trò cầu nối nên đã loại nhóm thứ nhất; giờ nhóm đó có chỗ đứng ở khâu gom cụm.

## 2. Nguyên tắc

1. **Tách vai trò.** Mỗi embedding được chọn theo tiêu chí của khâu nó phục vụ: ARI cho gom cụm, EER mức người cho cầu nối. Không dùng EER từng mẫu để loại một encoder nữa.
2. **Quyết định bằng nhãn thật.** Mọi thay đổi liên quan tới gom cụm được chọn trên: (a) v4 giữ ra 30 người (English); (b) **v1/v2 giữ ra theo người, phần Urdu/Hindi** — nhãn thật cho ngôn ngữ không phải English, gần Bangla nhất ta có. Pseudo-label SEL-01b chỉ để kiểm tra chéo (nó thiên vị hệ có gộp cụm: dự đoán g/En 20.2, thực tế 28.3).
3. **Mật độ giống dev.** Validation dùng khoảng 30 người mỗi "file", 50% target.
4. **Ngưỡng theo ngôn ngữ.** Ngưỡng hiệu chỉnh trên English không tự đúng cho Bangla (SET-04: ngưỡng 0.8 làm g/Bn tệ đi 4–6). Hiệu chỉnh cho ngôn ngữ không-English trên Urdu/Hindi.
5. **Một biến mỗi lần; một lượt CB cho mỗi thay đổi đã qua validation.** Progress phase còn khoảng 130 lượt; Evaluation phase chỉ 15.
6. **Luật và khai báo.** Không dùng danh sách trial để tính điểm, không train trên dev. Gom cụm không giám sát trong file test là transductive → ghi rõ trong system description; đã hỏi BTC ([email_organizers.md](email_organizers.md), câu 2). Luôn giữ một bản "sạch" (không gộp cụm) làm dự phòng.
7. **Kiểm tra luật cho encoder mới.** ReDimNet2 đa ngôn ngữ train trên VoxBlink2 + CN-Celeb; nếu các tập này có tiếng Bengali thì có thể vướng luật "không pretrain trên ngôn ngữ unheard" (câu 8 trong email). Trước khi có trả lời: dùng **ReDimNet2 vox2** (VoxCeleb2, ARI 0.947) làm bản chính, bản đa ngôn ngữ chỉ để so sánh.

## 3. Thí nghiệm, theo thứ tự

| # | Tên | Làm gì | Validation (nhãn thật) | Cổng qua | Chi phí |
|---|---|---|---|---|---|
| 1 | **SET-05 gom cụm giọng bằng ReDimNet2** | Thay ECAPA-192 bằng ReDimNet2 vox2 ở khâu [2]; ngưỡng hiệu chỉnh English trên v4, không-English trên Urdu/Hindi | v4 giữ ra 30 + v1/v2 Urdu/Hindi giữ ra | EER mức file tốt hơn ≥ 0.5 ở cả hai | CPU < 1 giờ, 1 lượt CB |
| 2 | **SET-06 ràng buộc một-một (optimal transport)** | Trong một file, mỗi cụm mặt ứng với nhiều nhất một cụm giọng. Chuẩn hoá ma trận cụm × cụm bằng Sinkhorn trước khi chấm. Chỉ dùng cụm + điểm cầu nối, không dùng danh sách trial | như #1 | như #1 | CPU < 1 giờ |
| 3 | **Cầu nối học ở mức người** | Train cầu nối với embedding trung bình theo người/video (prototype InfoNCE), khớp cách chấm lúc test | v4 giữ ra (EER mức người) | person-EER tốt hơn ≥ 0.5 | CPU vài giờ |
| 4 | **EXT-01 + gộp cụm** | Dữ liệu ngoài v1/v2 cho khâu [3] (giảm σ_p²), đánh giá ở mức người | EXT-02 (v1/v2 song ngữ) + v4 giữ ra | person-EER tốt hơn và không hại English | Đang trích v2 |
| 5 | Fusion sau gộp | Chọn cặp hệ cho từng cell sau khi #1–#4 ổn định | v4 giữ ra | ≥ 0.5 | nhẹ |

**Trạng thái 27/09** (chi tiết: nhật ký 48–51 trong [../Experiment/EXPERIMENT_LOG.md](../Experiment/EXPERIMENT_LOG.md)):

- #1 SET-05: đóng. Chênh ≤ 0.3, giữ ECAPA-192.
- #2 SET-06: đóng. Không ổn định giữa các split.
- #3 + #4 EXT-03 (B3P = v1 + v2 + SetProto): qua cổng trên nhãn thật (v4 +3.45 / +2.83, Urdu/Hindi +4.9). SET-07 (đưa vào s007 train đủ 70 người) được pseudo-arc dev chấm 22.12 so với 21.03. Nhưng CI của chênh lệch là [−3.7, +3.2], tức hòa; cần CB để phân xử.
- ARCH-01 (dạng cầu nối): không dạng nào trội. Trộn thêm CCA-16 (F3) cho +1.5…+3.6 ở ng trên nhãn thật, hòa ở g → SET-08, nhưng CB ra **22.30** (tệ hơn 0.62, g/Bn +2.25), đúng như pseudo-arc dự đoán.
- **Nguyên tắc mới (nhật ký 53–54):**
  - Ở progress phase, cổng cuối trước khi nộp là pseudo-arc trên dev, chỉ chọn giữa ít hệ đăng ký trước. Proxy nhãn thật (v4 giữ ra, Urdu/Hindi) dùng để sàng lọc.
  - Pseudo-arc được dựng từ cấu trúc danh sách trial, và LEAK-01 cho thấy nó gần bằng nhãn thật. Vì vậy **ở Evaluation phase không dùng nhãn giả để chọn hay trộn hệ**: cấu hình phải chốt từ dev trước ngày 11/11.
- SET-07 trên CB: 22.20 (ng/Bn −4.13, g/Bn +4.02). HYB-01 (ghép per-cell ba bản đã nộp) = 20.40 chắc chắn, chỉ dùng cho bảng progress.
- LANG-01 (chiếu bỏ hướng ngôn ngữ): hòa → đóng.
- NOISE-01: nhãn sạch. CEIL: phần dư của gom cụm < 1 EER.

**Ghi chú về #2.** Đây là giả định cấu trúc mạnh ("một người = một giọng trong file"). Nó đúng về bản chất dữ liệu (mỗi người chỉ có một giọng), không phải khai thác cách dựng danh sách trial. Nhưng nó cũng là giả định transductive mạnh nhất cho tới giờ; phải khai báo, và phải có validation nhãn thật trước khi nộp.

**Không làm:** thêm encoder mới cho vai trò cầu nối; tinh chỉnh kiến trúc/loss ở mức mẫu; các phép chuẩn hoá phân phối điểm (AS-norm, CORAL, CSLS).

## 4. Việc cần song song

- Gửi email BTC ([email_organizers.md](email_organizers.md)).
- Hoàn tất trích v2 (đang chạy, local).
- Chuẩn bị pipeline cho Evaluation phase: dữ liệu mới → trích ArcFace + ReDimNet2 + VGG + ECAPA-BTC → gom cụm → cầu nối → gộp → zip. Chạy thử toàn bộ trên dev trước ngày 11/11.
