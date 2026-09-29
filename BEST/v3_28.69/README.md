# Bản tốt nhất — FUSE-03 · CodaBench Overall **28.69** (nộp 2026-09-26)

> **Lịch sử: đã thay bởi SET-02 (v4_set02) → SET-13 (v5_set13) → pipeline Evaluation v6_eval.** Trạng thái hiện hành (29/09, nhật ký tới mục 75): cấu hình Evaluation là **SET-13 (CB 21.01) qua `BEST/v6_eval`**; xem [DECISIONS.md](../../docs/DECISIONS.md) và [OPEN_DIRECTIONS.md](../../docs/OPEN_DIRECTIONS.md).

Thứ hạng thay đổi liên tục; xem snapshot có ngày trong [../../README.md](../../README.md). Kế hoạch tiếp theo: [../../docs/PLAN_V4.md](../../docs/PLAN_V4.md).

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| FOP baseline của nhóm (000c) | 42.17 | 36.31 | 38.98 | 43.38 | 50.00 |
| Bản cũ trong `BEST/` (003c) | 31.34 | 25.60 | 27.88 | 34.01 | 37.87 |
| **FUSE-03 (bản này)** | **28.69** | **23.61** | **26.88** | **29.12** | **35.15** |
| Hạng 1 lúc nộp | 26.41 | 18.65 | 24.18 | 25.05 | 37.74 |

## 1. Tái tạo

```bash
cd BEST/v3_28.69
python build_best.py       # -> submission_best_28.69.zip, rank-corr 1.000000 với zip đã nộp ở cả 4 cell
```

Chỉ cần numpy / pandas / scipy và `../../kaggle_upload/dev/*_test.txt`. Không train, không cần GPU.
Toàn bộ công thức là dict `BEST_CONFIG` trong `build_best.py`; `check_policy` từ chối combiner lạ, score không đến
từ zip thành phần, hoặc quá 2 tham số fit trên dev (xem [PLAN_V4 §3](../../docs/PLAN_V4.md)).

## 2. Công thức: 4 hệ thành phần, ghép theo cell bằng rank

| Cell | Công thức | Vì sao |
|---|---|---|
| no_gender/English | mean(rank c007, rank c010B) | hai voice feature khác nhau (BTC-192, ECAPA-6144) sai ở trial khác nhau |
| gender/English | mean(rank c007, rank c010B) | như trên; g/En 30.96 → 29.12 |
| no_gender/Bangla | mean(rank c003, rank r2_mix) | hai hệ có cấu trúc lỗi khác nhau; r2_mix (speechbrain ECAPA-192) là expert bổ sung cho 003c ở Bangla |
| gender/Bangla | w(d)·rank c003 + (1−w(d))·rank r2_mix, w(d) = σ(0.5·ln d − 1.5) | **đổi trọng số thích nghi**, không phải mô hình độ dài: w(d) = 0.24 (2 s) → 0.33 (4.7 s) → 0.44 (12 s), tức nghiêng ~0.3/0.7 về r2_mix |

**Về cơ chế của r2_mix và gate.** r2_mix được train trên crop có phân phối độ dài giống Bangla, nhưng lợi ích
**không** đến từ việc chịu câu ngắn tốt hơn: ở g/Bn, r2_mix thua 003c ở câu 0–3 s (40.5 so với 36.7) và thắng lớn
ở câu dài (8–12 s: 28.3 so với 34.0; > 12 s: 21.1 so với 31.5). Gate lại cho 003c trọng số *tăng* theo độ dài,
ngược chiều với bằng chứng đó → chưa có bằng chứng cho bất kỳ cơ chế độ dài nào. Đối chứng
([gate_controls.py](../../Experiment/FUSE-03/gate_controls.py), log 41): OOF của trọng số tĩnh dao động −0.42 … +1.09
chỉ vì cách chọn lưới, gate đạt +1.25, độ dài bị xáo trộn +0.28 → gate và trọng số tĩnh không phân biệt được.
Điều chắc chắn: r2_mix là expert bổ sung, và CB xác nhận một lần g/Bn 36.10 → 35.15. Gate được **đóng băng**, không mở rộng.

Rank (không phải z-score) vì các hệ có thang điểm khác nhau và EER chỉ phụ thuộc thứ tự.
Mọi zip đều **lower = same** (đã kiểm chứng trên CodaBench, xem [../../docs/DATA.md](../../docs/DATA.md)).

### Các hệ thành phần (`components/`)

| File | Hệ | Feature | Recipe | Tái tạo từ đầu |
|---|---|---|---|---|
| `c003.zip` | EXP-003c | face VGG-4096 BTC + voice 192 BTC | InfoNCE, MLP 2 nhánh, 3 seed, 15 ep, + CCA k=4, per-file centering | [../reproduce.py](../reproduce.py) (CPU, ~3 phút) |
| `c007.zip` | EXP-007 | như trên | InfoNCE, MLP hid512/emb128/drop0.3, 40 ep, n=10, fusion CCA w=0.25 | `kaggle/FLAG_06_submit.ipynb` với `voice="given"` |
| `c010B.zip` | EXP-010B | face VGG BTC + **ECAPA-6144** (speechbrain, lớp ASP) | recipe 007 | `kaggle/FLAG_04_extract` → `FLAG_06_submit` với `voice="ecapa6144"` |
| `r2_mix.zip` | r2_ecapa192_mix | face VGG BTC + **speechbrain ECAPA-192** (`spkrec-ecapa-voxceleb`), train trên crop lấy mẫu độ dài theo dev Bangla | recipe 007, `voice_view="mix"` | `kaggle/FLAG_06_submit.ipynb` (RUNS mặc định) |

## 3. Cách các quyết định được chọn

- **SEL-01b — selector trên dev không nhãn** (nay là thư viện: `Experiment/SEL-01_selector/pseudo_eval.py`). Pseudo-identity dựng từ cụm khuôn mặt (ArcFace, ARI 0.991 trên train) và cụm giọng (ECAPA-192), nối qua việc xuất hiện chung trong trial. Kiểm định ngược trên 9 hệ có EER CodaBench thật: Spearman 0.95–0.996 mỗi cell; qua 3 lượt nộp, dự đoán lệch ≤ 0.9 EER mỗi cell. Code: [../../Experiment/SEL-01_selector/](../../Experiment/SEL-01_selector/).
- **Ứng viên cố định trước khi xem kết quả** (FUSE-01/02); tham số gate g/Bn chọn bằng 5 fold theo pseudo-identity, out-of-fold +1.25 EER, 4/5 fold thắng (FUSE-03).
- ⚠️ **Phải khai báo trong system description:** dev không nhãn được dùng (transductive) để chọn hệ thành phần và fit 2 tham số gate. Không có mô hình nào được train trên dev.

## 4. Đã thử và không dùng (sau bản 31.34)

| Hướng | Kết quả |
|---|---|
| Công thức hạng 1 FAME26 (linear + dropout .9 + AAM) | không cấu hình nào thắng MLP + InfoNCE |
| ArcFace (face, đã align) / ReDimNet2 (voice) | nhận dạng một modality tốt hơn rõ, nhưng ghép mặt–giọng **tệ hơn** — không phải lỗi kỹ thuật |
| Same-gender negatives, GRL gender, DANN language, score khớp giới | đều hại hoặc nằm trong nhiễu |
| Trọng số tĩnh tối ưu cho English | overfit (out-of-fold tệ hơn 0.5/0.5) |
| Làm mượt score bằng đồ thị kNN | English tệ hơn |
| + MAV-Celeb v3 (En/German) làm dữ liệu train | hại mọi cell: ánh xạ mặt–giọng phụ thuộc quần thể |

Đang chạy: MAV-Celeb v1 (Urdu) và v2 (Hindi) làm dữ liệu train ngoài (EXT-01, `kaggle/FLAG_08_external.ipynb`).
