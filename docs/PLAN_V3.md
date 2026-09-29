# Kế hoạch v3 — hợp nhất từ PROBLEMS.md và bản đề xuất bên ngoài

> **Lịch sử (26/09). Đã thay bởi PLAN_V4 → PLAN_V5 → PLAN_MODELS / PLAN_EVAL.** Trạng thái hiện hành (29/09, nhật ký tới mục 75): cấu hình Evaluation là **SET-13 (CB 21.01) qua `BEST/v6_eval`**; xem [DECISIONS.md](DECISIONS.md) và [OPEN_DIRECTIONS.md](OPEN_DIRECTIONS.md).

> **Đã thay bằng [PLAN_V4.md](PLAN_V4.md) (2026-09-26).** Giữ lại làm lịch sử; các mục ATTR / SSL-01 / GRAPH-01 / duration gate
> bên dưới đã được xử lý hoặc chuyển sang PLAN_V4.

Lập 2026-09-25. Nền: [PROBLEMS.md](PROBLEMS.md). Mốc: **30.13** (EXP-011).

## 0. Tiến độ (cập nhật 2026-09-26) — mốc hiện tại **28.69** (FUSE-03)

| # | Việc | Trạng thái | Kết quả |
|---|---|---|---|
| 1 | SEL-01 selector cho Bangla | ✅ Xong | SEL-01b (ArcFace + ECAPA-192) sai số ≤ 0.9 EER/cell qua 3 lượt nộp → hạ tầng chấm local |
| 2 | FUSE-01 rank fusion English | ✅ Nộp | 29.62 (g/En −1.84) |
| 3 | FUSE-02 rank fusion Bangla (003c + r2_mix) | ✅ Nộp | 28.93 |
| 4 | FUSE-03 trọng số tĩnh / gate độ dài (OOF theo identity) | ✅ Nộp | 28.69; trọng số tĩnh English overfit → giữ 0.5/0.5; giả thuyết "crop giúp câu ngắn" bị bác |
| 5 | GEN-01 score khớp giới tính | ❌ Đóng | dao động trong nhiễu |
| 6 | NB-4 ReDimNet2 (vox2 / đa ngôn ngữ) | ❌ Đóng | nhận giọng tốt hơn, ghép mặt–giọng không tốt hơn |
| 7 | GRAPH-01 làm mượt đồ thị | ❌ Đóng | mức fusion: English tệ hơn, ng/Bn +0.5 |
| 8 | EXT-01 dữ liệu ngoài MAV-Celeb | ◐ Đang chạy | v3 (châu Âu) hại mọi cell; v1 (Urdu) / v2 (Hindi) chạy lại sau khi sửa resample + quota Drive |
| – | ATTR (tuổi, pitch) | ☐ Chưa làm | |

**Kết luận xuyên suốt:** encoder mạnh hơn không giúp (ArcFace, ReDimNet2) → nút thắt là cầu nối mặt–giọng học từ 70 người;
ánh xạ này phụ thuộc quần thể (v3 châu Âu không chuyển sang v4).

## 1. SEL-01a đã chạy — selector cho Bangla qua được gate

Script: [../Experiment/SEL-01_selector/run.py](../Experiment/SEL-01_selector/run.py). Không train gì, không tốn lượt nộp.

**Cách làm.** Pseudo-identity chỉ dựng từ cấu trúc unimodal và từ việc face–voice xuất hiện cùng nhau trong trial, không dùng score cross-modal của hệ nào:
- gom cụm face trên cả 4 file dev (English và Bangla là cùng người);
- gom cụm voice theo từng ngôn ngữ;
- nối mỗi cụm voice với cụm face mà nó bị ghép cặp nhiều hơn hẳn mức ngẫu nhiên (mỗi file đúng 50% target).

Kết quả gán nhãn được 64–68% trial mỗi file, tỉ lệ positive 0.48–0.52 (khớp với 50% thật). Sau đó tính pseudo-EER cho từng zip đã nộp.

**Kiểm định ngược trên 8 hệ có EER CodaBench thật** (000c, 000d, 000e, 003b, 003c, 004, 007, 010B):

| Cell | Spearman | Cặp đúng thứ tự | 4 hệ sát nhau (003b/003c/007/010B) |
|---|---:|---:|---:|
| ng/En | 0.862 | 24/28 | 3/6 |
| **ng/Bn** | **0.970** | **26/28** | 5/6 |
| g/En | 1.000 | 28/28 | 6/6 |
| **g/Bn** | **0.976** | **27/28** | 5/6 |

- **Gate (|ρ| ≥ 0.6 và ≥ 75% cặp đúng) đạt ở cả 4 cell.** Selector chọn đúng ở đúng những chỗ mà internal English đã dẫn sai: 010B tệ hơn 003c ở Bangla, 003c tốt nhất ở cả hai cell Bangla.
- **Giới hạn** (bootstrap theo cụm face, [selector_pairs.csv](../Experiment/SEL-01_selector/selector_pairs.csv)): khoảng tin cậy 95% của hiệu hai hệ rộng khoảng ±3–5 EER. Nghĩa là selector **sắp hạng tốt** khi chênh lệch thật ≥ 1–2 EER, nhưng **không phân xử được** các hệ chênh < 1 EER (ví dụ 003b so với 003c ở ng/Bn: CB chênh 1.0, selector nhầm chiều). Chỉ có 30 identity nên đây là giới hạn cứng.
- ⇒ Vai trò của nó: **bộ lọc loại sớm** các thay đổi hại Bangla trước khi tốn lượt nộp. Nó **không** thay được CodaBench cho các quyết định sát nút.

### 1b. SEL-01b (ArcFace + ECAPA-192 tự trích làm pseudo-label) — thay thế SEL-01a

| Cell | Spearman (9 hệ) | Sai số tuyệt đối trung bình / lớn nhất so với CB |
|---|---:|---:|
| ng/En | 0.971 | 0.90 / 1.20 |
| **ng/Bn** | **0.996** | **0.53 / 1.07** |
| g/En | 0.967 | 0.90 / 1.60 |
| **g/Bn** | **0.954** | **0.32 / 0.60** |

Mọi cặp chênh ≥ 1 EER ở Bangla đều xếp đúng. Chạy: `SEL_LABELS=arc python Experiment/SEL-01_selector/run.py name=path.zip`.
**Quy tắc dùng:** (1) chỉ để chọn model, không train trên pseudo-label; (2) mỗi vòng chỉ chọn trong ít ứng viên, và thắng phải ≥ 1 EER mới coi là thật, vì chọn nhiều trên dev là overfit dev; (3) khai báo trong system description; (4) CB vẫn là xác nhận cuối.

## 2. Nhận xét về bản đề xuất bên ngoài

| Đề xuất | Quyết định | Lý do |
|---|---|---|
| Đóng AAM, fusion mới, ArcFace/AdaFace, CORAL, AS-norm, GRL, same-gender, blind DANN | ✅ Đồng ý | Khớp PROBLEMS.md §1 |
| PB-1 lên đầu, kiểm định ngược bắt buộc | ✅ Đã làm (§1) | |
| Selector A (DEV / IWCV trên pair feature) | ⏸ Hạ ưu tiên | (1) Ngôn ngữ tách được 88% (EDA-5), nên domain classifier gần như tách hoàn toàn và trọng số dồn vào vài trial (không có vùng chồng lấn giữa hai miền); (2) cần embedding của từng hệ cũ, mà các hệ cũ chỉ còn zip, nên không kiểm định ngược được. Selector pseudo-identity đã đạt gate |
| SND | ⏸ Hạ ưu tiên | Cũng cần embedding của hệ cũ để kiểm định ngược |
| Paired bootstrap | ✅ Có sửa | Phải resample **theo speaker / cụm**, không theo trial: trial của cùng một người không độc lập, resample theo trial cho CI hẹp giả tạo. §1 đã làm theo cụm |
| GRAPH-01 (lan truyền qua đồ thị ArcFace/ECAPA) | ✅ Giữ, kèm cảnh báo | **Vòng lặp:** nếu graph dùng cùng cấu trúc láng giềng unimodal với pseudo-label thì selector sẽ thưởng nó một cách giả tạo. GRAPH chỉ được chấm bằng CodaBench, hoặc bằng pseudo-label dựng từ encoder khác với encoder của graph |
| SSL-01 (scalar mix trên tất cả layer) | ✅ Giữ, **không cần trích lại** | Cache hiện có đã lưu mean từng layer (N×25×1024), nên scalar mix trên mean train được ngay. Muốn thêm std mới phải trích lại. Kỳ vọng vừa phải: WavLM unimodal EER 11.2 so với ECAPA 4.3; kết quả Microsoft có train back-end trên hàng nghìn speaker. **Chờ `r2_wavlm`** |
| ATTR (score thuộc tính mềm, EDA trước khi fusion) | ✅ Giữ | Tuổi/giới của mặt: model `genderage` có sẵn trong `buffalo_l` (đã tải). Pitch: F0 từ wav. Vì mặt không có "pitch", cần một hồi quy tuyến tính face → F0 trong cùng giới (70 speaker) |
| Rank fusion + trộn nhiều hệ trong mỗi cell | ✅ Làm ngay, rẻ | Trộn **offline từ các zip sẵn có**, lọc bằng SEL-01. ⚠️ Thử nhiều ứng viên trên cùng pseudo-label sẽ overfit selector, nên giới hạn ≤ 5 ứng viên rồi xác nhận bằng CB |
| Framework `runs/` + manifest, T4×2 hai process | ◐ Làm dần | Bắt buộc từ nay: **lưu embedding dev của mỗi run**, để các selector cần embedding kiểm định ngược được |
| Duration gate | ⏸ | Chờ `r2_ecapa192_mix` |

## 3. Thứ tự thực hiện

| # | Việc | Ở đâu | Tốn |
|---|---|---|---|
| 1 | Chấm trước `r2_*` và `sub_base` bằng SEL-01, rồi mới nộp | local | 0 |
| 2 | SEL-01b: dựng pseudo-label bằng **ArcFace + ECAPA-192 tự trích** (unimodal mạnh hơn), so với SEL-01a | local (cần 8 file `.npy` dev từ `flag2027-feats-v2`) | 0 |
| 3 | FUSE-01: rank fusion theo từng cell từ các zip sẵn có, SEL-01 lọc, nộp 1 lượt | local | 1 lượt nộp |
| 4 | ATTR-EDA: tuổi (face `genderage`), F0 voice; AUC trong cùng giới trên train | Kaggle, nhẹ | 0 |
| 5 | ATTR-02: score thuộc tính + score hiện tại, 2–4 trọng số | Kaggle | 1 lượt nộp |
| 6 | SSL-01 scalar mix (nếu `r2_wavlm` có tín hiệu ở Bangla) | Kaggle | 0–1 lượt nộp |
| 7 | GRAPH-01 (chỉ chấm bằng CB) | Kaggle | 1 lượt nộp |
| 8 | ImageBind zero-shot probe; hỏi BTC về dữ liệu ngoài | Kaggle / email | |
