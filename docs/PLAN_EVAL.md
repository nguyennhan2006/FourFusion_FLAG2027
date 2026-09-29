# Kế hoạch chuẩn bị Evaluation phase (11–18/11) — lập 28/09, cập nhật 29/09

**Mục tiêu:** khi dữ liệu Evaluation mở, ta có một quy trình đã chạy thử trọn vẹn. Quy trình đó tự quyết định dùng gom cụm (SET-13) hay chấm từng cặp (INDEP-01) cho từng file, và có sẵn số liệu cho biết khi nào gom cụm hỏng.

**Bối cảnh:**
- Gom cụm đáng 5.7 điểm trên dev: SET-13 21.01, INDEP-01 26.70 ([../Experiment/INDEP-01/README.md](../Experiment/INDEP-01/README.md)).
- Mức lợi đó chỉ có khi mỗi người xuất hiện nhiều lần trong file và cụm gom đúng người.

## Các việc

| ID | Việc | Chạy ở | Đầu ra | Phụ thuộc |
|---|---|---|---|---|
| **E1** | Pipeline Evaluation v6. Chế độ `set13 / indep / auto`; train thành phần một lần rồi dùng cho cả hai bản; `--n-models 15`; chẩn đoán cụm từng file; phép thử bất biến cho `indep` | CPU máy nhà | `BEST/v6_eval/` | — |
| **E2** | STRESS-01 phần A. Mô phỏng các dạng file Evaluation bằng 30 người v4 giữ ra (tiếng Anh, có nhãn): số mẫu mỗi người, phân bố lệch. So mức 0 / 1 / 2 và ngưỡng giọng tự thích nghi | CPU | `Experiment/STRESS-01/` + quy tắc cho `auto` | — |
| **E3** | Notebook **FLAG_12_features** (GPU, code đầy đủ trong notebook). Trích mọi đặc trưng pipeline cần (ArcFace, ECAPA-192/6144, ImageBind, tuổi) cho dữ liệu dạng BTC bất kỳ, và ArcFace + ECAPA cho người v1/v2 | Kaggle GPU | dataset `flag2027-feats-eval` | — |
| **E4** | STRESS-01 phần B. Urdu / Hindi / English của người v1/v2, file 60–150 người, cụm ước lượng thật, thành phần train trên đủ 70 người | CPU | bổ sung quy tắc `auto` | E3 |
| **E5a** | Sàng lọc rẻ cho E5: adapter tuyến tính trên embedding ImageBind đông cứng, train trên cặp v4 (+ v1/v2). E5 chỉ làm nếu nó tốt hơn zero-shot | CPU | `Experiment/IB-ADAPT/` | — |
| **E5** | Notebook **FLAG_13_iblora** (GPU, code đầy đủ). Fine-tune ImageBind bằng LoRA trên cặp mặt–giọng (hướng của đội hạng 2 FAME 2026), xuất embedding để đánh giá bằng VAL-70 / MODEL-01 mức người | Kaggle GPU | dataset `flag2027-iblora` | — |
| **E6** | Chạy thử toàn bộ: E3 trên dev → E1 → so với zip đã nộp (rank-corr ≈ 1) và đo thời gian | Kaggle + CPU | ghi chú trong `BEST/v6_eval/README.md` | E1, E3 |

**Thứ tự làm:** E3 (để chạy trên Kaggle song song) → E1 → E2 → E5. E4 và E6 làm khi có output của E3.

**Trạng thái (28/09):**

| ID | Trạng thái |
|---|---|
| E1 | **xong.** rank-corr 1.0000 với SET-13 / INDEP-01 khi n = 5; phép thử bất biến đạt; n = 15 chạy xong, 45 mạng đã cache ([../BEST/v6_eval/](../BEST/v6_eval/README.md)) |
| E2 | **xong** ([../Experiment/STRESS-01/](../Experiment/STRESS-01/README.md)): gom cụm tốt hơn ở mọi kịch bản; `auto` = SET-13 + cảnh báo; dự phòng là `noclus` |
| E3 | **xong, đã chạy trên Kaggle (28/09).** ArcFace / ECAPA / ImageBind / ViT-age khớp tuyệt đối mảng đang dùng (cosine 1.0000). w2v2 lỗi dưới transformers 5.0 + Jupyter (`__main__.__file__`); đã sửa và kiểm chứng lại với đúng phiên bản đó. **Lần chạy Evaluation phải dùng notebook đã sửa** |
| E4 | **xong.** Gom cụm tốt hơn ở Urdu / Hindi / English và ở file 60–150 người (−2.0…−6.5); không lần nào giọng bị gộp nhầm ([../Experiment/STRESS-01/](../Experiment/STRESS-01/README.md)) |
| E5a | **xong: không qua cổng** ([../Experiment/IB-ADAPT/](../Experiment/IB-ADAPT/README.md)) |
| E5 | **hoãn.** Chỉ làm lại khi có dữ liệu lớn hơn nhiều (X1 VoxCeleb), sau khi chạy lại E5a với dữ liệu đó |
| E6 | **xong.** v6 với đặc trưng từ output Kaggle FLAG_12 cho rank-corr 1.0000 với SET-13 và INDEP-01; phần chấm điểm mất 16.6 phút |
| E7 | **xong (29/09).** Các thí nghiệm CPU của deep-research-report 28/09: SetProto, SWA, IB-CCA đóng; v1/v2 chỉ còn ứng viên cho English; VoxCeleb chưa đạt cổng (nhật ký 74–75) |

**Còn lại trước khi chốt (3/11):**
1. Chạy lại `FLAG_12_features` bản đã sửa (w2v2) một lần trên Kaggle với dev, để xác nhận nó chạy trọn trên môi trường thật.
2. Nhóm xác nhận cấu hình: SET-13 qua v6, `--mode auto`, n = 15.
3. Tuỳ chọn: dựng + kiểm chứng ứng viên O1 (s007 + v1/v2 cho các cell English, [OPEN_DIRECTIONS.md](OPEN_DIRECTIONS.md)).
4. Khi có dữ liệu Evaluation:
   - kiểm tên file / định dạng danh sách trial (v6 đang theo định dạng dev);
   - kiểm trùng người giữa Evaluation và v1/v2 nếu O1 được dùng;
   - lượt nộp đầu: nếu EER > 50 thì đảo dấu.

## Quy ước cho notebook GPU

- **Code đầy đủ trong cell**: không import file .py của nhóm, không nhúng module dạng base64. Chỉ dùng thư viện cài bằng pip.
- **Chạy thử trên máy** (`FLAG_LOCAL=1`, dữ liệu mini) trước khi giao.
- **Chạy tiếp được sau khi lỗi**: gắn output của lần chạy trước làm input, mảng nào đã xong thì bỏ qua.
- **Thứ tự dòng** giữ đúng thứ tự trong file `.txt` của BTC (v4, Evaluation), hoặc đúng thứ tự trong `ext_*_meta.csv` (v1/v2), để khớp với đặc trưng đã có.

## Tiêu chí xong

- **E1:** `set13` cho rank-corr 1.0000 với SET-13 đã nộp khi n = 5; `indep` cho rank-corr 1.0000 với INDEP-01; phép thử bất biến đạt; chẩn đoán in ra cho từng file.
- **E3:** trên dev, cosine với các mảng đang có ≥ 0.999 (ArcFace, ECAPA, ImageBind, tuổi).
- **E2 / E4:** bảng EER theo từng kịch bản cho mức 0 / 1 / 2, và ngưỡng chẩn đoán cho `auto` rút ra từ bảng đó.
