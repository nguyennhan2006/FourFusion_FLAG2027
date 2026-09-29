# Bản tốt nhất — SET-02 · CodaBench **21.68** (2026-09-26)

> **Lịch sử: SET-02 đã được thay bởi SET-13 (v5_set13, CB 21.01). Pipeline cho Evaluation là `../v6_eval/`.** Trạng thái hiện hành (29/09, nhật ký tới mục 75): cấu hình Evaluation là **SET-13 (CB 21.01) qua `BEST/v6_eval`**; xem [DECISIONS.md](../../docs/DECISIONS.md) và [OPEN_DIRECTIONS.md](../../docs/OPEN_DIRECTIONS.md).

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| FUSE-03 (bản trước, `../v3_28.69/`) | 28.69 | 23.61 | 26.88 | 29.12 | 35.15 |
| **SET-02** | **21.68** | **12.50** | **21.34** | **28.31** | **24.59** |

Giải thích kỹ thuật: [../../docs/report_SET02/report_set02.pdf](../../docs/report_SET02/report_set02.pdf).

## Chạy

```bash
cd BEST/v4_set02
python pipeline.py --data ../../kaggle_upload --feats ../../kaggle/output/feats_v2 --out run_dev \
                   --reference ../../Experiment/SET-01/out/submission_SET02.zip
```

- `--data`: feature CSV của BTC (train + các file test).
- `--feats`: feature tự trích bằng `kaggle/FLAG_04_extract.ipynb` (ArcFace, speechbrain ECAPA-192 và 6144).
- Mỗi bước được cache trong `--out`; chạy lại chỉ làm phần còn thiếu. Bước 1 (train 3 thành phần × 5 model) mất khoảng 30 phút CPU.

**Đã kiểm chứng:** với các ma trận thành phần mà SET-02 đã dùng, bước 2–3 cho zip **rank-corr 1.0000** với bản đã nộp ở cả 4 cell.

## Các bước
1. **Thành phần:** 3 cầu nối theo recipe EXP-007, train trên 70 người: `s007` (VGG + voice BTC), `s010B` (VGG + ECAPA-6144), `r2mix` (VGG + ECAPA-192 + crop độ dài Bangla). Mỗi file test cho ra ma trận điểm đầy đủ mặt × giọng.
2. **Gom cụm** trong từng file test: mặt bằng ArcFace, giọng bằng ECAPA-192; ngưỡng hiệu chỉnh trên người train (0.75 / 0.70).
3. **Trung bình khối** (b = 0.99), rồi trộn theo thứ hạng: English = s007 + s010B, Bangla = r2mix + s007.

## Cho Evaluation phase (11–18/11)
1. Trích feature cho dữ liệu mới bằng `FLAG_04_extract` (cùng code, cùng encoder).
2. Chạy `pipeline.py` với `--data` / `--feats` trỏ tới dữ liệu mới.
3. Kiểm tra log: số cụm mỗi file phải hợp lý (khoảng số người trong file), zip qua `check_zip`, score theo chiều **lower = same**.

## Phải khai báo trong system description
- Gom cụm **không giám sát** ảnh mặt và giọng trong từng file test; không dùng nhãn, không dùng danh sách trial.
- **Bất biến với cách ghép cặp của danh sách trial:** điểm của cặp (mặt f, giọng v) được đọc ra từ ma trận đầy đủ mọi mặt × mọi giọng của file. Ma trận này tính chỉ từ tập ảnh và tập giọng, nên nếu ghép lại các trial khác theo cách khác thì điểm của (f, v) không đổi. Các phương pháp đếm số trial nối cụm mặt A với cụm giọng B (dùng cấu trúc danh sách trial) không có tính chất này.
- Chuẩn hoá mean theo từng file test.
- Encoder: VGGFace fc7 và `yangwang825/ecapa-tdnn-vox2` (của BTC); speechbrain `spkrec-ecapa-voxceleb` (192, 6144); ArcFace `buffalo_l` (chỉ để gom cụm).
