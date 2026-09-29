# Cấu hình Evaluation đề xuất — SET-13 · CodaBench **21.01** (2026-09-28)

> **SET-13 vẫn là cấu hình Evaluation, nhưng pipeline để chạy trên dữ liệu Evaluation là `../v6_eval/` (15 mạng, noclus / INDEP-01, chẩn đoán cụm, đọc output FLAG_12).** Trạng thái hiện hành (29/09, nhật ký tới mục 75): cấu hình Evaluation là **SET-13 (CB 21.01) qua `BEST/v6_eval`**; xem [DECISIONS.md](../../docs/DECISIONS.md) và [OPEN_DIRECTIONS.md](../../docs/OPEN_DIRECTIONS.md).

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| SET-02 (`../v4_set02/`) | 21.68 | 12.50 | 21.34 | 28.31 | 24.59 |
| SET-12 (+ ImageBind) | 21.45 | 11.64 | 21.05 | 23.01 | 30.11 |
| **SET-13 (+ ImageBind + tuổi)** | **21.01** | **11.31** | 20.91 | 23.22 | 28.61 |

**Chọn bằng validation độc lập, không bằng CB** (nhật ký 61, 66):
- v4 giữ ra, mức cụm: −6.8 / −6.5 (5/5 split);
- VAL-70 (người v1/v2, thành phần train trên 70 người), mức người: trung bình −7.4, thắng 10/10 file ở mọi ô.

## Khác SET-02 đúng một chỗ

Trong mỗi file test, ma trận đầy đủ mặt × giọng S của thành phần `s007` được thay bằng

```text
S' = z(0.5 z(S) + 0.5 z(IB)) + 0.1 z(AGE)
IB  = cosine giữa embedding ImageBind-huge của mặt và của giọng, mỗi bên trừ mean của chính file   (không train)
AGE = −|tuổi(mặt) − tuổi(giọng)|
      tuổi mặt  = kỳ vọng theo 9 nhóm tuổi FairFace của nateraw/vit-age-classifier
      tuổi giọng = đầu ra tuổi của audeering wav2vec2-large-robust × 100                          (không train)
```

`s010B`, `r2mix`, gom cụm (ArcFace / ECAPA-192, ngưỡng hiệu chỉnh trên người train), b = 0.99 và trộn theo hạng từng cell đều giữ nguyên.

## Chạy

```bash
cd BEST/v5_set13
python pipeline.py --data ../../kaggle_upload --feats ../../kaggle/output/feats_v2 \
                   --models ../../kaggle/output/feats_models --out run_dev \
                   --reference ../../Experiment/MODEL-01/out/submission_SET13.zip
```

**Đã kiểm chứng:** với `--members-from ../v4_set02/_dryrun` (các thành phần mà SET-02 dùng), zip ra **rank-corr 1.0000** với bản SET-13 đã nộp ở cả 4 cell (`_dryrun/`).
Không có `--members-from` thì các thành phần được train lại trên 70 người (khoảng 30 phút CPU) và lưu trong `--out`.

## Cho Evaluation phase (11–18/11)

Dùng **[../v6_eval/](../v6_eval/README.md)**, không dùng `pipeline.py` ở đây:
1. Trích đặc trưng cho dữ liệu mới bằng `kaggle/FLAG_12_features` (một notebook, code đầy đủ; thay FLAG_04 + FLAG_10).
2. `../v6_eval/pipeline.py --feats <output FLAG_12> --members-from ../v6_eval/_members` (SET-13, 15 mạng đã cache).
3. Đọc `decisions.csv`; nộp `submission.zip`.
4. Nếu cụm của một file có dấu hiệu gộp nhầm giọng: `--file-mode <file>=noclus`. Theo STRESS-01, bản dự phòng đúng là `noclus`, không phải INDEP-01 hay `--b 0` của bản này; INDEP-01 chỉ dùng khi có quy định cấm dùng các mẫu test khác.

## Phải khai báo trong system description

- **Gom cụm không giám sát** ảnh mặt và giọng trong từng file test; trừ mean theo file. Không dùng nhãn, không dùng danh sách trial. Điểm của một cặp không đổi khi các trial khác bị ghép lại theo cách khác.
- **Encoder và licence:**
  - VGGFace fc7 và `yangwang825/ecapa-tdnn-vox2` (của BTC);
  - speechbrain `spkrec-ecapa-voxceleb` (192, 6144);
  - ArcFace `buffalo_l` (chỉ để gom cụm);
  - **ImageBind-huge** (CC-BY-NC 4.0);
  - `nateraw/vit-age-classifier`;
  - **audeering wav2vec2-large-robust-24-ft-age-gender** (CC-BY-NC-SA 4.0; train trên Common Voice tiếng Đức, aGender, TIMIT, VoxCeleb2).
- Theo trả lời của BTC (nhật ký 63): mọi thứ trên đều được phép.
