# INDEP-01: chấm mỗi cặp độc lập (mức 0) · CodaBench **26.70**

| | Overall | ng/En | ng/Bn | g/En | g/Bn |
|---|---:|---:|---:|---:|---:|
| **INDEP-01** (mỗi cặp chấm độc lập) | **26.70** | 19.05 | 26.32 | 27.09 | 34.33 |
| SET-13 (có thống kê file + gom cụm), cấu hình Evaluation đề xuất | 21.01 | 11.31 | 20.91 | 23.22 | 28.61 |

- **INDEP-01 là gì:** SET-13 bỏ hết mọi bước nhìn sang các mẫu test khác. Điểm của một cặp (mặt, giọng) chỉ phụ thuộc vào đúng ảnh đó, đúng giọng đó và các hằng số tính từ tập train.
- **Dùng khi nào:** (1) nếu có quy định cấm dùng các mẫu test khác; (2) làm con số đo năng lực thật của mô hình, để báo trong system description. **Không** dùng làm bản dự phòng khi cụm hỏng: STRESS-01 cho thấy bản `noclus` (thống kê file, không cụm) tốt hơn hoặc bằng INDEP-01 trong mọi dạng file đã thử (mục 5).
- **Code:** [build.py](build.py). **Ngày:** 2026-09-28. **Nhật ký:** mục (68) trong [../EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md).

## 1. Ý tưởng

Một hệ chấm điểm có thể dựa vào test ở bốn mức ([../../docs/PLAN_MODELS.md](../../docs/PLAN_MODELS.md) §0b):

| Mức | Hệ được nhìn gì khi chấm cặp (f, v) | Ví dụ |
|---|---|---|
| **0** | Chỉ f, v và hằng số từ train | **INDEP-01** |
| 1 | Thống kê của cả file test: mean, độ lệch chuẩn, thứ hạng | trừ mean theo file, z-score / rank trong file |
| 2 | Cấu trúc của file: ai giống ai | gom cụm (SET-02, SET-13) |
| 3–4 | Danh sách trial (cặp nào đi cùng nhau) | không dùng |

SET-13 dùng mức 1 và 2. INDEP-01 giữ nguyên mọi thành phần đã học của SET-13 (cùng mô hình, cùng seed, cùng trọng số) và chỉ thay các bước mức 1–2 bằng hằng số từ train. Nhờ vậy, chênh lệch giữa hai bản đo đúng phần điểm do việc nhìn sang các mẫu test khác mang lại.

## 2. SET-13 và INDEP-01 khác nhau ở đâu

| Bước | SET-13 | INDEP-01 |
|---|---|---|
| Đặc trưng đầu vào của cầu nối | chuẩn hoá bằng train, rồi **trừ mean của file, cộng mean train** (`centre`, EXP-003c) | chỉ chuẩn hoá bằng train (mean / std / PCA của train) |
| CCA-4 | trừ mean của file | chuẩn hoá bằng train (`RidgeCCA.transform_*`) |
| ImageBind | mỗi bên trừ **mean của file** | mỗi bên trừ **mean của train** |
| Đưa điểm các mô hình về cùng thang | z-score trên toàn ma trận mặt × giọng **của file** | z-score với hằng số (μ, σ) đo trên **20 000 cặp ngẫu nhiên của train** |
| Trộn các thành phần trong một cell | trung bình **thứ hạng** trong file | trung bình **điểm đã chuẩn hoá** (hằng số cố định) |
| Gom cụm, trung bình khối (b = 0.99) | có | **không** |
| Tuổi (ViT-age, w2v2 age), thành phần s007 / s010B / r2mix, n = 5, seed, trọng số 0.5 / 0.1 / 0.25 | giống nhau | giống nhau |

## 3. Công thức

Ký hiệu `Z_x(·)` = `(· − μ_x) / σ_x`, trong đó μ_x và σ_x là mean và std của **cùng đại lượng x** đo trên 20 000 cặp (ảnh train, giọng train) chọn ngẫu nhiên (seed 0). Với 70 người, khoảng 1/70 số cặp là cùng người, nên thang đo gần với phân bố cặp khác người.

**Bước 1: ba thành phần.** Mỗi thành phần m ∈ {s007, s010B, r2mix}:
- 5 mạng MLP (công thức EXP-007: head MLP, dropout 0.3, embedding 128, 40 epoch), seed 1, 101, …, 401, train trên đủ 70 người.
- Điểm của thành phần là

  ```text
  s_m(f, v) = 0.75 · mean_i Z_{m,i}( u_i(f) · w_i(v) ) + 0.25 · Z_cca( cca_x(f) · cca_y(v) )
  ```

  - u_i, w_i là embedding mặt và giọng của mạng thứ i.
  - CCA là CCA tuyến tính 4 chiều, fit trên train.

| Thành phần | Mặt | Giọng |
|---|---|---|
| s007 | VGGFace 4096 (BTC) | ECAPA-192 (BTC) |
| s010B | VGGFace 4096 | ECAPA 6144 (speechbrain, tự trích) |
| r2mix | VGGFace 4096 | ECAPA-192 (speechbrain, tự trích), train trên đoạn cắt theo độ dài Bangla |

**Bước 2: thêm ImageBind và tuổi vào s007** (không train):

```text
IB(f, v)  = cos( IBface(f) − mean_train , IBaudio(v) − mean_train )          ImageBind-huge
AGE(f, v) = −| tuổi_mặt(f) − tuổi_giọng(v) |
            tuổi_mặt  = kỳ vọng trên 9 nhóm tuổi FairFace (nateraw/vit-age-classifier)
            tuổi_giọng = đầu ra tuổi của audeering wav2vec2-large-robust-24-ft-age-gender × 100
s007'     = Z_mix( 0.5 · Z(s_s007) + 0.5 · Z(IB) ) + 0.1 · Z(AGE)
```

**Bước 3: điểm cuối** (cao = cùng người):

```text
English:  score = ( Z(s007') + Z(s_s010B) ) / 2
Bangla:   score = ( Z(s_r2mix) + Z(s007') ) / 2
```

File nộp ghi `−score` (**thấp = cùng người**), giống mọi bản nộp khác ([../../docs/DATA.md](../../docs/DATA.md)).

**Tính chất kiểm chứng được:** mọi phép tính sau khi có hằng số train đều làm trên từng cặp. BatchNorm ở chế độ eval cũng chỉ dùng thống kê đã lưu. Vì vậy điểm của (f, v) **không đổi** khi file có 1 hay 10 000 trial, khi thêm hoặc bớt người, hay khi một người chiếm phần lớn file.
- Chỗ duy nhất gọi `centre(Z, mu)` truyền `mu` bằng chính mean của tập đó, nên `Z − mean + mean = Z` (chỉ sai số dấu phẩy động).
- Phép thử tự động: `BEST/v6_eval` chấm lại một nửa file và so sánh (chênh tối đa 5e-7).

## 4. Kết quả trên dev

| Hệ | Mức | ng/En | ng/Bn | g/En | g/Bn | Overall |
|---|---|---:|---:|---:|---:|---:|
| 000f (CCA tuyến tính, chuẩn hoá bằng train, 22/09) | 0 | 28.97 | 29.02 | 34.42 | 41.01 | 33.35 (CB) |
| FUSE-03 (không ImageBind) | 1 | 23.61 | 26.88 | 29.12 | 35.15 | 28.69 (CB) |
| `BEST/v5_set13 --b 0` (SET-13 không gom cụm) | 1 | 19.03 | 25.74 | 26.56 | 35.45 | 26.70 (pseudo-arc) |
| **INDEP-01** | **0** | **19.05** | **26.32** | **27.09** | **34.33** | **26.70 (CB)** |
| INDEP-01, dự đoán trước khi nộp | 0 | 18.11 | 26.10 | 26.03 | 34.07 | 26.08 (pseudo-arc) |
| SET-13 | 2 | 11.31 | 20.91 | 23.22 | 28.61 | 21.01 (CB) |

**Đọc kết quả:**
1. **Mức 0 ngang mức 1.** Chênh pseudo-arc từng cell giữa INDEP-01 và `--b 0` là −0.92 / +0.36 / −0.53 / −1.38, mọi khoảng tin cậy bootstrap đều chứa 0. Trừ mean và z / rank theo file không mang lại gì thêm khi đã chuẩn hoá bằng train.
2. **Gom cụm đáng 5.7 điểm** (26.70 → 21.01): ng/En −7.7, ng/Bn −5.4, g/En −3.9, g/Bn −5.7. Đây là phần phụ thuộc test duy nhất có giá trị.
3. **ImageBind + tuổi đã tự mang lại 2 điểm** so với FUSE-03 (bản mức 1 tốt nhất trước đó), mà không cần thống kê file.
4. **Khoảng cách ngôn ngữ** (Bangla − English): +7.3 (ng) và +7.2 (g). Gom cụm không thu hẹp được nó: SET-13 là +9.6 (ng) và +5.4 (g).
5. Pseudo-arc lệch CB +0.2…+1.1 mỗi cell, đúng tầm sai số đã biết (≤ 1.6).

## 5. Khi nào dùng INDEP-01 thay cho SET-13 (đã sửa theo STRESS-01, 28/09)

**Bản trước của mục này sai.** Nó cho rằng INDEP-01 an toàn hơn khi mỗi người có ít mẫu hoặc một người chiếm phần lớn file. [STRESS-01](../STRESS-01/README.md) (200 file mô phỏng, nhãn thật) bác điều đó:

| Dạng file (người tiếng Anh giữ ra) | INDEP-01 (L0) | Không cụm (L1, `noclus`) | SET-13 (L2) |
|---|---:|---:|---:|
| 1 mẫu mỗi người, ng / g | 28.60 / 31.21 | 23.09 / 29.64 | **22.53 / 29.26** |
| 2 mẫu mỗi người | 22.24 / 31.19 | **20.23 / 27.82** | 21.55 / 29.13 |
| một người chiếm khoảng 26% file | 25.62 / 33.55 | 23.21 / 34.22 | **17.90 / 27.90** |
| chỉ 5 người mỗi file | 23.19 / 31.06 | 22.63 / 25.88 | **16.16 / 24.22** |
| giống dev (30 người, 32 mẫu) | 22.92 / 33.63 | 22.27 / 33.34 | **16.99 / 28.10** |

- **SET-13 là lựa chọn mặc định cho mọi dạng file đã thử.**
- **Khi cụm của một file có dấu hiệu gộp nhầm giọng** (kiểu SET-04: ít cụm giọng hơn hẳn cụm mặt), bản dự phòng là `noclus`, không phải INDEP-01. `BEST/v6_eval` đánh dấu những file như vậy trong `decisions.csv`; đổi bằng `--file-mode <file>=noclus`.
- **INDEP-01 chỉ dùng khi có quy định cấm dùng các mẫu test khác.**
- Rủi ro gộp nhầm ở ngôn ngữ khác hoặc file đông người sẽ được đo ở STRESS-01 phần B.

**Khi Evaluation mở:** nộp SET-13 trước. Nếu còn lượt, nộp `noclus` để đo xem gom cụm có tác dụng trên dữ liệu mới không (trên dev là khoảng 5.7). Số lượt nên dành cho việc này tuỳ BTC tính lượt nào (câu 10 trong [../../docs/email_organizers.md](../../docs/email_organizers.md)).

## 6. Chạy

```bash
cd Experiment/INDEP-01
INDEP_THREADS=10 python build.py        # khoảng 14 phút CPU, 10 luồng
# -> out/submission_INDEP01.zip (qua check_zip)
```

**Đầu vào** (đường dẫn đang cố định trong `build.py`):

| Thư mục | Nội dung | Tạo bởi |
|---|---|---|
| `kaggle_upload/` | đặc trưng của BTC (VGGFace 4096, ECAPA-192) + danh sách trial dev | BTC |
| `kaggle/output/feats_v2/` | `voice_ecapa6144_*`, `voice_ecapa192_*` | `kaggle/FLAG_04_extract` |
| `kaggle/output/feats_models/` | `face_ibv_*`, `voice_iba_*`, `face_vitageprob_*`, `voice_w2v2agage_*` | `kaggle/FLAG_10_models` |

INDEP-01 **không cần ArcFace** vì không gom cụm.

**Trên dữ liệu mới:** dùng `BEST/v6_eval/pipeline.py --mode indep`. Nó cho đúng INDEP-01 (rank-corr 1.0000 trên dev khi n = 5) và nhận `--data / --feats` cho dữ liệu bất kỳ. `build.py` chỉ chạy trên 4 file dev.

## 7. Việc đã làm tiếp (28/09)

1. **Xong:** chế độ `--mode indep` trong [../../BEST/v6_eval/](../../BEST/v6_eval/README.md), dùng lại thành phần đã train. Phép thử bất biến chạy mỗi lần: chấm lại một nửa file, chênh tối đa 5e-7.
2. **Xong:** `--n-models 15` là mặc định của v6 (nhật ký 69).
3. **Phần A xong, phần B chờ dữ liệu:** [STRESS-01](../STRESS-01/README.md). Kết quả đã sửa lại mục 5.

## 8. Hạn chế

- **Hằng số chuẩn hoá lấy từ cặp train tiếng Anh**, mà các cặp này mô hình đã thấy khi train. Trên người mới hoặc trên Bangla, điểm có thể lệch.
  - **Dịch chuyển cả khối** không ảnh hưởng: EER tính riêng từng cell, mỗi cell chỉ một ngôn ngữ.
  - **Độ trải khác đi** thì trọng số tương đối giữa các thành phần khi trộn bị đổi, dù thứ tự bên trong từng thành phần giữ nguyên.
  - Trên dev, INDEP-01 vẫn ngang bản chuẩn hoá theo file, nên ảnh hưởng này hiện nhỏ.
- **Kém SET-13 khoảng 5.7 điểm** khi dữ liệu có dạng như dev.

## 9. Đoạn cho system description (tiếng Anh, dùng nếu nộp INDEP-01)

> Every trial is scored independently: the score of a (face, voice) pair depends only on that face, that voice and constants estimated on the training set, never on other test samples or on the trial list. Three face–voice bridges (VGGFace 4096 paired with the organiser ECAPA-192, speechbrain ECAPA 6144 and speechbrain ECAPA-192 trained on Bangla-length crops) are each an ensemble of 5 MLPs trained with a symmetric InfoNCE loss on the 70 training speakers, fused 0.75 / 0.25 with a 4-dimensional ridge CCA. The first bridge is further fused with the zero-shot cosine of ImageBind-huge face and audio embeddings (training-set mean removed) and an age-consistency term, −|face age − voice age|, from nateraw/vit-age-classifier and audeering wav2vec2-large-robust-24-ft-age-gender. All scores are standardised with the mean and standard deviation measured on 20 000 random training face–voice pairs, and two bridges are averaged per language. None of our own training uses Bengali data.

Encoder và licence cần khai báo giống SET-13 (xem [../../BEST/v5_set13/README.md](../../BEST/v5_set13/README.md)), trừ ArcFace.
