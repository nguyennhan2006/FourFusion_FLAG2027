# Dữ liệu FLAG 2027 (MAV-Celeb v4)

## Train (`Input/train_set.zip`)

```text
train_set/
├── features/
│   ├── faces/train_English_faces.csv    6485 × (4096 + label)   float32
│   └── voices/train_English_voices.csv  6485 × (192  + label)   float32
└── train_set/
    ├── train_English.txt                 6485 dòng
    ├── faces/English/<id>/<n>.jpg        ảnh mặt thô
    └── voices/English/<id>/<n>.wav       audio thô
```

`train_English.txt` mỗi dòng: `<pair_id> <label> <voice_path> <face_path> <speaker_id> <speaker_int>`

```text
yXceFNU4 1 voices/English/id028/00000.wav faces/English/id028/00000.jpg id028 61
```

- 70 speaker `id001`..`id070`, chỉ tiếng Anh. Face và voice CSV **align theo hàng**; cột cuối CSV là **speaker int** (0..69) và bằng cột cuối của txt (`<speaker_int>`, **không phải** video index — mỗi speaker chỉ có đúng 1 giá trị). `label` luôn = 1.
- ⚠️ **Voice feature bị trùng nhiều**: chỉ 4029 vector unique / 6485 hàng (1267 nhóm trùng, nhóm lớn nhất 24 hàng). Trùng luôn nằm trong cùng speaker và các face trong nhóm khác nhau → nhiều face crop dùng chung một đoạn audio. `id006`, `id052` chỉ có 2 mẫu; `id006/id030/id052` có ≤ 5 voice unique. Face trùng ít hơn (391 nhóm, max 5). Xem [../Experiment/EDA-000_raw/](../Experiment/EDA-000_raw/).
- Face 4096-D: 89% giá trị = 0, không âm → output post-ReLU (VGGFace fc7). Voice 192-D: zero-mean, |x| ≤ 55, không sparse (ECAPA-TDNN).
- **Encoder BTC đã tái tạo chính xác** (cos 1.0000 với CSV, EXPERIMENT_LOG (34)): face = VGGFace (Parkhi 2015) fc7 sau ReLU, ảnh BGR trừ mean (`vgg_face_dag.pth`); voice = speechbrain `yangwang825/ecapa-tdnn-vox2` (**không phải** `speechbrain/spkrec-ecapa-voxceleb`, cos 0.02), audio đọc bằng `librosa.load(sr=16000, mono=True)`. Code: `kaggle/flag_extract.py` (`VGGFace`, `OrganiserVoice`, `load_audio`). Toàn bộ wav v4 là 16 kHz mono; MAV-Celeb v1 có file 44.1 kHz.
- `Input/meta_file_train_set.csv`: `<id>,<m|f>` — 40 nam / 30 nữ. **Chưa được dùng trong EXP-000c** → dùng cho gender-constrained internal validation & hard-negative.
- Có ảnh/wav thô → có thể tự trích feature khác.

## Dev (`Input/dev_set.zip`)

```text
dev_set/
├── no_gender/
│   ├── English_test.txt   1008 dòng   |  features/English_test_{faces,voices}.csv
│   ├── Bangla_test.txt    1406 dòng   |  features/Bangla_test_{faces,voices}.csv
│   └── {English,Bangla}_test/{faces,voices}/   thô
└── gender/
    ├── English_test.txt    982 dòng   |  features/English_test_{faces,voices}.csv
    ├── Bangla_test.txt    1468 dòng   |  features/Bangla_test_{faces,voices}.csv
    └── {English,Bangla}_test/{faces,voices}/   thô
```

Mỗi dòng test: `<pair_id> <voice_path> <face_path>` — **không có nhãn**. Feature CSV dev không có cột label (4096 / 192 cột thuần).
Thứ tự hàng CSV = thứ tự dòng trong `*_test.txt`.

## Format nộp CodaBench

```text
submission.zip
├── no_gender/sub_score_v4_English_heard.txt
├── no_gender/sub_score_v4_Bangla_unheard.txt
├── gender/sub_score_v4_English_heard.txt
└── gender/sub_score_v4_Bangla_unheard.txt
```

Mỗi dòng `<pair_id> <score>`, không header. Overall = mean(4 EER).

## ⚠️ Score orientation (đã kiểm chứng thực nghiệm 2026-09)

| Nộp | Overall EER |
|---|---:|
| `-squared_L2` (theo Evaluation Plan "higher = match") | 57.83 |
| `+squared_L2` (theo convention FOP gốc) | **42.17** |

→ Scorer live hiện tại là **distance-oriented: lower = same speaker**. Nộp **raw +d²** (hoặc `−cosine`).
Local (sklearn `roc_curve`) phải dùng dấu ngược. Nếu organizer đổi scorer, phải re-validate bằng cùng một checkpoint.

⚠️ **Evaluation Plan viết ngược với scorer thực tế.** Plan nói score cao = tin là cùng người, nhưng phép thử
cùng-một-checkpoint ở trên cho thấy scorer live chấm ngược lại. **Thực nghiệm thắng văn bản.** Toàn bộ 7 lượt
nộp sau đó (42.17 → 35.79 → 34.46 → 33.35 → 31.96 → 31.34 → 30.90) đều dùng convention *lower = same*;
nếu convention sai thì mọi con số này đã phải là ~100 − EER (tức ~58–69), không phải 30–42.

Hệ quả: **không sửa dấu trong `write_submission()` dựa trên việc đọc Evaluation Plan.** Nếu muốn kiểm chứng
lại, cách rẻ nhất là nộp một lượt với dấu đảo và xem có ra ~69 không — nhưng phép thử đó đã làm rồi (57.83).

## Kích thước / thời gian

- Face CSV train 71 MB, voice 10 MB → load vào RAM thoải mái, train 1 epoch FOP trên GPU < vài giây.
- Full sweep EXP-000c (6 α × 50 epoch + retrain) chạy được trên Kaggle T4 trong dưới 1 giờ.
- Zip ảnh/wav thô ~2.4 GB: chỉ giải nén khi làm thí nghiệm re-extract feature (Pipeline D).
