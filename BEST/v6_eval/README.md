# Pipeline Evaluation v6: SET-13 và INDEP-01 trong một lần chạy, tự chọn theo từng file

Kế hoạch: [../../docs/PLAN_EVAL.md](../../docs/PLAN_EVAL.md) (việc E1).
- **SET-13:** CodaBench 21.01, có gom cụm. Tài liệu: [../v5_set13/README.md](../v5_set13/README.md).
- **INDEP-01:** CodaBench 26.70, chấm từng cặp. Tài liệu: [../../Experiment/INDEP-01/README.md](../../Experiment/INDEP-01/README.md).

## Khác v5 ở đâu

| | v5_set13 | **v6_eval** |
|---|---|---|
| Bản chạy ra | chỉ SET-13 (`--b 0` bỏ gom cụm nhưng vẫn dùng thống kê file) | SET-13, `noclus` **và** INDEP-01, cùng một bộ thành phần |
| Chọn theo từng file | không | `decisions.csv` (chẩn đoán cụm, cảnh báo) + `--file-mode <file>=<mode>` |
| Số mô hình mỗi thành phần | 5 | **15** mặc định (nhật ký 69: cùng kỳ vọng, bỏ được may rủi theo seed) |
| Cache thành phần | ma trận điểm của dev | **mạng đã train**, theo từng (thành phần, seed); dùng lại được cho dữ liệu mới |
| Đặc trưng | `--feats` (FLAG_04) + `--models` (FLAG_10) | `--feats` nhận nhiều thư mục; một thư mục output của **FLAG_12** là đủ |
| Tự kiểm tra | không | phép thử bất biến của INDEP; `decisions.csv` ghi lý do chọn cho từng file |

Seed là 1, 101, 201, …, giống `flag_v2.dev_scores`. Với `--n-models 5`, 5 mạng đầu chính là 5 mạng của các bản đã nộp.

## Chạy

```bash
cd BEST/v6_eval
python pipeline.py --data ../../kaggle_upload \
                   --feats ../../kaggle/output/feats_v2 ../../kaggle/output/feats_models \
                   --out _dev_n15 --members-from _members --n-models 15 --mode auto
```

- Lần đầu train 3 thành phần × 15 mạng, khoảng 40 phút CPU. Các mạng được lưu trong `_members/` và mọi lần chạy sau dùng lại.
- Để so với bản đã nộp, thêm `--reference set13=<zip> indep=<zip>`.

**Output trong `--out`:**
- `submission.zip`: bản đã chọn cho từng file;
- `submission_set13.zip`, `submission_noclus.zip`, `submission_indep.zip`: ba bản đầy đủ;
- `decisions.csv`: chẩn đoán cụm, bản được chọn và lý do, cho từng file;
- `invariance.json`: kết quả phép thử bất biến;
- `config.json`.

## Các chế độ và quy tắc `auto` (theo STRESS-01 phần A, nhật ký 71)

| `--mode` | Là gì | Dùng khi |
|---|---|---|
| `auto` (mặc định) | SET-13 cho mọi file; file nào có số cụm giọng / số cụm mặt < 0.8 thì bị **đánh dấu** trong `decisions.csv` | luôn dùng |
| `set13` | SET-13 cho mọi file, không cảnh báo | — |
| `noclus` | SET-13 bỏ cụm (b = 0): trừ mean và z-score theo file | cụm của một file có dấu hiệu gộp nhầm giọng: `--file-mode <file>=noclus` |
| `indep` | INDEP-01: từng cặp, hằng số từ train | chỉ khi có quy định cấm dùng các mẫu test khác |

**Vì sao `auto` không tự chuyển file nào:** [STRESS-01](../../Experiment/STRESS-01/README.md) thử 200 file mô phỏng có nhãn thật (1–32 mẫu mỗi người, 5–30 người mỗi file, một người chiếm khoảng 26% file). Kết quả:
- gom cụm tốt hơn chấm từng cặp ở mọi kịch bản, 0.7–11 EER;
- các file thua đều là file tí hon, và không dấu hiệu nào báo trước được;
- hai quy tắc tạm thời trước đây ("cụm lớn nhất > 25%", "cụm trung vị < 2") đều làm mất 2–6 EER, nên đã bị bỏ.

STRESS-01 phần B (người v1/v2, Urdu / Hindi / English, file đến 150 người) cũng cho gom cụm tốt hơn ở mọi kịch bản, và không lần nào giọng bị gộp nhầm. Cảnh báo được giữ như một lớp an toàn, không tự động chuyển.

## Kiểm chứng trên dev (28/09)

**Tái tạo bản đã nộp (n = 5, `_dev_n5/`):**
- `submission_set13.zip` cho rank-corr **1.0000** với SET-13 (CB 21.01) ở cả 4 cell;
- `submission_indep.zip` cho rank-corr **1.0000** với INDEP-01 (CB 26.70) ở cả 4 cell;
- phép thử bất biến: chấm lại một nửa file, chênh tối đa 5e-7.

**n = 15 (`_dev_n15/`, dùng cho Evaluation):** chẩn đoán cụm của cả 4 file dev đều bình thường, không có cảnh báo.

| File | Cụm mặt | Cụm giọng | Giọng / mặt | Cụm lớn nhất / file |
|---|---:|---:|---:|---:|
| no_gender/English | 29 | 41 | 1.41 | 0.11 |
| no_gender/Bangla | 32 | 39 | 1.22 | 0.09 |
| gender/English | 33 | 40 | 1.21 | 0.10 |
| gender/Bangla | 29 | 41 | 1.41 | 0.09 |

**Ước tính pseudo-arc** (chỉ để tham khảo, không dùng để chọn; sai số so với CB ≤ 1–1.6 mỗi cell):

| Zip | ng/En | ng/Bn | g/En | g/Bn | Overall |
|---|---:|---:|---:|---:|---:|
| set13, n = 5 (= SET-13 đã nộp) | 11.11 | 21.03 | 22.96 | 28.75 | 20.96 |
| **set13, n = 15** | 12.14 | 19.87 | 22.33 | 28.89 | **20.81** |
| noclus, n = 15 | 19.55 | 25.53 | 26.35 | 35.87 | 26.83 |
| indep, n = 15 | 18.31 | 25.74 | 26.88 | 34.14 | 26.27 |

- n = 15 và n = 5 ngang nhau trên dev: mọi khoảng tin cậy đều chứa 0. Đúng như VAL-70 dự đoán: cùng kỳ vọng, n = 15 bớt được phần may rủi theo seed.
- Trên dev, `noclus` và `indep` cũng ngang nhau. Trên file mô phỏng (STRESS-01), `noclus` ≥ `indep`.

**Chạy thử toàn bộ quy trình (E6, 28/09):**
- đặc trưng lấy từ output Kaggle của `kaggle/FLAG_12_features` (`kaggle/output/feats_eval`), tuổi giọng tạm từ `feats_models`;
- `submission_set13.zip` và `submission_indep.zip` cho rank-corr **1.0000** với bản đã nộp ở cả 4 cell;
- phần chấm điểm (mạng đã có trong cache) mất **16.6 phút** CPU;
- log: `_e6.log`.

## Cho Evaluation phase

1. Chạy `kaggle/FLAG_12_features` trên dữ liệu Evaluation (gắn thêm dữ liệu v4 để có mảng train).
2. Chạy `pipeline.py --feats <output FLAG_12> --members-from _members`. Các mạng đã train được dùng lại, chỉ phần chấm điểm chạy lại.
3. Đọc `decisions.csv`. Nếu có file bị cảnh báo, xem số cụm, rồi quyết định có chạy lại với `--file-mode <file>=noclus` hay không. Nộp `submission.zip`. Nếu còn lượt, nộp `submission_noclus.zip` để đo tác dụng của gom cụm trên dữ liệu mới.
4. **Tên file nộp và danh sách trial hiện vẫn theo định dạng dev** (`flag_lib.NAMES`, `load_dev`). Nếu dữ liệu Evaluation đặt tên khác, phải sửa hai chỗ đó.
