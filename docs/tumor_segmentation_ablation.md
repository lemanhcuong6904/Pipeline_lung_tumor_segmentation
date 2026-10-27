# Ablation phân đoạn u

Chọn `ABLATION` ở đầu `trainers/train_tumor_segmentation.py`:

| `ABLATION` | Dữ liệu | Mô hình | Lát cắt |
| --- | --- | --- | --- |
| `baseline` | Toàn khung hình, window wide | TransUNet một encoder, 1 kênh | Tất cả, gồm lát có và không có phổi |
| `lung_roi` | ROI đã crop phổi, window wide | TransUNet một encoder, 1 kênh | Chỉ lát có phổi |
| `multi_window` | ROI đã crop phổi, wide/lung/mediastinal | TransUNet ba encoder, 3 kênh | Chỉ lát có phổi |

`baseline` đọc ảnh và nhãn từ `data/nsclc_radiomics/processed`; hai chế độ còn
lại đọc từ `data/nsclc_radiomics/lung_cropped/processed`. Ảnh toàn khung đã
được resize về 256×256, không crop theo phổi. Model 1 kênh có cấu trúc tương
ứng `D:/2.5D Lung Tumor Segmentation/models/transunet.py`.

Các cờ cross-attention và channel-attention chỉ áp dụng cho `multi_window`.
Boundary loss và balanced sampling áp dụng cho cả ba chế độ. Khi bật balanced
sampling, mỗi epoch lấy mẫu nên không đảm bảo đi qua mọi lát. Để duyệt toàn bộ
lát train một lần mỗi epoch, đặt `BALANCED_TRAIN_SAMPLING = False` và
`TRAIN_BATCHES_PER_EPOCH = None`.

Khi `EXP_NAME` để trống, trainer chọn tên theo `ABLATION_EXPERIMENT_NAMES`;
nếu điền tên thì dùng tên đã điền. Hãy đặt tên riêng cho các thí nghiệm khác
nhau về attention, loss hoặc sampling để tránh ghi đè. Checkpoint, config và
`history_metrics.json` được lưu vào thư mục experiment tương ứng.

Đặt `CHECKPOINT_PATH` trong `inferences/infer_tumor_segmentation.py` tới
checkpoint cần đánh giá. Với `ABLATION = "auto"`, inference dùng chế độ đã
lưu trong checkpoint và chọn dataset, NIfTI tương ứng. Với checkpoint 1 kênh
không lưu chế độ, hãy đặt `ABLATION` trong inference thành `baseline` hoặc
`lung_roi`.

Thí nghiệm pipeline 3 và 4 dùng lại checkpoint phân đoạn `lung_roi`; thí nghiệm
5 dùng checkpoint `multi_window`. Trainer và inference phân đoạn hiện không tự
chạy bộ phân loại lát cắt ResNet50 hoặc mở rộng ±8 lát: cần thực hiện các bước
đó riêng khi đánh giá pipeline.

Để tránh chọn checkpoint bằng test set, đặt `VAL_SPLIT` của trainer tới
`val.txt`, và dành `test.txt` cho đánh giá cuối cùng.
