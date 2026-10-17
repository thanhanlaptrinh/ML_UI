> **Bản sửa lỗi nạp XGBoost trên Windows:** Flask nạp XGBoost qua `models/final/xgboost.ubj` (official save_model/load_model) và `xgboost_preprocess.pkl`, tránh khôi phục Booster từ pickle Linux. Giữ nguyên các PKL và trọng số cũ; không huấn luyện lại. Phải chép đủ cả gói, gồm manifest và audit đã cập nhật. Xem `docs/WINDOWS_FIX.md`.

# HomeValue Lab · nhóm 9 · bản sửa lượt 1

Đã thực hiện ở lượt 1: sao lưu SHA-256; bỏ Id (79 đặc trưng); tinh chỉnh bằng CV chỉ trên train; lưu đủ 5 pipeline chính thức; cập nhật suy luận, đồ thị và kiểm tra để hỗ trợ mục tiêu log1p. Tỷ giá hiển thị giữ nguyên.

**Chưa thực hiện ở lượt 1:** thang chẩn đoán 4 mức và tách chỉ số theo membership (lượt 2); hoàn thiện giao diện/quy trình/favicons/định dạng tiền (lượt 3); dọn repo, README và nghiệm thu cuối (lượt 4). Giao diện hiện tại chưa phải bản nghiệm thu cuối.

## Chạy bản đã huấn luyện sẵn (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
# Cửa sổ 1
.\.venv\Scripts\python.exe service\predict_service.py
# Cửa sổ 2
npm start
```

Mở http://localhost:3000 và Ctrl+F5. Cả ba tab đọc cùng 5 pipeline trong `models/final`, không tự huấn luyện.

## Tái lập từ đầu

Thứ tự phụ thuộc đúng là **backup → tune_models → train_all → prepare_vnd_audit → export_vnd**. `prepare_vnd_audit` cần trọng số cuối đã tồn tại nên phải chạy sau train_all, không chạy trước tuning.

```powershell
.\.venv\Scripts\python.exe scripts\backup_original.py
$env:LOKY_MAX_CPU_COUNT = "6"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\.venv\Scripts\python.exe notebooks_or_src_py\tune_models.py --force
.\.venv\Scripts\python.exe notebooks_or_src_py\train_all.py --force
.\.venv\Scripts\python.exe scripts\prepare_vnd_audit.py
.\.venv\Scripts\python.exe scripts\export_vnd.py
```

`--force` là lựa chọn ghi đè rõ ràng sau sao lưu. Không có cờ này, train_all dừng trước khi fit/ghi bất kỳ trọng số nào nếu đầu ra đã tồn tại. Sao lưu gốc không bao giờ bị thay thế kể cả khi dùng --force. `train_compare.py` chuyển sang cùng entry point an toàn; không còn chọn winner theo holdout.

Search dùng `n_jobs=-1`; biến LOKY_MAX_CPU_COUNT chỉ giới hạn số worker theo máy. RF/XGB bên trong dùng 1 luồng để tránh song song lồng nhau. Có checkpoint và `tune_models.py --resume` để tiếp tục các mô hình đã hoàn thành. Không đổi không gian/CV/phiên bản giữa các lần resume.

## Thiết kế thực nghiệm

Giữ split seed 42, 1.168 train và 292 holdout. KFold 5 folds, shuffle=True, seed 42, scoring=neg_mean_absolute_error. Decision Tree tìm toàn bộ grid 160 cấu hình. RF/GB/XGB tìm 40 cấu hình ngẫu nhiên cho mỗi nhánh giá gốc/log1p; biến đổi mục tiêu dùng TransformedTargetRegressor, chấm điểm sau expm1 về USD. LR giữ cấu hình gốc làm baseline.

Không gian có thu hẹp số cây/vòng/độ sâu để giới hạn thời gian; toàn bộ miền thực tế và lý do lưu trong tuning_selection.json và tuning_report.json. Cấu hình gốc bỏ Id cũng được CV làm đối chứng và giữ như một ứng viên an toàn. CV trước của mô hình cũ có Id được chạy lại trên đúng các fold shuffled để so sánh công bằng; khác CV lịch sử không shuffle trong CSV gốc.

XGBoost có thêm ứng viên điều chuẩn và early stopping. Mỗi fold CV tách validation **bên trong phần train của fold**, chọn số vòng ở đó rồi refit outer-train; không dùng holdout hay nhãn outer-validation để chọn số vòng. Số vòng cuối quyết định bằng validation tách từ toàn bộ train. Ứng viên này được so với các ứng viên khác bằng CV. Không dùng điều kiện MAE holdout/train > 2 để quyết định tuning tiếp vì điều đó sẽ dùng holdout cho lựa chọn.

Winner theo CV MAE USD nhỏ nhất, hòa dùng CV std nhỏ hơn rồi tên ID. Tệp selection được đóng băng và băm SHA-256 trước lần đánh giá holdout cuối. Các script verify/API chỉ tái lập dự đoán, không dùng các lần tái lập đó để sửa tham số. Holdout đã được xem ở phiên bản cũ nên chưa phải một bộ mới hoàn toàn độc lập.

## Tệp minh chứng lượt 1

- `models/_backup_original/manifest.json`: hash mọi bản sao gốc và hai CSV dữ liệu; kết quả cũ ở `results/baseline_original/`.
- `results/tuning_selection.json`: lựa chọn chỉ từ train/CV, không có chỉ số holdout.
- `results/tuning_report.json`, `results/tuning_summary.csv`: trước/sau, đối chứng không Id, fold scores/std, top5, tham số, thời gian, phiên bản, báo cáo holdout đóng băng.
- `models/final/manifest.json`: 5 file nén compress=3, hash/kích thước, đặc trưng, biến đổi mục tiêu, seed và winner CV. `best_model.pkl` là bản sao đúng byte của winner.
- `models/feature_defaults.json`: 79 mặc định từ train, không có Id, không dùng holdout tính median/mode.
- `results/actual_vs_predicted.png`: đủ 292 holdout, không chọn các căn dự đoán đẹp.
- `results/verification_round1.json`, `results/round1_test_results.json`: kiểm tra Id, backup, CV và log output thật của các script bắt buộc.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe scripts\verify_vnd.py
.\.venv\Scripts\python.exe scripts\verify_dynamic.py
.\.venv\Scripts\python.exe scripts\verify_training_report.py
.\.venv\Scripts\python.exe scripts\verify_round1.py
# Khi Flask và Node đã sẵn sàng:
node scripts\smoke_http.mjs
node scripts\smoke_dynamic_http.mjs
```

Bảng kết quả do code sinh tại `results/tuning_summary.csv`; không chép số kỳ vọng bằng tay. Các tài liệu README cũ được giữ lại đến lượt dọn repo nhưng có thể mô tả phiên bản trước.
