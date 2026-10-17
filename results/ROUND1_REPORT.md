# Lượt 1 — kết quả do code sinh

Winner theo CV train: **Gradient Boosting**.

MAE dưới đây ở USD gốc. UI giữ nguyên lớp quy đổi VND.

| Model | CV trước có Id | CV đối chứng không Id | CV sau | MAE holdout trước | MAE holdout sau | Target |
|---|---:|---:|---:|---:|---:|---|
| Linear Regression | 19,894.99 | 20,026.42 | 20,026.42 | 20,594.94 | 20,420.50 | raw |
| Decision Tree | 24,641.68 | 25,102.58 | 24,404.83 | 25,754.33 | 25,895.71 | raw |
| Random Forest | 18,276.92 | 18,265.39 | 17,337.20 | 17,634.87 | 16,880.35 | log1p |
| Gradient Boosting | 16,235.18 | 16,096.96 | 15,043.52 | 17,099.28 | 15,814.43 | raw |
| XGBoost | 16,329.83 | 16,175.36 | 15,393.44 | 15,822.88 | 16,178.73 | raw |

CV MAE tốt nhất: 16,235.178116 → 15,043.516960 USD (-7.3400%).
Winner MAE holdout/train = 2.619561; R² train − holdout = 0.084079.

## Output thật của các kiểm tra

```text
$ python scripts/verify_vnd.py
PASS 55 checks
Linear Regression Chưa thấy dấu hiệu rõ theo ngưỡng đã đặt
Decision Tree Có dấu hiệu overfit
Random Forest Có dấu hiệu overfit
Gradient Boosting Có dấu hiệu overfit
XGBoost Chưa thấy dấu hiệu rõ theo ngưỡng đã đặt

```

```text
$ python scripts/verify_dynamic.py
PASS 85 dynamic numerical checks

```

```text
$ python scripts/verify_training_report.py
PASS 53 training report checks

```

```text
$ python scripts/verify_round1.py
PASS 94 round1 checks

```

```text
$ node scripts/smoke_http.mjs
PASS HTTP: five real pipelines, prices/metrics, decomposition, CSV 150 rows and invalid model.

```

```text
$ node scripts/smoke_dynamic_http.mjs
PASS dynamic HTTP: arbitrary inputs, actual lookup beyond 30, complete 1459-row Kaggle test × 5, missing-label guards and true model explanations.

```

## Giới hạn và phạm vi

- Các baseline cũ được giữ trong models/_backup_original và results/baseline_original, kèm SHA-256. Dữ liệu gốc không sửa.
- 79 đặc trưng cuối, không Id. Kiểm tra thay Id trên cùng 200 căn và cả 5 pipeline cho chênh lệch chính xác bằng 0; kiểm tra thêm qua FlexibleEngine.
- CV trước có Id được tính lại bằng cùng KFold shuffled, khác CV lịch sử không shuffle. Giữ thêm đối chứng bỏ Id nhưng chưa đổi tham số.
- Không phải mọi chỉ số holdout đều cải thiện; RMSE và R² đầy đủ trong tuning_report/tuning_summary. Không retune dựa trên holdout.
- Winner còn khoảng cách train/holdout. Quy tắc chẩn đoán mới, tách membership chờ lượt 2; UI đầy đủ chờ lượt 3; dọn repo/nghiệm thu chờ lượt 4.
- XGBoost có ứng viên điều chuẩn/early stopping chỉ dùng train. Lựa chọn dựa trên CV, kể cả khi ứng viên đó không tốt hơn. Chi tiết fold_iterations trong tuning_report.
- Holdout đã được xem ở bản cũ; cần tập mới độc lập để củng cố kết luận tổng quát hóa.
- Tỷ giá VND giữ nguyên, không biến Ames thành dữ liệu thị trường Việt Nam.

## Cách chạy

Xem README.md: bản đã huấn luyện chỉ cần chạy Flask và Express; không cần tuning để trình chiếu. Muốn tái lập: backup_original.py → tune_models.py --force → train_all.py --force → prepare_vnd_audit.py → export_vnd.py → run_round1_checks.py.
