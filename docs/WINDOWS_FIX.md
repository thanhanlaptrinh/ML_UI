# Sửa lỗi XGBoostError: input stream corrupted

Cài pip và npm đã thành công. Traceback dừng tại XGBoosterUnserializeFromBuffer khi joblib khôi phục snapshot XGBoost. Tệp này vẫn đọc được ở môi trường Linux tạo mô hình; bản sửa dùng XGBRegressor.save_model/load_model định dạng UBJSON thay cho Booster pickle ở đường suy luận.

## Máy đã cài dependencies

1. Dừng Flask/Node nếu đang chạy.
2. Giải nén gói WindowsFix rồi chép **toàn bộ nội dung thư mục house-price-ml** vào thư mục repo hiện dùng. Giữ .venv và node_modules đã cài; không cần cài lại hoặc train lại.
3. Terminal 1: `.\.venv\Scripts\python.exe service\predict_service.py`.
4. Terminal 2: `npm start`.
5. Mở http://localhost:3000 rồi Ctrl+F5. Health: http://localhost:3000/api/health trả JSON status=ok, models=5.

Đừng chỉ chép file .py: models/final/manifest.json, xgboost.ubj, xgboost_preprocess.pkl và results/vnd_audit_source.json phải cùng bản. PKL gốc và hash của 5 trọng số không thay đổi; manifest chỉ thêm thông tin portable, audit được tái xuất theo manifest mới.

## Kiểm tra sau sửa

```powershell
.\.venv\Scripts\python.exe scripts\verify_portable_xgboost.py
.\.venv\Scripts\python.exe scripts\verify_vnd.py
.\.venv\Scripts\python.exe scripts\verify_dynamic.py
.\.venv\Scripts\python.exe scripts\verify_training_report.py
node scripts\smoke_http.mjs
node scripts\smoke_dynamic_http.mjs
```

Smoke HTTP cần cả hai server. verify_portable dùng dự đoán từ snapshot gốc khi môi trường đọc được; nếu Windows không đọc được snapshot thì đối chiếu tham chiếu đã xuất từ snapshot gốc, kiểm tra hash dữ liệu/trọng số, rồi chặn hoàn toàn Booster.__setstate__ để chứng minh đường API không còn phụ thuộc XGBoost pickle.

Môi trường kiểm tra tại nơi tạo bản sửa là Linux, không có Windows để chạy native. Đã kiểm tra loader bằng cách chủ động gây lỗi mọi lần unpickle Booster; đây là kiểm tra đúng đường gây lỗi chứ không phải tuyên bố đã chạy native Windows.

## Xuất lại portable sau lần train hợp lệ

train_all.py tự tạo portable sau khi fit. Nếu chỉ cần chuyển định dạng của một PKL đọc được: chạy `python scripts/export_portable_xgboost.py --force`, sau đó `python scripts/prepare_vnd_audit.py` và `python scripts/export_vnd.py`. Đây là chuyển định dạng, không phải huấn luyện. Không chạy exporter trên chính môi trường đang không đọc được snapshot; gói này đã chứa sẵn các file portable.

Tham khảo chính thức: https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
