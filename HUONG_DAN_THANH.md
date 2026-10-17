> Hướng dẫn lịch sử. Bản UI và API đang dùng xem README_DYNAMIC.md; không dùng lệnh huấn luyện trong tài liệu cũ để chạy demo mới.

# Phần thực hiện của Thành — demo với mô hình thật

Theo huong_dan_vinh_thanh.docx: Thành chạy service Python, server Express và kiểm tra giao diện. ZIP đã có best_model.pkl và feature_defaults.json nên không cần huấn luyện lại.

## Chạy trên Windows

Cần Python 3.11 trở lên (khuyến nghị 3.12) và Node.js. Giải nén, mở PowerShell trong thư mục có server.js.
Nếu dùng repo hiện tại: chép các file sửa từ ZIP vào repo của bạn; không ghi đè bằng cả một thư mục Git mới.

Cài thư viện một lần:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
```

Terminal 1 — giữ chạy:

```powershell
.\.venv\Scripts\python.exe service/predict_service.py
```

Terminal 2 — cùng thư mục dự án, giữ chạy:

```powershell
npm start
```

Mở http://localhost:3000, nhập ví dụ:
- LotArea: 8500 sqft
- GrLivArea: 1500 sqft
- OverallQual: 6
- YearBuilt: 2005
- Neighborhood: CollgCr
- GarageType: Attchd

Bấm Dự đoán giá nhà. Có thể để trống: service điền trung vị/mode cho các thuộc tính thiếu. Kết quả bằng USD, từ mô hình XGBoost đã có của nhóm.

Kiểm tra service: http://127.0.0.1:5001/health trả status ok.
Dừng bằng Ctrl+C trong cả hai terminal.

## Các file sửa

- requirements.txt: bổ sung thư viện; scikit-learn 1.9.1 khớp phiên bản lưu trong pickle.
- public/index.html: sửa lỗi inline display:none khiến kết quả bị ẩn; ghi đúng median/mode, USD, ví dụ khu vực và thêm viewport.
- service/predict_service.py: đường dẫn model theo vị trí file, tắt debug/reloader để nạp một lần.
- server.js: đường dẫn public theo vị trí file.
- package.json: thêm npm start.

Mô hình, dữ liệu, bảng kết quả và code huấn luyện của Vinh được giữ nguyên.
Xem KET_QUA_DEMO_THANH.json để biết kết quả kiểm tra thực tế trong môi trường của trợ lý.
Chỉ best_model.pkl được nạp ở demo; bảng so sánh năm mô hình nằm trong results/model_results.csv.

## Kết quả đã kiểm tra

Ví dụ sáu trường ở trên trả 173391.52 USD; để trống toàn bộ trả 162012.14 USD. Giá API trùng giá dự đoán trực tiếp từ model. Input LotArea="abc" trả HTTP 400. Script giao diện đã được chạy bằng jsdom với API thật, kết quả có display:block. Chưa kiểm tra trực quan bằng trình duyệt đồ họa; bạn cần mở localhost:3000 trên máy Windows để xác nhận bố cục.
