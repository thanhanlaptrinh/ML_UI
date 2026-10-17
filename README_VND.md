> Bản nâng cấp nhập liệu mới: xem [README_DYNAMIC.md](README_DYNAMIC.md). Nội dung dưới đây mô tả bản VND tham chiếu ban đầu.

# HomeValue Lab VND và kiểm chứng 30 căn

Bản này dùng dữ liệu và trọng số đã có của nhóm, bổ sung giao diện theo từng mô hình, phân rã cách dự đoán, bảng tính sai số và hai khu vực biểu đồ so sánh có điều khiển độc lập. Có sẵn đầy đủ 5 pipeline để chạy; **không cần huấn luyện lại để demo**.

## Chạy trên Windows

Mở PowerShell ở thư mục `house-price-ml` sau khi giải nén:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
```

Cửa sổ PowerShell thứ nhất:

```powershell
.\.venv\Scripts\python.exe service\predict_service.py
```

Cửa sổ PowerShell thứ hai, cùng thư mục dự án:

```powershell
npm start
```

Mở http://localhost:3000. Giữ cả hai cửa sổ khi demo; Ctrl+C để dừng. Node.js cần bản 18 trở lên, Python khuyến nghị 3.12 tương thích bộ thư viện đã lưu. Không chép `.venv`, `node_modules` hay `.git` từ máy khác vào dự án.

## Nội dung giao diện

- Trang **So sánh 5 mô hình**: biểu đồ 1 đối chiếu giá từng căn với nhãn thực tế, hoặc so sánh MAE trên toàn bộ 30 căn. Cột tốt nhất được tô xanh. Biểu đồ 2 cho chọn MAE, RMSE, R², MAPE và phạm vi train, holdout hoặc bộ 30 căn. Mỗi chỉ số có thang đo riêng; mỗi biểu đồ có nút cập nhật riêng. Chỉ số thấp hơn được chọn với MAE/RMSE/MAPE; cao hơn với R². Nếu bằng nhau trong sai số số thực, tất cả đồng hạng được đánh dấu.
- Mỗi mô hình có **một trang riêng**: nhập/chọn căn, tính dự đoán, xem giá nhãn và sai số; biểu đồ cột 30 cặp giá; biểu đồ train/test theo chỉ số; nhận xét dấu hiệu overfit/underfit; công thức và bảng tính từng căn.
- Mô hình tuyến tính hiện hệ số chặn và đóng góp tất cả đặc trưng; Decision Tree hiện đường đi đến lá; Random Forest hiện từng cây và trung bình; Gradient Boosting/XGBoost hiện các giá trị hiệu chỉnh và lũy kế từng vòng khi dự đoán.
- Sửa đặc trưng của một mẫu tạo ra căn giả định. Hệ thống bỏ nhãn “giá thực tế” cho căn đã sửa, vì không có giao dịch thật cho cấu hình mới đó. Khôi phục mẫu gốc để đối chiếu lại.

## VND và dữ liệu gốc

Giá gốc trong House Prices là USD. Bản này **quy đổi cả nhãn giá, dự đoán và chỉ số có đơn vị tiền sang VND**, không sửa target bên trong CSV và không đổi trọng số đã fit. Mặc định dùng 25.000 VND/USD, là tỷ giá cố định minh họa, không phải tỷ giá thời gian thực. Muốn dùng hệ số khác, đặt trước khi khởi động Flask:

```powershell
$env:USD_TO_VND="26000"
.\.venv\Scripts\python.exe service\predict_service.py
```

MAE và RMSE nhân tỷ giá; MSE nhân bình phương tỷ giá. R² và MAPE giữ nguyên. Mọi phép tính dùng số đầy đủ, chỉ làm tròn khi hiển thị. Nếu đổi tỷ giá, các file CSV/JSON xuất sẵn không tự đổi; tải bản mới qua nút trên giao diện hoặc chạy `scripts/export_vnd.py` trong cùng môi trường có biến USD_TO_VND.

Đây là nhà ở Ames, Iowa trong bộ Kaggle, **không phải 30 căn hộ có giao dịch VND tại Việt Nam**. Việc đổi tiền không làm mô hình trở thành mô hình thị trường Việt Nam.

## Bảo toàn trọng số và cách bổ sung

Các file dưới đây được giữ nguyên SHA-256 so với ZIP gốc đã cung cấp:

- `models/best_model.pkl`: XGBoost nguyên bản, tuyệt đối không fit lại hoặc ghi đè trong chức năng demo.
- `models/feature_defaults.json`.
- `data/train.csv`, `data/test.csv`.
- `results/model_results.csv`: bảng chỉ số USD lịch sử, được giữ làm bằng chứng đối chiếu; bảng quy đổi mới là `results/model_results_VND.csv`.

Repo gốc chỉ lưu **một** file trọng số của mô hình tốt nhất, không lưu bốn mô hình còn lại. Vì vậy bản này bổ sung `models/vnd_supplement/{linear,tree,forest,boosting}.pkl`, dùng nguyên cấu hình và cách chia của code cũ. Chỉ số toàn bộ holdout của bốn mô hình bổ sung khớp bảng lịch sử đến mức làm tròn đã lưu. Nguồn gốc của mỗi mô hình được ghi trên giao diện và trong bằng chứng JSON; không gọi bốn file bổ sung là trọng số được lưu từ lần chạy cũ.

**Không chạy lại `notebooks_or_src_py/train_compare.py` khi chỉ cần demo**, vì script lịch sử đó có thể ghi đè `best_model.pkl`. Các API mới chỉ đọc trọng số. `scripts/prepare_vnd_audit.py` chỉ fit file bổ sung còn thiếu; không ghi đè file bổ sung đã có, không thay đổi file gốc.

## Bộ 30 căn và ý nghĩa giá thực tế

Tái lập nguyên cách chia cũ: `train_test_split(test_size=0.2, random_state=42)` trên `data/train.csv`: 1.168 dòng train và 292 dòng holdout. Lấy 30 dòng từ holdout bằng `sample(n=30, random_state=20261015)`, không chọn mẫu theo độ đúng của dự đoán. Các ID cố định và chỉ mục được lưu trong `results/vnd_audit_source.json`; có kiểm tra không trùng ID train.

Nhãn thực tế là `SalePrice` trong CSV đã cung cấp. Cả 5 mô hình nhận toàn bộ đặc trưng của **đúng cùng bản ghi**, không thay phần còn lại bằng căn mặc định. ID cũng được giữ theo pipeline gốc để không đổi cách mô hình đã học. Dữ liệu thô, preprocessing, split và mô hình gốc đều có thể đối chiếu.

Code cũ đã dùng holdout để chọn best_model. Vì vậy các kết quả trình bày là kiểm tra tái lập và minh họa, không được gọi là kết quả cuối trên một tập chưa từng dùng trong lựa chọn mô hình. Không chỉnh mô hình để nó trùng giá 30 căn.

## Cách chứng minh phép tính

- `results/doi_chieu_30_can_VND.csv`: 150 dòng (30 căn × 5 mô hình), nhãn thực tế, dự đoán, sai số, sai số tuyệt đối, bình phương sai số và APE.
- `results/proof_vnd.json`: kết quả đầy đủ, hệ số quy đổi, ID train/holdout/demo, nguồn gốc mô hình và mã SHA-256.
- `results/model_results_VND.csv`: MAE, RMSE, MSE, R², MAPE cho train, holdout và demo30.
- `results/verification_vnd.json`: kết quả kiểm tra độc lập với sklearn và nhãn CSV gốc.

Công thức: eᵢ = ŷᵢ − yᵢ; MAE = Σ|eᵢ|/n; MSE = Σeᵢ²/n; RMSE = √MSE; R² = 1 − Σeᵢ²/Σ(yᵢ − ȳ)²; MAPE = Σ(|eᵢ|/yᵢ × 100%)/n. Bản dữ liệu này có nhãn giá dương nên MAPE không gặp mẫu giá bằng 0.

Kiểm tra lại mà không huấn luyện/ghi đè trọng số:

```powershell
.\.venv\Scripts\python.exe scripts\verify_vnd.py
# Khi Flask và Node đang chạy:
node scripts\smoke_http.mjs
```

Kiểm chứng này chứng minh các dự đoán đến từ file mô hình thực, phép phân rã khớp predict(), chỉ số khớp phép tính độc lập, nhãn khớp CSV gốc và trọng số gốc không bị đổi. **Nó không chứng minh dự đoán bằng đúng giá bán, không xác minh hợp đồng giao dịch bên ngoài và không bảo đảm giá thị trường hiện tại.**

## Chẩn đoán overfit và underfit

Dùng toàn bộ train và holdout để so sánh, có thêm baseline dự đoán giá trung bình train. Giao diện công khai ngưỡng minh họa:

- Dấu hiệu overfit: MAE holdout > 1,35 × MAE train **và** R² train − R² holdout > 0,08.
- Dấu hiệu underfit: R² train và holdout đều < 0,65 **và** MAE train ≥ 75% MAE baseline train.
- Nếu không thỏa các điều kiện, hiển thị “chưa thấy dấu hiệu rõ theo ngưỡng đã đặt”, không kết luận mô hình hoàn hảo.

Các ngưỡng này là cách chẩn đoán minh họa, không phải tiêu chuẩn phổ quát. Chỉ 30 mẫu không đủ để khẳng định chắc chắn overfit/underfit; cần learning curves/kiểm định bổ sung để kết luận mạnh hơn.
