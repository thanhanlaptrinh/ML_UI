## Bổ sung: trình bày kết quả của nhóm ở tab Train / Test

Mở http://localhost:3000: mặc định vào Train/Test. Bấm **Chạy kiểm nghiệm lại 5 mô hình · 292 căn** để chạy lại predict trên đúng 1.168 train / 292 holdout bằng 5 pipeline đã lưu. Không huấn luyện lại và không ghi đè trọng số.

Trang trình bày quy trình tiền xử lý/huấn luyện của repo, bảng lịch sử MAE/RMSE/R²/CV/thời gian từ `results/model_results.csv`, bảng chỉ số vừa tính lại, biểu đồ giá gốc–dự đoán của đủ 292 căn (phân trang), tỷ lệ sai lệch ≤5/10/20%, các căn sai lệch lớn nhất, overfit/underfit và giới hạn. Có tải minh chứng JSON / 1.460 dòng đối chiếu CSV (292 × 5).

Winner theo code nhóm là XGBoost (MAE holdout thấp nhất), nhưng Gradient Boosting có CV MAE thấp hơn theo bảng cũ. Không gọi winner là tối ưu toàn cục; không quy R² thành phần trăm dự đoán đúng. CV/thời gian là kết quả lịch sử, không phải huấn luyện vừa chạy.

Bấm **Ứng dụng XGBoost để dự đoán** để dùng chính `models/best_model.pkl`; hoặc **Đưa đủ 292 căn sang So sánh** để chấm lại trong tab 3. Cả ba tab dùng cùng đối tượng pipeline trong Flask; `POST /api/training-report` chỉ suy luận.

Repo được đối chiếu online tại commit `00e2ee36d4f9a8b2d18320644c0b2029860bdfb5`; chi tiết tại `results/repository_evidence.json`. XGBoost khớp byte; CSV khớp giá trị/types và mã nguồn khớp sau chuẩn hóa xuống dòng. Không đẩy thay đổi lên GitHub.

Kiểm tra bổ sung: `.venv\Scripts\python.exe scripts\verify_training_report.py`.

---

# Bản vá nhập liệu linh động và phân tách giao diện

Bản vá theo yêu cầu trong TXT giữ nguyên toàn bộ file dữ liệu gốc và cả 5 trọng số đã có ở bản VND. Các lần nhập mới chạy trực tiếp `.predict()` của mô hình; không lấy chỉ số 30 căn thay cho chỉ số của file mới. Không cần huấn luyện lại hoặc cài thêm thư viện ngoài requirements/package-lock hiện có.

## Chạy ứng dụng

PowerShell trong thư mục `house-price-ml`:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
```

Terminal 1:

```powershell
.\.venv\Scripts\python.exe service\predict_service.py
```

Terminal 2:

```powershell
npm start
```

**Không chạy lại `notebooks_or_src_py/train_compare.py` khi chỉ cần demo**, vì script lịch sử có thể ghi đè best_model.pkl. Các API mới chỉ suy luận.

Mở http://localhost:3000. Nếu chạy bản cũ trước đó, dừng hai tiến trình cũ bằng Ctrl+C rồi chạy lại và nhấn Ctrl+F5 trong trình duyệt. Bản mới mặc định mở tab dự đoán, không tự điền giá bán cho một căn nhập mới.

## Ba khu vực riêng

1. **Train / Test:** thống kê 1.168 train, 292 holdout, bộ 30 căn tham chiếu cố định; chỉ số train/holdout và chẩn đoán overfit/underfit gốc; nguồn dữ liệu và mã toàn vẹn.
2. **Dự đoán mô hình:** chọn một trong 5 mô hình; nhập bên trái, kết quả bên phải; phía dưới là biểu đồ cột so với baseline, MSE và phép tính đang dùng; dưới cùng là giai đoạn suy luận và trọng số/tham số tham chiếu.
3. **So sánh mô hình:** nhập CSV mới hoặc chọn bộ có sẵn, chọn 1–5 mô hình để chạy trên cùng các dòng; biểu đồ 1 có 5 màu, chuyển được giữa đường và cột nhóm, highlight mô hình có MAE tốt nhất nếu có nhãn; biểu đồ 2 chọn các chỉ số riêng. Hai nút cập nhật độc lập.

## Nhập một căn mới và tra giá thực tế

- Nhập thông tin trong tab dự đoán, chọn mô hình rồi nhấn **Dự đoán & tính lại**. Nếu có nhãn giá bán, nhập ô giá thực tế bằng VND.
- Có thể mở JSON bổ sung để gửi các cột khác như `TotalBsmtSF`, `GarageCars`, `KitchenQual`. Tên cột phải đúng schema cũ. Các cột hiển thị trên form được ưu tiên so với JSON bổ sung.
- Những cột không nhập dùng `feature_defaults.json` đã có. Số cột dùng mặc định được cảnh báo; kết quả đó là ước lượng trên đầu vào xấp xỉ, không phải dự đoán từ bộ đặc trưng đầy đủ.
- **Tra ID** tìm bất kỳ bản ghi nào có nhãn trong `data/train.csv`, không giới hạn 30 căn. **Lấy mẫu bất kỳ** cũng chọn từ toàn bộ các ID có nhãn. Bản ghi thuộc train được ghi rõ là đã dùng huấn luyện.
- Nếu sửa đặc trưng sau khi tra ID, nhãn giá tự tra bị xóa. Nếu nhập nhãn mới bằng tay, nguồn nhãn là “người dùng cung cấp”, không tự coi là giá giao dịch đã xác minh.
- Không thể tra giá bán thật của một cấu hình hoàn toàn mới chỉ bằng diện tích/năm xây. Cần dữ liệu giao dịch có nguồn. Ứng dụng chỉ tự xác nhận giá khớp **CSV gốc** khi toàn bộ đặc trưng khớp đúng bản ghi. Điều này không phải xác minh hợp đồng ngoài đời.

## CSV mới

CSV UTF-8, header dùng tên cột House Prices. Có thể dùng đầy đủ 80 đặc trưng để tránh điền mặc định, hoặc một phần đặc trưng và chấp nhận cảnh báo. Số dùng dấu chấm cho phần thập phân, không có dấu phân cách hàng nghìn; CSV dùng dấu phẩy phân cột và hỗ trợ các giá trị đặt trong ngoặc kép.

Hai lựa chọn cột nhãn, chỉ dùng một trong hai:

- `ActualPriceVND`: luôn là VND.
- `SalePrice`: chọn đúng USD hoặc VND trong ô **Đơn vị cột SalePrice** trước khi chạy. `SalePrice` trong Kaggle là USD.

Nếu không có cột nhãn, hoặc có dòng nhãn để trống, vẫn dự đoán mọi dòng. Chỉ số chỉ dùng những dòng có nhãn và ghi rõ số lượng. Không tự tạo nhãn cho dòng mới. Nếu toàn bộ file thiếu nhãn và không tra khớp được bản ghi gốc, không hiện chỉ số hay chọn “tốt nhất”. Ngoại lệ: một dòng đầy đủ đặc trưng và ID khớp chính xác CSV gốc có thể được tra nhãn tự động, và nguồn nhãn được ghi rõ.

Ví dụ CSV tối giản của người dùng:

```csv
SampleID,GrLivArea,LotArea,OverallQual,YearBuilt,ActualPriceVND
Can-A,2100,9000,8,2005,6500000000
Can-B,1800,8200,7,2000,5100000000
```

Hai giá trong ví dụ trên chỉ minh họa cú pháp nhãn do người dùng nhập, không phải giao dịch đã kiểm chứng. Nếu nhập giá tự đặt thì sai số cũng chỉ có ý nghĩa so với giá tự đặt đó.

Có hai file để thử tính năng nhập trong `public/examples/` và liên kết tải trên giao diện:

- `new_labeled_12_VND.csv`: 12 dòng **holdout gốc khác bộ 30**, đủ đặc trưng và nhãn gốc quy đổi VND. Dùng file này để chứng minh không bị rập khuôn 30 căn.
- `new_unlabeled_4.csv`: 4 dòng Kaggle test không có giá bán, dùng để kiểm tra không tự tạo giá thực tế.

Nút **Kaggle test.csv** nạp được toàn bộ 1.459 dòng test gốc. Cả file vẫn được dự đoán; biểu đồ chia trang 30 dòng để dễ đọc. `test.csv` cuộc thi không công khai SalePrice, nên chưa thể đánh giá đúng/sai trên file đó. `sample_submission.csv` chỉ minh họa định dạng nộp bài; không phải nhãn thật và không đủ đặc trưng để dự đoán, vì vậy không được dùng làm bằng chứng giá thực tế.

Tối đa 2.000 dòng và khoảng 4 MiB mỗi lượt. Nếu JSON đầy đủ lớn hơn giới hạn thì chia file. File mới nằm trong phiên trình duyệt, không ghi đè train/test hay trọng số. Có thể tải CSV/JSON kết quả mới để lưu lại.

## Chỉ số, trọng số và MSE

Chỉ số của lần nhập mới được tính trực tiếp từ dự đoán mới và nhãn hiện tại. Các dòng thiếu nhãn không được đưa vào mẫu số:

- MAE = Σ|ŷ−y|/n; MSE = Σ(ŷ−y)²/n; RMSE = √MSE; MAPE = Σ(|ŷ−y|/y × 100%)/n.
- R² chỉ hiện khi ít nhất hai nhãn có biến thiên. Một căn hoặc nhãn hằng số không bị gán R² giả.
- Với một căn: MAE = RMSE = |e| và MSE = e², không phải thước đo đại diện toàn thị trường.
- MSE cao/thấp được đối chiếu với baseline **trên chính các nhãn đang đánh giá**. Baseline luôn dự đoán trung bình giá train, không fit lại. Hiển thị phần trăm tăng/giảm MSE so baseline. Không có ngưỡng MSE cao/thấp phổ quát cho mọi thị trường.
- Bảng trọng số cho Linear Regression hiển thị hệ số VND, giá trị sau mã hóa và đóng góp. Random Forest có trọng số 1/số cây; boosting có learning_rate và số vòng; cây có lá và số mẫu train tại lá. Importance của các mô hình cây là importance toàn mô hình, không phải xác suất đúng hay đóng góp nhân quả cho căn đang nhập.
- Biểu đồ giai đoạn hiện giá lũy kế hoặc sai số bình phương khi có nhãn. Đây là **các phần của phép suy luận**, không phải lịch sử training loss. Thứ tự cộng đặc trưng tuyến tính không phải quá trình tối ưu.
- Không thể chẩn đoán overfit/underfit chắc chắn từ một căn hoặc chỉ một file test mới. Tab dự đoán dẫn tới đánh giá train/holdout gốc, không lấy nó làm kết luận của giá vừa nhập.

## API mới

- `GET /api/schema`: schema, các giá trị mặc định và danh sách ID gốc.
- `GET /api/record/1000`: tra đặc trưng đầy đủ và nhãn gốc VND của ID 1000.
- `GET /api/dataset/kaggle-test`: toàn bộ test.csv không nhãn.
- `POST /api/evaluate`: dự đoán 1–5 mô hình trên số dòng linh động, trả metric mới, nguồn nhãn từng dòng và cảnh báo. `explain: true` cho một dòng để phân rã mô hình.

```json
{"models":["linear","xgboost"],"target_currency":"VND","records":[{"features":{"GrLivArea":2100,"OverallQual":8},"actual_vnd":6500000000}],"explain":true}
```

Để kiểm tra một bản ghi gốc với đầy đủ đặc trưng: `records:[{"source_id":1000}]`. Nếu `features` ghi đè các giá trị của source_id thì chỉ tự dùng nhãn gốc nếu toàn bộ đặc trưng còn khớp.

## Kiểm tra lại

```powershell
.\.venv\Scripts\python.exe scripts\verify_dynamic.py
# Giữ Flask và Node đang chạy:
node scripts\smoke_dynamic_http.mjs
```

Các file gốc và trọng số không bị fit lại. Bằng chứng kiểm tra mới trong `results/verification_dynamic.json`; bằng chứng 30 căn cũ vẫn được giữ và đánh dấu là dữ liệu tham chiếu cũ. Mặc định dùng tỷ giá 25.000 VND/USD minh họa, không phải giá nhà Việt Nam hay tỷ giá trực tiếp.
