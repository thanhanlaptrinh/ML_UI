# Render: File gốc data/train.csv đã thay đổi

Đã đối chiếu dữ liệu trên GitHub với gói đã cung cấp: các giá trị CSV/types hoàn toàn giống nhau; chỉ khác CRLF/LF. Git chuẩn hóa xuống dòng đã thay hash byte dù không đổi số liệu. Kết quả đối chiếu lưu tại results/render_failure_diagnosis.json.

Bản sửa bổ sung .gitattributes: source text=auto; CSV, JSON và model artifacts -text để không chuyển đổi byte. Giữ nguyên dữ liệu/trọng số và kiểm tra SHA256, không sinh lại hash để hợp thức hóa dữ liệu đã thay. scripts/verify_git_artifacts.py thực sự commit với core.autocrlf=true rồi clone ở cả true/false, so SHA256 mọi file được bảo vệ và khởi tạo đủ 5 mô hình.

## Áp dụng trên máy Windows đã có repo

1. Giải nén bản RenderFix rồi chép toàn bộ nội dung thư mục house-price-ml vào repo local đang đẩy lên GitHub. Phải chép cả .gitattributes và các file data/results/models nguyên bản, không chỉ file code. Giữ .git, .venv, node_modules đang có. Không dùng trình soạn thảo mở rồi lưu lại CSV/JSON.
2. Trong PowerShell tại repo:

```powershell
.\.venv\Scripts\python.exe scripts\verify_vnd.py
git add .gitattributes
git add --renormalize .
git add .
git commit -m "Preserve artifact bytes and fix Render SHA256 startup"
git push
```

Không cần git push --force. Không sửa CSV, không chạy lại tune_models/train_all để xử lý lỗi này.
3. Render giữ service hiện tại. Vào Manual Deploy → Deploy latest commit và kiểm tra đang dùng commit mới, không phải commit cũ 02abae2.

Build Command giữ:

```bash
python -m pip install -r requirements.txt && python -m pip install gunicorn
```

Start Command giữ:

```bash
python -m gunicorn --chdir service predict_service:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
```

Giữ PYTHON_VERSION=3.12.12. Đợi Live rồi mở URL API + /health. API thành công mới tiếp tục service Node.

## Tái lập kiểm tra Git

```powershell
.\.venv\Scripts\python.exe scripts\verify_git_artifacts.py
```

Cần Git trong PATH. Đây là các repo thử tạm trên máy, không push lên GitHub và không sửa .git của repo thật.
