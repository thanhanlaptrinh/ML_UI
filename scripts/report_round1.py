"""Generate the round-1 comparison document strictly from recorded numeric outputs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'results/tuning_report.json').read_text());tests=json.loads((ROOT/'results/round1_test_results.json').read_text())
lines=['# Lượt 1 — kết quả do code sinh','',f"Winner theo CV train: **{r['models'][r['winner']]['name']}**.",'','MAE dưới đây ở USD gốc. UI giữ nguyên lớp quy đổi VND.','',
'| Model | CV trước có Id | CV đối chứng không Id | CV sau | MAE holdout trước | MAE holdout sau | Target |',
'|---|---:|---:|---:|---:|---:|---|']
for m in r['models'].values():lines.append('| '+m['name']+' | '+' | '.join(f'{v:,.2f}' for v in [m['original_with_Id_cv']['mae'],m['original_without_Id_cv']['mae'],m['selected']['cv_mae'],m['before_usd']['test']['mae'],m['after_usd']['test']['mae']])+' | '+m['selected']['variant']+' |')
old=min(m['original_with_Id_cv']['mae'] for m in r['models'].values());new=r['models'][r['winner']]['selected']['cv_mae'];winner=r['models'][r['winner']]
lines+=['',f'CV MAE tốt nhất: {old:,.6f} → {new:,.6f} USD ({(new/old-1)*100:+.4f}%).',f"Winner MAE holdout/train = {winner['holdout_train_mae_ratio']:.6f}; R² train − holdout = {winner['r2_gap']:.6f}.",'',
'## Output thật của các kiểm tra','']
for t in tests['results']:lines+=['```text','$ '+' '.join(t['command']),t['stdout'].rstrip(),t['stderr'].rstrip(),'```','']
lines+=['## Giới hạn và phạm vi','',
'- Các baseline cũ được giữ trong models/_backup_original và results/baseline_original, kèm SHA-256. Dữ liệu gốc không sửa.',
'- 79 đặc trưng cuối, không Id. Kiểm tra thay Id trên cùng 200 căn và cả 5 pipeline cho chênh lệch chính xác bằng 0; kiểm tra thêm qua FlexibleEngine.',
'- CV trước có Id được tính lại bằng cùng KFold shuffled, khác CV lịch sử không shuffle. Giữ thêm đối chứng bỏ Id nhưng chưa đổi tham số.',
'- Không phải mọi chỉ số holdout đều cải thiện; RMSE và R² đầy đủ trong tuning_report/tuning_summary. Không retune dựa trên holdout.',
'- Winner còn khoảng cách train/holdout. Quy tắc chẩn đoán mới, tách membership chờ lượt 2; UI đầy đủ chờ lượt 3; dọn repo/nghiệm thu chờ lượt 4.',
'- XGBoost có ứng viên điều chuẩn/early stopping chỉ dùng train. Lựa chọn dựa trên CV, kể cả khi ứng viên đó không tốt hơn. Chi tiết fold_iterations trong tuning_report.',
'- Holdout đã được xem ở bản cũ; cần tập mới độc lập để củng cố kết luận tổng quát hóa.',
'- Tỷ giá VND giữ nguyên, không biến Ames thành dữ liệu thị trường Việt Nam.','',
'## Cách chạy','',
'Xem README.md: bản đã huấn luyện chỉ cần chạy Flask và Express; không cần tuning để trình chiếu. Muốn tái lập: backup_original.py → tune_models.py --force → train_all.py --force → prepare_vnd_audit.py → export_vnd.py → run_round1_checks.py.','']
(ROOT/'results/ROUND1_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
print('\n'.join(lines[:14]))
