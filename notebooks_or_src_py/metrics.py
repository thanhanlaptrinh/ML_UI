"""
metrics.py — Các hàm đánh giá mô hình hồi quy: MAE, RMSE, R².
Người phụ trách: Thị Diễm.

Dùng lại trong train_compare.py khi so sánh 5 mô hình.

Cách test nhanh (không cần dữ liệu thật):
    python notebooks_or_src_py/metrics.py
"""

import numpy as np


def mae(y_true, y_pred) -> float:
    """Sai số tuyệt đối trung bình (Mean Absolute Error)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Căn bậc hai sai số bình phương trung bình (Root Mean Squared Error)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred) -> float:
    """Hệ số xác định R² — mức độ mô hình giải thích được biến thiên của dữ liệu."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def evaluate_all(y_true, y_pred) -> dict:
    """Gọi 1 lần, trả về cả 3 chỉ số — dùng trong train_compare.py."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ----------------------------------------------------------------------
# Test nhanh bằng dữ liệu giả (không cần train.csv thật)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    y_true_fake = [200000, 150000, 300000, 250000, 180000]
    y_pred_fake = [210000, 145000, 290000, 260000, 175000]

    print("=== Test 3 hàm metrics bằng dữ liệu giả ===")
    print("MAE :", mae(y_true_fake, y_pred_fake))
    print("RMSE:", rmse(y_true_fake, y_pred_fake))
    print("R2  :", r2_score(y_true_fake, y_pred_fake))
    print("\nGọi gộp bằng evaluate_all():")
    print(evaluate_all(y_true_fake, y_pred_fake))