"""
export_defaults.py — Tính giá trị mặc định (median cho cột số, mode cho cột phân loại)
từ phần huấn luyện (không dùng holdout), để dùng khi ứng dụng demo chỉ hỏi người dùng một số ít thuộc tính
quan trọng thay vì toàn bộ 79 cột của bộ House Prices.

Chạy sau khi đã có data/train.csv:
    python notebooks_or_src_py/export_defaults.py

Kết quả: models/feature_defaults.json
"""

import os
import json
import pandas as pd

from preprocess import load_and_split

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def compute_defaults(X: pd.DataFrame) -> dict:
    defaults = {}
    for col in X.columns:
        if X[col].dtype in ["int64", "float64"]:
            defaults[col] = float(X[col].median())
        else:
            mode_val = X[col].mode(dropna=True)
            defaults[col] = mode_val.iloc[0] if not mode_val.empty else "Missing"
    return defaults


def main():
    X_train, _, _, _ = load_and_split()
    defaults = compute_defaults(X_train)

    out_path = os.path.join(MODELS_DIR, "feature_defaults.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(defaults, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu {len(defaults)} giá trị mặc định vào: {out_path}")


if __name__ == "__main__":
    main()