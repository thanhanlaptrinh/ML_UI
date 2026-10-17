"""
eda.py — Phân tích khám phá dữ liệu (EDA) cho bộ House Prices.
Người phụ trách: Nguyễn Phúc Khang.

Cách chạy:
    python notebooks_or_src_py/eda.py

Yêu cầu: đã có file data/train.csv (tải từ Kaggle, xem README.md).
Kết quả: in thống kê ra terminal + lưu biểu đồ vào thư mục results/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# 0. Cấu hình đường dẫn
# ----------------------------------------------------------------------
DATA_PATH = os.path.join("data", "train.csv")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET = "SalePrice"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Đã đọc dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột.")
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("\n=== 1. Kiểu dữ liệu từng cột (df.info()) ===")
    df.info()

    print("\n=== 2. Thống kê mô tả các cột số (df.describe()) ===")
    print(df.describe())


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 3. Tỷ lệ missing values theo cột (chỉ cột có thiếu) ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"so_luong_thieu": missing, "ty_le_%": missing_pct})
    report = report[report["so_luong_thieu"] > 0].sort_values("ty_le_%", ascending=False)
    print(report)

    # Lưu ra CSV để đưa vào báo cáo / họp nhóm
    out_path = os.path.join(RESULTS_DIR, "missing_values_report.csv")
    report.to_csv(out_path, encoding="utf-8-sig")
    print(f"Đã lưu báo cáo missing values vào: {out_path}")
    return report


def duplicate_check(df: pd.DataFrame) -> None:
    n_dup = df.duplicated().sum()
    print(f"\n=== 4. Số bản ghi trùng lặp: {n_dup} ===")


def saleprice_distribution(df: pd.DataFrame) -> None:
    print("\n=== 5. Phân phối SalePrice ===")
    print(df[TARGET].describe())

    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET], kde=True, bins=40)
    plt.title("Phân phối SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Số lượng căn nhà")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "saleprice_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ phân phối SalePrice vào: {out_path}")


def correlation_with_target(df: pd.DataFrame, top_n: int = 15) -> None:
    print(f"\n=== 6. Top {top_n} thuộc tính số có tương quan mạnh nhất với SalePrice ===")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()[TARGET].sort_values(ascending=False)
    top_corr = corr.drop(TARGET).head(top_n)
    print(top_corr)

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        numeric_df[top_corr.index.tolist() + [TARGET]].corr(),
        annot=True, fmt=".2f", cmap="coolwarm", square=True
    )
    plt.title(f"Ma trận tương quan — Top {top_n} thuộc tính vs SalePrice")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu heatmap tương quan vào: {out_path}")


def main():
    df = load_data(DATA_PATH)
    basic_info(df)
    missing_values_report(df)
    duplicate_check(df)
    saleprice_distribution(df)
    correlation_with_target(df)
    print("\nHoàn tất EDA. Xem kết quả trong thư mục results/.")


if __name__ == "__main__":
    main()