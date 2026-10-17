"""
preprocess.py — Tiền xử lý dữ liệu House Prices: missing values, encoding, chia Train/Test.
Người phụ trách: Thị Diễm.

Dùng lại trong train_compare.py — mọi mô hình PHẢI dùng chung hàm build_pipeline()
và cùng RANDOM_SEED để việc so sánh 5 thuật toán công bằng.

Cách chạy thử độc lập:
    python notebooks_or_src_py/preprocess.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = os.path.join("data", "train.csv")
TARGET = "SalePrice"
RANDOM_SEED = 42  # cố định cho TẤT CẢ mô hình — không được đổi giữa chừng


def load_and_split(path: str = DATA_PATH, test_size: float = 0.2):
    """Đọc dữ liệu, tách X/y, chia Train/Test với seed cố định."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET, "Id"])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    print(f"Train: {X_train.shape[0]} dòng | Test: {X_test.shape[0]} dòng")
    return X_train, X_test, y_train, y_test


def build_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Xây bộ tiền xử lý dùng chung cho mọi mô hình:
    - Biến số: điền missing bằng median.
    - Biến phân loại: điền missing bằng nhãn 'Missing', rồi One-Hot Encoding.

    Lưu ý: đây chỉ là bước preprocessing chung ban đầu. Nếu Diễm quyết định
    một số cột có thứ bậc (Ordinal) thay vì One-Hot, sửa trong hàm này —
    KHÔNG sửa ở train_compare.py để tránh mỗi mô hình xử lý khác nhau.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    assert "Id" not in numeric_cols + categorical_cols, "Id is metadata only"
    return preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_pipeline(X_train)

    # Test thử fit_transform để chắc pipeline chạy không lỗi
    X_train_transformed = preprocessor.fit_transform(X_train)
    print(f"Sau khi tiền xử lý: {X_train_transformed.shape[1]} cột (đã One-Hot Encoding).")