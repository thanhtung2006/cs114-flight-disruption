"""Pipeline tiền xử lý và tạo đặc trưng."""
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu thô (xử lý missing, kiểu dữ liệu, trùng lặp...)."""
    df = df.drop_duplicates()
    # TODO: thêm các bước làm sạch
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo đặc trưng từ dữ liệu đã tiền xử lý."""
    # TODO: feature engineering (encoding, scaling, tạo cột mới...)
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Chạy toàn bộ: preprocess -> build_features."""
    return build_features(preprocess(df))
