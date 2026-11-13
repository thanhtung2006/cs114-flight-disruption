"""Pipeline huấn luyện model."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.pipelines import feature_eng_pipeline
from src.pipelines.utils import get_path


def build_model(cfg: dict):
    """Khởi tạo model từ config."""
    # TODO: hỗ trợ thêm model khác theo cfg["model"]["name"]
    return RandomForestClassifier(**cfg["model"]["params"])


def train(df: pd.DataFrame, cfg: dict):
    """Huấn luyện model trên DataFrame đã có đặc trưng, trả về (model, metrics)."""
    target = cfg["target"]
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg["model"]["test_size"], random_state=42
    )
    model = build_model(cfg)
    model.fit(X_train, y_train)
    metrics = {"val_accuracy": accuracy_score(y_val, model.predict(X_val))}
    return model, metrics


def run(cfg: dict) -> dict:
    """Đọc dữ liệu raw -> tạo đặc trưng -> train -> lưu model."""
    raw_path = get_path(cfg["data"]["raw_dir"]) / cfg["data"]["raw_file"]
    df = feature_eng_pipeline.run(pd.read_csv(raw_path))
    model, metrics = train(df, cfg)
    save_path = get_path(cfg["model"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return metrics
