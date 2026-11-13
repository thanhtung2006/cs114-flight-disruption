"""Pipeline dự đoán với model đã huấn luyện."""
import joblib
import pandas as pd

from src.pipelines import feature_eng_pipeline
from src.pipelines.utils import get_path


def predict(model, df: pd.DataFrame) -> pd.Series:
    """Dự đoán trên DataFrame đã có đặc trưng."""
    return pd.Series(model.predict(df), index=df.index, name="prediction")


def run(cfg: dict) -> pd.DataFrame:
    """Load model -> đọc dữ liệu -> tạo đặc trưng -> dự đoán -> lưu kết quả."""
    model = joblib.load(get_path(cfg["model"]["save_path"]))
    input_path = get_path(cfg["data"]["raw_dir"]) / cfg["data"]["inference_file"]
    df = feature_eng_pipeline.run(pd.read_csv(input_path))
    df = df.drop(columns=[cfg["target"]], errors="ignore")
    result = df.assign(prediction=predict(model, df))
    out_dir = get_path(cfg["data"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "predictions.csv", index=False)
    return result
