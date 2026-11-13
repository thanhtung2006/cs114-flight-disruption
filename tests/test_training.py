import numpy as np
import pandas as pd

from src.pipelines import training_pipeline


def test_train_returns_model_and_metrics():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=50), "x2": rng.normal(size=50)})
    df["target"] = (df["x1"] > 0).astype(int)
    cfg = {
        "target": "target",
        "model": {"params": {"n_estimators": 10, "random_state": 0}, "test_size": 0.2},
    }
    model, metrics = training_pipeline.train(df, cfg)
    assert hasattr(model, "predict")
    assert 0.0 <= metrics["val_accuracy"] <= 1.0
