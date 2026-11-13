"""Chạy dự đoán: python -m entrypoint.inference"""
from src.pipelines import inference_pipeline
from src.pipelines.utils import load_config


def main():
    cfg = load_config()
    result = inference_pipeline.run(cfg)
    print(f"Inference done: {len(result)} rows -> {cfg['data']['predictions_dir']}")


if __name__ == "__main__":
    main()
