"""Chạy huấn luyện: python -m entrypoint.train"""
from src.pipelines import training_pipeline
from src.pipelines.utils import load_config


def main():
    cfg = load_config()
    metrics = training_pipeline.run(cfg)
    print("Training done:", metrics)


if __name__ == "__main__":
    main()
