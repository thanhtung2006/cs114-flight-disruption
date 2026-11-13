"""Các hàm tiện ích dùng chung."""
import hashlib
import os
import platform
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Đọc file config YAML (đường dẫn tương đối tính từ thư mục gốc project)."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(relative: str | Path) -> Path:
    """Trả về đường dẫn tuyệt đối từ đường dẫn tương đối so với thư mục gốc."""
    return PROJECT_ROOT / relative


def file_md5(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Tính md5 của file (đọc theo khối để không tốn RAM)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_mb() -> float:
    """RAM (RSS) hiện tại của tiến trình, MB."""
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def peak_rss_mb() -> float:
    """RAM đỉnh của tiến trình từ lúc khởi động, MB.

    Windows có sẵn `peak_wset`; Linux/Kaggle dùng `ru_maxrss` (KB).
    """
    import psutil

    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / 2**20
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:  # pragma: no cover
        return info.rss / 2**20


def ram_status() -> dict:
    """RAM tổng và RAM trống của máy, GB."""
    import psutil

    vm = psutil.virtual_memory()
    return {"total_gb": round(vm.total / 2**30, 2), "available_gb": round(vm.available / 2**30, 2)}


def machine_info() -> dict:
    """Cấu hình máy và phiên bản thư viện chính (ghi kèm mọi số đo thời gian/RAM)."""
    import importlib.metadata as md

    import psutil

    versions = {}
    for pkg in ["pandas", "numpy", "pyarrow", "scikit-learn", "lightgbm", "psutil"]:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = "not installed"
    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_logical": psutil.cpu_count(logical=True) or os.cpu_count(),
        "python": sys.version.split()[0],
        **ram_status(),
        "versions": versions,
    }
