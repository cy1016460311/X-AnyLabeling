import os
import multiprocessing

from anylabeling.config import get_work_directory


def get_trainer_root_dir():
    return os.path.join(
        get_work_directory(), "xanylabeling_data", "trainer", "ultralytics"
    )


def get_data_path():
    return os.path.join(get_trainer_root_dir(), "data.yaml")


def get_dataset_path():
    return os.path.join(get_trainer_root_dir(), "datasets")


def get_settings_config_path():
    return os.path.join(get_trainer_root_dir(), "settings.json")


def get_default_project_dir():
    return os.path.join(get_trainer_root_dir(), "runs")


# UI configuration
DEFAULT_WINDOW_TITLE = "Ultralytics Training Platforms 🚀"
DEFAULT_WINDOW_SIZE = (1200, 800)  # (w, h)
ICON_SIZE_NORMAL = (32, 32)
ICON_SIZE_SMALL = (16, 16)

# Task configuration
TASK_TYPES = ["Classify", "Detect", "OBB", "Segment", "Pose"]
TASK_SHAPE_MAPPINGS = {
    "Classify": ["flags"],
    "Detect": ["rectangle"],
    "OBB": ["rotation"],
    "Segment": ["polygon"],
    "Pose": ["point"],
}
TASK_LABEL_MAPPINGS = {
    "Classify": "classify",
    "Detect": "hbb",
    "OBB": "obb",
    "Segment": "seg",
    "Pose": "pose",
}

# Training configuration
MIN_LABELED_IMAGES_THRESHOLD = 20
NUM_WORKERS = multiprocessing.cpu_count()
DEFAULT_TRAINING_CONFIG = {
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "workers": 8,
    "classes": "",
    "single_cls": False,
    "time": 0,
    "patience": 100,
    "close_mosaic": 10,
    "optimizer": "auto",
    "cos_lr": False,
    "amp": True,
    "multi_scale": False,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "dropout": 0.0,
    "fraction": 1.0,
    "rect": False,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 2.0,
    "save_period": -1,
    "val": True,
    "plots": False,
    "save": True,
    "resume": False,
    "cache": False,
}
OPTIMIZER_OPTIONS = [
    "auto",
    "SGD",
    "Adam",
    "AdamW",
    "NAdam",
    "RAdam",
    "RMSProp",
]
TRAINING_STATUS_COLORS = {
    "idle": "#6c757d",
    "training": "#6f42c1",
    "completed": "#28a745",
    "error": "#ffc107",
}
TRAINING_STATUS_TEXTS = {
    "idle": "Ready to train",
    "training": "Training in progress",
    "completed": "Training completed",
    "error": "Training error",
}


# Env Check
def is_torch_available() -> bool:
    try:
        import torch

        return hasattr(torch, "__version__")
    except Exception:
        return False


def is_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() if is_torch_available() else False
    except Exception:
        return False


def is_mps_available() -> bool:
    try:
        import torch

        return (
            torch.backends.mps.is_available()
            if is_torch_available()
            else False
        )
    except Exception:
        return False


def get_device_options():
    return (
        (["cuda"] if is_cuda_available() else [])
        + (["mps"] if is_mps_available() else [])
        + ["cpu"]
    )


def get_cuda_unavailable_reason() -> str:
    if not is_torch_available():
        return "PyTorch is not installed in the current runtime."

    try:
        import torch

        if torch.cuda.is_available():
            return ""

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices == "-1":
            return "CUDA is disabled by environment variable CUDA_VISIBLE_DEVICES=-1."

        if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda, "is_built"
        ):
            if not torch.backends.cuda.is_built():
                return (
                    "The current PyTorch build is CPU-only. Install a CUDA-enabled "
                    "PyTorch build to train on GPU."
                )

        if getattr(torch.version, "cuda", None) is None:
            return (
                "The current PyTorch build does not include CUDA support. "
                "Install a CUDA-enabled PyTorch build to train on GPU."
            )

        if torch.cuda.device_count() == 0:
            return (
                "CUDA support is present, but no usable NVIDIA GPU was detected "
                "by PyTorch."
            )

        return "CUDA is currently unavailable in the active runtime."
    except Exception as e:
        return f"Failed to detect CUDA availability: {e}"


IS_TORCH_AVAILABLE = is_torch_available()
IS_CUDA_AVAILABLE = is_cuda_available()
IS_MPS_AVAILABLE = is_mps_available()
DEVICE_OPTIONS = get_device_options()
