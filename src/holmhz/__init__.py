# src/holmhz/__init__.py

"""
HolmHz - Synthetic Image Detection System
==========================================

Hệ thống phát hiện ảnh tổng hợp bằng CNN với Explainable AI (Grad-CAM).

Modules:
    backbones   - Mạng trích xuất đặc trưng (EfficientNet-B0)
    detectors   - Bộ phát hiện (backbone + classification head)
    data        - Dataset classes và data transforms
    training    - Training loop, early stopping, schedulers
    losses      - Hàm mất mát (BCE)
    metrics     - Đánh giá (AUC, Accuracy)
    evaluation  - Benchmark và so sánh
    xai         - Giải thích mô hình (Grad-CAM)
    export      - Xuất model (ONNX)
    utils       - Tiện ích chung


"""
__version__ = "0.1.0"
