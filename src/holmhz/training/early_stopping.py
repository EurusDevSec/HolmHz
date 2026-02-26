"""
Early Stopping — dừng training khi metric không cải thiện.

Tại sao cần?
→ Chống overfitting: model "học thuộc" train data nhưng fail trên val
→ Tiết kiệm thời gian: không cần train hết 30 epoch nếu đã hội tụ
→ Tiết kiệm GPU quota: Kaggle 30h/tuần, Colab 4h/session

Pattern từ:
- CNNDetection: earlystop.py (đơn giản, patience-based)
- DeepfakeBench: trainer callback (phức tạp hơn)
- HolmHz: giữ đơn giản + thêm state_dict cho checkpoint resume
"""


class EarlyStopping:
    """Dừng training khi metric không cải thiện sau `patience` epochs.

    Dùng cho monitor=val_auc (mode="max" — AUC càng cao càng tốt).

    Args:
        patience: số epochs chờ trước khi dừng (default 5)
        mode: "max" (metric cao = tốt) hoặc "min" (metric thấp = tốt)
        min_delta: cải thiện tối thiểu để tính là "cải thiện" (default 0.0)

    Example:
        >>> es = EarlyStopping(patience=5, mode="max")
        >>> es(0.85)  # First epoch → always best
        False
        >>> es(0.86)  # Improved → reset counter
        False
        >>> es(0.84)  # Worse → counter = 1
        False
        >>> # ... 4 more epochs without improvement ...
        >>> es(0.83)  # counter = 5 → STOP!
        True
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        # Internal state
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False
        self.is_best = False

    def __call__(self, metric: float) -> bool:
        """Kiểm tra metric mới, cập nhật state.

        Args:
            metric: giá trị metric mới (ví dụ val_auc)

        Returns:
            True nếu nên dừng (patience hết), False nếu tiếp tục
        """
        if self.best_score is None:
            # Epoch đầu tiên — luôn là best
            self.best_score = metric
            self.is_best = True
        elif self._is_improvement(metric):
            # Metric cải thiện → reset counter
            self.best_score = metric
            self.counter = 0
            self.is_best = True
        else:
            # Không cải thiện → tăng counter
            self.counter += 1
            self.is_best = False
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        """Kiểm tra score mới có "đủ tốt hơn" best_score không."""
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:  # mode == "min"
            return score < self.best_score - self.min_delta

    def state_dict(self) -> dict:
        """Lưu state cho checkpoint resume.

        ⚠️ QUAN TRỌNG: Nếu không save state, resume sẽ reset counter
        → model train thêm patience epochs vô nghĩa.
        """
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state từ checkpoint.

        Gọi khi resume training để tiếp tục đếm patience.
        """
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.should_stop = state["should_stop"]

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, mode='{self.mode}', "
            f"counter={self.counter}, best={self.best_score})"
        )
