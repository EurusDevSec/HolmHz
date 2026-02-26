"""
Registry Pattern — Factory cho Detectors và Backbones.

Tại sao cần Registry?
→ Tạo model theo tên (string) thay vì import trực tiếp
→ Config YAML chỉ đổi model.name → code không đổi
→ Thêm model mới: chỉ cần @register, code cũ không sửa

Pattern từ DeepfakeBench: @DETECTOR.register_module()

Usage:
    # Đăng ký
    @DETECTOR_REGISTRY.register("efficientnet_b0")
    class EfficientNetDetector(BaseDetector):
        ...

    # Tạo
    model = DETECTOR_REGISTRY.build("efficientnet_b0", pretrained=True)

    # Liệt kê
    print(DETECTOR_REGISTRY.list())  # ["efficientnet_b0"]
"""

from __future__ import annotations

from typing import Any


class Registry:
    """Registry quản lý mapping: tên (str) → class.

    Arguments:
        name: Tên registry (cho error messages), ví dụ "detector", "backbone"
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type] = {}

    def register(self, name: str):
        """Decorator đăng ký class vào registry.

        Args:
            name: Tên dùng để lookup, ví dụ "efficientnet_b0"

        Returns:
            Decorator function

        Raises:
            ValueError: Nếu tên đã được đăng ký (tránh ghi đè nhầm)
        """
        def decorator(cls):
            if name in self._registry:
                raise ValueError(
                    f"'{name}' already registered in {self.name} registry. "
                    f"Existing: {self._registry[name].__name__}"
                )
            self._registry[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs) -> Any:
        """Tạo instance từ tên đã đăng ký.

        Args:
            name: Tên đã register, ví dụ "efficientnet_b0"
            **kwargs: Arguments truyền vào constructor của class

        Returns:
            Instance của class đã đăng ký

        Raises:
            KeyError: Nếu tên chưa được đăng ký
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name](**kwargs)

    def get(self, name: str) -> type:
        """Lấy class (không tạo instance) từ tên.

        Hữu ích khi muốn kiểm tra class trước khi tạo instance.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def list(self) -> list[str]:
        """Liệt kê tất cả tên đã đăng ký."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Kiểm tra tên đã đăng ký chưa: 'efficientnet_b0' in registry."""
        return name in self._registry

    def __len__(self) -> int:
        """Số lượng classes đã đăng ký."""
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', items={self.list()})"


# === Global Registries ===
# Sử dụng trong toàn project

BACKBONE_REGISTRY = Registry("backbone")
DETECTOR_REGISTRY = Registry("detector")
