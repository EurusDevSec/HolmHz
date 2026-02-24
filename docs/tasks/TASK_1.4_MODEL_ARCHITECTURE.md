## 💡 Context

> **Task ID**: S1-004  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: ~~28/02/2026~~ → **07/03/2026** (song song với S1-003)  
> **Assignee**: Hoàng  
> **Blocked by**: S1-001 (env)  
> **Blocks**: S1-005 (Training cần model)

> Implement EfficientNet-B0 binary classifier theo pattern rút ra từ 3 SOTA projects.
> Key design: backbone + Linear(1280→1) + Sigmoid, freeze backbone ban đầu.

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Engineer**, I want to **implement the EfficientNet-B0 detector with a clean backbone/detector separation and registry pattern** so that **I can easily swap backbones later (e.g., to CLIP) and the architecture follows proven patterns from DeepfakeBench.**

**Acceptance Criteria:**

- [ ] `EfficientNetBackbone` class wraps timm model, returns 1280-dim features
- [ ] `EfficientNetDetector` class = backbone + Dropout(0.3) + Linear(1280→1) + Sigmoid
- [ ] Registry pattern: `get_detector("efficientnet_b0")` trả về đúng class
- [ ] Abstract base classes: `BaseBackbone`, `BaseDetector` với interface rõ ràng
- [ ] `num_classes=1` + BCELoss (binary, pattern từ CNNDetection + UniversalFakeDetect)
- [ ] Unit test: forward pass shape check (batch=4, img=224x224) → output shape (4, 1)
- [ ] Model params count ≤ 6M

---

## 🛠️ Implementation

### Subtasks

- [ ] 1.4.1 Implement `src/holmhz/backbones/base.py` + `efficientnet.py`
- [ ] 1.4.2 Implement `src/holmhz/detectors/base.py` + `efficientnet_detector.py`
- [ ] 1.4.3 Implement `src/holmhz/utils/registry.py` (factory pattern)
- [ ] 1.4.4 Unit test `tests/test_backbones.py` + `tests/test_detectors.py`

### Branch & PR

- [ ] Branch: `feat/s1/model-architecture`
- [ ] PR Created
- [ ] All tests passed
- [ ] Lint clean

---

## 📝 Notes

> **Pattern cần follow:**
>
> ```python
> # Registry (từ DeepfakeBench)
> DETECTOR_REGISTRY = {}
> def register_detector(name):
>     def decorator(cls):
>         DETECTOR_REGISTRY[name] = cls
>         return cls
>     return decorator
>
> # Detector (từ UniversalFakeDetect pattern: CLIP + Linear)
> @register_detector("efficientnet_b0")
> class EfficientNetDetector(BaseDetector):
>     def __init__(self, config):
>         self.backbone = EfficientNetBackbone(pretrained=True)
>         self.head = nn.Sequential(
>             nn.Dropout(0.3),
>             nn.Linear(1280, 1)  # num_classes=1, binary
>         )
>     def forward(self, x):
>         features = self.backbone(x)   # (B, 1280)
>         logits = self.head(features)   # (B, 1)
>         return torch.sigmoid(logits)   # P(Fake) ∈ [0,1]
> ```

> **Tại sao num_classes=1 + Sigmoid thay vì 2-class:**
>
> - CNNDetection dùng pattern này → đơn giản, hiệu quả
> - UniversalFakeDetect cũng dùng Linear(768→1) + Sigmoid
> - Output trực tiếp P(Fake), dễ interpret cho user
