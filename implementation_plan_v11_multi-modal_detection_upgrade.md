# v11: Multi-Modal Detection Upgrade

Nâng cấp HolmHz từ 2-model ensemble → hệ thống phân tích đa tầng (multi-layer).
Mục tiêu: giảm false positive trên ảnh thật (iPhone, Facebook) từ ~15% xuống <5%.

## Phân tích hiện trạng

| Layer | Hiện tại (v10) | Mục tiêu (v11) |
|-------|:-:|:-:|
| Input | RGB image only | RGB + Frequency + Metadata |
| Models | 2 (EffNet + CLIP) | 3-4 (+ FFT + EXIF rule) |
| Explain | Grad-CAM | + Frequency heatmap |
| Speed | ~900ms | ~500ms (quantized) |

## User Review Required

> [!IMPORTANT]
> v11 chia 3 phase, mỗi phase có thể deploy **độc lập**. Recommend chạy Phase 1 trước (quick win), rồi Phase 2 (FFT — phù hợp khóa luận), Phase 3 (nếu cần thêm).

---

## Phase 1: Quick Wins (1-2 ngày) — EXIF Filter + Quantization

### Tại sao Phase 1 trước?
- EXIF filter: **0 training**, rule-based, fix ngay iPhone false positive
- Quantization: tăng tốc 3-5x, không giảm accuracy đáng kể

---

#### [NEW] [exif_analyzer.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/src/holmhz/analysis/exif_analyzer.py)

Rule-based EXIF metadata check:
```python
class EXIFAnalyzer:
    """Phân tích EXIF → confidence boost cho ảnh thật.
    
    Logic:
    - Có EXIF camera (Make, Model, FocalLength) → boost REAL +0.15
    - Có GPS data → boost REAL +0.05  
    - Có software tag (Photoshop/Lightroom) → neutral
    - Không có EXIF hoặc EXIF rỗng → không thay đổi (AI hoặc social media strip)
    """
    def analyze(self, image_path: str) -> dict:
        # Returns: {has_camera: bool, device: str, gps: bool, confidence_boost: float}
```

- Sử dụng `Pillow.ExifTags` (đã có sẵn trong deps)
- Không phải model, chạy <1ms
- Tích hợp vào `EnsemblePredictor.predict()` → điều chỉnh prob_fake

#### [MODIFY] [model_service.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/web/model_service.py)

- `EnsemblePredictor.predict()` thêm EXIF check
- Nếu EXIF có camera data → giảm prob_fake (boost REAL)
- Hiển thị EXIF info trong UI

#### [MODIFY] [app.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/web/app.py)

- Tab predict hiển thị EXIF metadata nếu có
- Tab explain hiển thị EXIF camera info

#### [NEW] [quantize_model.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/scripts/quantize_model.py)

- ONNX INT8 quantization cho EfficientNet (16MB → ~4MB, 3-5x faster)
- Sử dụng `onnxruntime.quantization`

---

## Phase 2: Frequency Domain Detector (3-5 ngày) — FFT Analysis

### Tại sao FFT?
- AI-generated images để lại **grid artifacts trong miền tần số** (đặc biệt GAN)
- Diffusion models có **spectral roll-off** khác biệt ở high-frequency
- Immune với JPEG compression (frequency info preserved after re-encode)
- **Rất phù hợp cho khóa luận/báo cáo khoa học**

---

#### [NEW] [freq_detector.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/src/holmhz/detectors/freq_detector.py)

```python
class FrequencyDetector(BaseDetector):
    """FFT-based detector — phân tích phổ tần số.
    
    Architecture:
    Image → FFT → Log Amplitude Spectrum → Small CNN (3 conv layers) → Linear → logit
    
    Input processing:
    1. Convert to grayscale
    2. Apply 2D FFT (torch.fft.fft2)
    3. Shift zero-frequency to center (fftshift)
    4. Take log(|amplitude| + 1)
    5. Feed 1-channel spectrum image to CNN
    
    CNN backbone: 3 conv blocks (32→64→128) + Global Average Pool + Linear(128→1)
    Total params: ~200K (very lightweight)
    """
```

- Trainable: ~200K params (rất nhẹ, train nhanh)
- Input: 224x224 grayscale → FFT spectrum → CNN
- Register: `DETECTOR_REGISTRY.register("freq_fft")`

#### [NEW] [freq_transforms.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/src/holmhz/data/freq_transforms.py)

- `get_freq_transforms()`: image → grayscale → FFT → log amplitude
- Tích hợp với DataLoader hiện có

#### [NEW] [train_v11_freq.yaml](file:///r:/_Projects/Eurus_Workspace/HolmHz/configs/train_v11_freq.yaml)

- Config train FrequencyDetector: 20 epochs, batch 64, lr 0.001
- Sử dụng cùng dataset v8 (không cần data mới)

#### [MODIFY] [model_service.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/web/model_service.py)

- Thêm `FreqPredictor` class
- [EnsemblePredictor](file:///r:/_Projects/Eurus_Workspace/HolmHz/web/model_service.py#128-188) nâng cấp: 3 models + EXIF
- Weight mới: EffNet 0.30 + CLIP 0.40 + FFT 0.20 + EXIF boost

#### [MODIFY] [app.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/web/app.py)

- UI hiện thêm FFT spectrum visualization
- Tab explain: show frequency heatmap cạnh Grad-CAM

#### [NEW] [KAGGLE_V11_TRAINING.md](file:///r:/_Projects/Eurus_Workspace/HolmHz/docs/KAGGLE_V11_TRAINING.md)

- Kaggle guide train FrequencyDetector
- Cùng dataset, chỉ thêm frequency preprocessing

---

## Phase 3: Multi-Scale Patch Analysis (5-7 ngày) — Nâng cao

> [!NOTE]
> Phase 3 là optional, chạy nếu Phase 1+2 chưa đủ performance target.

#### SRM Noise Filter + Patch-based Detection

- Cắt ảnh thành 4-16 patches ở resolution gốc (không resize 224x224)
- Apply SRM filter (3 high-pass kernels) → extract noise patterns
- MLP classifier trên noise features
- Giúp phát hiện AI artifacts ở chi tiết nhỏ (texture da, hạt nhiễu)

---

## Verification Plan

### Phase 1
```
# Test EXIF analyzer
python -c "from holmhz.analysis.exif_analyzer import EXIFAnalyzer; ..."

# Test quantized model speed
python scripts/quantize_model.py
python -c "# compare latency: FP32 vs INT8"
```

### Phase 2
```
# Train frequency detector on Kaggle
# Download best_v11_freq.pt
# Test ensemble with 3 models + EXIF
python web/app.py  # verify ensemble mode shows 3 models
```

### End-to-End
- Test iPhone photos (HEIC → JPEG) → target: <5% false positive
- Test Facebook re-compressed photos → target: <5% false positive
- Test AI-generated (Gemini, DALL-E, SD) → target: >95% detection

---

## Kiến trúc tổng thể v11

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│ EfficientNet │     │  CLIP ViT-L  │     │  FFT CNN   │
│  (ID: 0.30) │     │ (OOD: 0.40)  │     │ (Freq:0.20)│
└──────┬──────┘     └──────┬───────┘     └─────┬──────┘
       │                   │                   │
       └─────────┬─────────┴───────┬──────────┘
                 ▼                 ▼
         ┌──────────────┐  ┌────────────┐
         │   Weighted   │  │   EXIF     │
         │   Average    │←─│  Boost     │
         │  (0.3+0.4+0.2) │  │ (+0.15)   │
         └──────┬───────┘  └────────────┘
                ▼
         ┌──────────────┐
         │  Final Pred  │
         │ REAL / FAKE  │
         └──────────────┘
```
