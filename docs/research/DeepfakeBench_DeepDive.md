# DeepfakeBench Deep Dive

## Overview

**DeepfakeBench** is a comprehensive benchmark framework for deepfake detection. It provides a modular architecture to train and evaluate various state-of-the-art detection methods on multiple datasets.

- **Repository**: [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)
- **Key Features**:
  - Unified codebase for 30+ detection methods.
  - Integrated datasets (FaceForensics++, DeepFakeDetection, etc.).
  - YAML-based configuration for easy experimentation.
  - Evaluation metrics (AUC, EER, etc.).

## Setup & configuration

The setup was challenging due to numerous dependencies and some outdated requirements.

### Dependencies

Successfully installed most dependencies on Python 3.12 (Windows), though some required workarounds:

- `torch`, `torchvision` (Standard)
- `efficientnet_pytorch`: Required for EfficientNet backbones.
- `timm`: Required for Vision Transformers and other backbones.
- `kornia`, `imageio`, `loralib`, `transformers`, `einops`, `scikit-image`.
- **Issues**:
  - `dlib`: Failed to install (requires C++ build tools). Mocked for inference.
  - `imgaug`: Compatible issues with NumPy 2.0 (`np.sctypes` removed).
  - `tensorboard`: Protocol Buffers version mismatch causes import crashes.

### Workarounds

To run inference without full installation:

1. **Mocking**: Used `unittest.mock` to bypass `tensorboard` and `dlib` in the inference script.
2. **registry**: Modified `detectors/__init__.py` to avoid importing broken modules (like `sladd_detector` which triggers `imgaug` crash).
3. **Manual Weights**: Downloaded `efficientnet-b4` weights manually and bypassed the internal download mechanism.

## Running Inference

Created a standalone inference script `training/test_deepfakebench.py` that utilizes the `EfficientDetector` class.

### Command

```bash
cd prac/ai-experiments/deepfake-detection/DeepfakeBench/training
python test_deepfakebench.py [image_path]
```

### Script Logic

1. Loads `efficientnetb4.yaml` config.
2. Initializes `EfficientDetector` (EfficientNet-B4 backbone).
3. Loads weights from `pretrained/effnb4_best.pth`.
4. Preprocesses image (Resize to 256x256, Normalize).
5. Outputs "Fake" or "Real" with probability.

## Code Analysis

### Architecture

- **Detector**: `AbstractDetector` (Base class) -> `EfficientDetector`.
  - `features(x)`: Extracts embeddings.
  - `classifier(x)`: Fully connected layer.
  - `forward(x)`: Combines features + classifier.
- **Backbone**: `EfficientNetB4` wrapper around `efficientnet_pytorch`.
- **Registry**: Uses a decorator-based registry (`@DETECTOR.register_module`) to manage models, making the code highly modular but harder to trace statically.

### Configuration

Managed via `training/config/detector/*.yaml`.

- Hyperparameters (batch size, learning rate).
- Augmentations (compression, blurring).
- Dataset selection.

## Validation Results (User Data)

Tests were conducted on local images providing a realistic assessment of the model's capabilities on modern AI-generated content (Gemini/Flux) versus standard datasets.

| Image Category | Source            | Prediction | Fake Prob  | Analysis                                                                      |
| :------------- | :---------------- | :--------- | :--------- | :---------------------------------------------------------------------------- |
| **Fake**       | Gemini Generated  | `FAKE`     | **0.5076** | **Borderline Fail**: Barely crossed the 0.5 threshold. The model is guessing. |
| **Fake**       | AI Gen (Unknown)  | `REAL`     | **0.4339** | **False Negative**: Failed to detect artifacts.                               |
| **Real**       | Camera (IMG_2344) | `REAL`     | **0.1883** | **Pass**: Correctly identified real photo with confidence.                    |

### Interpretation

The `EfficientNet-B4` model (trained on FaceForensics++) struggles significantly with modern diffusion-based images (Gemini). It lacks the "generalization" capabilities seen in CLIP-based models (UniversalFakeDetect).

## Findings for HolmHz

1. **Modularity**: The `Registry` pattern and config system are excellent for scaling a project with many models. We should adopt a simplified version of this.
2. **Dependency Management**: DeepfakeBench is "heavy". For HolmHz, we should strictly limit dependencies and favor modern, well-maintained libraries (e.g., `timm` over `efficientnet_pytorch` if possible/maintained).
3. **Preprocessing**: The framework relies heavily on `dlib` for face extraction. We need a robust, lighter alternative or a pre-processing pipeline that runs independently (e.g. `mediapipe`).
4. **Generalization**: The model trained on `FF-NT` (FaceForensics NeuralTextures) performed poorly on `ProGAN` images (CNNDetection). This emphasizes the need for diverse training data or generalized features (like `UniversalFakeDetect`).

## Next Steps

- Evaluate `UniversalFakeDetect` vs `DeepfakeBench` architectures.
- Decide on the "Backbone" for HolmHz (EfficientNet vs CLIP vs ResNet).
