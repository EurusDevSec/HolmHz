"""
Frequency Domain Detector — FFT-based AI image detection.

AI-generated images leave distinctive artifacts in the frequency domain:
- GANs: periodic grid patterns in the amplitude spectrum
- Diffusion models: different spectral roll-off at high frequencies
- All AI: phase spectrum inconsistencies vs natural images

Architecture:
  Image → Grayscale → FFT2D → Log(|Amplitude|) + Phase → 2-channel → Small CNN → logit

CNN backbone (custom, ~200K params):
  Conv2d(2, 32, 3) → BN → ReLU → MaxPool
  Conv2d(32, 64, 3) → BN → ReLU → MaxPool
  Conv2d(64, 128, 3) → BN → ReLU → AdaptiveAvgPool(7)
  Flatten → Linear(128*7*7, 256) → ReLU → Dropout → Linear(256, 1)

References:
  [1] Frank et al. "Leveraging Frequency Analysis for Deep Fake Image Recognition" (ICML 2020)
  [2] Dzanic et al. "Fourier Spectrum Discrepancies in Deep Network Generated Images" (NeurIPS 2020)
  [3] FreqNet (AAAI 2024): Source-agnostic frequency features for deepfake detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from holmhz.detectors.base import BaseDetector


class FrequencyCNN(nn.Module):
    """Small CNN for frequency spectrum classification.

    Input: [B, 2, 224, 224] — 2 channels: log amplitude + phase
    Output: [B, 1] — logit (fake score)

    Total params: ~215K (very lightweight, trains fast)
    """

    def __init__(self, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()

        # 3 conv blocks with batch normalization
        self.features = nn.Sequential(
            # Block 1: [B,2,224,224] → [B,32,112,112]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: [B,32,112,112] → [B,64,56,56]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: [B,64,56,56] → [B,128,7,7]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(7),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class FrequencyDetector(BaseDetector):
    """FFT-based detector for AI-generated image detection.

    Converts RGB images to frequency domain (FFT) and classifies
    using the amplitude + phase spectrum.

    This detector is immune to JPEG compression because frequency
    domain information is preserved after re-encoding.

    Args:
        dropout: dropout rate for classifier head (default: 0.3)
        use_phase: if True, use both amplitude and phase (2-channel input)
                   if False, use only amplitude (1-channel input)
    """

    def __init__(self, dropout: float = 0.3, use_phase: bool = True, **kwargs):
        super().__init__()
        self.use_phase = use_phase
        in_channels = 2 if use_phase else 1
        self.cnn = FrequencyCNN(in_channels=in_channels, dropout=dropout)

    def _rgb_to_gray(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB [B,3,H,W] to grayscale [B,1,H,W].

        Uses standard luminance weights: 0.299*R + 0.587*G + 0.114*B
        """
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
        return (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

    def _compute_spectrum(self, gray: torch.Tensor) -> torch.Tensor:
        """Compute FFT spectrum from grayscale image.

        Args:
            gray: [B, 1, H, W] grayscale image

        Returns:
            spectrum: [B, 1, H, W] (amplitude only) or [B, 2, H, W] (amp + phase)
        """
        # Remove channel dim for FFT: [B, H, W]
        img = gray.squeeze(1)

        # 2D FFT
        fft = torch.fft.fft2(img)

        # Shift zero frequency to center
        fft_shifted = torch.fft.fftshift(fft)

        # Log amplitude spectrum: log(|F| + 1) for better dynamic range
        amplitude = torch.log(torch.abs(fft_shifted) + 1.0)

        # Normalize amplitude to [0, 1] per sample
        b = amplitude.shape[0]
        amp_flat = amplitude.view(b, -1)
        amp_min = amp_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        amp_max = amp_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        amplitude = (amplitude - amp_min) / (amp_max - amp_min + 1e-8)

        if self.use_phase:
            # Phase spectrum: angle(F) normalized to [0, 1]
            phase = torch.angle(fft_shifted)
            phase = (phase + torch.pi) / (2 * torch.pi)  # [-π, π] → [0, 1]

            # Stack: [B, 2, H, W]
            return torch.stack([amplitude, phase], dim=1)
        else:
            # Only amplitude: [B, 1, H, W]
            return amplitude.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: RGB image → FFT spectrum → CNN → logit.

        Args:
            x: [B, 3, H, W] — normalized RGB images

        Returns:
            logits: [B, 1] — raw scores (before sigmoid)
        """
        gray = self._rgb_to_gray(x)
        spectrum = self._compute_spectrum(gray)
        logits = self.cnn(spectrum)
        return logits
