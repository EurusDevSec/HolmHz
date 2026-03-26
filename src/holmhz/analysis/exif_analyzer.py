"""
EXIF Analyzer — Rule-based metadata analysis for Real/Fake detection.

Phân tích EXIF metadata để hỗ trợ phân loại:
- Ảnh thật từ camera/phone luôn có EXIF (Make, Model, FocalLength, GPS)
- Ảnh AI-generated thường KHÔNG có EXIF hoặc có metadata giả lập
- Social media (Facebook, Instagram) XÓA EXIF khi upload

Logic (Hard Constraint — theo feedback chuyên gia):
- Có camera EXIF → giảm threshold (giảm mạnh prob_fake)
- Không có EXIF → NEUTRAL (không phạt — có thể là ảnh thật bị strip)
- Có AI software tag → tăng nhẹ prob_fake

Tham khảo:
- EXIF là lớp lọc đầu tiên (rule-based), chạy <1ms
- Không thay thế model nhưng bổ trợ rất tốt cho False Positive

Usage:
    analyzer = EXIFAnalyzer()
    result = analyzer.analyze(image)  # PIL Image hoặc file path
    # result = {has_camera: True, device: "Apple iPhone 15 Pro", ...}
"""

from pathlib import Path
from typing import Optional, Union

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


# Camera manufacturers — ảnh có Make thuộc list này → chắc chắn từ camera thật
KNOWN_CAMERA_MAKERS = {
    "apple", "samsung", "google", "huawei", "xiaomi", "oppo", "vivo",
    "sony", "canon", "nikon", "fujifilm", "panasonic", "olympus",
    "leica", "hasselblad", "pentax", "ricoh", "motorola", "oneplus",
    "lg", "nokia", "realme", "honor", "asus", "nothing",
}

# AI/editing software — nếu có tag này trong Software → có thể là AI hoặc edited
AI_SOFTWARE_TAGS = {
    "midjourney", "dall-e", "dalle", "stable diffusion", "stablediffusion",
    "novelai", "artbreeder", "deepai", "runway", "firefly",
    "comfyui", "automatic1111", "invoke", "flux",
}

# Photo editing software — neutral (không phạt, ảnh thật cũng chỉnh sửa)
EDITING_SOFTWARE = {
    "photoshop", "lightroom", "snapseed", "vsco", "afterlight",
    "darkroom", "procreate", "gimp", "capture one", "affinity",
}


class EXIFAnalyzer:
    """Rule-based EXIF metadata analyzer.

    Analyze EXIF to produce a confidence adjustment for the ensemble.

    Strategy (Hard Constraint):
    - Strong camera EXIF → scale down prob_fake by multiplier (e.g., 0.5x)
    - GPS present → additional scale down (e.g., 0.85x)
    - No EXIF → neutral (multiplier = 1.0)
    - AI software tag → scale up (e.g., 1.2x)
    """

    def __init__(
        self,
        camera_multiplier: float = 0.5,    # p_fake *= 0.5 nếu có camera
        gps_multiplier: float = 0.85,       # p_fake *= 0.85 thêm nếu có GPS
        ai_software_multiplier: float = 1.2,  # p_fake *= 1.2 nếu có AI software
    ):
        self.camera_mult = camera_multiplier
        self.gps_mult = gps_multiplier
        self.ai_mult = ai_software_multiplier

    def analyze(self, image: Union[Image.Image, str, Path]) -> dict:
        """Analyze EXIF metadata from image.

        Args:
            image: PIL Image, file path str, or Path object

        Returns:
            {
                has_camera: bool,       # Có camera Make/Model?
                device: str|None,       # "Apple iPhone 15 Pro" hoặc None
                has_gps: bool,          # Có GPS data?
                software: str|None,     # Software tag (nếu có)
                is_ai_software: bool,   # Software là AI generator?
                is_editing_software: bool,
                multiplier: float,      # Multiplier cho prob_fake
                exif_summary: str,      # Human-readable summary
            }
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(str(image))
            except Exception:
                return self._empty_result("Could not open image")

        # Extract EXIF
        exif_data = self._extract_exif(image)
        if not exif_data:
            return self._empty_result("No EXIF data found")

        # Parse fields
        make = exif_data.get("Make", "").strip()
        model = exif_data.get("Model", "").strip()
        software = exif_data.get("Software", "").strip()
        focal_length = exif_data.get("FocalLength")
        exposure_time = exif_data.get("ExposureTime")
        f_number = exif_data.get("FNumber")
        has_gps = "GPSInfo" in exif_data

        # Device string
        device = None
        if make and model:
            device = f"{make} {model}"
        elif model:
            device = model
        elif make:
            device = make

        # Check if camera is known
        has_camera = False
        if make:
            has_camera = make.lower() in KNOWN_CAMERA_MAKERS
        if not has_camera and device:
            # Fuzzy match
            device_lower = device.lower()
            has_camera = any(m in device_lower for m in KNOWN_CAMERA_MAKERS)

        # Additional camera evidence: focal length + exposure = real camera
        has_optics = focal_length is not None and (exposure_time is not None or f_number is not None)
        if has_optics and not has_camera:
            has_camera = True  # Has optical data → real camera

        # Check software
        is_ai = False
        is_editing = False
        if software:
            sw_lower = software.lower()
            is_ai = any(ai in sw_lower for ai in AI_SOFTWARE_TAGS)
            is_editing = any(ed in sw_lower for ed in EDITING_SOFTWARE)

        # Calculate multiplier (Hard Constraint)
        multiplier = 1.0
        if is_ai:
            multiplier *= self.ai_mult   # Increase prob_fake
        elif has_camera:
            multiplier *= self.camera_mult  # Decrease prob_fake strongly
            if has_gps:
                multiplier *= self.gps_mult  # Even more decrease

        # Summary
        parts = []
        if device:
            parts.append(f"📷 {device}")
        if has_gps:
            parts.append("📍 GPS")
        if software and (is_ai or is_editing):
            parts.append(f"💻 {software}")
        if has_camera:
            parts.append(f"→ REAL boost ({multiplier:.2f}x)")
        elif is_ai:
            parts.append(f"→ AI suspicion ({multiplier:.2f}x)")

        exif_summary = " | ".join(parts) if parts else "EXIF present but no camera data"

        return {
            "has_camera": has_camera,
            "device": device,
            "has_gps": has_gps,
            "software": software if software else None,
            "is_ai_software": is_ai,
            "is_editing_software": is_editing,
            "multiplier": multiplier,
            "exif_summary": exif_summary,
        }

    def _extract_exif(self, image: Image.Image) -> Optional[dict]:
        """Extract EXIF as {tag_name: value} dict."""
        try:
            raw_exif = image._getexif()
            if raw_exif is None:
                return None

            exif = {}
            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                exif[tag_name] = value
            return exif
        except (AttributeError, Exception):
            # Image format doesn't support EXIF or corrupt data
            return None

    def _empty_result(self, reason: str = "") -> dict:
        """Return neutral result (no EXIF = neutral, NOT suspicious)."""
        return {
            "has_camera": False,
            "device": None,
            "has_gps": False,
            "software": None,
            "is_ai_software": False,
            "is_editing_software": False,
            "multiplier": 1.0,  # Neutral — không phạt
            "exif_summary": reason or "No EXIF data",
        }
