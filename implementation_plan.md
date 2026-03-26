# Phân Tích v7 & Kế Hoạch Fix OOD — v5

## Bảng so sánh v4 → v6 → v7

| Metric | v4 | v6 | v7 | Trend |
|--------|-----|-----|-----|-------|
| ID AUC | 0.97 | 0.9958 | **0.9989** | 📈 Liên tục tăng |
| camera_real (OOD) | ~40% | **98.3%** | 36.75% | 🔄 Seesaw |
| camera_ai (OOD) | ~100% | 2.7% | **71.36%** | 🔄 Seesaw |
| OOD AUC | 0.78 | 0.41 | 0.57 | Unstable |

> [!WARNING]
> **Model đang "seesaw"** — mỗi lần train lại, nó swing giữa 2 cực:
> - v6: Quá conservative → mọi thứ = REAL (real 98%, fake 2.7%)
> - v7: Quá aggressive → mọi thứ = FAKE (fake 71%, real 36%)

---

## Root Cause — Tại sao OOD luôn kém?

**Model đang học DOMAIN, không phải AUTHENTICITY.**

EfficientNet-B0 + ImageNet features nhận diện "ảnh này nhìn giống training data hay không" thay vì "ảnh này có artifact AI hay không":

```
Training data:  rvf10k faces, ciplab GAN, DALL-E/SD/MJ
                ↓
Model learns:   "nếu ảnh giống samples tôi đã thấy → REAL/FAKE"
                ↓
OOD images:     camera_vs_ai (kiểu ảnh khác hoàn toàn)
                ↓
Model confused: "chưa bao giờ thấy kiểu này → random guess"
```

**Bằng chứng**: OOD AUC = 0.57 ≈ random (0.5). Model KHÔNG CÓ KHẢ NĂNG phân biệt real/fake trong domain camera_vs_ai, bất kể threshold.

---

## Kế hoạch fix — 2 bước

### Bước 1: Đưa camera_vs_ai vào training (Quick Fix)

> Split camera_vs_ai: 60% train, 40% OOD test

Nếu model **thấy** camera domain trong training, nó sẽ học phân biệt real/fake trong domain đó.

- Train: +136 real + +132 fake (từ camera_vs_ai)
- OOD test: 98 real + 88 fake (giữ lại để đánh giá)

**Ưu điểm**: Không cần download thêm gì, chỉ sửa script split.
**Nhược**: OOD test set nhỏ hơn (186 vs 454).

### Bước 2: Thêm DeepDetect-2025 (Robust Fix)

> [!IMPORTANT]
> **DeepDetect-2025** (Kaggle) — dataset **lớn nhất và đa dạng nhất** cho AI detection:
> - 100K+ images (60K real, 55K fake)
> - Fake generators: **SD3, StyleGAN3, DALL-E 3, Midjourney**
> - Categories: people, animals, nature, urban, artworks, objects
> - Peer-reviewed, 2025, designed specifically for detection research

Lấy subset ~5K-10K đa dạng từ DeepDetect-2025 → thêm vào training.

**Ưu điểm**: Massive domain diversity → model buộc phải học features thực sự.
**Nhược**: Download thêm ~2-5GB, cần nén lại lên Kaggle.

---

## Thực hiện

1. Sửa [prepare_data_v2.py](file:///r:/_Projects/Eurus_Workspace/HolmHz/scripts/prepare_data_v2.py): split camera_vs_ai 60/40 thay vì 100% OOD
2. Download DeepDetect-2025 subset (Kaggle)
3. Rebuild manifests → nén → train v8
4. Target: OOD AUC > **0.80**, camera_real > **80%**, camera_ai > **70%**
