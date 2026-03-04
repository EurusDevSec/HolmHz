## 💡 Context

# Task 1.2 — Data Collection ✅ COMPLETED

> **Task ID**: S1-002  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ✅ COMPLETED (25/02/2026 — trước target 02/03)  
> **Created**: 10/02/2026  
> **Completed**: 25/02/2026  
> **Target**: ~~24/02/2026~~ → ~~02/03/2026~~ → ✅ 25/02  
> **Assignee**: Hoàng + Luân  
> **Blocked by**: ~~Không~~ Task 1.1 ✅ DONE  
> **Blocks**: S1-003 (Data Pipeline cần data) → ĐÃ UNBLOCK

> Thu thập dataset đa nguồn: Real + GAN + Diffusion.  
> **Kết quả**: 27,680 ảnh (26,500 train + 1,180 OOD test), tất cả resize 224×224 PNG.  
> **Budget thực tế**: $0 (tất cả nguồn miễn phí).

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **collect a diverse dataset covering both GAN and Diffusion-generated images** so that **the HolmHz model can detect modern AI-generated images, unlike the 3 SOTA models that all failed on Diffusion content.**

**Acceptance Criteria:**

- [x] ≥6k ảnh Real → **12,000** (CIFAKE Real 7K + FFHQ 5K) ✅
- [x] ≥5k ảnh Diffusion Fake → **9,500** (CIFAKE Fake 7K + SD v1.5 2.5K) ✅
- [x] ≥3k ảnh GAN Fake → **5,000** (StyleGAN từ 140k-real-and-fake) ✅
- [x] OOD test set → **1,180 ảnh**: Flux 80 + Real camera 100 + tristanzhang_fake 500 + real_pexels 500 ✅
- [x] Tất cả ảnh resize về 224×224 PNG → **27,680/27,680 valid** ✅
- [x] Folder structure: `data/processed/{train,ood_test}/` ✅
- [x] File `data/manifests/dataset_stats.json` → ✅ `all_criteria_pass: true`
- [x] Data integrity: `validate_dataset.py` → 0 corrupt, 0 wrong size ✅

> **Thay đổi so với kế hoạch ban đầu:**
>
> - Gemini OOD: ❌ Không có — `imagen-3.0-generate-001` deprecated, `gemini-2.5-flash-image` cần paid billing
> - DALL-E / Midjourney riêng: ❌ Không có — tristanzhang_fake đã chứa mixed SD+MJ+DALLE
> - Real camera: Dùng Unsplash API (100 portrait) thay vì tự chụp
> - Flux: HF Inference API (FLUX.1-schnell 57 ảnh + SD v1.5 fallback 23 ảnh) = 80 tổng
> - StyleGAN: Subset 5K từ Kaggle 140k-real-and-fake thay vì scrape thispersondoesnotexist.com

---

## 🛠️ Implementation

### Subtasks

- [x] 1.2.1 Download CIFAKE dataset (120K ảnh, Kaggle 1-click, ~500MB) — **Luân** ✅
- [x] 1.2.2 Download FFHQ subset (5K real, Kaggle mirror) + FFHQ full 52K backup — **Luân** ✅
- [x] 1.2.3 Subset StyleGAN faces (5K từ 140k-real-and-fake Kaggle dataset) — Hoàng ✅
- [x] 1.2.4 Self-generate SD v1.5 ảnh trên Colab (2.5K, `runwayml/stable-diffusion-v1-5`) — Hoàng ✅
- [x] 1.2.5 Chuẩn bị OOD: Flux 80 + Real camera 100 + tristanzhang 1K — Hoàng ✅
- [x] 1.2.6 Resize tất cả raw → 224×224 PNG vào `data/processed/` — `resize_all.py` ✅
- [x] 1.2.7 Tạo `dataset_stats.json` — `dataset_stats.py` → ALL CRITERIA PASS ✅
- [x] 1.2.8 Validate data integrity — `validate_dataset.py` → 27,680/27,680 valid ✅

### Branch & PR

- [x] Branch: `feat/s1/data-collection` ✅
- [ ] PR Created (pending commit)
- [x] Data integrity check → 0 corrupt, 0 wrong size, 0 zero bytes ✅
- [x] `dataset_stats.json` updated ✅

### Scripts đã tạo

| Script                         | Mô tả                                                      |
| ------------------------------ | ---------------------------------------------------------- |
| `scripts/subset_cifake.py`     | Random subset 7K từ CIFAKE (seed=42)                       |
| `scripts/subset_ffhq.py`       | Random subset 5K từ FFHQ                                   |
| `scripts/subset_stylegan.py`   | Subset 5K StyleGAN từ 140k-real-and-fake                   |
| `scripts/subset_ood_kaggle.py` | Subset tristanzhang_fake 500 + real_pexels 500             |
| `scripts/resize_all.py`        | Resize all raw → 224×224 PNG (có resume support)           |
| `scripts/dataset_stats.py`     | Tạo `data/manifests/dataset_stats.json` + acceptance check |
| `scripts/validate_dataset.py`  | Kiểm tra corrupt, wrong size, zero bytes                   |

---

## 📝 Notes

> **Kết quả thực tế (25/02/2026):**
>
> | Folder                        | Nguồn                           | Số ảnh     | Resolution gốc    |
> | ----------------------------- | ------------------------------- | ---------- | ----------------- |
> | `train/real/cifake`           | CIFAKE Real subset              | 7,000      | 32×32             |
> | `train/real/ffhq`             | FFHQ Kaggle mirror              | 5,000      | 512×512           |
> | `train/fake_gan/stylegan`     | 140k-real-and-fake subset       | 5,000      | 256×256           |
> | `train/fake_diffusion/cifake` | CIFAKE Fake subset              | 7,000      | 32×32             |
> | `train/fake_diffusion/sd15`   | Self-gen Colab                  | 2,500      | 512×512           |
> | `ood_test/tristanzhang_fake`  | tristanzhang32 test/fake        | 500        | 1024×1024         |
> | `ood_test/real_pexels`        | tristanzhang32 test/real        | 500        | ~4480×6272        |
> | `ood_test/flux`               | HF API FLUX.1-schnell + SD v1.5 | 80         | 1024×1024         |
> | `ood_test/real_camera`        | Unsplash API portraits          | 100        | ~400×446          |
> | **TỔNG**                      |                                 | **27,680** | **→ 224×224 PNG** |

> **Dữ liệu backup (raw, chưa dùng):**
>
> - `data/raw/cifake/` — 120K ảnh gốc (dùng subset 7K mỗi loại)
> - `data/raw/140k_real_and_fake/` — 140K StyleGAN (dùng subset 5K)
> - `data/raw/real/ffhq_full/` — 52K FFHQ (dùng subset 5K)
> - Có thể tăng data nếu AUC thấp

> **Budget thực tế**: $0 (Kaggle, HuggingFace, Unsplash — tất cả miễn phí).

> **→ Next**: Task 1.3 Data Pipeline — viết PyTorch Dataset class đọc từ `data/processed/`
