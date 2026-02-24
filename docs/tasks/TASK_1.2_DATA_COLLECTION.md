## 💡 Context

# Task 1.2 — Data Collection (Revised 24/02/2026)

> **Task ID**: S1-002  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: ~~24/02/2026~~ → **02/03/2026** (1 tuần từ hôm nay)  
> **Assignee**: Hoàng + Luân  
> **Blocked by**: ~~Không~~ Task 1.1 ✅ DONE  
> **Blocks**: S1-003 (Data Pipeline cần data)

> Thu thập dataset đa nguồn: Real + GAN + Diffusion.
> ⚠️ BÀI HỌC #1: Training data quyết định thành bại. Diffusion data là BẮT BUỘC.
>
> ⚠️ **Cập nhật 24/02/2026**: Thay GenImage (~50GB, khó download) bằng CIFAKE (Kaggle, ~500MB, 1-click).
> Giảm scope: 45k → 15-20k ảnh — vẫn đủ cho EfficientNet-B0 (model nhẹ).

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **collect a diverse dataset covering both GAN and Diffusion-generated images** so that **the HolmHz model can detect modern AI-generated images, unlike the 3 SOTA models that all failed on Diffusion content.**

**Acceptance Criteria:**

- [ ] ≥6k ảnh Real (CIFAKE Real subset + FFHQ subset)
- [ ] ≥5k ảnh Diffusion Fake (CIFAKE Fake + SD v1.5 self-gen) — **ưu tiên cao nhất**
- [ ] ≥3k ảnh GAN Fake (StyleGAN faces)
- [ ] OOD test set: 100-200 ảnh Gemini + 100-200 ảnh Flux + 200 real camera
- [ ] Ảnh trong `imgs/` folder đã copy vào OOD test set
- [ ] Tất cả ảnh resize về cùng size (224x224 hoặc 256x256)
- [ ] Folder structure: `data/raw/{real,fake_gan,fake_diffusion,ood_test}/`
- [ ] File `data/manifests/dataset_stats.json` ghi số lượng mỗi loại

---

## 🛠️ Implementation

### Subtasks

- [ ] 1.2.1 Download CIFAKE dataset 🌟 (Real+Fake, Kaggle 1-click, ~500MB) — **Luân thực hiện**
- [ ] 1.2.2 Download FFHQ subset (3-5k real, Kaggle mirror) — **Luân thực hiện**
- [ ] 1.2.3 Download/scrape StyleGAN faces (3k, thispersondoesnotexist.com hoặc Kaggle) — Hoàng
- [ ] 1.2.4 Self-generate SD v1.5 ảnh trên Colab (2-3k, dùng diffusers miễn phí) — Hoàng
- [ ] 1.2.5 Chuẩn bị OOD: Gemini (100-200) + Flux (100-200) + Real camera (200) — Hoàng

### Branch & PR

- [ ] Branch: `feat/s1/data-collection`
- [ ] PR Created
- [ ] Data integrity check (không bị corrupt, đúng format)
- [ ] `dataset_stats.json` updated

---

## 📝 Notes

> **Nguồn download (CẬP NHẬT 24/02/2026):**
>
> **⭐ CIFAKE (ƯU TIÊN SỐ 1):**
>
> - Link: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
> - 120k ảnh (60k real + 60k AI), ~500MB, download 1 click trên Kaggle
> - Ảnh 32×32 (cần resize lên 224×224 — vẫn ok vì EfficientNet-B0 sẽ học features)
>
> **FFHQ:**
>
> - https://www.kaggle.com/datasets (tìm "FFHQ" — nhiều mirror nhỏ)
> - Subset 3-5k real faces, chất lượng cao 1024×1024
>
> **StyleGAN faces:**
>
> - https://thispersondoesnotexist.com (scrape script) hoặc Kaggle "fake faces"
>
> **SD v1.5 self-generate:**
>
> - Dùng `diffusers` library trên Colab/Kaggle, chạy pipeline generate 2-3k ảnh miễn phí
> - Script: `from diffusers import StableDiffusionPipeline` → generate with random prompts
>
> **OOD test (thủ công):**
>
> - Gemini: Vào gemini.google.com, tạo thủ công 100-200 ảnh (miễn phí)
> - Flux: replicate.com (free tier) hoặc flux1.ai — generate 100-200 ảnh
> - ~~GenImage: ❌ BỎ — quá lớn ~50GB, không thực tế cho SV~~
> - ~~DFFD: ❌ BỎ — cần xin access, mất thời gian~~

> **Hướng dẫn cho Luân:**
>
> 1. Vào Kaggle, tìm "CIFAKE", click Download → giải nén vào `data/raw/cifake/`
> 2. Vào Kaggle, tìm "FFHQ 70000", download subset 3-5k → `data/raw/real/ffhq/`
>    Hoàng sẽ viết script chi tiết + validation.

> **Budget ước tính:** ~~GenImage ~50GB, Flux API ~$5-10~~ → **$0** (tất cả miễn phí).
