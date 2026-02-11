## 💡 Context

# new task 1.2

> **Task ID**: S1-002  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: 24/02/2026 (2 tuần, song song với S1-001)  
> **Assignee**: Hoàng + Luân  
> **Blocked by**: Không (download song song với setup env)  
> **Blocks**: S1-003 (Data Pipeline cần data)

> Thu thập dataset đa nguồn: Real + GAN + Diffusion.
> ⚠️ BÀI HỌC #1: Training data quyết định thành bại. Diffusion data là BẮT BUỘC.

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **collect a diverse dataset covering both GAN and Diffusion-generated images** so that **the HolmHz model can detect modern AI-generated images, unlike the 3 SOTA models that all failed on Diffusion content.**

**Acceptance Criteria:**

- [ ] ≥10k ảnh Real (FFHQ subset)
- [ ] ≥10k ảnh Diffusion Fake (GenImage subset) — **ưu tiên cao nhất**
- [ ] ≥5k ảnh GAN Fake (StyleGAN2 faces)
- [ ] OOD test set: 200-500 ảnh Gemini + 200-500 ảnh Flux
- [ ] Ảnh trong `imgs/` folder đã copy vào OOD test set
- [ ] Tất cả ảnh resize về cùng size (224x224 hoặc 256x256)
- [ ] Folder structure: `data/raw/{real,fake_gan,fake_diffusion,ood_test}/`
- [ ] File `data/manifests/dataset_stats.json` ghi số lượng mỗi loại

---

## 🛠️ Implementation

### Subtasks

- [ ] 1.2.1 Download FFHQ subset (10k real) — **Luân thực hiện** theo script/hướng dẫn
- [ ] 1.2.2 Download GenImage subset (Diffusion fake, 10k) — Hoàng
- [ ] 1.2.3 Download/Generate StyleGAN2 faces (5k) — Hoàng
- [ ] 1.2.4 Chuẩn bị OOD test set: Flux/Gemini/SDXL (1k) — Hoàng
- [ ] 1.2.5 Copy ảnh Gemini/Flux thực tế từ folder `imgs/` — Hoàng

### Branch & PR

- [ ] Branch: `feat/s1/data-collection`
- [ ] PR Created
- [ ] Data integrity check (không bị corrupt, đúng format)
- [ ] `dataset_stats.json` updated

---

## 📝 Notes

> **Nguồn download:**
>
> - FFHQ: https://github.com/NVlabs/ffhq-dataset (Kaggle mirror nhanh hơn)
> - GenImage: https://github.com/GenImage-Dataset/GenImage
> - StyleGAN2: https://github.com/NVlabs/stylegan2 (hoặc generate bằng pretrained)
> - Gemini: Gemini API generate hoặc manual tạo trên gemini.google.com
> - Flux: Flux.1 API hoặc Replicate.com

> **Hướng dẫn cho Luân:**
> Chỉ cần download FFHQ theo link/script, giải nén vào đúng folder.
> Hoàng sẽ viết script download + hướng dẫn chi tiết.

> **Budget ước tính:** GenImage ~50GB raw, cần subset 10k. Flux API ~$5-10.
