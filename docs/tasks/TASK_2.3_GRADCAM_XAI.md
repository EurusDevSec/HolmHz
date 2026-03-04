## 💡 Context

> **Task ID**: S2-003  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 2 - Evaluation + XAI + Benchmark  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: ~~28/03/2026~~ → **07/04/2026** (song song với S2-002)  
> **Assignee**: Hoàng  
> **Blocked by**: S1-006 (trained model)  
> **Blocks**: Không

> Tích hợp Grad-CAM để giải thích model predictions bằng heatmap.
> XAI là core feature của HolmHz — thể hiện model nhìn vào đâu khi phân loại.

---

## 🤖 AI Refined

> **User Story:**

> As a **Researcher**, I want to **integrate Grad-CAM to visualize which regions the model focuses on when classifying real vs fake** so that **I can explain model decisions, validate correctness, and create an engaging XAI gallery for the report.**

**Acceptance Criteria:**

- [ ] `pytorch-grad-cam` integrated, chạy được trên EfficientNet-B0
- [ ] Heatmap overlay function: input image → output image với heatmap chồng lên
- [ ] XAI gallery: 50 samples (25 real + 25 fake), mỗi sample gồm: original → heatmap → overlay
- [ ] Heatmap validate: fake images nên highlight vùng mặt/artifacts; real images nên highlight ít hơn
- [ ] Script `scripts/explain.py` chạy trên bất kỳ ảnh

---

## 🛠️ Implementation

### Subtasks

- [ ] 2.3.1 Implement `src/holmhz/xai/gradcam.py` (wrapper quanh pytorch-grad-cam)
- [ ] 2.3.2 Implement `src/holmhz/xai/utils.py` (heatmap overlay, gallery generation)
- [ ] 2.3.3 Script `scripts/explain.py` (CLI: input image/folder → output heatmap)
- [ ] 2.3.4 Generate XAI gallery (50 samples) → `outputs/xai_gallery/`

### Branch & PR

- [ ] Branch: `feat/s2/gradcam-xai`
- [ ] PR Created
- [ ] Gallery generated
- [ ] Sample heatmaps look reasonable

---

## 📝 Notes

> **Target layer cho EfficientNet-B0:**
>
> ```python
> from pytorch_grad_cam import GradCAM
> from pytorch_grad_cam.utils.image import show_cam_on_image
>
> # EfficientNet-B0: target layer = last conv block
> target_layer = model.backbone.model.conv_head  # hoặc features[-1]
> cam = GradCAM(model=model, target_layers=[target_layer])
> ```

> **Gallery layout:**
>
> ```
> outputs/xai_gallery/
> ├── real/
> │   ├── 001_original.png
> │   ├── 001_heatmap.png
> │   └── 001_overlay.png
> └── fake/
>     ├── 001_original.png
>     ├── 001_heatmap.png
>     └── 001_overlay.png
> ```

> **Kỳ vọng heatmap:**
>
> - Fake (GAN): highlight vùng mắt, răng, tóc (artifacts thường ở đây)
> - Fake (Diffusion): pattern khác GAN — có thể highlight texture, edges
> - Real: heatmap phân tán hơn, ít focus vào 1 vùng
> - Nếu heatmap ALL random → model chưa learn được meaningful features
