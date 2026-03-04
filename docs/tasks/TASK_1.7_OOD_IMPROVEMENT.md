## Context

> **Task ID**: S1-007
> **Phase**: Phase 1 - Data + Model Development
> **Sprint**: Sprint 1 - Data + Baseline Training (extended)
> **Status**: !! NOT STARTED
> **Created**: 26/02/2026
> **Updated**: 26/02/2026
> **Target**: **07/03/2026**
> **Assignee**: Hoang
> **Blocked by**: S2-001 (Task 2.1 Evaluation Pipeline) DONE 26/02
> **Blocks**: S2-002 (Benchmark SOTA), S2-003 (Grad-CAM XAI)
> **Milestone**: M2 -- Final model (AUC >= 0.90 ID, >= 0.75 OOD)

> Cai thien khong nang OOD generalization cua model EfficientNet-B0.
> Task 2.1 da phat hien: ID AUC = 0.9979, OOD AUC = 0.4812 (te hon random).
> Model bieu hien FAKE-bias nghiem trong: real_pexels 8.6%, real_camera 12% accuracy.
> Can fix DATA + AUGMENTATION + RETRAIN de dat OOD AUC >= 0.75.

---

## Van de phat hien tu Task 2.1 -- OOD Generalization Failure

> **Ket qua evaluation 26/02/2026:**
>
> | Set       | AUC    | Acc    | F1     | N     |
> | --------- | ------ | ------ | ------ | ----- |
> | In-Domain | 0.9979 | 0.9814 | 0.9830 | 3,975 |
> | OOD       | 0.4812 | 0.4805 | 0.6255 | 1,180 |
>
> **Per-source OOD breakdown:**
>
> | Source            | Acc    | N   | Label | Van de                 |
> | ----------------- | ------ | --- | ----- | ---------------------- |
> | flux              | 0.9500 | 80  | Fake  | Cao vi model bias FAKE |
> | tristanzhang_fake | 0.8720 | 500 | Fake  | Cao vi model bias FAKE |
> | real_pexels       | 0.0860 | 500 | Real  | NGHIEM TRONG -- gan 0% |
> | real_camera       | 0.1200 | 100 | Real  | NGHIEM TRONG -- gan 0% |
>
> **Root cause analysis:**
>
> 1. Training Real chi co 2 nguon: cifake (32x32 upscale, objects) + ffhq (face only)
> 2. Model hoc shortcut: "anh giong cifake/ffhq preprocessing = Real, khac = Fake"
> 3. Khong co anh Real da dang (phong canh, do vat, high-res camera) trong training
> 4. Augmentation p=0.3 qua nhe -- model van giu duoc shortcut features

---

## AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **improve OOD generalization of my deepfake detector by diversifying training data and strengthening augmentations** so that **the model achieves AUC >= 0.75 on OOD test set and can correctly classify real-world photos as Real, not FAKE.**

**Acceptance Criteria:**

- [ ] OOD AUC >= 0.75 (hien tai: 0.4812)
- [ ] OOD Accuracy >= 0.70 (hien tai: 0.4805)
- [ ] real_pexels accuracy >= 0.60 (hien tai: 0.0860)
- [ ] real_camera accuracy >= 0.50 (hien tai: 0.1200)
- [ ] ID AUC van >= 0.95 (hien tai: 0.9979 -- khong giam qua nhieu)
- [ ] Retrain tren Kaggle T4 thanh cong
- [ ] Re-evaluate bang pipeline Task 2.1 (scripts/test.py)
- [ ] Cap nhat CONTEXT.md voi ket qua moi

---

## Implementation

### Strategy -- 3 huong dong thoi

```
1. MO RONG TRAINING REAL (quan trong nhat)
   ----------------------------------------
   Hien tai: cifake(4,927 real) + ffhq(3,500 face)
   => Chi co 32x32 objects + faces preprocessed

   Them:
   a) 140k_real_and_fake dataset: 50,000 real (256x256, da dang)
      => Subset 3,000-5,000 anh diverse content
   b) Split real_pexels: 500 => 300 train + 200 test-OOD
      => Model duoc thay anh high-res da dang khi train

2. TANG CUONG AUGMENTATION (chong shortcut learning)
   --------------------------------------------------
   Hien tai: p=0.3, JPEG quality 60-100, blur 3-7

   Sua:
   a) Tang p=0.5 cho OneOf (JPEG/blur/noise)
   b) JPEG quality range: (30, 100) -- aggressive hon
   c) Them RandomResizedCrop -- pha resolution artifacts
   d) Them Downscale (0.5-0.9) -- mo phong nhieu resolution
   e) Tang ColorJitter p=0.5

3. FINE-TUNE STRATEGY
   -------------------
   Phase 1 (5 epochs): Freeze backbone, train head voi data moi
   Phase 2 (10 epochs): Unfreeze all, LR=1e-4, train toan bo

   Hoac: Full fine-tune tu dau voi data moi (don gian hon)
```

### Subtasks

- [ ] 1.7.1 Chuan bi data moi: subset 140k_real (3,000 real diverse)
- [ ] 1.7.2 Split real_pexels: 300 train + 200 test-OOD
- [ ] 1.7.3 Update resize_all.py + build_splits.py cho data moi
- [ ] 1.7.4 Tang cuong augmentation trong transforms.py
- [ ] 1.7.5 Update configs/train.yaml (v2 config)
- [ ] 1.7.6 Upload data moi len Kaggle dataset
- [ ] 1.7.7 Retrain tren Kaggle T4 (Phase 1 + Phase 2)
- [ ] 1.7.8 Download checkpoint moi ve local
- [ ] 1.7.9 Re-evaluate bang scripts/test.py
- [ ] 1.7.10 Cap nhat docs (CONTEXT.md, PROJECT_PLAN.md)

### Branch & PR

- [ ] Branch: `fix/s1/ood-improvement`
- [ ] PR Created
- [ ] OOD AUC >= 0.75
- [ ] ID AUC >= 0.95
- [ ] real_pexels accuracy >= 0.60
- [ ] All 20 existing tests still pass

---

## Notes

> **Du lieu hien co (co the dung ngay, khong can download):**
>
> | Source             | Location                                    | Images | Size    | Type |
> | ------------------ | ------------------------------------------- | ------ | ------- | ---- |
> | 140k_real_and_fake | data/raw/140k_real_and_fake/.../train/real/ | 50,000 | 256x256 | Real |
> | 140k_real_and_fake | data/raw/140k_real_and_fake/.../train/fake/ | 50,000 | 256x256 | Fake |
> | real_pexels (raw)  | data/raw/ood_test/real_pexels/              | 500    | ~4000px | Real |
> | real_camera (raw)  | data/raw/ood_test/real_camera/              | 100    | ~1000px | Real |
>
> **Du lieu da processed (dang dung):**
>
> | Source   | Location                              | In manifest | Train | Val   | Test  |
> | -------- | ------------------------------------- | ----------- | ----- | ----- | ----- |
> | cifake   | processed/train/real/cifake           | train.json  | 4,927 | 1,026 | 1,047 |
> | ffhq     | processed/train/real/ffhq             | train.json  | 3,500 | 750   | 750   |
> | stylegan | processed/train/fake_gan/stylegan     | train.json  | 3,500 | 750   | 750   |
> | cifake   | processed/train/fake_diffusion/cifake | train.json  | 4,873 | 1,074 | 1,053 |
> | sd15     | processed/train/fake_diffusion/sd15   | train.json  | 1,750 | 375   | 375   |
>
> **Tinh toan du lieu moi:**
>
> ```
> Hien tai train: 18,550 (8,427 real + 10,123 fake)
> Them:
>   + 3,000 real tu 140k dataset (diverse objects, 256x256)
>   + 300 real tu real_pexels (high-res natural photos)
> Sau khi them: ~21,850 (11,727 real + 10,123 fake)
>   => Ty le real/fake: 54/46 (gan can bang)
> ```
>
> **Augmentation changes:**
>
> | Parameter         | Cu (v1) | Moi (v2) | Ly do                       |
> | ----------------- | ------- | -------- | --------------------------- |
> | OneOf p           | 0.3     | 0.5      | Tang kha nang pha artifacts |
> | JPEG quality min  | 60      | 30       | Aggressive JPEG compression |
> | GaussianBlur max  | 7       | 9        | Blur manh hon               |
> | ColorJitter p     | 0.3     | 0.5      | Da dang mau sac hon         |
> | Downscale         | (none)  | 0.25-0.9 | Mo phong multi-resolution   |
> | RandomResizedCrop | (none)  | 0.7-1.0  | Pha spatial artifacts       |
>
> **Kaggle training estimate:**
>
> ```
> Dataset v2: ~21,850 train samples
> Batch size: 32
> Batches/epoch: ~683
> Phase 1 (5 epochs, freeze): ~20 min
> Phase 2 (10 epochs, unfreeze): ~40 min
> Total: ~1 hour (well within Kaggle 30h/week quota)
> ```
>
> **Risk: ID accuracy co the giam**
>
> Khi them data diverse + augmentation manh hon, ID AUC co the giam tu 0.998 xuong ~0.96-0.97.
> Day la trade-off chap nhan duoc: OOD tang > ID giam nhe.
> Target: ID >= 0.95, OOD >= 0.75.
