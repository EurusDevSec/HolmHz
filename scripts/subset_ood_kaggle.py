# scripts/subset_ood_kaggle.py
"""
Lọc random subset từ bộ tristanzhang32/ai-generated-images-vs-real-images
cho OOD test set của HolmHz.

Cấu trúc thực tế (đã tải xong):
  data/raw/ood_test/tristanzhang_fake/  (6,000 ảnh: SD+MidJourney+DALL-E mixed)
  data/raw/ood_test/real_pexels/        (6,000 ảnh: Pexels+Unsplash)

⚠️ Dataset gốc KHÔNG có subfolder riêng per generator (MidJourney/DALL-E/SD).
   Tất cả fake nằm flat trong 1 folder. Chấp nhận điều này — đây là OOD data.

Script này giữ lại `keep` ảnh ngẫu nhiên, xóa phần còn lại để tiết kiệm ổ đĩa.
Nếu muốn dùng tất cả 6K ảnh mỗi folder → không cần chạy script này.
"""
import random
from pathlib import Path

random.seed(42)

OOD_ROOT = Path("data/raw/ood_test")


def subset_in_place(folder: Path, keep: int) -> None:
    """
    Giữ lại `keep` ảnh ngẫu nhiên trong folder, xóa phần còn lại.
    Dùng khi folder đã có đủ ảnh nhưng muốn giảm xuống.
    """
    if not folder.exists():
        print(f"  ❌ Không tìm thấy folder: {folder}")
        return

    images = sorted(
        list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        + list(folder.glob("*.jpeg")) + list(folder.glob("*.webp"))
    )
    print(f"  Found {len(images)} images in {folder.name}")

    if len(images) <= keep:
        print(f"  ✅ Đủ/ít hơn {keep} → giữ nguyên tất cả")
        return

    to_keep = set(random.sample(images, keep))
    removed = 0
    for img in images:
        if img not in to_keep:
            img.unlink()
            removed += 1

    print(f"  Kept {keep}, removed {removed} → {folder.name}: {keep} ảnh")


if __name__ == "__main__":
    fake_dir = OOD_ROOT / "tristanzhang_fake"
    real_dir = OOD_ROOT / "real_pexels"

    if not fake_dir.exists() or not real_dir.exists():
        print("❌ Chưa có folder tristanzhang_fake/ hoặc real_pexels/")
        print("   → Copy test/fake và test/real từ Kaggle trước (xem Bước 5.0)")
        exit(1)

    print("✂️  Subset tristanzhang_fake (giữ 500 ảnh)...")
    subset_in_place(fake_dir, keep=500)

    print("\n✂️  Subset real_pexels (giữ 500 ảnh)...")
    subset_in_place(real_dir, keep=500)

    # Tóm tắt
    print(f"\n{'='*50}")
    print("📊 OOD Test Set Summary:")
    for name in ["tristanzhang_fake", "real_pexels", "gemini", "flux", "real_camera"]:
        p = OOD_ROOT / name
        n = len(list(p.glob("*.*"))) if p.exists() else 0
        status = "✅" if n > 0 else "⏳"
        print(f"  {status} {name}: {n} ảnh")
    print("\nBước tiếp: tạo thêm Gemini + Flux thủ công (xem guide Bước 5.2 và 5.3)")
