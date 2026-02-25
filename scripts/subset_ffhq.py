# scripts/subset_ffhq.py — Chọn random 5K ảnh khuôn mặt từ FFHQ
import shutil
import random
from pathlib import Path

random.seed(42)

src = Path("data/raw/real/ffhq_full")  # Folder chứa FFHQ đã giải nén
dst = Path("data/raw/real/ffhq")
dst.mkdir(parents=True, exist_ok=True)

all_images = sorted(
    list(src.glob("**/*.png")) + list(src.glob("**/*.jpg"))
)

if len(all_images) == 0:
    print(f"❌ Không tìm thấy ảnh trong {src}")
    print("Kiểm tra lại: giải nén FFHQ vào data/raw/real/ffhq_full/")
    exit(1)

count = min(5000, len(all_images))
selected = random.sample(all_images, count)

for img_path in selected:
    shutil.copy2(img_path, dst / img_path.name)

print(f"✅ Copied {count} FFHQ images to {dst}")