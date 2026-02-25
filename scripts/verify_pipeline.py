"""Verify data pipeline hoạt động — chạy nhanh, không cần notebook."""

from holmhz.data import create_dataloader, get_dataset_info


def main():
    # === Xem info các split ===
    print("=" * 60)
    print("VERIFY DATA PIPELINE — HolmHz")
    print("=" * 60)

    for name, path in [
        ("Train", "data/manifests/train.json"),
        ("Val", "data/manifests/val.json"),
        ("Test ID", "data/manifests/test_id.json"),
        ("Test OOD", "data/manifests/test_ood.json"),
    ]:
        info = get_dataset_info(path)
        print(
            f"  {name:10s}: {info['total']:6d} ảnh | "
            f"{info['label_ratio']} | "
            f"sources: {list(info['sources'].keys())}"
        )

    # === Load 1 batch để verify shape/dtype ===
    print("\n--- Loading 1 batch from val.json ---")
    loader = create_dataloader(
        "data/manifests/val.json",
        batch_size=32,
        is_training=False,
        num_workers=0,
    )
    batch = next(iter(loader))

    print(f"  Batch image shape : {batch['image'].shape}")
    print(f"  Batch label shape : {batch['label'].shape}")
    print(f"  Image dtype       : {batch['image'].dtype}")
    print(f"  Image range       : [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")
    print(f"  Labels (first 8)  : {batch['label'][:8].tolist()}")
    print(f"  Sources (first 4) : {batch['source'][:4]}")
    print("\n✅ Data pipeline working!")


if __name__ == "__main__":
    main()
