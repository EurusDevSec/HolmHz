class Dataset:
    def __init__(self, ten_dataset):
        self.ten_dataset = ten_dataset

    def bao_cao(self):
        return f"Day la dataset {self.ten_dataset}"


class ImageDataset(Dataset):
    def __init__(self, ten_dataset: str, so_luong_anh: int):
        super().__init__(ten_dataset)
        self.so_luong_anh = so_luong_anh

    def bao_cao(self):
        return f"Dataset hinh anh {self.ten_dataset} co {self.so_luong_anh} anh."

class AudioDataset(Dataset):
    def __init__(self, ten_dataset: str, so_gio_thu_am: int):
        super().__init__(ten_dataset)
        self.so_gio_thu_am = so_gio_thu_am

    def bao_cao(self):
        return f"Dataset am thanh {self.ten_dataset} co {self.so_gio_thu_am} Gio"


if __name__ == "__main__":
    ds1 = ImageDataset("Iris", 1000)
    ds2 = ImageDataset("MNIST", 2000)
    ds3 = AudioDataset("AudioTrack", 70)
    ds4 = AudioDataset("VietNam SoundTrack", 10)

    danh_sach_lon_xon = [ds1, ds2, ds3, ds4]

    for danh_sach in danh_sach_lon_xon:
        print(danh_sach.bao_cao())
