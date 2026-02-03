class KhoAnh:
    def __init__(self, duong_dan_thu_muc):
        self.thu_muc = duong_dan_thu_muc
        self.danh_sach_anh = ["anh1.jpg", "anh2.jpg", "anh3.jpg"]


    def __len__(self):
        return len(self.danh_sach_anh)
    
    def __getitem__(self, key):
        ten_anh = self.danh_sach_anh[key]
        return f"Dang lay anh o {self.thu_muc}/ {ten_anh}"
    


kho_cua_toi = KhoAnh("R:/Data/HolmHz")

print(f"Tong so anh: {len(kho_cua_toi)}")
print(kho_cua_toi[0])