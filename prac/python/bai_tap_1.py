class ChoNghiepvu:
    def __init__(self, ten, khu_vuc_truc):
        self.ten= ten
        self.khu_vuc_truc = khu_vuc_truc
        self.so_lan_sua = 0

    def phat_hien_trom(self):
        print(f"Gau Gau! co trom co khu vuc {self.khu_vuc_truc}")
        self.so_lan_sua += 1


Dog_1 = ChoNghiepvu("Micky", "Cong chinh")

Dog_1.phat_hien_trom()
Dog_1.phat_hien_trom()
print(Dog_1.so_lan_sua)
