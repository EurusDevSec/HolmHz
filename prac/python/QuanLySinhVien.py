class SinhVien:
    def __init__(self, ho_ten: str, tuoi: int, diem_tb: float):
        self.ho_ten = ho_ten
        self.tuoi = tuoi
        self.diem_tb = diem_tb

    def xep_loai(self):
        if self.diem_tb >=8:
            return "gioi"
        elif self.diem_tb >=6.5:
            return "kha"
        elif self.diem_tb >=5:
            return "trung_binh"

        return "yeu"

    def Gioi_thieu(self):
        return f"Toi ten la {self.ho_ten}, {self.tuoi} tuoi, xep loai {self.xep_loai()}"



if __name__ == "__main__":
    sinhvien_1 = SinhVien("le van A", 18, 9.5)
    sinhvien_2 = SinhVien("Nguyen Van cuong", 20, 2)

    print(sinhvien_1.Gioi_thieu())
    print(sinhvien_2.Gioi_thieu())
