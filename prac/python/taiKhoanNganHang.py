class TaiKhoan:
    def __init__(self, ten_chu_tai_khoan: str):
        self.ten_chu_tai_khoan = ten_chu_tai_khoan
        self.__so_du = 0

    def nap_tien(self, so_tien: float):
        if so_tien > 0:
            self.__so_du +=so_tien
            print(f"Ting Ting !! Nap vao thanh cong, So tien hien tai cua ban la {self.__so_du}")
        else:
            print("So tien khong hoop le")
    def rut_tien(self, so_tien: float):
        if so_tien > 0 and so_tien <=self.__so_du:
            self.__so_du-=so_tien
            print(f"Ting Ting!!! rut tien thanh cong, so tien con lai cua ban la {self.__so_du}")
        elif so_tien <= self.__so_du:
            print("khong đủ tiền")
        else:
            print("So tien khoong hoop le")

    def Xem_so_du(self):
        return self.__so_du
    
if __name__ == "__main__":
    tk = TaiKhoan("Hoang")
    tk.nap_tien(1000)
    tk.nap_tien