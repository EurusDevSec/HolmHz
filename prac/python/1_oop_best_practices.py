class Robot:

    def __init__(self, ten ,mau_sac):

        self.ten = ten
        self.mau_sac = mau_sac
        self.pin = 100


    def chao(self):
        print(f"xin chao, ta la {self.ten}, mau {self.mau_sac}")

    def chay (self):
        self.pin = self.pin -10
        print(f"{self.ten} dang chay... Pin con {self.pin}%")



robot1 = Robot("Wall-E", "Vang")
robot2= Robot("Baymax", "Trang")

robot1.chao()
robot2.chao()

robot1.chay()
print(robot2.pin)
