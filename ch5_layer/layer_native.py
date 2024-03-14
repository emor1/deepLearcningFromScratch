# 乗算レイヤ

class MullLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y # xとyをひっくり返す
        dy = dout * self.x # xとyをひっくり返す

        return dx, dy

# りんごの例
if __name__ == "__main__":
    apple = 100
    apple_num = 2
    tax = 1.1

    # layer
    mul_apple_layer = MullLayer()
    mul_tax_layer = MullLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)
    print(f"{price}")

    # backward
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dappl_num = mul_apple_layer.backward(dapple_price)

    print(f"dapple: {dapple}, dapple_num: {dappl_num}, dtax: {dtax}")