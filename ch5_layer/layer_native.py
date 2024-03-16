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


# 加算レイヤ
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

# りんごの例, りんご2こ、みかん３個
if __name__ == "__main__":
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    # layer
    mul_apple_layer = MullLayer()
    mul_orange_layer = MullLayer()

    add_apple_orange_layer = AddLayer()

    mul_tax_layer = MullLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)
    print(f"{price}")

    # backward
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dappl_num = mul_apple_layer.backward(dapple_price)

    print(f"dapple: {dapple}, dapple_num: {dappl_num}, dorange: {dorange}, dourange_num:{dorange_num}, dtax: {dtax}")