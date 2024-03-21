"""_summary_
文字認識アプリケーションの実装

ここを参考に前処理を追加する
https://www.yakupro.info/entry/handwritten-digit-preprocess

そもそも今の入力方法ではできない可能性がある
CNNで再度試してみた方が良い

"""

import sys, os
import numpy as np
import pygame
from pygame.locals import *
import cv2

sys.path.append(os.pardir)
from two_layer_net_backp import TwoLayerNet

# init
pygame.init()
screen = pygame.display.set_mode((600, 400))
gray = (122,122,122)
black = (0, 0, 0)

# init DNN
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, load=True)


# cellのx、yの数
input_x = 28
input_y = 28

# cellの初期化
input_field = [[(0,0,0)]*input_x for i in range(input_y)]

cell_size = 10      # cellの大きさ
offset_x = 20       # 描画画面のxのオフセット下はyのオフセット
offset_y = 50
isPressed = False

# Reset buttonの初期設定
button_x = 100
button = pygame.Rect(offset_x+((cell_size*input_x)/2)-button_x/2, offset_y+cell_size*input_y+10, button_x, 50)  # creates a rect object

#STEP1.フォントの用意
font = pygame.font.SysFont(None, 25)

#STEP2.テキストの設定
text1 = font.render("Reset", True, (0,0,0))
text2 = font.render("Prediction: ", True, (0,0,0))

cell = [0 for x in range(input_x*input_y)]

trial = 0
predict = " "

while True:
    screen.fill(gray)
    pygame.draw.rect(screen, (200, 200, 200), button)
    screen.blit(text1, (offset_x+((cell_size*input_x)/2)-25, offset_y+cell_size*input_y+25))
    screen.blit(text2, (offset_x+cell_size*input_x+50, 80))

    # prediction text
    font = pygame.font.SysFont(None, 200)
    prediction = font.render(predict, True, (0,0,0))
    screen.blit(prediction, (offset_x+cell_size*input_x+100, 120))

    index = 0
    for i in range(input_x):
        for j in range(input_y):
            pygame.draw.rect(screen, input_field[i][j], (i*cell_size+offset_x, j*cell_size+offset_y, cell_size, cell_size), )
            cell[index] = np.mean(input_field[j][i])
            index+=1
    np_cell = np.array(cell)/255.0


    pygame.display.update()
    for event in pygame.event.get():
        # 描画の処理
        if event.type ==MOUSEMOTION and isPressed:
                mouse_Pos = pygame.mouse.get_pos()
                if mouse_Pos[0] >= cell_size*input_x+offset_x or mouse_Pos[1] >= cell_size*input_y + offset_y or mouse_Pos[0] < offset_x or mouse_Pos[1] < offset_y:
                    continue

                # マウスがいるセルを計算
                x = int((mouse_Pos[0]-offset_x)/cell_size)
                y = int((mouse_Pos[1]-offset_y)/cell_size)

                # マウスがあるセルを真っ白に、上下左右は+50明るくする
                input_field[x][y] = (230,230,230)
                if x+1<input_x:
                    lst = np.array(input_field[x+1][y])
                    lst += 50
                    lst = np.clip(lst, 0, 255)
                    input_field[x+1][y] = tuple(lst)
                if y+1 < input_y:
                    lst = np.array(input_field[x][y+1])
                    lst += 50
                    lst = np.clip(lst, 0, 255)
                    input_field[x][y+1] = tuple(lst)
                if x-1 >= 0:
                    lst = np.array(input_field[x-1][y])
                    lst += 50
                    lst = np.clip(lst, 0, 255)
                    input_field[x-1][y] = tuple(lst)
                if y-1 >= 0:
                    lst = np.array(input_field[x][y-1])
                    lst += 50
                    lst = np.clip(lst, 0, 255)
                    input_field[x][y-1] = tuple(lst)
                # 予測
                predict = str(np.argmax(network.predict(np_cell)))
                print(network.predict(np_cell))

        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                isPressed = True

            # リセットボタンが押された時の処理
            if button.collidepoint(event.pos):
                # キャンバスを真っ白にする
                input_field = [[(0,0,0)]*input_x for i in range(input_y)]
                predict = " "

        elif event.type == MOUSEBUTTONUP:
            isPressed = False

        if event.type == QUIT:
            pygame.quit()
            sys.exit()