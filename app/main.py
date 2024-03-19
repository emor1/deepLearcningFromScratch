"""_summary_
文字認識アプリケーションの実装
"""

import sys
import numpy as np
import pygame
from pygame.locals import *

# init
pygame.init()
screen = pygame.display.set_mode((600, 400))
gray = (122,122,122)
black = (0, 0, 0)

input_x = 28
input_y = 28

input_field = [[(0,0,0)]*input_x for i in range(input_y)]
print(input_field)

cell_size = 10
offset_x = 20
offset_y = 50
isPressed = False

button_x = 100
button = pygame.Rect(offset_x+((cell_size*input_x)/2)-button_x/2, offset_y+cell_size*input_y+10, button_x, 50)  # creates a rect object
#STEP1.フォントの用意  
font = pygame.font.SysFont(None, 25)

#STEP2.テキストの設定
text1 = font.render("Reset", True, (0,0,0))

while True:
    # if pygame.mouse.get_pressed():
    screen.fill(gray)
    pygame.draw.rect(screen, (200, 200, 200), button)
    screen.blit(text1, (offset_x+((cell_size*input_x)/2)-25, offset_y+cell_size*input_y+25))

    for i in range(input_x):
        for j in range(input_y):
            pygame.draw.rect(screen, input_field[i][j], (i*cell_size+offset_x, j*cell_size+offset_y, cell_size, cell_size), )

    pygame.display.update()
    for event in pygame.event.get():
        if event.type ==MOUSEMOTION and isPressed:
                mouse_Pos = pygame.mouse.get_pos()
                if mouse_Pos[0] >= cell_size*input_x+offset_x or mouse_Pos[1] >= cell_size*input_y + offset_y or mouse_Pos[0] < offset_x or mouse_Pos[1] < offset_y:
                    continue
                x = int((mouse_Pos[0]-offset_x)/cell_size)
                y = int((mouse_Pos[1]-offset_y)/cell_size)
                input_field[x][y] = (255,255,255)

        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                isPressed = True

            if button.collidepoint(event.pos):
                input_field = [[(0,0,0)]*input_x for i in range(input_y)]

        elif event.type == MOUSEBUTTONUP:
            isPressed = False
                # print(x, y)
                # print(input_field)


        if event.type == QUIT:
            pygame.quit()
            sys.exit()