"""_summary_
文字認識アプリケーションの実装
"""

import sys
import pygame
from pygame.locals import *

# init
pygame.init()
screen = pygame.display.set_mode((600, 400))
gray = (122,122,122)
black = (0, 0, 0)


while True:
    # if pygame.mouse.get_pressed():
    screen.fill(gray)

    for i in range(28):
        for j in range(28):
            pygame.draw.rect(screen, black, (i*9+20, j*9+50, 8, 8), )

    pygame.display.update()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()