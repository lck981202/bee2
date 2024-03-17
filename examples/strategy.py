import cv2
import torch
import warnings
import argparse
import numpy as np
from collections import Counter
from collections import deque
from math import atan2, degrees
from PIL import Image, ImageDraw, ImageFont
import time
import math



def tlbr_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format 求物体中心的坐标
    return midpoint


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

#求角度信息
def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    # if x == 0:
    #     x += 1
    #     return math.degrees(math.atan2(y, x))
    # else:
    #     return math.degrees(math.atan2(y, x))
    return degrees(atan2(y, x))

def vector_position(x, y):
    z = x-y
    if z > 0:
        return 1
    else:
        return -1


def get_size_with_pil(label, size=25):
    font = ImageFont.truetype("./configs/simkai.ttf", size, encoding="utf-8")  # simhei.ttf 导入中文模块
    return font.getsize(label)


#为了支持中文，用pil
def put_text_to_cv2_img_with_pil(cv2_img, label, pt, color):
    pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./configs/simkai.ttf", 25, encoding="utf-8") #simhei.ttf
    draw.text(pt, label, color,font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def get_size_with_pil(label, size):
    font = ImageFont.truetype("./weights/simkai.ttf", size, encoding="utf-8")
    return font.getsize(label)

colors = np.array([
    [1, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0]
])

def get_color(c,x, max):
    ratio = (x/max)*5
    i = math.floor(ratio)
    j = math.ceil(ratio)
    ratio -= i
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
    return r

def compute_color_for_labels(class_id, class_total=80):
    offset = (class_id+0)*1234567 % class_total;
    red = get_color(2,offset, class_total)
    green = get_color(1, offset, class_total)
    blue = get_color(0, offset, class_total)
    return (int(red*256), int(green*256), int(blue*256))