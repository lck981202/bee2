#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2023/7/11 下午17:02
# @Author : Leichaokai
import warnings

import cv2
import torch
import argparse
import numpy as np
from utils.log import get_logger

import math
from PIL import Image, ImageFont, ImageDraw
from boxmot.tracker_zoo import create_tracker
from collections import Counter,deque

#求框的中点
def tlbr_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format 求物体中心的坐标
    return midpoint

#判断相交状态
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

#求角度信息
def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))

#导入中文模块
def get_size_with_pil(label,size=25):
    font = ImageFont.truetype("./configs/simkai.ttf", size, encoding="utf-8")  # simhei.ttf 导入中文模块
    return font.getsize(label)

#为了支持中文，用pil
def put_text_to_cv2_img_with_pil(cv2_img,label,pt,color):
    pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./configs/simkai.ttf", 25, encoding="utf-8") #simhei.ttf
    draw.text(pt, label, color,font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式

#颜色模块
colors = np.array([
    [1,0,1],
    [0,0,1],
    [0,1,1],
    [0,1,0],
    [1,1,0],
    [1,0,0]
    ]);

def get_color(c, x, max): #获取颜色
    ratio = (x / max) * 5;
    i = math.floor(ratio);
    j = math.ceil(ratio);
    ratio -= i;
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    return r;

class yolo_reid():
    def __init__(self, cfg, args, path):
        self.logger = get_logger("root")
        self.args = args
        self.video_path = path
        use_cuda = self.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)





    def boxmot_work(self):
        idx_frame = 0
        paths = {}
        track_cls = 0
        last_track_id = -1
        total_track = 0
        angle = -1
        total_counter = 0
        out_count = 0  # 外出计数
        in_count = 0  # 进入计数
        class_counter = Counter()  # 存储每个检测类别的数量
        already_counted = deque(maxlen=30)  # 短期内储存已计数的id，deque储存可迭代的对象接口
        total_time = 0
        in_count_id = deque(maxlen=80)
        tracking_num_output = 0


