#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/18 下午6:02
# @Author : Leichaokai
# 尝试性文件 无法运行

import os
import cv2
import torch
import warnings
import argparse
import numpy as np

import os
#import onnxruntime as ort
from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from utils.torch_utils_1 import time_synchronized
from bee_detect_yolov8 import Bee_detect
from boxmot.tracker_zoo import create_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils_1 import select_device, load_classifier, time_synchronized
# count
from collections import Counter
from collections import deque
import math
from PIL import Image, ImageDraw, ImageFont

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
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
    return math.degrees(math.atan2(y, x))


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

#随机获取不同颜色的id框(可视化界面)
def compute_color_for_labels(class_id,class_total=80):
    offset = (class_id + 0) * 123457 % class_total;
    red = get_color(2, offset, class_total);
    green = get_color(1, offset, class_total);
    blue = get_color(0, offset, class_total);
    return (int(red*256),int(green*256),int(blue*256))


class yolo_reid():
    def __init__(self, cfg, args, path):
        self.logger = get_logger("root")
        self.args = args
        self.video_path = path
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        self.bee_detect = Bee_detect(self.args, self.video_path)
        imgsz = check_imgsz(args.img_size, s=32)  # self.model.stride.max())  # check img_size
        self.dataset = LoadImages(self.video_path, img_size=imgsz)
        self.boxmot_use = create_tracker(cfg, args.sort, use_cuda=use_cuda)

    def deep_sort(self): #定义deep_sort类 初始化各种标记
        idx_frame = 0
        results = []
        paths = {}
        track_cls = 0
        last_track_id = -1
        total_track = 0
        angle = -1
        total_counter = 0
        out_count = 0 # 外出计数
        in_count = 0 # 进入计数
        class_counter = Counter()   # 存储每个检测类别的数量
        already_counted = deque(maxlen=30)   # 短期内储存已计数的id，deque储存可迭代的对象接口
        total_time = 0
        in_count_id = deque(maxlen=80)
        tracking_num_output = 0
        for video_path, img, ori_img, vid_cap in self.dataset:
            idx_frame += 1
            # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)
            t1 = time_synchronized()

            # yolo目标检测
            bbox_xywh, cls_conf, cls_ids, xy = self.bee_detect.detect(video_path, img, ori_img, vid_cap)

            # deepsort跟踪，跟新帧
            outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

            # 1.视频种画线(构建两个位置坐标的集合，其中第一个为横坐标，第二个为纵坐标)
            # test2-4位置线
            # line_1 = [(int(0.79 * ori_img.shape[0]), int(0.325 * ori_img.shape[1])), (int(1.52 * ori_img.shape[0]), int(0.325 * ori_img.shape[1]))] #蜂箱上端线
            # line_2 = [(int(1.52 * ori_img.shape[0]), int(0.3723 * ori_img.shape[1])), (int(0.79 * ori_img.shape[0]), int(0.3723 * ori_img.shape[1]))] #蜂箱进出口重合线
            # line_3 = [(int(0.79 * ori_img.shape[0]), int(0.3723 * ori_img.shape[1])), (int(0.79 * ori_img.shape[0]), int(0.325 * ori_img.shape[1]))] #左端线
            # line_4 = [(int(1.52 * ori_img.shape[0]), int(0.325 * ori_img.shape[1])), (int(1.52 * ori_img.shape[0]), int(0.3723 * ori_img.shape[1]))] #右端线
            # # test_6位置线
            line_1 = [(int(0.59 * ori_img.shape[0]), int(0.3 * ori_img.shape[1])), (int(1.25 * ori_img.shape[0]), int(0.3 * ori_img.shape[1]))]  # 蜂箱上端线
            line_2 = [(int(1.25 * ori_img.shape[0]), int(0.343 * ori_img.shape[1])), (int(0.59 * ori_img.shape[0]), int(0.343 * ori_img.shape[1]))]  # 蜂箱进出口重合线
            line_3 = [(int(0.59 * ori_img.shape[0]), int(0.343 * ori_img.shape[1])), (int(0.59 * ori_img.shape[0]), int(0.3 * ori_img.shape[1]))]  # 左端线
            line_4 = [(int(1.25 * ori_img.shape[0]), int(0.3 * ori_img.shape[1])), (int(1.25 * ori_img.shape[0]), int(0.343 * ori_img.shape[1]))]  # 右端线
            # #oval_try = []
            # (该参数是蜂箱口线的可视化操作的可视化设置，第一个参数是导入背景图、第二三个参数是导入两个端点的位置坐标、第四个是线的颜色、第五个参数是线的宽度)
            cv2.line(ori_img, line_1[0], line_1[1], (255, 0, 255), 2)
            cv2.line(ori_img, line_2[0], line_2[1], (255, 0, 255), 2)
            cv2.line(ori_img, line_3[0], line_3[1], (255, 0, 255), 2)
            cv2.line(ori_img, line_4[0], line_4[1], (255, 0, 255), 2)
            #cv2.ellipse(ori_img, (300, 300), (240, 100), 0, 0, 360, (0, 250, 0), 3)

            # 2. 统计蜜蜂数量
            for track in outputs: #output为框线坐标和id的二维坐标集合

                bbox = track[:4] # 框线四个方向坐标
                track_id = track[-1] # 获box的id
                midpoint = tlbr_midpoint(bbox) #将目标框转化为中心点
                origin_midpoint = (midpoint[0], ori_img.shape[0] - midpoint[1])  # get midpoint respective to botton-left
                tracking_num_output += 1
                if track_id not in paths: #path字典是路径，将id储存在字典中
                    paths[track_id] = deque(maxlen=2) #某一id仅仅保存轨迹值
                    total_track = track_id

                paths[track_id].append(midpoint)
                previous_midpoint = paths[track_id][0]
                origin_previous_midpoint = (previous_midpoint[0], ori_img.shape[0] - previous_midpoint[1])

                #判断1线进入线进入情况
                if intersect(midpoint, previous_midpoint, line_1[0], line_1[1]) and track_id not in already_counted:
                    class_counter[track_cls] += 1
                    total_counter += 1
                    last_track_id = track_id;
                    # 经过线跳红
                    cv2.line(ori_img, line_1[0], line_1[1], (0, 0, 255), 10) #通过线闪动且变粗变红
                    already_counted.append(track_id)  # 将已经触碰的蜜蜂id记录

                    angle = vector_angle(origin_midpoint, origin_previous_midpoint) #计算角度

                    if angle > 0:
                        out_count += 1
                        already_counted.remove(track_id)
                    if angle < 0:
                        in_count += 1
                        in_count_id.append(track_id)

                #判断2线进入
                elif intersect(midpoint, previous_midpoint, line_2[0], line_2[1]) and track_id not in already_counted:
                    class_counter[track_cls] += 1
                    total_counter += 1
                    last_track_id = track_id;
                    cv2.line(ori_img, line_2[0], line_2[1], (0, 0, 255), 10)

                    already_counted.append(track_id)  # Set already counted for ID to true.

                    angle = vector_angle(origin_midpoint, origin_previous_midpoint)

                    if angle < 0:
                        out_count += 1
                        already_counted.remove(track_id)
                    if angle > 0:
                        in_count += 1
                        in_count_id.append(track_id)

                # 判断3线进入
                elif intersect(midpoint, previous_midpoint, line_3[0], line_3[1]) and track_id not in already_counted:
                    class_counter[track_cls] += 1
                    total_counter += 1
                    last_track_id = track_id;
                    cv2.line(ori_img, line_3[0], line_3[1], (0, 0, 255), 10)

                    already_counted.append(track_id)  # Set already counted for ID to true.

                    angle = vector_angle(origin_midpoint, (0.6 * ori_img.shape[0], 0.56 * ori_img.shape[1]))

                    if angle > 90 :
                        out_count += 1
                        already_counted.remove(track_id)
                    if angle < 90:
                        in_count += 1
                        in_count_id.append(track_id)

                # 判断4线进入
                elif intersect(midpoint, previous_midpoint, line_4[0], line_4[1]) and track_id not in already_counted:
                    class_counter[track_cls] += 1
                    total_counter += 1
                    last_track_id = track_id;
                    cv2.line(ori_img, line_4[0], line_4[1], (0, 0, 255), 10)

                    already_counted.append(track_id)  # Set already counted for ID to true.

                    angle = vector_angle(origin_midpoint, (0.7 * ori_img.shape[1], 0.56 * ori_img.shape[1]))

                    if angle < 90:
                        out_count += 1
                        already_counted.remove(track_id)
                    if angle > 90:
                        in_count += 1
                        in_count_id.append(track_id)

                #判断是否在1处徘徊
                elif intersect(midpoint, previous_midpoint, line_1[0], line_1[1]) and track_id in in_count_id:

                    class_counter[track_cls] -= 1
                    total_counter -= 1

                    cv2.line(ori_img, line_2[0], line_2[1], (0, 0, 255), 10)

                    angle_2 = vector_angle(origin_midpoint, origin_previous_midpoint)
                    if angle_2 > 0:
                        in_count -= 1
                        already_counted.remove(track_id)
                        in_count_id.remove(track_id)

                #判断是否在2处徘徊
                elif intersect(midpoint,previous_midpoint, line_2[0], line_2[1]) and track_id in in_count_id:

                    class_counter[track_cls] -= 1
                    total_counter -= 1

                    cv2.line(ori_img, line_2[0], line_2[1], (0, 0, 255), 10)

                    angle_2 = vector_angle(origin_midpoint, origin_previous_midpoint)
                    if angle_2 < 0:
                        in_count -= 1
                        already_counted.remove(track_id)
                        in_count_id.remove(track_id)

                #判断是否在3处徘徊
                elif intersect(midpoint,previous_midpoint, line_3[0], line_3[1]) and track_id in in_count_id:

                    class_counter[track_cls] -= 1
                    total_counter -= 1

                    cv2.line(ori_img, line_3[0], line_3[1], (0, 0, 255), 10)

                    angle_2 = vector_angle(origin_midpoint, (0.6 * ori_img.shape[0], 0.56 * ori_img.shape[1]))
                    if angle_2 > 90:
                        in_count -= 1
                        already_counted.remove(track_id)
                        in_count_id.remove(track_id)

                #判断是否在4处徘徊
                elif intersect(midpoint,previous_midpoint, line_2[0], line_2[1]) and track_id in in_count_id:

                    class_counter[track_cls] -= 1
                    total_counter -= 1

                    cv2.line(ori_img, line_4[0], line_4[1], (0, 0, 255), 10)

                    angle_2 = vector_angle(origin_midpoint, (0.7 * ori_img.shape[1], 0.56 * ori_img.shape[1]))
                    if angle_2 < 90:
                        in_count -= 1
                        already_counted.remove(track_id)
                        in_count_id.remove(track_id)

                if len(paths) > 50:
                    del paths[list(paths)[0]]

            # 3. 绘制检测框
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_img = draw_boxes(ori_img, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.boxmot_use._xyxy_to_tlwh(bb_xyxy))

                # results.append((idx_frame - 1, bbox_tlwh, identities))
            print("yolo+deepsort:", time_synchronized() - t1)

            # 4. 绘制统计信息
            label = "蜜蜂总数: {}".format(str(total_track))
            t_size = get_size_with_pil(label, 25)
            x1 = 20
            y1 = 50
            color = compute_color_for_labels(2)
            cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))

            label = "穿过检测区蜜蜂数: {} ({} 外出, {} 进入)".format(str(total_counter), str(out_count), str(in_count))
            t_size = get_size_with_pil(label, 25)
            x1 = 20
            y1 = 100
            color = compute_color_for_labels(2)
            cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))

            if last_track_id >= 0:
                label = "实时更新: 蜜蜂{}号以{}方式穿过框".format(str(last_track_id), str("外出") if angle >= 0 else str('进入'))
                t_size = get_size_with_pil(label, 25)
                x1 = 20
                y1 = 150
                color = compute_color_for_labels(2)
                cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
                ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))

            end = time_synchronized()

            total_time = total_time + end - t1

            if self.args.display:
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.logger.info("{}/time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}"
                             .format(idx_frame, end - t1, 1 / (end - t1),
                                     bbox_xywh.shape[0], len(outputs)))
        print("进出蜜蜂总数: {}".format(str(total_counter)),",",
              "识别总用时：{}s".format(str(total_time)),",",
              "离开蜂箱蜜蜂总数：{}只".format(str(out_count)),",",
              "进入蜂箱的蜜蜂总数：{}只".format(str(in_count)), ",",
              "每帧平均跟踪蜜蜂数目：{}".format(str(tracking_num_output/idx_frame)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='./videos/bee_test_8.mp4', type=str)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1") #外界设备
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') #内部设备
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/v5s_150_best_2.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deep_sort
    parser.add_argument("--sort", default=True, help='True: sort model, False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=True, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--gpu", dest="use_cuda", action="store_false", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    yolo_reid = yolo_reid(cfg, args, path=args.video_path)
    with torch.no_grad():
        yolo_reid.deep_sort()
