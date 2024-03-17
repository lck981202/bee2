from pathlib import Path
import torch
import cv2
import argparse

import torch

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from boxmot.utils import logger as LOGGER
from boxmot.utils.torch_utils import select_device

from utils.torch_utils_1 import time_synchronized

from ultralytics.yolo.engine.model import YOLO, TASK_MAP
from ultralytics.yolo.utils import SETTINGS, colorstr, ops, is_git_dir, IterableSimpleNamespace
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.engine.predictor import BasePredictor

from multi_yolo_backend import MultiYolo
from examples.bee_detect_yolov8 import Bee_detect

from collections import Counter
from collections import deque

def tlbr_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format 求物体中心的坐标
    return midpoint

def on_predict_start(predictor):
    predictor.trackers = []

    predictor.tracker_outputs = [None] * predictor.dataset.bs  # 后面一个参数是列表的长度
    predictor.args.tracking_config = \
        ROOT / \
        'boxmot' / \
        opt.tracking_method / \
        'configs' / \
        (opt.tracking_method + '.yaml')
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.args.device,
            predictor.args.half
        )
        predictor.trackers.append(tracker)
    predictor.create_tracker.update()

@torch.no_grad()
def run(args):
    # 实例化YOLO模型，选择yolov8n或者yolo_model作为参数
    model = YOLO(args['yolo_model'] if 'v8' in str(args['yolo_model']) else 'yolov8n')
    # 复制模型的overrides属性
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)

    # extract task predictor从task_map提取任务的预测器
    predictor = model.predictor

    # combine default predictor args with custom, preferring custom
    combined_args = {**predictor.args.__dict__, **args}
    # overwrite default args
    predictor.args = IterableSimpleNamespace(**combined_args)
    predictor.args.device = select_device(args['device'])
    LOGGER.info(args)

    # setup source and model设置数据来源和模型
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)
    predictor.setup_source(predictor.args.source)

    predictor.args.imgsz = check_imgsz(predictor.args.imgsz, stride=model.model.stride, min_dim=2)  # check image size
    predictor.save_dir = increment_path(Path(predictor.args.project) / predictor.args.name,
                                        exist_ok=predictor.args.exist_ok)

    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (predictor.save_dir / 'labels' if predictor.args.save_txt else predictor.save_dir).mkdir(parents=True,
                                                                                                 exist_ok=True)
    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(
            imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz))
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = 0, [], None, (
    ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
    predictor.add_callback('on_predict_start', on_predict_start)
    predictor.run_callbacks('on_predict_start')
    model = MultiYolo(
        model=model.predictor.model if 'v8' in str(args['yolo_model']) else args['yolo_model'],
        device=predictor.device,
        args=predictor.args
    )  #ops是调入边界框信息

    cap = cv2.VideoCapture(predictor.args.source)  # 使用视频文件路径初始化VideoCapture对象

    #初始化多个元素
    tracking_num_output = 0
    paths = {}
    frame_idx = 0
    total_track = 0

    for video_path, img, frame, cap in predictor.dataset:
        t1 = time_synchronized()

        xyxy, conf, cls, id = predictor.Boxes(xyxy, conf, cls, id)
        #调用results包中Boxes
        outputs = predictor.create_tracker.update()

        line_1 = [(int(0.59 * frame.shape[0]), int(0.3 * frame.shape[1])),
                  (int(1.25 * frame.shape[0]), int(0.3 * frame.shape[1]))]
        line_2 = [(int(1.25 * frame.shape[0]), int(0.343 * frame.shape[1])),
                  (int(0.59 * frame.shape[0]), int(0.343 * frame.shape[1]))]  # 蜂箱进出口重合线
        line_3 = [(int(0.59 * frame.shape[0]), int(0.343 * frame.shape[1])),
                  (int(0.59 * frame.shape[0]), int(0.3 * frame.shape[1]))]  # 左端线
        line_4 = [(int(1.25 * frame.shape[0]), int(0.3 * frame.shape[1])),
                  (int(1.25 * frame.shape[0]), int(0.343 * frame.shape[1]))]  # 右端线

        for track in outputs:
            bbox = track[:4]
            track_id = track[-1]
            midpoint = tlbr_midpoint(bbox)
            origin_midpiont = (midpoint[0], frame.shape[0] - midpoint[1])
            tracking_num_output += 1
            if track_id not in paths:
                paths[track_id] = deque(maxlen=2)
                total_track = track_id

            paths[track_id].append(midpoint)
            previous_midpoint = paths[track_id][0]
            origin_previous_midpiont = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])


        while cap.isOpened():
            frame_idx += 1

            ret, frame = cap.read()  # 读取一帧
            if not ret:
                break
            if predictor.args.show:
                # 在这里进行线段绘制的操作


                cv2.line(frame, line_1[0], line_1[1], (0, 255, 0), 2)  # 在每一帧上绘制线段
                cv2.line(frame, line_2[0], line_2[1], (0, 255, 0), 2)
                cv2.line(frame, line_3[0], line_3[1], (0, 255, 0), 2)
                cv2.line(frame, line_4[0], line_4[1], (0, 255, 0), 2)

                cv2.imshow('Video', frame)  # 显示帧

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 键退出
                break

        cap.release()
        cv2.destroyAllWindows()
    print(frame_idx)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'best.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='examples/test_6.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_false', help='display tracking video results')
    parser.add_argument('--save', action='store_true', help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--hide-label', action='store_false', help='hide labels when show')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true', help='save tracking results in a txt file')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)