import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
####################################################
import dota_process
import shapely.geometry as geometry
####################################################


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        min_x = min(box1[0], box2[0])
        max_x = max(box1[2], box2[2])
        min_y = min(box1[1], box2[1])
        max_y = max(box1[3], box2[3])
        box1_w = box1[2] - box1[0]
        box1_h = box1[3] - box1[1]
        box2_w = box2[2] - box2[0]
        box2_h = box2[3] - box2[1]
    else:
        min_x = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        max_x = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        min_y = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        max_y = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        box1_w = box1[2]
        box1_h = box1[3]
        box2_w = box2[2]
        box2_h = box2[3]
    union_w = max_x - min_x
    union_h = max_y - min_y
    inter_w = box1_w + box2_w - union_w
    inter_h = box1_h + box2_h - union_h
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    box1_area = box1_w * box1_h
    box2_area = box2_w * box2_h
    inter_area = inter_w * inter_h
    union_area = box1_area + box2_area - inter_area
    return inter_area/union_area


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        min_x = torch.min(boxes1[0], boxes2[0])
        max_x = torch.max(boxes1[2], boxes2[2])
        min_y = torch.min(boxes1[1], boxes2[1])
        max_y = torch.max(boxes1[3], boxes2[3])
        box1_w = boxes1[2] - boxes1[0]
        box1_h = boxes1[3] - boxes1[1]
        box2_w = boxes2[2] - boxes2[0]
        box2_h = boxes2[3] - boxes2[1]
    else:
        min_x = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        max_x = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        min_y = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        max_y = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        box1_w = boxes1[2]
        box1_h = boxes1[3]
        box2_w = boxes2[2]
        box2_h = boxes2[3]
    union_w = max_x - min_x
    union_h = max_y - min_y
    inter_w = box1_w + box2_w - union_w
    inter_h = box1_h + box2_h - union_h
    mask = ((inter_w <= 0) + (inter_h <= 0) > 0)
    box1_area = box1_w * box1_h
    box2_area = box2_w * box2_h
    inter_area = inter_w * inter_h
    inter_area[mask] = 0
    union_area = box1_area + box2_area - inter_area
    return inter_area/union_area


# def nms_cls(boxes, nms_thresh):
#     if len(boxes) == 0:
#         return boxes
#     det_confs = torch.zeros(len(boxes))
#     for i in range(len(boxes)):
#         det_confs[i] = 1-boxes[i][4]
#     _,sortIds = torch.sort(det_confs)
#     out_boxes = []
#     for i in range(len(boxes)):
#         box_i = boxes[sortIds[i]]
#         if box_i[4] > 0:
#             out_boxes.append(box_i)
#             for j in range(i+1, len(boxes)):
#                 box_j = boxes[sortIds[j]]
#                 if (box_j[4] != 0) & (box_i[6] == box_j[6]):
#                     if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
#                         box_j[4] = 0
#     return out_boxes

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if box_j[4] != 0:
                    if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                        box_j[4] = 0
    return out_boxes


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def plot_boxes(img, boxes, savename=None, class_names=None):

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype('times.ttf', 20)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb, font=fnt)
        draw_rect(draw, [x1, y1, x2, y2], rgb, 4)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img


def draw_rect(drawcontext, xy, color=None, width=1):
    offset = 1
    for i in range(0, width):
        drawcontext.rectangle(xy, outline=color)
        xy[0] = xy[0] - offset
        xy[1] = xy[1] + offset
        xy[2] = xy[2] + offset
        xy[3] = xy[3] - offset


def get_color(c, x, max_val):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);
    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)


def truths_length(truths):
    for i in range(800):
        if truths[i][1] == 0:
            return i
    return 800


####################################################
def read_truths_args(lab_path, min_box_scale, shape):
    new_truths = []
    objects = dota_process.read_dota_gt(lab_path)
    for obj in objects:
        gtbox = geometry.Polygon([(obj['box'][0], obj['box'][1]),
                                   (obj['box'][2], obj['box'][3]),
                                   (obj['box'][4], obj['box'][5]),
                                   (obj['box'][6], obj['box'][7])])
        out_box = gtbox.exterior.bounds
        x = abs(out_box[2] + out_box[0]) / (2. * shape[0])
        y = abs(out_box[1] + out_box[3]) / (2. * shape[1])
        w = abs(out_box[2] - out_box[0]) / shape[0]
        h = abs(out_box[3] - out_box[1]) / shape[1]
        if (max(w, h) < min_box_scale) or ((w*h) < (min_box_scale*min_box_scale)):
            continue
        c = dota_process.classnames.index(obj['name'])
        new_truths.append([c, x, y, w, h])
    return np.array(new_truths)
####################################################


def load_class_names(names_file):
    class_names = []
    fp = open(names_file, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        line = line.strip('\n')
        class_names.append(line)
    return class_names


def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0'
    fp = open(datacfg, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        line = line.strip('\n')
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def file_lines(file_path):
    fd = open(file_path, 'r')
    file_lines = fd.readlines()
    fd.close()
    lines = file_lines.__len__()
    for idx in range(lines):
        if file_lines[lines-idx-1].strip('\n') == '':
            lines = lines - 1
        else:
            break
    return lines


def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
