#!/usr/bin/python
# encoding: utf-8
import random
from PIL import Image
import numpy as np
####################################################
import shapely.geometry as geometry
import dota_process
####################################################


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    return im


def rand_scale(s):
    scale = random.uniform(1, s)
    if random.randint(1, 10000) % 2:
        return scale
    return 1./scale


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


####################################################
def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height
    ow = img.width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    flip = random.randint(1, 10000) % 2
    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    sized = cropped.resize(shape)

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, swidth, sheight, pleft, ptop
####################################################


def fill_truth_detection(labpath, w, h, flip, swidth, sheight, pleft, ptop):
    max_boxes = 800
    cc = 0
    label = np.zeros((max_boxes, 5))

    left = max(pleft, 0)
    right = min((pleft + swidth), w)
    up = max(ptop, 0)
    down = min((ptop + sheight), h)

    objects = dota_process.read_dota_gt(labpath)
    if objects is None:
        return label
    imgbox = geometry.Polygon([(left, up), (right, up), (right, down),
                               (left, down)])

    for obj in objects:
        gtbox = geometry.Polygon([(obj['box'][0], obj['box'][1]),
                                  (obj['box'][2], obj['box'][3]),
                                  (obj['box'][4], obj['box'][5]),
                                  (obj['box'][6], obj['box'][7])])
        if gtbox.area <= 0:
            continue
        inter_box, half_iou = dota_process.calc_half_iou(gtbox, imgbox)
        if half_iou == 1:
            boxInsub = dota_process.boxorig2sub(pleft, ptop, obj['box'])
        elif half_iou > 0.3:
            inter_box = geometry.polygon.orient(inter_box, sign=1)
            out_box = list(inter_box.exterior.coords)[0: -1]
            if len(out_box) < 4:
                continue
            out_box2 = []
            for i in range(len(out_box)):
                out_box2.append(out_box[i][0])
                out_box2.append(out_box[i][1])

            if len(out_box) == 5:
                out_box2 = dota_process.box5_box4(out_box2)
            elif len(out_box) > 5:
                continue
            out_box2 = dota_process.choose_best_point_order_fit_another(out_box2, obj['box'])
            boxInsub = dota_process.boxorig2sub(left, up, out_box2)

            for index, item in enumerate(boxInsub):
                if index % 2 == 0:
                    if item <= 1:
                        boxInsub[index] = 1
                    elif item >= swidth:
                        boxInsub[index] = swidth
                elif index % 2 == 1:
                    if item <= 1:
                        boxInsub[index] = 1
                    elif item >= sheight:
                        boxInsub[index] = sheight
        else:
            continue
        length = max(np.abs(np.subtract(boxInsub[0], boxInsub[4])), np.abs(np.subtract(boxInsub[1], boxInsub[5])))
        boxInsub = [(boxInsub[0], boxInsub[1]), (boxInsub[2], boxInsub[3]), (boxInsub[4], boxInsub[5]),
                    (boxInsub[6], boxInsub[7])]

        if (length / min(swidth, sheight)) < (5.0/704.0) and geometry.Polygon(boxInsub).area < 15:
            continue
        rect_minx = geometry.Polygon(boxInsub).bounds[0]
        rect_miny = geometry.Polygon(boxInsub).bounds[1]
        rect_maxx = geometry.Polygon(boxInsub).bounds[2]
        rect_maxy = geometry.Polygon(boxInsub).bounds[3]
        if max(abs(rect_maxx - rect_minx)/swidth, abs(rect_maxy - rect_miny)/sheight) < (5.0/704.0) \
                or min(abs(rect_maxx - rect_minx)/swidth, abs(rect_maxy - rect_miny)/sheight) <= 0:
            continue
        label[cc][0] = dota_process.classnames.index(obj['name'])
        if flip:
            label[cc][1] = 0.999 - abs(rect_maxx + rect_minx) / (2.0 * swidth)
        else:
            label[cc][1] = abs(rect_maxx + rect_minx) / (2.0 * swidth)
        label[cc][2] = abs(rect_maxy + rect_miny) / (2.0 * sheight)
        label[cc][3] = abs(rect_maxx - rect_minx) / swidth
        label[cc][4] = abs(rect_maxy - rect_miny) / sheight
        cc += 1
        if cc >= 800:
            break
    label = np.reshape(label, (-1))
    return label
####################################################


def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    ####################################################
    img, flip, swidth, sheight, pleft, ptop = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, img.width, img.height, flip, swidth, sheight, pleft, ptop)
    ####################################################
    return img, label
