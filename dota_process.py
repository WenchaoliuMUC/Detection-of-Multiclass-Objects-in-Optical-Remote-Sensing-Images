import shapely.geometry as geometry
import numpy as np
import math

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
              'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
              'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']


########################################################
#    parse the dota ground truth in the format:
#    [x1, y1, x2, y2, x3, y3, x4, y4]
########################################################
def read_dota_gt(filename):

    objects = parse_dota_box(filename)
    for obj in objects:
        obj['box'] = vertex_rect(obj['box'])
        obj['box'] = list(map(int, obj['box']))
    return objects


########################################################
# parse the dota ground truth in the format:
# [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
########################################################
def parse_dota_box(filename):
    objects = []
    fd = open(filename, 'r')
    txt_lines = fd.readlines()
    fd.close()
    for idx in range(txt_lines.__len__()):
        line = txt_lines[idx]
        if line.strip('\n') != '':
            splitlines = line.strip().split(' ')
            object_struct = {}
            if len(splitlines) < 9:
                continue
            if len(splitlines) >= 9:
                    object_struct['name'] = splitlines[8]
            if len(splitlines) == 9:
                object_struct['difficult'] = '0'
            elif len(splitlines) >= 10:
                object_struct['difficult'] = splitlines[9]
            object_struct['box'] = [(float(splitlines[0]), float(splitlines[1])),
                                    (float(splitlines[2]), float(splitlines[3])),
                                    (float(splitlines[4]), float(splitlines[5])),
                                    (float(splitlines[6]), float(splitlines[7]))]
            gtbox = geometry.Polygon(object_struct['box'])
            object_struct['area'] = gtbox.area
            objects.append(object_struct)

    return objects


def vertex_rect(box):
    outbox = [box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]]
    return outbox


def calc_half_iou(box1, box2):
    inter_box = box1.intersection(box2)
    inter_area = inter_box.area
    box1_area = box1.area
    half_iou = inter_area / box1_area
    return inter_box, half_iou


def boxorig2sub(left, up, box):
    boxInsub = np.zeros(len(box))
    for i in range(int(len(box)/2)):
        boxInsub[i * 2] = int(box[i * 2] - left)
        boxInsub[i * 2 + 1] = int(box[i * 2 + 1] - up)
    return boxInsub


def box5_box4(box):
    distances = [calc_line_length((box[i * 2], box[i * 2 + 1]), (box[(i + 1) * 2], box[(i + 1) * 2 + 1])) for i in range(int(len(box)/2 - 1))]
    distances.append(calc_line_length((box[0], box[1]), (box[8], box[9])))
    pos = np.array(distances).argsort()[0]
    count = 0
    outbox = []
    while count < 5:
        if count == pos:
            outbox.append((box[count * 2] + box[(count * 2 + 2) % 10])/2)
            outbox.append((box[(count * 2 + 1) % 10] + box[(count * 2 + 3) % 10])/2)
            count = count + 1
        elif count == (pos + 1) % 5:
            count = count + 1
            continue
        else:
            outbox.append(box[count * 2])
            outbox.append(box[count * 2 + 1])
            count = count + 1
    return outbox


def calc_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def choose_best_point_order_fit_another(box1, box2):
    x1 = box1[0]
    y1 = box1[1]
    x2 = box1[2]
    y2 = box1[3]
    x3 = box1[4]
    y3 = box1[5]
    x4 = box1[6]
    y4 = box1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(box2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]
