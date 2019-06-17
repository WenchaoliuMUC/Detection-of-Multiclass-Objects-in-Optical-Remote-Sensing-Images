import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from region_loss import RegionLoss
from config import *
from iorn.modules import ORConv2d
from iorn_bn import ORBatchNorm2d
from torch.autograd import Variable


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors,
                     x_start=0, y_start=0, imgwidth=0, imgheight=0, validation=False):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).\
        transpose(0, 1).contiguous().view(5 + num_classes, batch * num_anchors * h * w)

    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])
    cls_confs = torch.nn.Softmax(dim=1)((Variable(output[5:5 + num_classes].transpose(0, 1)))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    box_idx = torch.nonzero(det_confs > conf_thresh)

    if not validation:
        for b_idx in range(batch):
            boxes = []
            for ind in box_idx:
                if (ind >= b_idx * sz_hwa) and (ind < sz_hwa * (b_idx + 1)):
                    det_conf = det_confs[ind][0]
                    bcx = xs[ind][0]
                    bcy = ys[ind][0]
                    bw = ws[ind][0]
                    bh = hs[ind][0]
                    cls_max_conf = cls_max_confs[ind][0]
                    cls_max_id = cls_max_ids[ind][0]
                    box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                    boxes.append(box)
            all_boxes.append(boxes)
    else:
        for b_idx in range(batch):
            boxes = []
            for ind in box_idx:
                if (ind >= b_idx * sz_hwa) and (ind < sz_hwa * (b_idx + 1)):
                    det_conf = det_confs[ind][0]
                    bcx = xs[ind][0]
                    bcy = ys[ind][0]
                    bw = ws[ind][0]
                    bh = hs[ind][0]
                    cls_max_conf = cls_max_confs[ind][0]
                    cls_max_id = cls_max_ids[ind][0]
                    box = [(x_start + (bcx / w) * 1024.0) / imgwidth, (y_start + (bcy / h) * 1024.0) / imgheight,
                           ((bw / w) * 1024.0) / imgwidth, ((bh / h) * 1024.0) / imgheight, det_conf, cls_max_conf,
                           cls_max_id]
                    # box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                    boxes.append(box)
            all_boxes.append(boxes)
    return all_boxes


class MaxPoolStride(nn.Module):
    def __init__(self):
        super(MaxPoolStride, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


class EmpotyModule(nn.Module):
    def __init__(self):
        super(EmpotyModule, self).__init__()

    def forward(self, x):
        return x


class deformDarknet(nn.Module):
    def __init__(self, cfgfile):
        super(deformDarknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks)
        self.loss = self.models[len(self.models) - 1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks) - 1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'iorn_convolutional' or block['type'] == 'convolutional' or\
                    block['type'] == 'maxpool':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'trans_conv':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 3:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x = torch.cat((x1, x2, x3), 1)
                    outputs[ind] = x
            elif block['type'] == 'region':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x

    def create_network(self, blocks):
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'iorn_convolutional':
                conv_id = conv_id + 1
                iorn_id = 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                activation = block['activation']
                pad = int(block['pad'])
                dilate = int(block['dilate'])
                stride = int(block['stride'])
                nOrientation = int(block['nOrientation'])
                model = nn.Sequential()
                if batch_normalize:
                    if iorn_id == 1:
                        model.add_module('conv{0}'.format(conv_id),
                                     ORConv2d(prev_filters, filters // nOrientation,
                                              arf_config=(1, nOrientation), kernel_size=3,
                                              padding=pad, stride=stride, dilation=dilate))
                    else:
                        model.add_module('conv{0}'.format(conv_id),
                                         ORConv2d(prev_filters // nOrientation, filters // nOrientation,
                                                  arf_config=nOrientation, kernel_size=3,
                                                  padding=pad, stride=stride, dilation=dilate))
                    model.add_module('bn{0}'.format(conv_id), ORBatchNorm2d(filters // nOrientation, nOrientation))
                else:
                    if iorn_id == 1:
                        model.add_module('conv{0}'.format(conv_id),
                                         ORConv2d(prev_filters, filters / nOrientation, arf_config=nOrientation,
                                                  kernel_size=3, padding=1))
                    else:
                        model.add_module('conv{0}'.format(conv_id),
                                         ORConv2d(prev_filters // nOrientation, filters // nOrientation,
                                                  arf_config=nOrientation, kernel_size=3, padding=1))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'trans_conv':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.ConvTranspose2d(prev_filters, filters, kernel_size, stride,
                                                        pad, output_padding=1, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.ConvTranspose2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert (layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                elif len(layers) == 3:
                    assert (layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]]
                out_filters.append(prev_filters)
                models.append(EmpotyModule())
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors) // loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            # if ind == 24:
            #     break
            if block['type'] == 'net':
                continue
            elif block['type'] == 'iorn_convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'convolutional' or block['type'] == 'trans_conv':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'region':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)
        ind = -1
        for blockId in range(1, cutoff + 1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'iorn_convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'convolutional' or block['type'] == 'trans_conv':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'region':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()
