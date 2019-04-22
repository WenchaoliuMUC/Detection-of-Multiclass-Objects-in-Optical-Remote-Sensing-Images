import torch
from utils import convert2cpu


def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block = None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks


def load_conv(buf, start, conv_model):
    if conv_model.bias is not None:
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
    num_w = conv_model.weight.numel()
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view(conv_model.weight.shape)); start = start + num_w
    return start


def load_deform_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view(conv_model.weight.shape))
    start = start + num_w

    num_w = conv_model.layer_1.weight.numel()
    conv_model.layer_1.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view(conv_model.layer_1.weight.shape))
    start = start + num_w
    return start


def save_conv(fp, conv_model):
    if conv_model.weight.is_cuda:
        if conv_model.bias is not None:
            convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        if conv_model.bias is not None:
            conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def save_deform_conv(fp, conv_model):
    if conv_model.weight.is_cuda:
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
        convert2cpu(conv_model.layer_1.weight.data).numpy().tofile(fp)
    else:
        conv_model.weight.data.numpy().tofile(fp)
        conv_model.layer_1.weight.data.numpy().tofile(fp)


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()

    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view(conv_model.weight.shape)); start = start + num_w
    if conv_model.bias is not None:
        num_w_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_w_b]).view(conv_model.bias.shape)); start = start + num_w_b
    return start


def load_bn(buf, start, bn_model):
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    return start


def save_bn(fp, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)


def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
        if conv_model.bias is not None:
            convert2cpu(conv_model.bias.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)
        if conv_model.bias is not None:
            conv_model.bias.data.numpy().tofile(fp)
