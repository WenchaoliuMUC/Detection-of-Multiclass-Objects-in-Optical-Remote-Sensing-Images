from __future__ import print_function
import sys
if len(sys.argv) != 4:
    print('Usage:')
    print('python train.py datacfg cfgfile weightfile')
    exit()

import torch.optim as optim
from torchvision import transforms
import dataset
from utils import *
from config import parse_cfg
from deform_darknet import deformDarknet, get_region_boxes
from torch.autograd import Variable

# Training settings
datacfg       = sys.argv[1]
cfgfile       = sys.argv[2]
weightfile    = sys.argv[3]

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

trainlist     = data_options['train']
testlist      = data_options['valid']
backupdir     = data_options['backup']
nsamples      = file_lines(trainlist)
gpus          = data_options['gpus']
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])
subdiv        = int(net_options['subdivisions'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
max_epochs    = max_batches*batch_size//nsamples+1
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
save_interval = 1   # epoches
dot_interval  = 70  # batches

# Test parameters
conf_thresh   = 0.3
nms_thresh    = 0.4
iou_thresh    = 0.5
############################################################
if not os.path.exists(backupdir):
    os.mkdir(backupdir)
############################################################
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
############################################################
model = deformDarknet(cfgfile)
############################################################
init_epoch = int(weightfile.split('/')[-1].split('.')[0])
############################################################
model.load_weights(weightfile)
#################################s###########################
region_loss = model.loss
region_loss.seen = model.seen
processed_batches = model.seen//batch_size

init_width = model.width
init_height = model.height
############################################################
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]), train=False),
    batch_size=batch_size//subdiv, shuffle=False, **kwargs)

train_loader = torch.utils.data.DataLoader(
    dataset.listDataset(trainlist, shape=(init_width, init_height),
                        shuffle=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]),
                        train=True,
                        seen=model.seen,
                        batch_size=batch_size // subdiv,
                        num_workers=num_workers // subdiv),
    batch_size=batch_size // subdiv, shuffle=False, **kwargs)
############################################################
if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
############################################################
optimizer = optim.Adam(model.parameters(), lr=learning_rate/batch_size, weight_decay=decay*batch_size)
# optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate/batch_size, weight_decay=decay*batch_size)


def train(epoch):
    global processed_batches
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()
        data, target = Variable(data), Variable(target)
        if batch_idx % subdiv == 0:
            processed_batches = processed_batches + 1
            optimizer.zero_grad()
        output = model(data)
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss = region_loss(output, target, batch_idx)
        loss.backward()
        if batch_idx % subdiv == 1:
            optimizer.step()
    if (epoch+1) % save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch + 1))


def test():
    model.eval()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    num_classes = cur_model.num_classes
    anchors = cur_model.anchors
    num_anchors = cur_model.num_anchors
    total = 0.0
    proposals = 0.0
    correct = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, requires_grad=False)
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)
            total = total + num_gts
            for j in range(len(boxes)):
                if boxes[j][4] > conf_thresh:
                    proposals = proposals + 1
            for k in range(num_gts):
                box_gt = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], 1.0, 1.0, truths[k][0]]
                best_iou = 0
                best_j = 0
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou > iou_thresh and boxes[best_j][4] > conf_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct + 1

    precision = 1.0 * correct / (proposals + eps)
    recall = 1.0 * correct / (total + eps)
    fscore = 2.0 * precision * recall / (precision + recall + eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
    fp = open('./log.dat', 'a')
    fp.write("precision: %f, recall: %f, fscore: %f\n" % (precision, recall, fscore))
    fp.close()


if __name__ == '__main__':
    for epoch in range(init_epoch, max_epochs):
        train(epoch)
        if epoch % 4 == 3:
            test()