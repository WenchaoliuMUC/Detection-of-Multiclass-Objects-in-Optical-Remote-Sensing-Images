from utils import *
from deform_darknet import deformDarknet, get_region_boxes
from torchvision import transforms
from torch.autograd import Variable

seed = int(time.time())
torch.manual_seed(seed)


def write_boxes(img, file_name, boxes, class_names=None, Result_dir=None):
    file_class = []
    for i in range(len(class_names)):
        file_class.append(open(Result_dir + '/Task2_'+class_names[i]+'.txt', 'a'))

    width = img.width
    height = img.height

    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        cls_id = box[6]
        conf = box[4]*box[5]
        file_class[cls_id].write(file_name.split('/')[-1].split('.')[0]+' '+str(conf)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+'\n')

    for i in range(len(class_names)):
        file_class[i].close()


def detect(model, weightfile, imgfile, Result_dir):

    conf_thresh = 0.01
    nms_thresh = 0.4
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    namesfile = 'data/dota.names'
    model.eval()
    use_cuda = 1
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.cuda.manual_seed(seed)
        model.cuda()
#################################################################
    img_list_file = open(imgfile)
    img_list = img_list_file.readlines()
    img_list_file.close()
    # img_list = os.listdir(imgfile)
#################################################################
    for imgpath in img_list:
        imgpath = imgpath.strip('\n')
        img = Image.open(imgpath).convert('RGB')
        x_idx = range(0, img.width, 1024-512)
        y_idx = range(0, img.height, 1024-512)
        all_boxes = []
        for x_start in x_idx:
            for y_start in y_idx:
                x_stop = x_start + 1024
                if x_stop > img.width:
                    x_start = img.width - 1024
                    x_stop = img.width
                y_stop = y_start + 1024
                if y_stop > img.height:
                    y_start = img.height - 1024
                    y_stop = img.height
                croped_img = img.crop((x_start, y_start, x_stop, y_stop))
                croped_img = transforms.ToTensor()(croped_img)
                croped_img = torch.unsqueeze(croped_img, 0)
                croped_img = Variable(croped_img, requires_grad=False)
                output = model(croped_img.cuda()).data
                boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors,
                                         x_start, y_start, img.width, img.height, validation=True)[0]
                all_boxes = all_boxes + boxes
        boxes = nms(all_boxes, nms_thresh)
        class_names = load_class_names(namesfile)
        # write_boxes(imgpath, boxes, Result_dir)
        write_boxes(img, imgpath, boxes, class_names, Result_dir)
        print("save results of %s" % imgpath)


if __name__ == '__main__':
    workdir = './'
    cfgfile = workdir + 'cfg/orn_4_dota.cfg'
    model = deformDarknet(cfgfile)
    imgfile = '/home/lwc/my_prj/DOTA/val/val/images/val_list.txt'

    weightfile_list = open(workdir + 'backup/weight_list.txt').readlines()

    num_weight = weightfile_list.__len__()
    for idx_weight in range(num_weight):
        weightfile = workdir + 'backup/' + weightfile_list[idx_weight].strip('\n')
        Result_dir = workdir + 'backup/' + weightfile_list[idx_weight].split('.')[0]
        if not os.path.exists(Result_dir):
            os.mkdir(Result_dir)
        detect(model, weightfile, imgfile, Result_dir)

