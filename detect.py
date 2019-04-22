from utils import *
from deform_darknet import deformDarknet, get_region_boxes
from torchvision import transforms
from torch.autograd import Variable


seed = int(time.time())
torch.manual_seed(seed)


def detect(model, weightfile, imgfile, Result_dir):

    conf_thresh = 0.5
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
        plot_boxes(img, boxes, os.path.join(Result_dir, imgpath.split('/')[-1]), class_names)


if __name__ == '__main__':
    workdir = './'
    cfgfile = workdir + 'cfg/orn_4_dota.cfg'
    model = deformDarknet(cfgfile)
    imgfile = './test_img/test_img_list.txt'

    weightfile_list = open(workdir + 'backup/test_weight_list.txt').readlines()

    num_weight = weightfile_list.__len__()
    for idx_weight in range(num_weight):
        weightfile = workdir + 'backup/' + weightfile_list[idx_weight].strip('\n')
        Result_dir = workdir + 'final_result/test_img/'
        if not os.path.exists(Result_dir):
            os.mkdir(Result_dir)
        detect(model, weightfile, imgfile, Result_dir)

