import os
from PIL import Image

ori_img_path = '/home/lwc/my_prj/DOTA/val_1024/trainsplit/images/'
ori_label_path = '/home/lwc/my_prj/DOTA/val_1024/trainsplit/labels/'
null_label_path = '/home/lwc/my_prj/DOTA/val_1024/trainsplit/null_labels/'
null_img_path = '/home/lwc/my_prj/DOTA/val_1024/trainsplit/null_images/'
img_list = os.listdir(ori_img_path)
if not os.path.exists(null_img_path):
    os.makedirs(null_img_path)
if not os.path.exists(null_label_path):
    os.makedirs(null_label_path)

for img_name in img_list:
    name = img_name.strip('\n')
    ori_img = ori_img_path + name
    ori_label_name = ori_label_path + name.strip('.png') + '.txt'
    ori_label_file = open(ori_label_name, 'r')
    sss = ori_label_file.readlines().__len__()
    if sss == 0:
        os.system('mv ' + ori_label_name + ' ' + null_label_path)
        os.system('mv ' + ori_img + ' ' + null_img_path)

    ori_label_file.close()
