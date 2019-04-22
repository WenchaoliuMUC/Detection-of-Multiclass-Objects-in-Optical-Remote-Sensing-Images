import os
img_dir = '/home/lwc/my_prj/dota/train/'
if __name__ == '__main__':
    img_file = os.listdir(os.path.join(img_dir, 'images'))
    img_num = img_file.__len__()

    img_train = open(os.path.join(img_dir, 'train.txt'), 'w')
    for idx in range(img_num):
        img_train.write(img_dir + 'images/' + img_file[idx]+'\n')
    img_train.close()
