import os
import random

from PIL import Image
from torch.utils import data
import torchvision.transforms as tfs



class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, format='.jpg'):
        super(RESIDE_Dataset, self).__init__()
        self.train = train                                                 # /imgs/haze/xxx.jpg
        self.format = format                                               # haze下面一堆文件夹,每个文件夹下面有两个文件夹 一个是haze 一个是clear
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'haze'))
        self.haze_imgs = [os.path.join(path, 'haze', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]                              # img就是这张haze的图片
        id = img.split('\\')[-1]                                 # id是格式化以后的序号

        clear_name = id                                          # 配对图像中的图片名和有雾的图像是一模一样的

        clear = Image.open(os.path.join(self.clear_dir, clear_name))    # 打开有雾图片对应的没有雾的图片

        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))   # 把这两张图片转换成RGB图像返回
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)

