import os

import cv2

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import os
import glob
import SimpleITK as sitk
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
from dataset import Dataset
import argparse
import os
from glob import glob


import torch


from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

# LiST数据集为例，data是volume，lable是segmementation，nii格式
# 加r，可以将路径转换为原始字符串，不用变为‘\\’
def crop(image):
    g = np.zeros([880, 880])
    if image.shape[0] > 880:
        for i in range(880):
            for j in range(880):
                g[i][j] = image[i][j]
    if image.shape[0] < 880:
        for i in range(880):
            for j in range(880):
                if i <= image.shape[0]:
                    g[i][j] = image[i][j]
                else:
                    g[i][j] = 0
    return g
def trans_value(image):
    for i in range(880):
            for j in range(880):
                if  image[i][j] > 0:
                    image[i][j] = 1
    return image
#这个是我测试的部分，没有实际意义
data_path = r'data_mit\dataset\train\image'
label_path = r'data_mit\dataset\train\groundtruth'
dataname_list = os.listdir(label_path)
dataname_list.sort()
'''for i in range(len(dataname_list)):
    ori_data = sitk.ReadImage(os.path.join(label_path, dataname_list[i]))  # 读取一个数据
    data1 = sitk.GetArrayFromImage(ori_data)  # 获取数据的array
    print(data1.shape)'''
ori_data = sitk.ReadImage(os.path.join(label_path, dataname_list[3]))  # 读取一个数据
data1 = sitk.GetArrayFromImage(ori_data)
print(data1[1, :, :].shape)
img=crop(data1[1, :, :])
img=trans_value(img)
print(img.shape)
print(img.max())
print(img.min())
'''plt.imshow(data1[1, :, :])
plt.show()
for i in range(5):
    plt.imshow(data1[:, :, i])
    plt.show()
print(data1[3,:,:].max())
print(data1[3,:,:].min())
'''
 #打印数据name、shape、某一个位置的元素的值（z,y,x）
'''train_dataset = Dataset(
        img_ids=dataname_list,
        img_dir=os.path.join('data_mit', 'dataset', 'train','image'),
        mask_dir=os.path.join('data_mit', 'dataset', 'train','groundtruth'),
        )
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True)
for i, data in enumerate(train_loader):
	# i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    print("第 {} 个Batch \n{}".format(i, data))


img_ids = glob(os.path.join('inputs', 'dsb2018_96', 'images', '*' + '.png'))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
print(img_ids)
train_dataset = Dataset(
        img_ids=img_ids,
        img_dir=os.path.join('inputs', 'dsb2018_96', 'images'),
        mask_dir=os.path.join('inputs', 'dsb2018_96', 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=1,
        )
print(train_dataset)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=True)
data_path = r'inputs\mit_ai_2021_course_2_project_1_dataset_train_1\dataset\train\image'
label_path = r'inputs\mit_ai_2021_course_2_project_1_dataset_train_1\dataset\train\groundtruth'
dataname_list = os.listdir(label_path)
dataname_list = os.listdir(label_path)
dataname_list.sort()
train_dataset = CreateNiiDataset(
        path=data_path,
        img_ids=dataname_list,
        )
print(train_dataset)'''


