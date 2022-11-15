import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

import archs
from metrics import iou_score
import SimpleITK as sitk
from torchvision import transforms


def crop(image):
    g = np.zeros([880, 880])
    if image.shape[0] >= 880:
        for i in range(880):
            for j in range(880):
                g[i][j] = image[i][j]
    if image.shape[0] < 880:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                g[i][j] = image[i][j]
    return g


def trans_value(image):
    for i in range(880):
        for j in range(880):
            if image[i][j] > 0:
                image[i][j] = 1
    return image

os.environ['KMP_DUPLICATE_LIB_OK']='True'
data_path = r'data_mit\dataset\train\image'
data_path_ = r'data_mit\dataset\train\groundtruth'
dataname_list = os.listdir(data_path)
dataname_list.sort()
img = sitk.ReadImage(os.path.join(data_path, dataname_list[4]))
mask= sitk.ReadImage(os.path.join(data_path_,'mask_'+dataname_list[4]))
img = sitk.GetArrayFromImage(img)
mask=sitk.GetArrayFromImage(mask)
img=img[1, :, :]
mask=mask[1,:,:]
img=crop(img)
mask=crop(mask)
mask=trans_value(mask)
img=img.reshape(1,880,880)
mask = mask.reshape(1, 880, 880)
img = img.astype('float32') / 2500
img=img.astype(np.float32)
mask=mask.astype(np.float32)
img=img.reshape(1,1,880,880)
img=torch.from_numpy(img)
img=img.cuda()
model = archs.NestedUNet(num_classes=1)
model = model.cuda()
checkpoint=torch.load('models/dsb2018_96_NestedUNet_woDS/model.pth')
model.load_state_dict(checkpoint)
model.eval()
output = model(img)
iou = iou_score(output, mask)
print('iou',iou)
output=output.reshape(880,880)
output = output.data.cpu().numpy()
mean=np.array([0.5])
std=np.array([0.5])
output=std*output+mean
output = np.clip(output, 0, 1)
print('output',output.max(),output.min())
plt.imshow(output)
plt.savefig('img/seg.png')
plt.show()
