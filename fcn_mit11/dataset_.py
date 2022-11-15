import os

import cv2
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import ConcatDataset
import SimpleITK as sitk

class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
class CreateNiiDataset(Dataset):
    def __init__(self, path,img_ids):
        self.path1 = path # parameter passing
        self.img_ids = img_ids

    def crop(self, image, crop_size):
            shp = image.shape
            scl = [int((shp[0] - crop_size[0]) / 2), int((shp[1] - crop_size[1]) / 2)]
            image_crop = image[scl[0]:scl[0] + crop_size[0], scl[1]:scl[1] + crop_size[1]]
            return image_crop


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img1 = sitk.ReadImage(os.path.join(self.path1, img_id))
        data1 = sitk.GetArrayFromImage(img1)

        if data1.shape[0] != 880:
            data1 = self.crop(data1, [880, 880])

        if np.min(data1)<0:
            data1 = (data1 - np.min(data1))/(np.max(data1)-np.min(data1))


        data = {}
        data1 = data1[np.newaxis, np.newaxis, :, :]
        data1_tensor = torch.from_numpy(np.concatenate([data1,data1,data1], 1))
        data1_tensor = data1_tensor.type(torch.FloatTensor)
        print(data1_tensor)
        #data['A'] = data1_tensor # should be a tensor in Float Tensor Type

        #data['A_paths'] = [os.path.join(self.path1, self.A, file)] # should be a list, with path inside
        return data

    def load_data(self):
        return self

    def __len__(self):
        return len(self.imgs)


