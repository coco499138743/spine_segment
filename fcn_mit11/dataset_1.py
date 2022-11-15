import os
import SimpleITK as sitk
import cv2
import numpy as np
import torch
import torch.utils.data
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir,transform):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)


    def trans_value(self,image):
        for i in range(880):
            for j in range(880):
                if image[i][j] == 0:
                    image[i][j] = 0
                elif image[i][j] <= 10:
                    image[i][j] = 1
                elif image[i][j] > 10:
                    image[i][j] = 2
        return image

    def mask_to_onehot(slef,mask, palette):
        semantic_map = []
        for colour in range(len(palette)):
            equality = np.equal(mask, palette[colour])
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
        return semantic_map

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img= sitk.ReadImage(os.path.join(self.img_dir,img_id))
        mask= sitk.ReadImage(os.path.join(self.mask_dir,'mask_'+img_id))
        img = sitk.GetArrayFromImage(img)
        mask=sitk.GetArrayFromImage(mask)
        for i in range(12):
            img=img[i, :, :]
            mask=mask[i,:,:]
            img=cv2.resize(img,(880,880),interpolation=cv2.INTER_NEAREST)
            mask=cv2.resize(mask,(880,880),interpolation=cv2.INTER_NEAREST)
            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)  # 这个包比较方便，能把mask也一并做掉
                img = augmented['image']
                mask = augmented['mask']
            mask=self.trans_value(mask)
            img=img.reshape(1,880,880)
            mask = mask.reshape(880, 880,1)
            palette = [[0], [1], [2]]
            mask= self.mask_to_onehot(mask, palette)
            mask = mask.transpose([2, 0, 1])
            img = img.astype('float32') / 2500
            return img.astype(np.float32), mask.astype(np.float32)


