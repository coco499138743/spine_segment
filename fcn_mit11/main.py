import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
from glob import glob

from dataset_1 import Dataset

import torch
import argparse
import os
from glob import glob

from torch import optim

import losses
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from metrics import iou_score
from utils import AverageMeter
import torch
import torch.backends.cudnn as cudnn
import archs
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
import numpy as np
import random

# create model
model = archs.NestedUNet(num_classes=1)
model = model.cuda()
params = filter(lambda p: p.requires_grad, model.parameters())



data_path = r'data_mit\dataset\train\image'
label_path = r'data_mit\dataset\train\groundtruth'
dataname_list = os.listdir(data_path)
dataname_list.sort()
train_dataset = Dataset(
        img_ids=dataname_list,
        img_dir=os.path.join('data_mit', 'dataset', 'train','image'),
        mask_dir=os.path.join('data_mit', 'dataset', 'train','groundtruth'),
        )
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    drop_last=True)

def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        print('---------',input.shape)
        print('---------', target.shape)
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()
    torch.save(model.state_dict(), 'models/mit/model.pth')
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
#train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
criterion=losses.BCEDiceLoss().cuda()
optimizer = optim.SGD(params, lr=0.001, momentum=0.9,
                      nesterov=False, weight_decay= 0.0001)
for epoch in range(1):
    print('Epoch [%d/50]' % (epoch))
    train_log = train(train_loader, model, criterion, optimizer)
    torch.cuda.empty_cache()