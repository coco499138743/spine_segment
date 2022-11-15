# 项目1验证代码
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import archs
import losses
from dataset_1 import Dataset
from metrics import *
from losses import *
from losses import SoftDiceLoss
LOSS = False
batch_size=2
palette = [[0], [1], [2]]
import SimpleITK as sitk

data_path = r'data_mit\dataset\train\image'
dataname_list = os.listdir(data_path)
dataname_list.sort()
_, val_img_ids = train_test_split(dataname_list, test_size=0.2, random_state=41)
val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('data_mit', 'dataset', 'train', 'image'),
        mask_dir=os.path.join('data_mit', 'dataset', 'train', 'groundtruth'))

val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True)


def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + eps) / (unionset + eps)

    return loss.sum() / N
def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x
# 验证用的模型名称
vgg_model = archs.VGGNet(requires_grad=True, show_params=False)
model = archs.FCNs(pretrained_net=vgg_model, n_class=3)
model = model.cuda()
checkpoint=torch.load('models/FCN/model.pth')
model.load_state_dict(checkpoint)
model.eval()


def auto_val(model):
    # 效果展示图片数
    iters = 0
    SIZES = 8
    imgs = []
    preds = []
    gts = []
    dices = 0
    tumor_dices = 0
    bladder_dices = 0
    for i, (img, mask) in enumerate(val_loader):
        im = img
        img = img.cuda()
        print(img.shape,mask.shape)
        model = model.cuda()
        pred = model(img)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()
        iters += 2
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        bladder_dice = diceCoeff(pred[:,1,:, :], mask[:,1,:, :])
        tumor_dice =diceCoeff(pred[:,2,:, :], mask[:,2,:, :])
        mean_dice = (bladder_dice + tumor_dice) / 2
        dices += mean_dice
        tumor_dices += tumor_dice
        bladder_dices += bladder_dice
        acc = accuracy(pred, mask)
        p = precision(pred, mask)
        r = recall(pred, mask)
        print('mean_dice={:.4}, bladder_dice={:.4}, tumor_dice={:.4}, acc={:.4}, p={:.4}, r={:.4}'
              .format(mean_dice.item(), bladder_dice.item(), tumor_dice.item(),
                      acc, p, r))
        mask = mask.cpu().detach().numpy()[1].transpose([1, 2, 0])
        mask = onehot_to_mask(mask, palette)
        pred = pred.cpu().detach().numpy()[1].transpose([1, 2, 0])
        pred = onehot_to_mask(pred, palette)
        mask = mask.reshape(880, 880)
        pred=pred.reshape(880,880)
        im = im[1].numpy().transpose([1, 2, 0])
        im = im.reshape(880, 880)
        if len(imgs) < SIZES:
            imgs.append(im)
            preds.append(pred)
            gts.append(mask)

    print('##############',np.array(imgs).shape)
    print('##############', np.array(preds).shape)
    print('##############', np.array(gts).shape)
    val_mean_dice = dices / (len(val_loader) / batch_size)
    val_tumor_dice = tumor_dices / (len(val_loader) / batch_size)
    val_bladder_dice = bladder_dices / (len(val_loader) / batch_size)
    print('Val Mean Dice = {:.4}, Val Bladder Dice = {:.4}, Val Tumor Dice = {:.4}'
          .format(val_mean_dice, val_bladder_dice, val_tumor_dice))
    #画图
    fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(18,4*6))
    for row_num in range(6):
        ax[row_num][0].imshow(imgs[row_num])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(preds[row_num])
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(gts[row_num])
        ax[row_num][2].set_title("Target image")
    plt.savefig('img/valitation.png')
    plt.show()


if __name__ == '__main__':
    # val(model)
    auto_val(model)
