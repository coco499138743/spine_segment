import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.nn as nn
'''try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        #input = torch.sigmoid(input)
        input=torch.softmax(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
def CE_Loss(inputs, target, num_classes=3):
    n, c, h, w = inputs.size()
    nt, ht, wt,ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim = -1), temp_target)
    return CE_loss
def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ct ,ht, wt= target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

        # --------------------------------------------#
        #   计算dice loss
        # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss
def dice_coef(output, target, smooth=1):
    output = torch.sigmoid(output)
    mean_loss = 0
    for i in range(output.size(1)):
        intersection = torch.sum(target[:,:,:,i] * output[:,:,:,i], axis=-1)
        union = torch.sum(target[:,:,:,i], axis=-1) + torch.sum(output[:,:,:,i], axis=-1)
    mean_loss =mean_loss+ (2. * intersection + smooth) / (union + smooth)
    print('mean_loss',mean_loss)
    return torch.mean(mean_loss, axis=0)
def Dice_loss(target, output):
    score=dice_coef(output,target, smooth=1)
    return 1 - score



def categorical_dice(Y_pred, Y_gt, weight_loss):
    """
    multi label dice loss with weighted
    Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_pred is softmax result
    Y_gt:[None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_gt is one hot result
    weight_loss: numpy array of shape (C,) where C is the number of classes,eg:[0,1,1,1,1,1]
    :return:
    """
    # print('Y_pred.shape',Y_pred.shape)
    # print('Y_gt.shape',Y_gt.shape)
    weight_loss = np.array(weight_loss)
    smooth = 1.e-5
    smooth_tf = torch.constant(smooth, torch.float32)
    Y_pred = torch.cast(Y_pred, torch.float32)
    Y_gt = torch.cast(Y_gt, torch.float32)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    # print('intersection shape',numerator.shape) intersection shape (?, 64, 96, 96, 6)
    numerator = torch.reduce_sum(numerator, axis=(1, 2, 3))
    # print('after reduce_sum intersection shape', numerator.shape) after reduce_sum intersection shape (?, 6)
    denominator = Y_gt + Y_pred
    denominator = torch.reduce_sum(denominator, axis=(1, 2, 3))
    gen_dice_coef = torch.reduce_mean(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    # print('gen_dice_coef',gen_dice_coef.shape) gen_dice_coef (6,)
    loss = -torch.reduce_mean(weight_loss * gen_dice_coef)
    return loss

'''

class SoftDiceLoss(_Loss):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def diceCoeffv2(slef,pred, gt, eps=1e-5, activation='sigmoid'):
        r""" computational formula：
            dice = (2 * tp) / (2 * tp + fp + fn)
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

        tp = torch.sum(gt_flat * pred_flat, dim=1)
        fp = torch.sum(pred_flat, dim=1) - tp
        fn = torch.sum(gt_flat, dim=1) - tp
        loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        return loss.sum() / N

    def diceCoeff(self,pred, gt, eps=1e-5, activation='sigmoid'):
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
    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(1, self.num_classes):
            class_dice.append(self.diceCoeffv2(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

