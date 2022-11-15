# unet网络训练代码
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import archs
import losses
from dataset_2 import Dataset
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from metrics import iou_score, diceCoeff
from utils import AverageMeter, str2bool

from collections import OrderedDict
from glob import glob

ARCH_NAMES = archs.__all__


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='FCN')
    # ,help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=880, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=880, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='SoftDiceLoss',
                        help='loss: ')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'mean_dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
        output = torch.sigmoid(output)
        output[output < 0.5] = 0
        output[output > 0.5] = 1
        bladder_dice = diceCoeff(output[:, 1, :, :], target[:, 1, :, :], activation=None).cpu().item()
        tumor_dice = diceCoeff(output[:, 2, :, :], target[:, 2, :, :], activation=None).cpu().item()
        mean_dice = (bladder_dice + tumor_dice) / 2

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['mean_dice'].update(mean_dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('mean_dice', avg_meters['mean_dice'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('mean_dice', avg_meters['mean_dice'].avg)
                        ]
                       )


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'mean_dice': AverageMeter()
                  }

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                output = torch.sigmoid(output)
                output[output < 0.5] = 0
                output[output > 0.5] = 1
                bladder_dice = diceCoeff(output[:, 1, :, :], target[:, 1, :, :], activation=None).cpu().item()
                tumor_dice = diceCoeff(output[:, 2, :, :], target[:, 2, :, :], activation=None).cpu().item()
                mean_dice = (bladder_dice + tumor_dice) / 2

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['mean_dice'].update(mean_dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('mean_dice', avg_meters['mean_dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('mean_dice', avg_meters['mean_dice'].avg)
                        ])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % ('dsb2018_96', config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        criterion = losses.SoftDiceLoss(num_classes=3, activation='sigmoid').cuda()
    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    vgg_model = archs.VGGNet(requires_grad=True, show_params=False)
    model = archs.FCNs(pretrained_net=vgg_model, n_class=3)
    model = model.cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    data_path = r'data_mit\dataset\train\image'
    dataname_list = os.listdir(data_path)
    dataname_list.sort()
    train_img_ids, val_img_ids = train_test_split(dataname_list, test_size=0.2, random_state=41)
    train_transform = Compose([
        transforms.Flip(),])
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('data_mit', 'dataset', 'train', 'image'),
        mask_dir=os.path.join('data_mit', 'dataset', 'train', 'groundtruth'),
        transform=train_transform
    )
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('data_mit', 'dataset', 'train', 'image'),
        mask_dir=os.path.join('data_mit', 'dataset', 'train', 'groundtruth'),
        transform = None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('val_loss', []),
        ('bice', []),
        ('val_bice', []),

    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f  - val_loss %.4f  - bice %.4f -val_bice %.4f'
              % (train_log['loss'], val_log['loss'], train_log['mean_dice'], val_log['mean_dice']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['bice'].append(train_log['mean_dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_bice'].append(val_log['mean_dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)


        trigger += 1
        torch.cuda.empty_cache()
        if val_log['mean_dice'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model1.pth' %
                       config['name'])
            best_iou = val_log['mean_dice']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
