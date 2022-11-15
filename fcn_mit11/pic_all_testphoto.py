import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torchvision
from PIL\
    import Image
import archs
import archs_newunet
import archs_unet as archs1
import metrics
import SimpleITK as sitk
def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x
def trans_value(image):
    for i in range(880):
        for j in range(880):
            if image[i][j] == 0:
                image[i][j]=0
            elif image[i][j]<10:
                image[i][j] = 1
            elif image[i][j]>=10 :
                image[i][j]=2
    return image
def mask_to_onehot(mask, palette):
    semantic_map = []
    for colour in range(len(palette)):
        equality = np.equal(mask, palette[colour])
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map
os.environ['KMP_DUPLICATE_LIB_OK']='True'
data_path = r'data_mit\dataset\train\image'
data_path_ = r'data_mit\dataset\train\groundtruth'
dataname_list = os.listdir(data_path)
dataname_list.sort()
img = sitk.ReadImage(os.path.join(data_path, dataname_list[8]))
mask= sitk.ReadImage(os.path.join(data_path_,'mask_'+dataname_list[8]))
img = sitk.GetArrayFromImage(img)
mask=sitk.GetArrayFromImage(mask)
img=img[6, :, :]
mask=mask[6,:,:]
img1=cv2.resize(img,(880,880),interpolation=cv2.INTER_NEAREST)
mask=cv2.resize(mask,(880,880),interpolation=cv2.INTER_NEAREST)
mask=trans_value(mask)
img=img1.reshape(1,880,880)
#mask = mask.reshape(880, 880,1)
palette = [[0], [1], [2]]
#mask=mask_to_onehot(mask,palette)
img = img.astype('float32') / 2500
img=img.astype(np.float32)
#mask=mask.astype(np.float32)
img=img.reshape(1,1,880,880)
img=torch.from_numpy(img)
img=img.cuda()
print("=> creating model %s" % 'FCNs')
vgg_model = archs.VGGNet(requires_grad=True, show_params=False)
model1 = archs.FCNs(pretrained_net=vgg_model, n_class=3)
model1 = model1.cuda()
checkpoint=torch.load('models/FCN/model.pth')
model1.load_state_dict(checkpoint)
model1.eval()
mask1 = mask.reshape(880, 880,1)
palette = [[0], [1], [2]]
mask1=mask_to_onehot(mask1,palette)
mask1=mask1.astype(np.float32)
mask1=torch.from_numpy(mask1)
mask1 = torch.transpose(mask1, 1, 2)
mask1 = torch.transpose(mask1, 0, 1)
mask1=mask1.unsqueeze(0)
mask1=mask1.cuda()
output = model1(img)
output = torch.sigmoid(output)
output[output < 0.5] = 0
output[output > 0.5] = 1
print(output.shape,mask.shape)
bladder_dice = metrics.diceCoeff(output[:, 1, :, :], mask1[:, 1, :, :], activation=None).cpu().item()
tumor_dice = metrics.diceCoeff(output[:, 2, :, :], mask1[:, 2, :, :], activation=None).cpu().item()
mean_dice = (bladder_dice + tumor_dice) / 2
print('mean_dice',mean_dice)
output=output.reshape(3,880,880)
output = torch.transpose(output, 0, 1)
output = torch.transpose(output, 1, 2)
output = output.cpu().detach().numpy()
output=onehot_to_mask(output,palette)
output=output.reshape(880,880)
print("=> creating model %s" % 'UNET++')
model2 = archs1.__dict__['NestedUNet'](3,1,deep_supervision=False)
model2 = model2.cuda()
checkpoint=torch.load('models/UNET++/model.pth')
model2.load_state_dict(checkpoint)
model2.eval()
output1 = model2(img)
output1 = torch.sigmoid(output1)
output1[output1 < 0.5] = 0
output1[output1 > 0.5] = 1
print(output1.shape,mask.shape)
bladder_dice = metrics.diceCoeff(output1[:, 1, :, :], mask1[:, 1, :, :], activation=None).cpu().item()
tumor_dice = metrics.diceCoeff(output1[:, 2, :, :], mask1[:, 2, :, :], activation=None).cpu().item()
mean_dice2 = (bladder_dice + tumor_dice) / 2
print('mean_dice',mean_dice2)
output1=output1.reshape(3,880,880)
output1 = torch.transpose(output1, 0, 1)
output1 = torch.transpose(output1, 1, 2)
output1 = output1.cpu().detach().numpy()
output1=onehot_to_mask(output1,palette)
output1=output.reshape(880,880)



print("=> creating model %s" % 'UNET')
model3 = archs1.__dict__['UNet'](3,1,deep_supervision=False)
model3 = model3.cuda()
checkpoint=torch.load('models/UNET/model.pth')
model3.load_state_dict(checkpoint)
model3.eval()
output2 = model3(img)
output2 = torch.sigmoid(output2)
output2[output2 < 0.5] = 0
output2[output2 > 0.5] = 1
print(output2.shape,mask.shape)
bladder_dice = metrics.diceCoeff(output2[:, 1, :, :], mask1[:, 1, :, :], activation=None).cpu().item()
tumor_dice = metrics.diceCoeff(output2[:, 2, :, :], mask1[:, 2, :, :], activation=None).cpu().item()
mean_dice3 = (bladder_dice + tumor_dice) / 2
print('mean_dice',mean_dice3)
output2=output2.reshape(3,880,880)
output2 = torch.transpose(output2, 0, 1)
output2 = torch.transpose(output2, 1, 2)
output2 = output2.cpu().detach().numpy()
output2=onehot_to_mask(output2,palette)
output2=output2.reshape(880,880)

print("=> creating model %s" % 'UNET_new')
model4 = archs_newunet.U_Net(3,1).cuda()
model4 = model4.cuda()
checkpoint=torch.load('models/UNET_new/model.pth')
model4.load_state_dict(checkpoint)
model4.eval()
o1 = torch.ones(1, 880, 880)
o2 = torch.zeros(2, 880, 880)
output3 = torch.cat((o1, o2), 0)
output3 = output3.unsqueeze(0)
for i in [432,0]:
    sample = img[:, :, i:i + 448, 216:664]
    output3[:, :, i:i + 448, 216:664] = model4(sample)
output3=output3.cuda()
print(output3.shape,mask.shape)
output3 = torch.sigmoid(output3)
output3[output3 <= 0.5] = 0
output3[output3 > 0.5] = 1
bladder_dice = metrics.diceCoeff(output3[:, 1, :, :], mask1[:, 1, :, :], activation=None).cpu().item()
tumor_dice = metrics.diceCoeff(output3[:, 2, :, :], mask1[:, 2, :, :], activation=None).cpu().item()
mean_dice4 = (bladder_dice + tumor_dice) / 2
print('mean_dice',mean_dice4)
output3=output3.reshape(3,880,880)
output3 = torch.transpose(output3, 0, 1)
output3 = torch.transpose(output3, 1, 2)
output3 = output3.cpu().detach().numpy()
output3=onehot_to_mask(output3,palette)
output3=output3.reshape(880,880)



fig=plt.figure()
ax1=fig.add_subplot(161)
ax1.imshow(img1)
plt.xticks([]),plt.yticks([])
#ax1.set_title("Image")
ax2=fig.add_subplot(162)
ax2.imshow(mask)
plt.xticks([]),plt.yticks([])
#ax2.set_title("groundtruth")
ax3=fig.add_subplot(163)
ax3.imshow(output)
#ax3.set_title("dice:%s"%round(mean_dice,3))
plt.xticks([]),plt.yticks([])
ax4=fig.add_subplot(164)
ax4.imshow(output1)
#ax4.set_title("dice:%s"%round(mean_dice2,3))
plt.xticks([]),plt.yticks([])
ax5=fig.add_subplot(165)
ax5.imshow(output2)
#ax5.set_title("dice:%s"%round(mean_dice3,3))
plt.xticks([]),plt.yticks([])
ax6=fig.add_subplot(166)
ax6.imshow(output3)
#ax6.set_title("dice:%s"%round(mean_dice4,3))
plt.xticks([]),plt.yticks([])
plt.savefig('img/valitation3.png')
plt.show()



