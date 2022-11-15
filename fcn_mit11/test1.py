import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import archs
import SimpleITK as sitk
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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
data_path = r'data_mit\dataset\train\image'
data_path_ = r'data_mit\dataset\train\groundtruth'
dataname_list = os.listdir(data_path)
dataname_list.sort()
img = sitk.ReadImage(os.path.join(data_path, dataname_list[4]))
mask= sitk.ReadImage(os.path.join(data_path_,'mask_'+dataname_list[4]))
img = sitk.GetArrayFromImage(img)
mask=sitk.GetArrayFromImage(mask)
def mask_to_onehot(mask, palette):
    semantic_map = []
    for colour in range(len(palette)):
        equality = np.equal(mask, palette[colour])
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map
def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x
img=img[1, :, :]
mask=mask[1,:,:]
img=cv2.resize(img,(880,880),interpolation=cv2.INTER_NEAREST)
mask=cv2.resize(mask,(880,880),interpolation=cv2.INTER_NEAREST)
mask=trans_value(mask)
mask = mask.reshape(880, 880,1)
palette = [[0], [1], [2]]
mask=mask_to_onehot(mask,palette)
img=img.reshape(1,880,880)
img = img.astype('float32') / 2500
img=img.astype(np.float32)
img=img.reshape(1,1,880,880)
img=torch.from_numpy(img)
img=img.cuda()
print("=> creating model %s" % 'NestedUNet')
model = archs.__dict__['NestedUNet'](3,1,deep_supervision=False)
model = model.cuda()
checkpoint=torch.load('models/dsb2018_96_NestedUNet_woDS/model.pth')
model.load_state_dict(checkpoint)
model.eval()
output = model(img)
output=output.reshape(3,880,880)
output = torch.sigmoid(output)
output[output < 0.5] = 0
output[output >= 0.5] = 1
output = output.cpu().detach().numpy()
output1=output[0]
output2=output[1]
output3=output[2]
plt.imshow(output3)
plt.show()

'''output = torch.sigmoid(output)
for i in range(2):
    for m in range(880):
        for n in range(880):
            if output[i][m][n]<=0.5:
                output[i][m][n]=0
            else:output[i][m][n]=1
output = torch.transpose(output, 0, 1)
output = torch.transpose(output, 1, 2)
output = output.cpu().detach().numpy()
output=onehot_to_mask(output,palette)
print(output.shape)
output=output.reshape(880,880)'''
