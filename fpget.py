import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
 

feature=[]
def viz(module, input, output):
    feature.append(output.clone().detach())

def feature_conv_hook(submoudle): 
    if isinstance(submoudle,torch.nn.Conv2d):
        submoudle.register_forward_hook(viz)

def feature_index_get(model,block_name):
    index=-1
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    for name , m in model.named_modules():
        if block_name in name:
            index=int(block_name[-1])
        feature_conv_hook(m)
    return index           

def featuremap_get(index):
    featuremap=feature[index]
    return featuremap
    





##########################test#######################################
 
# if __name__ == '__main__':

#     transform = transforms.Compose([transforms.ToPILImage(),
#                             transforms.Resize((224, 224)),
#                             transforms.ToTensor(),
#                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                  std=[0.229, 0.224, 0.225])
#                             ])
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model_dict['vgg8'](num_classes=2).to(device)
#     img=cv2.imread('F:\\Desktop\\LZ\\AE28B28F22E07D383F1C18B14122A58F.jpg')
#     img=np.array(img)
#     img=transform(img)
#     print(img.shape)
#     img=img.unsqueeze(0).to(device)
#     index=feature_index_get(model,"block2")
#     with torch.no_grad():
#         model(img)
#     featuremap=featuremap_get(index)
#     print(featuremap.shape)
#     show_num=np.minimum(8,featuremap.size()[1])
#     for i in range(2*show_num):
#         plt.subplot(2, 8, i+1)
#         plt.imshow(featuremap[0][i].cpu().numpy())
#     plt.show()
#     print(index)