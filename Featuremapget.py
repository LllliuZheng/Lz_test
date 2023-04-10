import torch
import torchvision as tv 

import torchvision.transforms as transforms
import torch.nn as nn
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import os
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("F:\\Desktop\\PEBKD-master\\"))))
from models import model_dict
class FeatureExtractor(nn.Module):
    def __init__(self,model,layers):
        super(FeatureExtractor,self).__init__()
        self.model=model
        self.layers=layers
    
    def forward(self,x):
        outputs={}
        for name,module in self.model._modules.items():
            if "classifier" in name:
                x=x.view(x.size(0),-1)
                x=module(x)
            
            
            print(name)
            if self.layers is None or name in self.layers and "classifier" not in name:
                outputs[name]=x

        return outputs

def get_picture(pic_name,transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img,(256,256))
    img = np.asanyarray(img , dtype=np.float32)
    return transform(img)

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def get_feature(model):
    pic_dir="F:\\Desktop\\头像\\AE28B28F22E07D383F1C18B14122A58F.jpg"
    transforms=tv.transforms.ToTensor()
    img=get_picture(pic_dir,transforms)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img=img.unsqueeze(0)
    img=img.to(device)
    exact_list=["block1","block2"]
    dst="F:\\Desktop\\Saving"
    therd_size=256
    exactor=FeatureExtractor(model,exact_list)
    outs=exactor(x=img)
    outs(img)
    for k,v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            if 'fc' in k:
                continue
        
        feature=features.data.cpu().numpy()
        feature_img=feature[i,:,:]
        feature_img=np.asarray(feature_img*255,dtype=np.uint8)

        dst_path=os.path.join(dst,k)

        make_dirs(dst_path)
        feature_img=cv2.applyColorMap(feature_img,cv2.COLORMAP_JET)
        if feature_img.shape[0]<therd_size:
            tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
            tmp_img = feature_img.copy()
            tmp_img = cv2.resize(tmp_img, (therd_size,therd_size), interpolation =  cv2.INTER_NEAREST)
            cv2.imwrite(tmp_file, tmp_img)

        dst_file = os.path.join(dst_path, str(i) + '.png')
        cv2.imwrite(dst_file, feature_img)

if __name__ == '__main__':
    model_test = model_dict['vgg8'](num_classes=1)
    get_feature(model_test)