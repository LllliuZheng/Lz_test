import torch
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
from models import model_dict
# B, C, H, W
a = torch.arange(2 * 2).view(1,2,2,1)
# print(a)
# print(a.shape)

# b = rearrange(a, 'b c h w -> b c (h w)')
# print(b)
# print(b.shape)
# path="F:\\Desktop\\头像\\AE28B28F22E07D383F1C18B14122A58F.jpg"
# img=plt.imread(path)
# img=torch.tensor(img)
# print(torch.tensor([0,0]))
# plt.imshow(img)
# plt.show()
a=torch.rand([2,2,3])
x=a.view(a.size(0),-1)
print(a.size(0))
print(a)
print(x)
print(x.size())