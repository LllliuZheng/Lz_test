import torch
from torch.nn import MSELoss
import math
from einops import rearrange

def cut_featuremap(feat):
    bs, num, c, h, w = feat.size()
    new_h = h // 2
    new_w = w // 2
    # [bs,c,w,h] => [bs,c*4,w//2,h//2]
    feature_map = torch.cat((
                            feat[:, :, :, :new_h, :new_w], feat[:, :, :, :new_h, new_w:], feat[:, :, :, new_h:, :new_w],
                            feat[:, :, :, new_h:, new_w:]), dim=1)


    feature_map = feature_map.reshape(bs, 4 * num, c, new_h, new_w)
    return feature_map


def MYAPloss(feat_t,feat_s,loss_fn,T1_percent,T2_percent=0.5,T3_percent=0.5):
    loss = 0.0
    mse_matrix = loss_fn(feat_s,feat_t.detach())
    bs, c, h, w = feat_t.size()
    print("h:%d"%h)
    for i in range(int(math.log2(h))):
        # change the feature shape, however, this is not divide equally into four parts.
        feat_t = rearrange(feat_t.detach(),'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=2, w1=2)
        feat_s = rearrange(feat_s,'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=2, w1=2)
        bs_f, c_f, h_f,w_f = feat_t.size()
        if i == int(math.log2(h))-1:
            mse_matrix = loss_fn(feat_s,feat_t.detach()).sum()#增加.sum()
            print(mse_matrix.shape)
            loss += mse_matrix

        else:
            if i == 0:
                tem_num = int(T1_percent * mse_matrix.size(1))
            if i == 1:
                tem_num = int(T2_percent * mse_matrix.size(1))
            mse_matrix = loss_fn(feat_s,feat_t.detach())
            _, index = mse_matrix.mean(dim=(2,3)).topk(tem_num,-1)
            print(index.shape,mse_matrix.shape)
            tem_mask = torch.zeros(bs_f, c_f)
            tem_mask[torch.arange(bs_f)[:,None], index] = 1
            tem_mask = tem_mask.reshape(bs_f, c_f,1,1)
            # tem_mask = tem_mask.reshape(bs_f, c_f, h_f,w_f)#.cuda()
            require_mask = 1-tem_mask
            loss += (mse_matrix * require_mask).sum()
            feat_t = feat_t * tem_mask
            feat_s = feat_s * tem_mask
            print('yes')
        

    return loss




def aploss(feat_t, feat_s, loss_fn, T):
    loss = 0.0
    bs, c, h, w = feat_t.size()
    feat_t = feat_t.reshape(bs, 1, c, h, w)
    feat_s = feat_s.reshape(bs, 1, c, h, w)
    cut_feat_t=feat_t
    cut_feat_s=feat_s
    print()
    times = int(math.log2(h))
    print(times)
    for i in range(times):
        cut_feat_t = cut_featuremap(cut_feat_t)
        cut_feat_s = cut_featuremap(cut_feat_s)
        mse_matrix = loss_fn(cut_feat_t, cut_feat_s)
        mse_mask = mse_matrix.mean(dim=(2, 3, 4)) > T
        # mse_matrix = mse_matrix * mse_mask.reshape(bs,4,1,1,1)
        print(mse_matrix.mean(dim=(2, 3, 4)).size())
        # 小于T的计入损失
        loss += (mse_matrix * mse_mask.reshape(bs, 4 ** (i + 1), 1, 1, 1).logical_not()).sum()
        # print(loss)
        # feat_t = cut_feat_t * mse_mask.reshape(bs, 4 ** (i + 1), 1, 1, 1)
        # feat_s = cut_feat_s * mse_mask.reshape(bs, 4 ** (i + 1), 1, 1, 1)
        # print(feat_s.size())
    return loss
    # print(mse_mask[0])

    # mse_matrix = loss_fn(cut_feat_t,cut_feat_s).mean(dim=(2,3,4))
    # print(mse_matrix.size())






if __name__ == "__main__":
    feat_t = torch.randn(4, 6, 8, 8)
    feat_s = torch.randn(4, 6, 8, 8)
    Threshold = 0.5
    loss_fn = torch.nn.MSELoss(reduction='none')
    print(aploss(feat_t, feat_s, loss_fn, Threshold))
    # MYAPloss(feat_t, feat_s, loss_fn, Threshold)