import torch, torch.nn.functional as F
from typing import Tuple
def edge_fast(x):  # 复用你前面的 edge_detect_torch 配置
    gx = x[:,:,1:,:] - x[:,:,:-1,:]
    gy = x[:,:,:,1:] - x[:,:,:,:-1]
    gx = F.pad(gx, (0,0,0,1))
    gy = F.pad(gy, (0,1,0,0))
    grad = gx.abs() + gy.abs()
    return torch.exp(-1e-9/(grad.pow(4)+1e-10))

def diff3(ms4, pan, alpha, scale_factor):
    if scale_factor == 1:
        ms_r  = ms4
        pan_r = F.interpolate(pan, scale_factor=0.25, mode='bilinear', align_corners=True)
    elif scale_factor == 2:
        ms_r  = F.interpolate(ms4, scale_factor=2.0, mode='bilinear', align_corners=True)
        pan_r = F.interpolate(pan,  scale_factor=0.5, mode='bilinear', align_corners=True)
    elif scale_factor == 4:
        ms_r  = F.interpolate(ms4, scale_factor=4.0, mode='bilinear', align_corners=True)
        pan_r = pan
    alpha = alpha.view(1,-1,1,1).to(ms4)
    I = (ms_r*alpha).sum(1, keepdim=True)
    MT = ms_r + pan_r - I
    W_mi = edge_fast(ms_r)
    W_pi = edge_fast(pan_r)
    m_sum = ms_r.mean(1, keepdim=True).clamp_min(1e-12)
    W_m = ms_r/m_sum * (0.5*W_mi + 0.5*W_pi)
    MT_1 = MT*W_m
    return MT_1, pan_r

