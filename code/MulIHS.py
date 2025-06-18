import numpy as np
import cv2
import torch
import torch.nn as nn
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image
def image_gradient(img):
    H, W = img.shape
    gx = np.pad(np.diff(img, axis=0), ((0,1),(0,0)), 'constant')
    gy = np.pad(np.diff(img, axis=1), ((0,0),(0,1)), 'constant')
    gradient = abs(gx) + abs(gy)
    return gradient
def edge_dect(img):
    nam=1e-9
    apx=1e-10
    return np.exp( -nam / ( (image_gradient(img)**4)+apx ) )

def edge_dect_batch1(imgs):
    batch_size, channels, H, W = imgs.shape
    edges = np.zeros((batch_size, channels, H, W), dtype=np.float32)
    for i in range(batch_size):
        for j in range(channels):
            img = imgs[i,j]  # 取出每个样本的单通道图像
            edges[i,j] = edge_dect(img)
    return edges
def downsample(tensor, scale_factor=0.5):
    
    # 保存原始batch的大小
    batch_size, channels, height, width = tensor.shape
    
    # 创建一个用于存储输出的空张量
    output = torch.zeros((batch_size, channels, int(height * scale_factor), int(width * scale_factor)), device=tensor.device)
    
    # 对每个图像（每个batch）进行高斯模糊和降采样
    for i in range(batch_size):
        for j in range(channels):
            image = tensor[i, j].cpu().numpy() 
            downsampled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            output[i, j] = torch.from_numpy(downsampled_image).float().to(tensor.device)
    
    return output
def mulIHS(ms4, pan, alpha, scale_factor):
    if scale_factor == 1:
        pan = downsample(pan, scale_factor=0.25)
    if scale_factor == 2:
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ms4 = upsample(ms4)
        pan = downsample(pan, scale_factor=0.5)
    if scale_factor == 4:
        upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        ms4 = upsample(ms4)

    ms4 = ms4.detach().cpu().numpy()
    pan = pan.detach().cpu().numpy()

    batch_size, channels, H, W = ms4.shape
    alpha = alpha.reshape(1, -1, 1, 1)  # 形状变为 (1, 4, 1, 1)
    I = alpha * ms4
    I = np.sum(I, axis=1, keepdims=True)
    MT = ms4 + pan - I
    # Edge detection operator
    W_mi = edge_dect_batch1(ms4)#(B,4,H,W)
    W_pi = edge_dect_batch1(pan)#(B,1,H,W)
    m_sum = np.zeros_like(W_pi)
    m_sum = m_sum[0:1,:,:,:]

    gamma = 0.5  
    W_m = np.zeros_like(W_mi)
    for i in range(batch_size):
        m_sum = np.mean(ms4[i], axis=0, keepdims=True)
        m_sum = np.where(m_sum == 0, np.inf, m_sum)
        for j in range(channels):
            W_m[i,j] = ms4[i, j] / m_sum * ( gamma * W_mi[i, j] + (1-gamma) * W_pi[i,0])
    MT_1 = np.zeros_like(MT)
    for i in range(batch_size):
        for j in range(channels):
            MT_1[i, j] = MT[i, j] * W_m[i, j]
    MT_1 = torch.from_numpy(MT_1).cuda().float()
    pan = torch.from_numpy(pan).cuda().float()
    return MT_1, pan