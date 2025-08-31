import cv2
import numpy as np
from scipy.io import savemat
from scipy.optimize import minimize, LinearConstraint, Bounds
from tifffile import imread
import torch
from torch.nn import functional as F
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

'''def split(pan, size):
    st = []
    for i in range(size):
        for j in range(size):
            st.append(pan[i::size,j::size])
    return np.stack(st, axis=-1)'''

#当前文件夹指资源管理器打开的文件夹，不是当前程序所在文件夹，或者可以直接查看终端PS D:\MyCo\test\IHS\TSMF-Net-main>
#当前文件夹. 上一级..
ms4 = imread('./dataset/ms4.tif').astype("float32")
#msf = cv2.resize(ms4, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
msf = ms4
print(msf.shape, msf.dtype, np.min(msf), np.max(msf))
msf = to_tensor( msf )  
#msf = msf.reshape((-1, 4))
print(msf.shape, msf.dtype, np.min(msf), np.max(msf))
#savemat('./dataset/msf.mat', {'msf': msf})


pan = imread('./dataset/pan.tif').astype("float32")
#pan = split(pan, 2)
pan = torch.from_numpy(pan).unsqueeze(0).unsqueeze(0)
pan = F.max_pool2d(pan, kernel_size=4, stride=4)
pan = pan.squeeze().numpy()
print(pan.shape, pan.dtype, np.min(pan), np.max(pan))

pan = to_tensor( pan )   
#pan = pan.reshape((-1, 1))
print(pan.shape, pan.dtype, np.min(pan), np.max(pan))
#savemat('./dataset/pan.mat', {'pan': pan})
#cv2.imshow('Max Pooled Image', (pan* 255).astype(np.uint8))

#pan_uint8 = (pan * 255).astype(np.uint8)
#filename = 'example_gray.jpg'
#saved = cv2.imwrite(filename, pan_uint8)

msf = np.transpose(msf, (2, 0, 1))
def objective(alpha, msf, pan):
    
    alpha = alpha.reshape(-1, 1, 1)
    ms_jiaquan = alpha * msf
    I_m = np.sum(ms_jiaquan, axis=0, keepdims=False)#通过设置 keepdims=True，保持了结果的维度为 (3200, 3320, 1)，而不会直接变成 (3200, 3320)。
    PT = pan - I_m
   # MT = (1 - alpha) * msf
    target = np.linalg.norm(PT)**2
    return target

# 定义初始解
#alpha_init = np.array([0.25, 0.25, 0.25, 0.25])
#alpha_init = np.array([0.37, 0.10, 0.10, 0.43])
#alpha_init = np.array([0.1, 0.24, 0.14, 0.52]), *2的
alpha_init = np.array([0.18, 0.27, 0.11, 0.59])#*4的

#constraint = LinearConstraint([[1, 1, 1, 1]], [1], [1])#系数，下界，上界
bounds = Bounds([0, 0, 0, 0], [1, 1, 1, 1])
# 进行优化
result = minimize(objective, alpha_init, args=(msf, pan), bounds=bounds)

# 输出结果
print("Optimal alpha:", result.x)
print("Optimal value:", result.fun)

"""
(2001, 2101, 4) float32 108.0 999.0
(2001, 2101, 4) float32 0.0 1.0
(2001, 2101) float32 176.0 1023.0
(2001, 2101) float32 0.0 1.0
Optimal alpha: [0.         0.13025627 0.4394764  0.54001404]
Optimal value: 17224.185912316814

(2001, 2101, 4) float32 108.0 999.0
(2001, 2101, 4) float32 0.0 1.0
(2001, 2101) float32 176.0 1023.0
(2001, 2101) float32 0.0 1.0
Optimal alpha: [0.07957024 0.26066495 0.18944403 0.53352954]
Optimal value: 18607.414014873684

alpha_init = np.array([0.18, 0.27, 0.11, 0.59])
(2001, 2101, 4) float32 108.0 999.0
(2001, 2101, 4) float32 0.0 1.0
(2001, 2101) float32 176.0 1023.0
(2001, 2101) float32 0.0 1.0
Optimal alpha: [0.         0.13025639 0.43947589 0.54001455]
Optimal value: 17224.185912325443
"""


