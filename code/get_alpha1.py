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
ms4 = imread('./dataset/ms4.tif').astype("float32")
msf = ms4
print(msf.shape, msf.dtype, np.min(msf), np.max(msf))
msf = to_tensor( msf )  
print(msf.shape, msf.dtype, np.min(msf), np.max(msf))

pan = imread('./dataset/pan.tif').astype("float32")
pan = torch.from_numpy(pan).unsqueeze(0).unsqueeze(0)
pan = F.max_pool2d(pan, kernel_size=4, stride=4)
pan = pan.squeeze().numpy()
print(pan.shape, pan.dtype, np.min(pan), np.max(pan))

pan = to_tensor( pan )   
print(pan.shape, pan.dtype, np.min(pan), np.max(pan))
msf = np.transpose(msf, (2, 0, 1))
def objective(alpha, msf, pan):
    
    alpha = alpha.reshape(-1, 1, 1)
    ms_jiaquan = alpha * msf
    I_m = np.sum(ms_jiaquan, axis=0, keepdims=False)
    PT = pan - I_m
    target = np.linalg.norm(PT)**2
    return target
alpha_init = np.array([0.18, 0.27, 0.11, 0.59])#*4的
bounds = Bounds([0, 0, 0, 0], [1, 1, 1, 1])
result = minimize(objective, alpha_init, args=(msf, pan), bounds=bounds)

print("Optimal alpha:", result.x)
print("Optimal value:", result.fun)



