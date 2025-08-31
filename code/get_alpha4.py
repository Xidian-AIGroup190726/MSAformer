import cv2
import numpy as np
from scipy.io import savemat
from scipy.optimize import minimize, LinearConstraint, Bounds
from tifffile import imread
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


ms4 = imread('./dataset/ms4.tif').astype("float32")
msf = cv2.resize(ms4, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
print(msf.shape, msf.dtype, np.min(msf), np.max(msf))
msf = to_tensor( msf )  

print(msf.shape, msf.dtype, np.min(msf), np.max(msf))

pan = imread('./dataset/pan.tif').astype("float32")

print(pan.shape, pan.dtype, np.min(pan), np.max(pan))

print(pan.shape, pan.dtype, np.min(pan), np.max(pan))

msf = np.transpose(msf, (2, 0, 1))
def objective(alpha, msf, pan):
    
    alpha = alpha.reshape(-1, 1, 1)
    ms_jiaquan = alpha * msf
    I_m = np.sum(ms_jiaquan, axis=0, keepdims=False)
    PT = pan - I_m
    target = np.linalg.norm(PT)**2
    return target

alpha_init = np.array([0.25, 0.25, 0.25, 0.25])
bounds = Bounds([0, 0, 0, 0], [1, 1, 1, 1])
result = minimize(objective, alpha_init, args=(msf, pan), bounds=bounds)
print("Optimal alpha:", result.x)
print("Optimal value:", result.fun)


