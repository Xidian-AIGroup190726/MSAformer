import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tifffile import imread
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional as TF
TRAIN_BATCH_SIZE = 42   # 每次喂给的数据量
TEST_BATCH_SIZE = 42
ALL_BATCH_SIZE = 42
ms4_np = imread('./dataset/ms4.tif').astype("float32")
pan_np =  imread('./dataset/pan.tif').astype("float32")
train_label_np = np.load("./dataset/train.npy")
test_label_np = np.load("./dataset/test.npy")
# ms4与pan图补零  (给图片加边框）
Ms4_patch_size = 32  # ms4截块的边长
Interpolation = cv2.BORDER_REFLECT_101
top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)

train_label_np = train_label_np - 1
test_label_np = test_label_np - 1
label_element, element_count = np.unique(train_label_np, return_counts=True)
Categories_Number = len(label_element) - 1  # 数据的类别数
label_row, label_column = np.shape(train_label_np)  # 获取标签图的行、列

def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)  # [800*830, 2] 二维数组
ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

count = 0
for row in range(label_row): 
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if int(train_label_np[row][column]) != 255:
            ground_xy_train.append([row, column])
            label_train.append(train_label_np[row][column])
        if int(test_label_np[row][column]) != 255:
            ground_xy_test.append([row, column])
            label_test.append(test_label_np[row][column])

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)

# 训练数据与测试数据，数据集内打乱
shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]


label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

r = ms4_np[:, :,0]
g = ms4_np[:, :,1]
b = ms4_np[:, :,2]
nr = ms4_np[:, :,3]
r = to_tensor(r)
g = to_tensor(g)  
b = to_tensor(b)  
nr = to_tensor(nr)  
ms4 = np.stack([r, g, b, nr], axis=2)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)

class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)      # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]
        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)

class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]
        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)
train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=16, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=16, drop_last=True)
all_data_loader = DataLoader(dataset=all_data, batch_size=ALL_BATCH_SIZE, shuffle=False, num_workers=4)
