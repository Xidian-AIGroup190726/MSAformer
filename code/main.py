import torch
import os
import torch.optim as optim
from Dataprocess import train_loader
from Dataprocess import test_loader
from model import Total_model

from utils import setup_seed, train_model, test_model
import logging
#-------------------定义超参数----------------------------------------------
EPOCH = 30  # 训练多少轮次
BATCH_SIZE = 42# 每次喂给的数据量实际应该在数据加载里面改
LR = 0.0001 # 学习率
seed_num = 3407
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 是否用GPU环视cpu训练
torch.cuda.set_per_process_memory_fraction(0.8)
opt = {
    'SDFD_options': {
        'embed_dim': 48,
        'embed_dim_paths':[48, 72, 84],
        'img_size_paths':[32, 64, 128],
        'num_heads': 6,
        'expansion_factor': 2,
        'split_size': [8, 8],
        'proj_drop_rate' : 0.12,
        'attn_drop_rate' : 0.12,
        'drop_paths' : [0.16, 0.14, 0.12, 0.08, 0.04, 0.0]
        }
    }


def main():
    #设置随机种子
    setup_seed(seed_num)

    if_cuda = torch.cuda.is_available()
    print("if_cuda=",if_cuda)
    gpu_count = torch.cuda.device_count()
    print("gpu_count=",gpu_count)

    #定义优化器
    model = Total_model(opt)
    model = model.cuda()


    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    logging.basicConfig(level=logging.INFO, filename='training.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')
    # 调用训练和测试
    for epoch in range(1, EPOCH+1):
        train_model(model, train_loader, optimizer, epoch)
        if (epoch > 19 and epoch % 5 == 0) or epoch  == 12 or epoch == 18:
            torch.save(model.state_dict(), f'./model_epoch_{epoch}.pt')
            test_model(model, test_loader)

if __name__ == '__main__':
    main()


