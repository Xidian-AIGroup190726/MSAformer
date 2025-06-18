import torch.nn as nn
from SDFD import SDFD
from MulIHS import mulIHS
from tokenizer import Tokenizer
import torch
import numpy as np
from DF import SqueezeBodyEdge
from Restnet import ResNet18
opt = {
    'SDFD_options': {
        'embed_dim': 48,
        'embed_dim_paths':[48, 72, 84],
        'img_size_paths':[32, 64, 128],
        'num_heads': 6,
        'expansion_factor': 2,
        'split_size': [16, 32],
        'proj_drop_rate' : 0.12,
        'attn_drop_rate' : 0.12,
        'drop_paths' : [0.16, 0.14, 0.12, 0.08, 0.04, 0.0]
        }
    }
class Total_model(nn.Module):
    def __init__(self, 
                 opt,
                 x_input_channels=4,
                 y_input_channels=1,
                 kernel_size=3,
                 img_size_paths=opt['SDFD_options']['img_size_paths'],
                 positional_embedding=None,
                 embed_dim = opt['SDFD_options']['embed_dim'],
                 embed_dim_paths = opt['SDFD_options']['embed_dim_paths'],
                *args, **kwargs):
        super(Total_model, self).__init__()
        self.tokenizer1 = nn.ModuleList([
                                Tokenizer(n_input_channels=x_input_channels,
                                   n_output_channels=embed_dim,
                                   kernel_size=kernel_size,
                                   stride= max(1, (kernel_size // 2) - 1),
                                   padding=max(1, (kernel_size // 2)),
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=2,
                                   conv_bias=False)for i in range(3)])
        self.tokenizer2=nn.ModuleList([
                                Tokenizer(n_input_channels=y_input_channels,
                                   n_output_channels=embed_dim,
                                   kernel_size=kernel_size,
                                   stride= max(1, (kernel_size // 2) - 1),
                                   padding=max(1, (kernel_size // 2)),
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=2,
                                   conv_bias=False)for i in range(3)])
        self.SDFD = nn.ModuleList([
            SDFD(
            embed_dim=opt['SDFD_options']['embed_dim_paths'][i],
            num_heads=opt['SDFD_options']['num_heads'],
            expansion_factor=opt['SDFD_options']['expansion_factor'],
            split_size=opt['SDFD_options']['split_size'],
            proj_drop_rate=opt['SDFD_options']['proj_drop_rate'],
            attn_drop_rate=opt['SDFD_options']['attn_drop_rate'],
            drop_paths=opt['SDFD_options']['drop_paths'][i],

            sequence_length1=self.tokenizer1[i].sequence_length(n_channels=x_input_channels,
                                                           height=img_size_paths[i],
                                                           width=img_size_paths[i]),
            sequence_length2=self.tokenizer2[i].sequence_length(n_channels=y_input_channels,
                                                           height=img_size_paths[i],
                                                           width=img_size_paths[i]),
            seq_pool=True,
            positional_embedding=positional_embedding
        )for i in range(3)])  
        self.conv_init_edge40 = nn.Sequential(nn.Conv2d(4, embed_dim//2, 3, 1, 1),nn.BatchNorm2d(embed_dim//2),nn.LeakyReLU())
        #self.conv_init_edge30 = nn.Sequential(nn.Conv2d(3, embed_dim//4, 3, 1, 1),nn.BatchNorm2d(embed_dim//4),nn.LeakyReLU())
        self.conv_init_edge10 = nn.Sequential(nn.Conv2d(1, embed_dim//2, 3, 1, 1),nn.BatchNorm2d(embed_dim//2),nn.LeakyReLU())
        self.conv_x3 = nn.Sequential(nn.Conv2d(embed_dim_paths[2], embed_dim_paths[2], 3, 1, 1),nn.BatchNorm2d(embed_dim_paths[2]),nn.LeakyReLU())
        #torch.Size([10, 24, 16, 16]) torch.Size([10, 36, 32, 32]) torch.Size([10, 42, 64, 64])
        self.down_dim2 = nn.Sequential(
        nn.Conv2d(embed_dim_paths[0], embed_dim_paths[1], 1, 1, 0),
        nn.BatchNorm2d(embed_dim_paths[1]),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(embed_dim_paths[1], embed_dim_paths[1], 1, 1, 0),
        nn.BatchNorm2d(embed_dim_paths[1]),
        nn.Dropout(p=opt['SDFD_options']['drop_paths'][2])
        )
        self.down_dim4 = nn.Sequential(
        nn.Conv2d(embed_dim_paths[1], embed_dim_paths[2], 1, 1, 0),
        nn.BatchNorm2d(embed_dim_paths[2]),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(embed_dim_paths[2], embed_dim_paths[2], 1, 1, 0),
        nn.BatchNorm2d(embed_dim_paths[2]),
        nn.Dropout(p=opt['SDFD_options']['drop_paths'][2])
        )        
        self.edge_sv1 = SqueezeBodyEdge(embed_dim_paths[1])
        self.edge_sv2 = SqueezeBodyEdge(embed_dim_paths[2])
        self.fc = ResNet18(embed_dim + embed_dim_paths[2])
    def forward(self, msf, pan):  
        msf = msf.float()
        pan = pan.float()

        alpha1 = np.array([0., 0.13, 0.44, 0.54]).astype(np.float32)
        alpha2 = np.array([0.1, 0.24, 0.14, 0.52]).astype(np.float32)
        alpha4 = np.array([0.18, 0.27, 0.11, 0.59]).astype(np.float32)
        MT_1, PT_1 = mulIHS(msf, pan, alpha1, scale_factor = 1)
        MT_2, PT_2 = mulIHS(msf, pan, alpha2, scale_factor = 2)
        MT_4, PT_4 = mulIHS(msf, pan, alpha4, scale_factor = 4)

        #upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        #msm = upsample(msm)
        MTcat0 = self.conv_init_edge40(MT_4)
        #MTcat = self.conv_init_edge30(msm)
        PTcat = self.conv_init_edge10(PT_4)
        MT_1 = self.tokenizer1[0](MT_1)
        PT_1 = self.tokenizer2[0](PT_1)
        MT_2 = self.tokenizer1[1](MT_2)
        PT_2 = self.tokenizer2[1](PT_2)
        MT_4 = self.tokenizer1[2](MT_4)
        PT_4 = self.tokenizer2[2](PT_4)
#从这开始改！

        x3, MT_1up, PT_1up = self.SDFD[0](MT_1, PT_1, last1 = None, last3 = None)
        #print(MT_1up.shape, MT_2.shape)#torch.Size([10, 1024, 12]) torch.Size([10, 1024, 24])
        x6, MT_2up, PT_2up = self.SDFD[1](MT_2, PT_2, last1 = MT_1up, last3 = PT_1up)
        x9, _, _ = self.SDFD[2](MT_4, PT_4, last1 = MT_2up, last3 = PT_2up)
        #print(x3.shape, x6.shape, x9.shape)#torch.Size([10, 24, 16, 16]) torch.Size([10, 36, 32, 32]) torch.Size([10, 42, 64, 64])
        x3 = self.down_dim2(x3)
        x3 = self.edge_sv1(x3,x6)
        x3 = self.down_dim4(x3)
        x3 = self.edge_sv2(x3,x9)
        x3 = self.conv_x3(x3)
        x3 = torch.concat([x3, PTcat, MTcat0], dim=1).float()
        cls_prd = self.fc(x3)
        return cls_prd
    def cuda(self):
        self.tokenizer1.cuda()
        self.tokenizer2.cuda()
        self.conv_init_edge40.cuda()
        #self.conv_init_edge30.cuda()
        self.conv_init_edge10.cuda()
        self.conv_x3.cuda()
        self.SDFD.cuda()
        self.down_dim2.cuda()
        self.down_dim4.cuda()
        self.edge_sv1.cuda()
        self.edge_sv2.cuda()
        self.fc.cuda()
        return self


if __name__ == '__main__':
    model = Total_model(opt)
    model = model.cuda()
    msf = torch.randn((4, 4, 32, 32)).cuda()
    pan = torch.randn((4, 1, 128, 128)).cuda()
    Output = model(msf, pan)
    print(Output.shape)