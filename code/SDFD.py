import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
from timm.models.layers import DropPath
def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class ASC(nn.Module):
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.up_dim2 = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim*2),
            nn.GELU()
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1)
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )

    def forward(self, x, H, W, don = False):
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        conv_x = self.conv1(x)#conv_x(B, C, H, W)
        spatial_map = self.spatial_interaction(x).permute(0, 2, 3, 1).contiguous().view(B, L, 1)
        channel_map = self.channel_interaction(x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        conv_x = conv_x + conv_x * torch.sigmoid(spatial_map)
        conv_x = conv_x.transpose(-2,-1).contiguous().view(B, C, H, W)

        conv_x = self.conv2(conv_x)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        conv_x = conv_x + conv_x * torch.sigmoid(channel_map)
        if don == False:
            conv_x = self.proj(conv_x)
        else:
            conv_x = conv_x.transpose(-2,-1).contiguous().view(B, C, H, W)
            conv_x = self.up_dim2(conv_x)
            conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, int(L/4), int(C*2))
        conv_x = self.proj_drop(conv_x)
        return conv_x

class Inverted_Attention(nn.Module):
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx

        head_dim = dim // num_heads if dim >= num_heads else 1
        self.scale = head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        """
        Input: Image (B, C, H, W)
        Output: Window Partition (B', N, C)
        """
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv1,qkv2, H, W):
        q1, k1, v1, v2= qkv1[0], qkv1[1], qkv1[2], qkv1[3]
        q2, k2= qkv2[0], qkv2[1]
        B, L, C = q1.shape   
        assert L == H * W, "flatten img_tokens has wrong size"
        q1 = self.im2win(q1, H, W)
        k1 = self.im2win(k1, H, W)
        v1 = self.im2win(v1, H, W)
        q2 = self.im2win(q2, H, W)
        k2 = self.im2win(k2, H, W)
        v2 = self.im2win(v2, H, W)
        attn_co1 = (q2 @ k1.transpose(-2, -1))* self.scale
        attn_co2 = (k2 @ q1.transpose(-2, -1))* self.scale
        attn_sp1 = 2 - attn_co1 - attn_co2
        attn_sp1 = nn.functional.softmax(attn_sp1, dim=-1, dtype=attn_sp1.dtype)
        attn_sp1 = self.attn_drop(attn_sp1)        
        v = v1 + v2
        sp1 = (attn_sp1 @ v)
        sp1 = sp1.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)
        sp1 = windows2img(sp1, self.H_sp, self.W_sp, H, W)
        return sp1
class Adaptive_Inverted_Attention(nn.Module):
    def __init__(self, dim, num_heads, split_size=[8,16], proj_drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.qkv = nn.Linear(dim, dim *4)
        self.branch_num = 2

        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj3 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attns = nn.ModuleList([
                Inverted_Attention(
                    dim//2 if dim >= 4 else 1, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2 if dim >= 4 else 1,
                    attn_drop=attn_drop)
                for i in range(self.branch_num)])
             
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, kernel_size=1)
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )

    def forward(self, x, y, H, W):
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        qkv1 = self.qkv(x).reshape(B, -1, 4, C).permute(2, 0, 1, 3) # 3, B, HW, C
        qkv2 = self.qkv(y).reshape(B, -1, 4, C).permute(2, 0, 1, 3) # 3, B, HW, C###########################
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv1 = qkv1.reshape(4*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv1 = F.pad(qkv1, (pad_l, pad_r, pad_t, pad_b)).reshape(4, B, C, -1).transpose(-2, -1) # l r t b
        qkv2 = qkv2.reshape(4*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv2 = F.pad(qkv2, (pad_l, pad_r, pad_t, pad_b)).reshape(4, B, C, -1).transpose(-2, -1) # l r t b###############之前是reshape2？？？？
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W
        sp11 = self.attns[0](qkv1[:,:,:,:C//2], qkv2[:,:,:,:C//2], _H, _W)
        sp11 = sp11[:, :H, :W, :].reshape(B, L, C//2)

        sp12 = self.attns[1](qkv1[:,:,:,C//2:], qkv2[:,:,:,C//2:],  _H, _W)
        sp12 = sp12[:, :H, :W, :].reshape(B, L, C//2)

        sp1 = torch.cat([sp11,sp12], dim=2)
        x = x.reshape(B, C, L).contiguous().view(B, C, H, W)
        y = y.reshape(B, C, L).contiguous().view(B, C, H, W)
        conv_x1 = self.dwconv1(x)
        channel_map = self.channel_interaction(conv_x1).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        sp1 = sp1 + sp1 * torch.sigmoid(channel_map)

        sp1 = self.proj1(sp1)
        sp1 = self.proj_drop(sp1)
     
        return sp1

class Self_Attention(nn.Module):
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx

        head_dim = dim // num_heads if dim >= num_heads else 1
        self.scale = head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv1,qkv2, qkv3, H, W):
        q1, k1, v1= qkv1[0], qkv1[1], qkv1[2]
        q2, k2, v2= qkv2[0], qkv2[1], qkv2[2]
        q3, k3, v3= qkv3[0], qkv3[1], qkv3[2]
        B, L, C = q1.shape   
        assert L == H * W,"flatten img_tokens has wrong size"
        q1 = self.im2win(q1, H, W)
        k1 = self.im2win(k1, H, W)
        v1 = self.im2win(v1, H, W)
        q2 = self.im2win(q2, H, W)
        k2 = self.im2win(k2, H, W)
        v2 = self.im2win(v2, H, W)
        q3 = self.im2win(q3, H, W)
        k3 = self.im2win(k3, H, W)
        v3 = self.im2win(v3, H, W)
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn3 = (q3 @ k3.transpose(-2, -1)) * self.scale
        attn = attn1 + attn2 + attn3
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)        
        v = v1 + v2 + v3
        fu = (attn @ v)
        fu = fu.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)
        fu = windows2img(fu, self.H_sp, self.W_sp, H, W)  # B H' W' C
        return fu
class Adaptive_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, split_size=[8,16], proj_drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.qkv = nn.Linear(dim, dim *3)
        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attns = nn.ModuleList([
                Self_Attention(
                    dim//2 if dim >= 4 else 1, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2 if dim >= 4 else 1,
                    attn_drop=attn_drop)
                for i in range(self.branch_num)])
             
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, kernel_size=1)
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        #print("ASA中的input形状", x.shape, y.shape)
        qkv1 = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        qkv2 = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C###########################\\
        qkv3 = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C###########################\\
        # V without partition
        #print("ASA中的qkv2形状", qkv2.shape)
        #v1 = qkv1[1].transpose(-2,-1).contiguous().view(B, C, H, W)

        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv1 = qkv1.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv1 = F.pad(qkv1, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        qkv2 = qkv2.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv2 = F.pad(qkv2, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        qkv3 = qkv3.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv3 = F.pad(qkv3, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W
        fu11 = self.attns[0](qkv1[:,:,:,:C//2], qkv2[:,:,:,:C//2], qkv3[:,:,:,:C//2], _H, _W)
        fu11 = fu11[:, :H, :W, :].reshape(B, L, C//2)

        fu12 = self.attns[1](qkv1[:,:,:,C//2:], qkv2[:,:,:,C//2:], qkv3[:,:,:,C//2:],  _H, _W)
        fu12 = fu12[:, :H, :W, :].reshape(B, L, C//2)

        fu = torch.cat([fu11,fu12], dim=2)
        x = x.reshape(B, C, L).contiguous().view(B, C, H, W)
        conv_x1 = self.dwconv1(x)
        spatial_map = self.spatial_interaction(conv_x1).permute(0, 2, 3, 1).contiguous().view(B, L, 1)
        channel_map = self.channel_interaction(conv_x1).permute(0, 2, 3, 1).contiguous().view(B, 1, C)


        fu = fu + fu * torch.sigmoid(spatial_map)
        fu = fu.transpose(-2,-1).contiguous().view(B, C, H, W)
        fu = self.dwconv2(fu)
        fu = fu.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        fu = fu + fu * torch.sigmoid(channel_map)
        fu = self.proj(fu)
        fu = self.proj_drop(fu)
        return fu



class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class  Inverted_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, split_size=[8,16], expansion_factor=2,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm = norm_layer(dim)
        #Spatial cross transformer block
        self.attn1 = Adaptive_Inverted_Attention(
            dim, num_heads=num_heads, split_size=split_size, 
            proj_drop=drop_path, attn_drop=attn_drop)

        #Channel cross transformer block
        self.attn2 = nn.ModuleList([
            Adaptive_Self_Attention(
            dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=drop_path)for i in range(2)])
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn =FFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer, drop = drop_path)
    def forward(self, x, y, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """        
        H , W = x_size
        #print(x.shape,y.shape)
        sp1 = self.attn1(self.norm(x), self.norm(y), H, W)
        sp1 = self.attn2[0](self.norm(sp1), H, W)
        sp1 = self.attn2[1](self.norm(sp1), H, W)
        sp1 = self.drop_path(sp1)
        sp1 = sp1 + self.drop_path(self.ffn(sp1, H, W))#只在feedforward时候加残差快
        sp1 = self.norm(sp1)        
        return sp1
    

class ResidualGroup(nn.Module):
    """ 
    总核心
    """
    def __init__(   self,
                    dim,
                    num_heads,
                    split_size=[4,8],
                    expansion_factor=2,
                    proj_drop=0.1,
                    attn_drop=0.1,
                    drop_paths=0.16,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    ):
        super().__init__()

        self.conv = nn.ModuleList([
        ASC(
            dim = dim,
            proj_drop=drop_paths,
            )for i in range(2)])
        
        self.blocks = Inverted_Self_Attention(
            dim=dim,
            num_heads=num_heads,
            split_size = split_size,
            expansion_factor=expansion_factor,
            attn_drop=attn_drop,
            drop_path=drop_paths,
            act_layer=act_layer,
            norm_layer=norm_layer,
            )
        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim//2, 1, 1, 0),
                nn.BatchNorm2d(dim//2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(dim//2,dim//2 , 1, 1, 0),
                nn.BatchNorm2d(dim//2),
        )for i in range(2)])
    def forward(self, msf, pan, x_size, last1, last3):
        H, W = x_size
        conv1 = self.conv[0]
        conv3 = self.conv[1]
        if last1 != None:
            msf = torch.concat([msf, last1], dim = 2).float()
            pan = torch.concat([pan, last3], dim = 2).float()

        B, L, C = msf.shape
        msf = conv1(msf, H, W, don = False)
        pan = conv3(pan, H, W, don = False)
        z = self.blocks(msf, pan, x_size)
        z = z.transpose(-2,-1).contiguous().view(B, C, H, W)     
        msf = msf.transpose(-2,-1).contiguous().view(B, C, H, W)   
        pan = pan.transpose(-2,-1).contiguous().view(B, C, H, W)     

        msf = self.upsample[0](msf).permute(0, 2, 3, 1).contiguous().view(B, L*4, C//2)
        pan = self.upsample[1](pan).permute(0, 2, 3, 1).contiguous().view(B, L*4, C//2)

        return z, msf, pan

class SDFD(nn.Module):
    """ 
    整体建构
    """
    def __init__(self,
                 embed_dim = 24,
                 split_size=[4,8],
                 num_heads= 6,
                 expansion_factor=2,
                 proj_drop_rate = 0.16,
                 drop_paths = 0.16,
                 attn_drop_rate=0.16,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,

                 seq_pool=True,
                 positional_embedding='learnable',
                 sequence_length1=None,
                 sequence_length2=None,
                 sequence_length3=None,
                **kwargs):
        super().__init__()
        self.dim = embed_dim
        #位置编码
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'

        self.embed_dim = embed_dim
        self.sequence_length1 = sequence_length1#msf
        self.sequence_length2 = sequence_length2#pan
        
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert sequence_length1 is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        
        if not seq_pool:
            sequence_length1 += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embed_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = nn.Linear(self.embed_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb1 = Parameter(torch.zeros(1, sequence_length1, embed_dim),
                                                requires_grad=True)
                self.positional_emb2 = Parameter(torch.zeros(1, sequence_length2, embed_dim),
                                                requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb1, std=0.2)
                nn.init.trunc_normal_(self.positional_emb2, std=0.2)
            else:
                self.positional_emb1 = Parameter(self.sinusoidal_embedding(sequence_length1, embed_dim),
                                                requires_grad=False)
                self.positional_emb2 =  Parameter(self.sinusoidal_embedding(sequence_length2, embed_dim),
                                                requires_grad=False)
            self.positional_emb1 = None
            self.positional_emb2 = None
        self.dropout = nn.Dropout(p=0.16)

        self.layer = ResidualGroup(
            dim=embed_dim,
            num_heads=num_heads,
            split_size=split_size,
            expansion_factor=expansion_factor,
            proj_drop=proj_drop_rate,
            attn_drop=attn_drop_rate,
            drop_paths=drop_paths,
            act_layer=act_layer,
            norm_layer=norm_layer,
            )

    def forward(self, x, y,  last1 = None, last3 = None):
        """
        Input: x msf ,z fu: (B, 4, H, W) ,y pan: (B, 1, H, W)
        output: x(B, 64, H/2, W/2), msf_init, pan_init(B, 16, H, W)
        
        """
        B, L, C = x.shape
        H = int(L**0.5)
        W = H
        torch.cuda.empty_cache()
        if self.positional_emb1 is None and self.positional_emb2 is None and x.size(1) < self.sequence_length1 and y.size(1)<self.sequence_length2 :
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)
            y = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)
        if not self.seq_pool:
            cls_token1 = self.class_emb.expand(x.shape[0], -1, -1)
            cls_token2 = self.class_emb.expand(y.shape[0], -1, -1)
            x = torch.cat((cls_token1, x), dim=1)
            y = torch.cat((cls_token2, y), dim=1)
            
        if self.positional_emb1 is not None and self.positional_emb2 is not None:
            x+=self.positional_emb1
            y+=self.positional_emb2
        x = self.dropout(x)
        y = self.dropout(y)
        x_size = [H, W]
        x = self.layer(x, y, x_size, last1, last3)
        return x
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
