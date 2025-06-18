import torch
import torch.nn as nn
import torch.nn.functional as F
from Restnet import ResBlk
class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane):
        super(SqueezeBodyEdge, self).__init__()
        self.flow_make = nn.Sequential(
            nn.Conv2d(inplane, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True))
        self.fine_edge = ResBlk(inplane*2, inplane) 
        self.sigmoid_edge = nn.Sigmoid()
        
    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=True)
        return output
    
    def forward(self, x, texture):
        torch.cuda.empty_cache()
        size = x.size()[2:]
        flow = self.flow_make(x)
        seg_body = self.flow_warp(x, flow, size)
        seg_edge = x - seg_body
        fine_edge = self.fine_edge(torch.cat([seg_edge, texture], dim = 1))
        fine_edge = self.sigmoid_edge(fine_edge)
        final = seg_body + fine_edge
        torch.cuda.empty_cache()
        return final

