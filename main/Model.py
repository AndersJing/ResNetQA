#!/usr/bin/env python3
# encoding: utf-8

import torch
import torch.nn as nn

class ResNet2DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, *args, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        padding = (dilation*(kernel_size-1)//2, dilation*(kernel_size-1)//2)
        self.blocks = nn.Sequential(
            nn.Sequential(nn.InstanceNorm2d(in_channels), nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, bias=False)),
            nn.ELU(inplace=True),
            nn.Sequential(nn.InstanceNorm2d(in_channels), nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, bias=False)),
        )
        self.activate = nn.ELU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x    
        
class ResNet2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, n_layer, *args, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResNet2DResidualBlock(out_channels, out_channels, kernel_size, dilation) for _ in range(n_layer)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
    
class ResNet2D(nn.Module):
    '''
    ResNet 2D
    '''
    def __init__(self, in_channels, deepths, 
                        kernel_size=5, channel_size=64, dilation=2,
                        *args, **kwargs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, channel_size, kernel_size=kernel_size, padding=2, bias=False),
            nn.InstanceNorm2d(channel_size),
            nn.ELU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            *[ResNet2DLayer(channel_size, channel_size, kernel_size, dilation, n_layer=n) for n in deepths]       
        ])
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResNet1DLayer(nn.Module):
    """
    ResNet 1D layer
    """
    def __init__(self, in_channels, out_channels, groups, 
                 kernel_size=5, dilation=2, *args, **kwargs):
        super().__init__()
        padding = dilation*((kernel_size-1)//2)
        self.blocks = nn.Sequential(
            nn.Sequential(nn.InstanceNorm1d(in_channels), nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups, dilation=dilation, padding=padding, bias=False)),
            nn.ELU(inplace=True),
            nn.Sequential(nn.InstanceNorm1d(in_channels), nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups, dilation=dilation, padding=padding, bias=False)),
        )
        self.activate = nn.ELU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
        

class QAModel(nn.Module):
    """
    Dilated ResNets model for both local and global quality assessment.
        Args:
            ResNet2D_in (int): channel size of 2D features.
            ResNet2D_deepths (int): deepth of ResNet2D blocks.
            ResNet1D_in (int): size of 1D features.
            ResNet1D_layer (int): number of ResNet1D layer.
    """
    def __init__(self, 
                 ResNet2D_in=21, ResNet2D_deepths=[2,3,3,2],  
                 ResNet1D_in=52, ResNet1D_layer=8, 
                 *args, **kwargs):

        print(kwargs)
        
        super().__init__()

        # ResNet2D layers
        self.resnet2d = ResNet2D(ResNet2D_in, ResNet2D_deepths)
        self.resnet2d_out = self.resnet2d.blocks[-1].blocks[-1].out_channels

        # ResNet2D output pooling
        self.pool_row = torch.nn.AdaptiveAvgPool2d((None, 1))
        self.pool_col = torch.nn.AdaptiveAvgPool2d((1, None))
        
        # ResNet1D layers
        resnet1d_group = self.resnet2d_out*2 + ResNet1D_in
        self.resnet1d = nn.ModuleList([
            *[ResNet1DLayer(resnet1d_group, resnet1d_group, groups=resnet1d_group)
                for _ in range(ResNet1D_layer)]
        ])

        # ResNet1D output pooling
        self.pool_1d = nn.AdaptiveAvgPool1d((1))

        # linear layer
        self.local_linear = nn.Linear(resnet1d_group, 1)
        self.global_linear = nn.Linear(resnet1d_group, 1)

        # sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        # ResNet2D
        x = self.resnet2d(x2)

        # pooling
        x_row = self.pool_row(x)
        x_row = x_row.permute(0,1,3,2).reshape(x_row.size()[0], x_row.size()[1]*x_row.size()[3], x_row.size()[2])
        x_col = self.pool_col(x)
        x_col = x_col.reshape(x_col.size()[0], x_col.size()[1]*x_col.size()[2], x_col.size()[3])
        
        # Concatenates pooled pairwise features with sequencial features
        x = torch.cat((x_row, x_col, x1), dim=1)

        # ResNet1D
        for block in self.resnet1d: x = block(x)
        
        # linear
        y_local = self.local_linear(x.permute(0,2,1))
        y_global = self.global_linear(self.pool_1d(x).view(x.size()[0], -1))

        # sigmoid
        y_local = self.sigmoid(y_local)
        y_global = self.sigmoid(y_global)
        
        return y_global, y_local

