import math
import os
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

import imagenet


def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        
        padding = (kernel_size-1) // 2
        assert 2  * padding == kernel_size-1, f"parameters incorrect. kernel={kernel_size}, padding={padding}"
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, padding=padding, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # add init weights
    
    def forward(self, x):
        return self.net(x)
    

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
            kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # add init weights
        
    def forward(self, x):
        return self.net(x)
    
class ConvDecomposed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.net = nn.Sequential(
            DepthwiseConv(in_channels=in_channels, kernel_size=kernel_size),
            PointwiseConv(in_channels=in_channels, out_channels=out_channels)
        )
        
        # add init weights
        
    def forward(self, x):
        return self.net(x)
        
        
# use this in mobilenet.py
# replace .view() in mobilent.py

class NNConv5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, interpolation_scale_factor):
        super().__init__()
        
        self.interpolation_scale_factor = interpolation_scale_factor
        
        self.net = ConvDecomposed(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        
    def forward(self, x):
        x = self.net(x)
        return F.interpolate(x, scale_factor=self.interpolation_scale_factor, mode='nearest')
    
    
class Model(nn.Module):
    def __init__(self, pretrained=True, decoder_kernel_size=5, decoder_interpolation_scale_factor=2):
        super().__init__()
        
        mobilenet = imagenet.mobilenet.MobileNet()
        if pretrained:
            pretrained_path = os.path.join('cifar100.pth')
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.state_dict()

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            #mobilenet.load_state_dict(new_state_dict)
            mobilenet.load_state_dict(state_dict)
        else:
            mobilenet.apply(weights_init)

        for i in range(14):
            setattr( self, 'conv{}'.format(i), mobilenet.model[i])
            
        self.decode_conv1 = NNConv5(in_channels=1024, out_channels=512, 
                                    kernel_size=decoder_kernel_size,
                                    interpolation_scale_factor=decoder_interpolation_scale_factor)
       
        self.decode_conv2 = NNConv5(in_channels=512, out_channels=256, 
                                    kernel_size=decoder_kernel_size,
                                    interpolation_scale_factor=decoder_interpolation_scale_factor)
        
        self.decode_conv3 = NNConv5(in_channels=256, out_channels=128, 
                                    kernel_size=decoder_kernel_size,
                                    interpolation_scale_factor=decoder_interpolation_scale_factor)
    
        self.decode_conv4 = NNConv5(in_channels=128, out_channels=64, 
                                    kernel_size=decoder_kernel_size,
                                    interpolation_scale_factor=decoder_interpolation_scale_factor)
        
        self.decode_conv5 = NNConv5(in_channels=64, out_channels=32, 
                                    kernel_size=decoder_kernel_size,
                                    interpolation_scale_factor=decoder_interpolation_scale_factor)
        
        self.decode_conv6 = PointwiseConv(in_channels=32, out_channels=1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        for i in range(14):
            layer = getattr(self, f'conv{i}')
            x = layer(x)
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        for i in range(1,6):
            layer = getattr(self, f'decode_conv{i}')
            x = layer(x)
            if i==4:
                x = x + x1
            elif i==3:
                x = x + x2
            elif i==2:
                x = x + x3
        x = self.decode_conv6(x)
        return x