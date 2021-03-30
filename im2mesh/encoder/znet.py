import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler

#https://github.com/vlkniaz/Z_GAN
# Defines the Znet.
# |num_downs|: number of downsamplings in ZNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1 => 1x64x64x64
# at the bottleneck
class znet(nn.Module): 
    def __init__(self, input_nc=3, output_nc=64, num_downs=7, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], fruxel_depth=16):
        super(znet, self).__init__()
        self.gpu_ids = gpu_ids
        
        # construct unet structure
        znet_block = ZnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            z_stride = 2
            #if i % 2 == 0 and fruxel_depth == 16:
            #    z_stride = 1
            znet_block = ZnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=znet_block, norm_layer=norm_layer, use_dropout=use_dropout, z_stride=z_stride)
        znet_block = ZnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=znet_block, norm_layer=norm_layer)
        znet_block = ZnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=znet_block, norm_layer=norm_layer)
        znet_block = ZnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=znet_block, norm_layer=norm_layer)
        znet_block = ZnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=znet_block, outermost=True, norm_layer=norm_layer)
        # znet_block = ZnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=znet_block, norm_layer=norm_layer)

        self.model = znet_block

    def forward(self, input):
        
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            
            # print("input",input.shape)
            # input = self.fc_in(input)
            # print("input",input.shape)
            # exit(1)
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ZnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, z_kernel=4, z_stride=2):
        super(ZnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(False)
        upnorm = nn.BatchNorm3d(outer_nc)#norm_layer(outer_nc)
        view = View()

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=(1,1,1), stride=(1,1,1),
                                        padding=(0,0,0))
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=(z_stride*2,4,4), stride=(z_stride,2,2),
                                        padding=(1,1,1), bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + [view] + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=(z_stride*2,4,4), stride=(z_stride,2,2),
                                        padding=(1,1,1), bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print(x.data.shape)
        if self.outermost:
            # temp = self.model(x) 
            # print('out---')
            # print(temp.data.shape)
            
            return self.model(x)
        else:
            a = x.unsqueeze(2)
            b = self.model(x)
            while a.data.shape[2] != b.data.shape[2]:
                a = torch.cat([a, a], 2)
            return  torch.cat([a, b], 1)

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
    def forward(self, input):
        x = input.unsqueeze(2)
        return x