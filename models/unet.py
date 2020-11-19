import math
import torch
from torch import nn
from torch.nn import functional as F
import utils.utils as utils


def get_norm_class(normtype: str, num_features: int):
    if 'group' in normtype:
        ngroups = int(normtype.replace('group',''))
        return nn.GroupNorm(num_groups=ngroups, num_channels=num_features)
    elif normtype == 'batch':
        return nn.BatchNorm2d(num_features=num_features)
    else:
        raise NotImplementedError(normtype)


def get_activation_class(activation_type: str):
    if activation_type=='relu':
        return nn.ReLU()
    elif activation_type == 'swish':
        return utils.Swish()
    else:
        raise NotImplementedError(activation_type)


class Unet(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, num_middles: int = 2, ngf: int = 64, norm: str = 'batch', activation: str = 'relu', use_dropout: bool = False, use_attention: bool = False, device: str = 'cpu'):
        """Create a Unet Module

        Parameters:
            input_nc (int) -- the number of channels in input images/features
            output_nc (int) -- the number of channels in output images/features
            outer_nc (int) -- the number of filters in the outer conv layer
            num_middles (int) -- the number of intermediate layers
            ngf (int) -- the number of filters in the first layer
            user_dropout (bool) -- if use dropout layers.
            device (str) -- device name
        """
        super(Unet, self).__init__()
        # construct unet structure
        unet_block = UnetSkipModule(ngf * 8, ngf * 8, ngf * 4, input_nc=None, norm=norm, activation=activation, submodule=None, innermost=True, use_dropout=use_dropout, use_attention=use_attention)  # add the innermost layer
        for i in range(num_middles):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipModule(ngf * 8, ngf * 8, ngf * 4, input_nc=None, norm=norm, activation=activation, submodule=unet_block, use_dropout=use_dropout, use_attention=use_attention)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipModule(ngf * 4, ngf * 8, ngf * 4, input_nc=None, norm=norm, activation=activation, submodule=unet_block, use_dropout=use_dropout, use_attention=use_attention)
        unet_block = UnetSkipModule(ngf * 2, ngf * 4, ngf * 4, input_nc=None, norm=norm, activation=activation, submodule=unet_block, use_dropout=use_dropout, use_attention=use_attention)
        unet_block = UnetSkipModule(ngf, ngf * 2, ngf * 4, input_nc=None, norm=norm, activation=activation, submodule=unet_block, use_dropout=use_dropout, use_attention=use_attention)
        self.model = UnetSkipModule(output_nc, ngf, ngf * 4, input_nc=input_nc, norm=norm, activation=activation, submodule=unet_block, outermost=True, use_dropout=use_dropout)  # add the outermost layer

        self.embedding_weight = torch.exp(torch.arange(0, ngf//2) * -(math.log(10000) / (ngf//2 - 1))).to(device)
        self.embedding = nn.Sequential(
            nn.Linear(ngf, ngf * 4),
            get_activation_class(activation_type=activation),
            nn.Linear(ngf * 4, ngf * 4)
            )

    def forward(self, input: dict):
        x = input['x']
        t = input['t']
        t = t.view(t.size(0), 1) * self.embedding_weight.view(1, self.embedding_weight.size(0))
        t = torch.cat([torch.sin(t), torch.cos(t)], dim=1)
        return self.model({'x': x, 't': self.embedding(t)})


class UnetSkipModule(nn.Module):
    def __init__(self, outer_nc: int, inner_nc: int, emb_nc: int, input_nc = None, norm: str = 'batch', activation: str = 'relu', submodule = None, outermost: bool = False, innermost: bool = False, use_attention: bool = False, use_dropout: bool = False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipModule) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            use_attention (bool) -- if use attention layers.
            use_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipModule, self).__init__()
        if input_nc is None:
            input_nc = outer_nc
        if outermost:
            self.down = ModuleWrap(nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))
            self.up = ModuleWrap(nn.Sequential(
                get_norm_class(normtype=norm, num_features=inner_nc * 2),
                get_activation_class(activation_type=activation),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
                ))
        elif innermost:
            model = [ResnetBlock(input_nc, emb_nc, 3, norm=norm, activation=activation, use_dropout=use_dropout)]
            if use_attention:
                model += [AttentionBlock(input_nc, enable_resolutions=[16])]
            model += [
                get_norm_class(normtype=norm, num_features=input_nc),
                get_activation_class(activation_type=activation),
                nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
                ]
            self.down = nn.Sequential(*model)
            model = [ResnetBlock(inner_nc, emb_nc, 3, norm=norm, activation=activation, use_dropout=use_dropout)]
            if use_attention:
                model += [AttentionBlock(input_nc, enable_resolutions=[16])]
            self.up = nn.Sequential(*model)
        else:
            model = [ResnetBlock(input_nc, emb_nc, 3, norm=norm, activation=activation, use_dropout=use_dropout)]
            if use_attention:
                model += [AttentionBlock(input_nc, enable_resolutions=[16])]
            model += [
                get_norm_class(normtype=norm, num_features=input_nc),
                get_activation_class(activation_type=activation),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, padding_mode='reflect')    
                ]
            self.down = nn.Sequential(*model)
            model = [ResnetBlock(inner_nc * 2, emb_nc, 3, norm=norm, activation=activation, use_dropout=use_dropout)]
            if use_attention:
                model += [AttentionBlock(inner_nc * 2, enable_resolutions=[16])]
            model += [
                get_norm_class(normtype=norm, num_features=inner_nc * 2),
                get_activation_class(activation_type=activation),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
                ]
            self.up = nn.Sequential(*model)

        self.innermost = innermost
        self.outermost = outermost
        self.submodule = submodule

    def forward(self, input: dict):
        x = input['x']
        t = input['t']
        h = self.down({'x': x, 't': t})
        if self.innermost:
            return torch.cat([x, self.up({'x': h, 't': t})], 1)
        h = self.submodule({'x': h, 't': t})
        if self.outermost:
            return self.up({'x': h, 't': t})
        return torch.cat([x, self.up({'x': h, 't': t})], 1)


class ResnetBlock(nn.Module):
    def __init__(self, input_nc: int, emb_nc: int, kernel_size: int = 3, norm: str = 'batch', activation: str = 'relu', use_dropout: bool = False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipModule) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            user_dropout (bool) -- if use dropout layers.
        """
        super(ResnetBlock, self).__init__()
        self.first_layer = nn.Sequential(
            get_norm_class(normtype=norm, num_features=input_nc),
            get_activation_class(activation_type=activation),
            nn.Conv2d(input_nc, input_nc, kernel_size=kernel_size, padding=((kernel_size-1)//2), padding_mode='reflect')
            )
        model = [
            get_norm_class(normtype=norm, num_features=input_nc),
            get_activation_class(activation_type=activation)
            ]
        if use_dropout:
            model += [nn.Dropout(0.5)]
        model += [nn.Conv2d(input_nc, input_nc, kernel_size=kernel_size, padding=((kernel_size-1)//2), padding_mode='reflect')]
        self.second_layer = nn.Sequential(*model)
        self.embedding = nn.Sequential(
                get_activation_class(activation_type=activation),
                nn.Linear(emb_nc, input_nc)
            )
    
    def forward(self, input: dict):
        x = input['x']
        t = input['t']
        h = self.first_layer(x)
        s = list(h.size())
        s[-1] = 1
        s[-2] = 1
        h += self.embedding(t).view(tuple(s)).expand(h.size())
        return x + self.second_layer(h)


class AttentionBlock(nn.Module):
    def __init__(self, input_nc: int, norm: str = 'batch', enable_resolutions=[]):
        super(AttentionBlock, self).__init__()
        self.query = nn.Sequential(
            get_norm_class(normtype=norm, num_features=input_nc),
            TensorLinear(input_nc, input_nc)
        )
        self.key = nn.Sequential(
            get_norm_class(normtype=norm, num_features=input_nc),
            TensorLinear(input_nc, input_nc)
        )
        self.value = nn.Sequential(
            get_norm_class(normtype=norm, num_features=input_nc),
            TensorLinear(input_nc, input_nc)
        )
        self.last_layer = TensorLinear(input_nc, input_nc)
        self.enable_resolutions = enable_resolutions

    def forward(self, input):
        x = input
        B, C, H, W = x.size()
        if len(self.enable_resolutions)>0 and H not in self.enable_resolutions:
            return x
        q = self.query(x)
        v = self.value(x)
        k = self.key(x)
        w = (torch.einsum('bchw,bcix->bhwix', q, k) * (int(C) ** (-0.5))).view(B, H, W, H * W)
        w = F.softmax(w).view(B, H, W, H, W)
        h = torch.einsum('bhwix,bcix->bchw', w, v)
        return x + self.last_layer(h)


class TensorLinear(nn.Linear):
    def forward(self, input):
        x = input
        y = torch.tensordot(x.permute(0,2,3,1), self.weight, dims=1) + self.bias
        return y.permute(0,3,1,2)


class ModuleWrap(nn.Module):
    def __init__(self, module):
        super(ModuleWrap, self).__init__()
        self.module = module

    def forward(self, input:dict):
        return self.module(input['x'])