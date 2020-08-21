import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler


class DepthToSpace(nn.Module):
    """Pytorch implementation of tf.nn.depth_to_space"""
    def __init__(self, block_size: int):
        super().__init__()
        self.bs = block_size

    def forward(self, input: dict):
        x = input['x']
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToDepth(nn.Module):
    """Pytorch implementation of tf.nn.space_to_depth"""
    def __init__(self, block_size: int):
        super().__init__()
        self.bs = block_size

    def forward(self, input: dict):
        x = input['x']
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to('cuda:{}'.format(gpu_ids[0]))
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_mat(w):
        if init_type == 'normal':
            init.normal_(w.data, 0.0, init_gain)
        elif init_type == 'xavier':
            init.xavier_normal_(w.data, gain=init_gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(w.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(w.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if hasattr(m, 'weight'):
                init_mat(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    print('initialize network of %s with %s' % (net.__class__.__name__, init_type))
    net.apply(init_func)  # apply the initialization function <init_func>


# losses: same format as |losses| of plot_current_losses
def print_current_losses(epoch, iters, losses, t_comp, t_data, log_name):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
        log_name (str) -- log filename
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


def save_images(images, paths):
    """Save images to the disk.

    Parameters:
        images (Tensor) -- Tensor images (B, C, H, W) or (B, H, W)
        paths (array) -- save paths
    """
    if images.size(1)==1:
        images = images.repeat(1, 3, 1, 1)
    images = (images.cpu().float().numpy() + 1.) * 0.5
    images[images>1] = 1
    images[images<0] = 0
    images = np.transpose(np.uint8(images*255), (0, 2, 3, 1))
    for idx, image in enumerate(images):
        save_image = Image.fromarray(image)
        save_image.save(paths[idx])
