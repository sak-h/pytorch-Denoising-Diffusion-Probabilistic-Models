import os
import os.path
import numpy as np
import torch
from torch import nn
import models.unet as unet
import utils.visualizer as visualizer
import utils.utils as utils


class DDPModel:
    """ Denoising Diffusion Probabilistic Models
        @article{ho2020denoising,
            title={Denoising Diffusion Probabilistic Models},
            author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
            year={2020},
            journal={arXiv preprint arxiv:2006.11239}
        }
        written for PyTorch with borrowing from the official implementation:
            https://github.com/hojonathanho/diffusion
    """
    def __init__(self, opt):
        super(DDPModel, self).__init__()

        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print('Directory created: %s' % self.save_dir)

        betas = self.get_beta_schedule(opt.beta_schedule, opt.beta_start, opt.beta_end, opt.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod_prev.shape == betas.shape

        self.betas = torch.tensor(betas)
        self.alphas_cumprod = torch.tensor(alphas_cumprod)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32, device=self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod), dtype=torch.float32, device=self.device)
        self.log_one_minus_alphas_cumprod = torch.tensor(np.log(1. - alphas_cumprod), dtype=torch.float32, device=self.device)
        self.sqrt_recip_alphas_cumprod = torch.tensor(np.sqrt(1. / alphas_cumprod), dtype=torch.float32, device=self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.tensor(np.sqrt(1. / alphas_cumprod - 1), dtype=torch.float32, device=self.device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = torch.tensor(posterior_variance, dtype=torch.float32, device=self.device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32, device=self.device)
        self.posterior_mean_coef1 = torch.tensor(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), dtype=torch.float32, device=self.device)
        self.posterior_mean_coef2 = torch.tensor((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), dtype=torch.float32, device=self.device)

        assert isinstance(betas, np.ndarray) and (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.loss_criteria = nn.MSELoss(reduction='none')
        self.loss_type = opt.loss_type

        # setup denoise model
        model = []
        if opt.block_size != 1:
            model += [utils.SpaceToDepth(opt.block_size)]
        model += [unet.Unet(opt.input_nc, opt.input_nc, num_middles=1, ngf=opt.ngf, use_dropout=opt.dropout, use_attention=opt.attention, device=self.device)]
        if opt.block_size != 1:
            model += [utils.SpaceToDepth(opt.block_size)]
        self.denoise_model = utils.init_net(nn.Sequential(*model), opt.init_type, opt.init_gain, opt.gpu_ids)

        if opt.phase == 'train':
            # setup optimizer, visualizer, and learning rate scheduler
            self.optimizer = torch.optim.Adam(self.denoise_model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.visualizer = visualizer.Visualizer(opt)
            self.scheduler = utils.get_scheduler(self.optimizer, opt)
            self.lr_policy = opt.lr_policy
        else:
            self.image_size = (opt.batch_size, opt.input_nc, opt.load_size, opt.load_size)
            self.denoise_model.train(False)

        if opt.phase == 'interpolate':
            self.mix_rate = opt.mix_rate

    def setup(self, opt):
        """Load and print networks"""
        if opt.phase == 'train':
            self.schedulers = utils.get_scheduler(self.optimizer, opt)
        if opt.phase in ['sample', 'interpolate'] or opt.resume:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_denoise_model(load_suffix)
        self.print_networks(opt.verbose)

    def save_denoise_model(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int or str) -- current epoch; used in the file name '%s_net.pth' % epoch
        """
        save_filename = '%s_net.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.denoise_model.module.cpu().state_dict(), save_path)
            self.denoise_model.cuda(self.gpu_ids[0])
        else:
            torch.save(self.denoise_model.cpu().state_dict(), save_path)

    def load_denoise_model(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int or str) -- current epoch; used in the file name '%s_net.pth' % epoch
        """
        load_filename = '%s_net.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)
        if isinstance(self.denoise_model, torch.nn.DataParallel):
            net = self.denoise_model.module
        else:
            net = self.denoise_model
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.denoise_model.parameters():
            num_params += param.numel()
        if verbose:
            print(self.denoise_model)
        print('[Denoise Model] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    def compute_visuals(self, step):
        """Print the current state on log directory

        Parameters:
            step (int) -- global step
        """
        self.visualizer.add_values([{'name': 'loss', 'x': step, 'y': self.losses}])
        images = [
            {'name': 'x_start', 'step': step, 'data': self.x_start},
            {'name': 'noise',   'step': step, 'data': self.noise},
            {'name': 'x_noisy', 'step': step, 'data': self.x_noisy},
            {'name': 'x_recon', 'step': step, 'data': self.x_recon}
        ]
        self.visualizer.add_images(images)

    def get_current_losses(self):
        return {'losses': self.losses}

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.images = input['images'].to(self.device)
        self.paths = input['paths']

    def train(self):
        """called by the training script"""
        B = self.images.size(0)
        self.timestep = torch.randint(self.num_timesteps, size=tuple([B]), dtype=torch.long, device=self.device)
        self.x_start = self.images
        losses = self.p_losses(x_start=self.x_start, t=self.timestep)
        assert losses.size(0) == self.timestep.size(0) == B
        self.losses = losses.mean()
        self.optimizer.zero_grad()
        self.losses.backward()
        self.optimizer.step()

    def sample(self):
        """called by the sampling script"""
        with torch.no_grad():
            return self.p_sample_loop(shape=self.image_size, noise_fn=torch.randn)

    def interpolate(self):
        """called by the interpolation script"""
        with torch.no_grad():
            B = self.images.size(0)
            assert B%2 == 0
            B = B//2
            x0_1 = self.images[:B]
            x0_2 = self.images[B:]
            xt_1 = self.q_sample(x0_1, torch.full(tuple([B]), self.num_timesteps-1, dtype=torch.long, device=self.device))
            xt_2 = self.q_sample(x0_2, torch.full(tuple([B]), self.num_timesteps-1, dtype=torch.long, device=self.device))
            xt_mix = torch.tensor((1.-self.mix_rate), dtype=torch.float32, device=self.device) * xt_1 + torch.tensor(self.mix_rate, dtype=torch.float32, device=self.device) * xt_2
            for i in reversed(range(self.num_timesteps - 1)):
                xt_mix = self.p_sample(x=xt_mix.detach(), t=torch.full(tuple([B]), i, dtype=torch.long, device=self.device), noise_fn=torch.randn)
            assert xt_mix.size() == x0_1.size()
            return xt_mix

    def get_interpolate_filename(self):
        assert len(self.paths) == 2
        basename1, _ = os.path.splitext(os.path.basename(self.paths[0]))
        basename2, _ = os.path.splitext(os.path.basename(self.paths[1]))
        return basename1 + '+' + basename2 + '.png'

    @staticmethod
    def _extract(a, t, x_shape):
        """Extract some coefficients at specified timesteps, then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Parameters:
            a (Tensor): target alphas
            t (Tensor): timestep
            x_shape (tuple): reshape size (B, C, H, W) or (B, H, W)
        """
        B = t.size(0)
        assert x_shape[0] == B
        out = torch.gather(a, 0, t)
        assert out.size(0) == B
        shape = list(x_shape)
        shape[0] = B
        for i in range(1, len(shape)):
            shape[i] = 1
        return out.view(shape)

    @staticmethod
    def _noise_like(shape, noise_fn=torch.randn, repeat=False, dtype=torch.float32):
        repeat_noise = lambda: noise_fn((1, *shape[1:]), dtype=dtype).repeat(shape[0])
        noise = lambda: noise_fn(shape, dtype=dtype)
        return repeat_noise() if repeat else noise()

    def q_sample(self, x_start, t, noise=None):
        """Diffuse the data (t == 0 means diffused for 1 step)

        Parameters:
            x_start (Tensor): image for start
            t (Tensor): timestep
            noise (Tensor): noise image 
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.size() == x_start.size()
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.size()) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.size()) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        """Training loss calculation

        Parameters:
            x_start (Tensor): image for start
            t (Tensor): timestep
            noise (Tensor): noise image 
        """
        B = x_start.size(0)
        assert t.size(0) == B
        if noise is None:
            noise = torch.randn_like(x_start)
        self.noise = noise
        assert self.noise.size() == x_start.size() and self.noise.type() == x_start.type()
        self.x_noisy = self.q_sample(x_start=x_start, t=t, noise=self.noise)
        self.x_recon = self.denoise_model({'x': self.x_noisy, 't': t})
        assert self.x_noisy.size() == x_start.size()
        assert self.x_recon.size() == x_start.size()
        if self.loss_type == 'noisepred':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            losses = self.loss_criteria(self.noise, self.x_recon).view(B, -1).mean(dim=1)
        else:
            raise NotImplementedError(loss_type)
        assert losses.dim() == 1 and losses.size(0) == B
        return losses

    def p_sample_loop(self, shape, noise_fn=torch.randn):
        """Generate samples

            shape (tuple): image shape (B, C, H, W) or (B, H, W)
            noise_fn (function): noise function 
        """
        #i_0 = torch.tensor(self.num_timesteps - 1, dtype=torch.int32, device=self.device)
        assert isinstance(shape, (tuple, list))
        img_0 = noise_fn(shape, dtype=torch.float32, device=self.device)
        for i in reversed(range(self.num_timesteps)):
            img_0 = self.p_sample(x=img_0.detach(), t=torch.full(tuple([shape[0]]), i, dtype=torch.long, device=self.device), noise_fn=noise_fn)
        assert img_0.size() == shape
        return img_0

    def p_sample(self, x, t, noise_fn, clip_denoised=True, repeat_noise=False):
        """Sample from the model

        Parameters:
            x (Tensor): noisy image
            t (Tensor): timestep
            noise_fn (function): noise function
            clip_denoised (bool): clip saturated values
            repeat_noise (bool): use same noise along the batch  
        """
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = self._noise_like(x.size(), noise_fn, repeat_noise).to(self.device)
        assert noise.size() == x.size()
        # no noise when t == 0
        s = list(x.size())
        for i in range(1, len(s)):
            s[i] = 1
        nonzero_mask = (t!=0).float().view(tuple(s))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def p_mean_variance(self, x, t, clip_denoised: bool):
        if self.loss_type == 'noisepred':
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_model({'x': x, 't': t}))
        else:
            raise NotImplementedError(self.loss_type)

        if clip_denoised:
            x_recon[x_recon>1] = 1
            x_recon[x_recon<-1] = -1

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        assert model_mean.size() == x_recon.size() == x.size()
        assert posterior_variance.size() == posterior_log_variance.size() == (x.size(0), 1, 1, 1)
        return model_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        assert x_t.size() == noise.size()
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.size()) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.size()) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)

        Parameters:
            x_start (Tensor): image for start
            x_t (Tensor): image at t
            t (Tensor): timestep
        """
        assert x_start.size() == x_t.size()
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.size()) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.size()) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.size())
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.size())
        assert (posterior_mean.size(0) == posterior_variance.size(0) == posterior_log_variance_clipped.size(0) == x_start.size(0))
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @staticmethod
    def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
        if beta_schedule == 'quad':
            betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
        elif beta_schedule == 'linear':
            betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == 'warmup10':
            betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
        elif beta_schedule == 'warmup50':
            betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
        elif beta_schedule == 'const':
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas

    @staticmethod
    def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        warmup_time = int(num_diffusion_timesteps * warmup_frac)
        betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
        return betas

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        if self.lr_policy == 'plateau':
            self.scheduler.step(self.metric)
        else:
            self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)