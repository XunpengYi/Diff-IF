import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import random
from .loss import Grad_loss
from .loss import SSIM_loss

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        refinement_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.refinement_fn = refinement_fn
        self.loss_type = loss_type
        self.ddim_timesteps = 4
        self.ddim_eta = 1.0
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)
        else:
            raise NotImplementedError()
        self.grad_loss = Grad_loss()
        self.ssim_loss = SSIM_loss()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(
                torch.cat((condition_x['vis'], condition_x['ir'], x), dim=1), noise_level))
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        denoise_x0 = x_recon
        x_recon = self.refinement_fn(torch.cat([condition_x['vis'], condition_x['ir']], 1), x_recon, noise_level)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, x_recon, denoise_x0

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance, x0, denoise_x0 = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp(), x0, denoise_x0

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        x = x_in['vis']
        shape = x.shape
        img = torch.randn(shape, device=device)
        ret_img = x
        x0_img = x
        denoise_x0_img = x
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x0, denoise_x0 = self.p_sample(img, i, condition_x=x_in)
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
                x0_img = torch.cat([x0_img, x0], dim=0)
                denoise_x0_img = torch.cat([denoise_x0_img, denoise_x0], dim=0)
        if continous:
            return ret_img, x0_img, denoise_x0_img
        else:
            return ret_img[-1], x0_img, denoise_x0_img

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    @torch.no_grad()
    def super_resolution_ddim(self, x_in, continous=False):
        return self.p_sample_loop_ddim(x_in, continous)

    @torch.no_grad()
    def p_sample_loop_ddim(self, x_in, continous=False, log_time_flag=False):
        device = self.betas.device

        ddim_timesteps = self.ddim_timesteps
        c = self.num_timesteps // ddim_timesteps
        ddim_timestep_seq = list(reversed(range(self.num_timesteps - 1, -1, -c)))
        ddim_timestep_seq = np.asarray(ddim_timestep_seq)
        ddim_timestep_prev_seq = np.append(np.array([-1]), ddim_timestep_seq[:-1])

        sample_inter = (1 | (ddim_timesteps // 10))
        x = x_in['vis']
        shape = x.shape
        b, c, h, w = x_in['vis'].shape
        img = torch.randn(shape, device=device)
        ret_img = x
        x0_img = x
        denoise_x0_img = x
        for i in tqdm(reversed(range(0, self.ddim_timesteps)), desc='sampling loop time step',
                      total=self.ddim_timesteps):
            t = torch.full((b,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((b,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, img.shape)
            if i == 0:
                alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t)
            else:
                alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, img.shape)

            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(b, 1).to(x.device)

            pred_noise = self.denoise_fn(torch.cat([x_in['vis'], x_in['ir'], img], dim=1), noise_level)
            x_recon = self.predict_start_from_noise(img, t=t, noise=pred_noise)
            x_recon.clamp_(-1., 1.)
            denoise_x0 = x_recon
            
            pred_x0 = self.refinement_fn(torch.cat([x_in['vis'], x_in['ir']], 1), x_recon, noise_level)
            
            pred_x0.clamp_(-1., 1.)

            sigmas_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(pred_x0)
            img = x_prev

            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
                x0_img = torch.cat([x0_img, x_recon], dim=0)
                denoise_x0_img = torch.cat([denoise_x0_img, denoise_x0], dim=0)
        
        if continous:
            return ret_img, x0_img, denoise_x0_img
        else:
            return ret_img[-1], x0_img, denoise_x0_img

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None, clip_denoised=True):
        x_start = x_in['fusion']

        x_start_vis = x_in['vis']
        x_start_ir = x_in['ir']
        [b, _, _, _] = x_start_vis.shape

        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1], self.sqrt_alphas_cumprod_prev[t], size=b)
        ).to(x_start_vis.device)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        # compute the fusion image diffusion
        pred_noise = self.denoise_fn(torch.cat((x_start_vis, x_start_ir, x_noisy), dim=1), continuous_sqrt_alpha_cumprod)

        loss_eps = 8 * self.loss_func(pred_noise, noise)
        pred_x0 = self.predict_start_from_noise(x_noisy, t-1, pred_noise)
        if clip_denoised:
            pred_x0.clamp_(-1., 1.)
            
        pred_x0_detach = pred_x0.detach()
        refine_x0 = self.refinement_fn(torch.cat([x_start_vis, x_start_ir], 1), pred_x0_detach, continuous_sqrt_alpha_cumprod)

        if clip_denoised:
            refine_x0.clamp_(-1., 1.)

        max_img = torch.max(x_start_vis, x_start_ir)
        loss_max = 4 * self.loss_func(refine_x0, max_img)
        loss_grad = 5 * self.grad_loss(refine_x0, x_start_vis, x_start_ir)
        loss_ssim = self.ssim_loss(refine_x0, x_start_vis) + self.ssim_loss(refine_x0, x_start_ir)
        loss_simple = loss_max + loss_grad + loss_ssim
        loss_x0 = 2 * self.loss_func(refine_x0, x_start)

        loss = loss_eps + loss_x0 + loss_simple
        return loss, loss_eps, loss_max, loss_grad, loss_ssim, loss_x0

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    def expand_Y_channel(self, rgb_tensor):
        y_channel = 0.299 * rgb_tensor[:, 0, :, :] + 0.587 * rgb_tensor[:, 1, :, :] + 0.114 * rgb_tensor[:, 2, :, :]
        y_channel = y_channel.unsqueeze(1)
        expanded_tensor = y_channel.expand(-1, 3, -1, -1)
        return expanded_tensor