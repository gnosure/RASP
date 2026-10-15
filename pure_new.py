import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import diff2clf, clf2diff, normalize
import random

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()


class Purifica(torch.nn.Module):
    def __init__(self, clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device,
                 amplitude_cut_range, phase_cut_range, delta, forward_noise_steps,
                 radius_min=2.0, radius_max=6.0, radius_gamma=1.0,
                 lambda_amp=1.0, lambda_phase=0.25, guidance_scale=3e-4,
                 guidance_start_ratio=0.6, use_guidance=False,
                 use_adaptive_radius=False, adaptive_radius_mix=0.5,
                 adaptive_radius_threshold=0.15, adaptive_radius_temperature=0.05):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.device = device
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.max_timestep = max_timestep
        self.attack_steps = attack_steps
        self.sampling_method = sampling_method
        self.amplitude_cut_range = amplitude_cut_range
        self.phase_cut_range = phase_cut_range
        self.delta=delta
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.radius_gamma = radius_gamma
        self.lambda_amp = lambda_amp
        self.lambda_phase = lambda_phase
        self.guidance_scale = guidance_scale
        self.guidance_start_ratio = guidance_start_ratio
        self.use_guidance = use_guidance
        self.use_adaptive_radius = use_adaptive_radius
        self.adaptive_radius_mix = adaptive_radius_mix
        self.adaptive_radius_threshold = adaptive_radius_threshold
        self.adaptive_radius_temperature = adaptive_radius_temperature
        if not 0.0 <= self.adaptive_radius_mix <= 1.0:
            raise ValueError('adaptive_radius_mix must be between 0 and 1')
        if self.adaptive_radius_temperature <= 0:
            raise ValueError('adaptive_radius_temperature must be positive')
        assert sampling_method in ['ddim', 'ddpm']
        if self.sampling_method == 'ddim':
            self.eta = 0
        elif self.sampling_method == 'ddpm':
            self.eta = 1
        self.is_imagenet = is_imagenet
        self.forward_noise_steps = forward_noise_steps

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def get_dynamic_radius(self, t, total_t):
        if total_t <= 0:
            return float(self.radius_max)
        progress = 1.0 - float(t) / float(total_t)
        progress = max(0.0, min(1.0, progress))
        return self.radius_min + (self.radius_max - self.radius_min) * (progress ** self.radius_gamma)

    def get_guidance_multiplier(self, t, total_t):
        if total_t <= 0:
            return 0.0
        normalized_t = float(t) / float(total_t)
        if normalized_t < self.guidance_start_ratio:
            return 0.0
        active_span = max(1e-8, 1.0 - self.guidance_start_ratio)
        return (normalized_t - self.guidance_start_ratio) / active_span

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if type(t) == int:
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x

    def denoising_process(self,ori_x, x, seq):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = x
        for i, j in zip(reversed(seq), reversed(seq_next)):

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            dynamic_radius = self.get_dynamic_radius(i, max(seq))
            x0_t = self.amplitude_phase_exchange_torch(ori_x, x0_t, freq_radius=dynamic_radius)

            if self.use_guidance:
                guidance_multiplier = self.get_guidance_multiplier(i, max(seq))
                if guidance_multiplier > 0:
                    x0_t, _ = self.apply_frequency_guidance(
                        ori_x, x0_t, i, max(seq), guidance_multiplier
                    )
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt

    def preprocess(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(x_diff, noised_x, self.attack_steps[i])

        x_clf = diff2clf(x_diff)
        return x_clf

    def classify(self, x):
        logits = self.clf(x)
        return logits

    def forward(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(x_diff,noised_x, self.attack_steps[i])

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        else:
            x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return logits
    

    def compute_fft(self,image):  
        amplitude_channels = []  
        phase_channels = []  

        for channel in range(3):  
            f = torch.fft.fft2(image[channel, :, :])  
            fshift = torch.fft.fftshift(f)  
            amplitude = torch.abs(fshift)  
            amplitude_channels.append(amplitude)  
            phase = torch.angle(fshift)  
            phase_channels.append(phase + torch.pi)  

        return amplitude_channels, phase_channels

    def compute_fft_tensor(self, x):
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        amplitude = torch.abs(fft)
        phase = torch.angle(fft)
        return amplitude, phase

    def build_frequency_mask(self, h, w, radius, device):
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        cy, cx = h // 2, w // 2
        dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
        return (dist <= radius).float()[None, None, :, :]

    def build_batch_frequency_mask(self, h, w, radius, device):
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        cy, cx = h // 2, w // 2
        dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
        if not torch.is_tensor(radius):
            radius = torch.tensor([radius], device=device)
        return (dist[None, None, :, :] <= radius.reshape(-1, 1, 1, 1)).float()

    def wrap_phase(self, phase_diff):
        return torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

    def get_adaptive_radius(self, base_radius, amplitude_ref, phase_ref, amplitude_pred, phase_pred):
        batch, channel, height, width = amplitude_ref.shape
        base_radius = torch.full(
            (batch,), float(base_radius), device=amplitude_ref.device, dtype=amplitude_ref.dtype
        )
        if not self.use_adaptive_radius:
            return base_radius

        max_radius_mask = self.build_batch_frequency_mask(
            height, width, self.radius_max, amplitude_ref.device
        )
        num_values = max_radius_mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * channel

        log_amplitude_ref = torch.log1p(amplitude_ref)
        log_amplitude_pred = torch.log1p(amplitude_pred)
        amplitude_diff = torch.abs(log_amplitude_pred - log_amplitude_ref)
        amplitude_diff = amplitude_diff / log_amplitude_ref.abs().clamp_min(1e-6)
        phase_diff = torch.abs(self.wrap_phase(phase_pred - phase_ref)) / torch.pi

        amplitude_score = (amplitude_diff * max_radius_mask).sum(dim=(1, 2, 3)) / num_values
        phase_score = (phase_diff * max_radius_mask).sum(dim=(1, 2, 3)) / num_values
        score_weight = max(1e-8, self.lambda_amp + self.lambda_phase)
        frequency_score = (
            self.lambda_amp * amplitude_score + self.lambda_phase * phase_score
        ) / score_weight

        temperature = max(1e-8, self.adaptive_radius_temperature)
        consistency = torch.sigmoid((self.adaptive_radius_threshold - frequency_score) / temperature)
        spectral_radius = self.radius_min + (self.radius_max - self.radius_min) * consistency
        radius = (1.0 - self.adaptive_radius_mix) * base_radius + self.adaptive_radius_mix * spectral_radius
        return torch.clamp(radius, min=self.radius_min, max=self.radius_max)

    def low_pass_exchange(self,amplitude_channels, amplitude_channels_0_t, cut_range=None):
        filtered_amplitude_channels = []
        if cut_range is None:
            cut_range = self.amplitude_cut_range
        for i in range(3):
            rows, cols = amplitude_channels[i].shape
            u = np.arange(-cols // 2, cols // 2)
            v = np.arange(-rows // 2, rows // 2)
            U, V = np.meshgrid(u, v)
            frequency_map = np.sqrt(U ** 2 + V ** 2)
            low_frequency = (frequency_map <= cut_range)
            low_frequency = torch.from_numpy(low_frequency).to(self.device)
            amplitude_channels_0_t[i] = torch.where(low_frequency, amplitude_channels[i], amplitude_channels_0_t[i])
            filtered_amplitude_channels.append(amplitude_channels_0_t[i])
        return filtered_amplitude_channels


    def phase_low_pass_exchange(self,phase_channels, phase_channels_0_t, cut_range=None):
        filtered_amplitude_channels = []
        if cut_range is None:
            cut_range = self.phase_cut_range
        for i in range(3):
            rows, cols = phase_channels[i].shape
            u = np.arange(-cols // 2, cols // 2)
            v = np.arange(-rows // 2, rows // 2)
            U, V = np.meshgrid(u, v)
            frequency_map = np.sqrt(U ** 2 + V ** 2)
            low_frequency = (frequency_map <= cut_range)
            low_frequency = torch.from_numpy(low_frequency).to(self.device)
            phase_channels_0_t[i] = torch.where(low_frequency, phase_channels[i], phase_channels_0_t[i])
            phase_channels_0_t[i][low_frequency] = torch.clip(phase_channels_0_t[i][low_frequency],phase_channels[i][low_frequency]-self.delta,phase_channels[i][low_frequency]+self.delta)
            filtered_amplitude_channels.append(phase_channels_0_t[i])
        return filtered_amplitude_channels
    

    
    def phase_exchange(self,phase_channels,phase_channels_0_t):
        exchanged_phase_channels = []
        for i in range(3):
            rows, cols = phase_channels[i].shape
            exchange_matrix = self.generate_frequency_exchange_matrix(rows, cols)
            phase_channels_0_t[i][exchange_matrix] = phase_channels[i][exchange_matrix]
            exchanged_phase_channels.append(phase_channels_0_t[i])
        return exchanged_phase_channels
    
    def phase_clip(self,phase_channels,phase_channels_0_t,delta=0.6):
        phase_channels_clip=[]
        for i in range(3):
            phase_channels_clip.append(np.clip(phase_channels_0_t[i],phase_channels[i]-delta,phase_channels[i]+delta))
        return phase_channels_clip

    
    def reconstruct_image(self,filtered_amplitude_channels, phase_channels):
        reconstructed_image = []
        for channel in range(3):
            amplitude = filtered_amplitude_channels[channel]
            phase = phase_channels[channel]-torch.pi
            fshift_filtered = amplitude * torch.exp(1j * phase)
            f_ishift = torch.fft.ifftshift(fshift_filtered)
            img_reconstructed = torch.fft.ifft2(f_ishift)
            img_reconstructed = torch.abs(img_reconstructed)
            img_reconstructed = torch.clip(img_reconstructed,0,255)
            reconstructed_image.append(img_reconstructed/255)
        return torch.stack(reconstructed_image,dim=2)

        
    
    def apply_frequency_guidance(self, ori_x, x0_t, t, total_t, guidance_multiplier):
        with torch.enable_grad():
            x_ref = torch.clamp(diff2clf(ori_x), 0.0, 1.0).detach()
            x_pred = torch.clamp(diff2clf(x0_t), 0.0, 1.0).detach().requires_grad_(True)

            _, _, h, w = x_pred.shape
            amp_ref, phase_ref = self.compute_fft_tensor(x_ref)
            amp_pred, phase_pred = self.compute_fft_tensor(x_pred)
            base_radius = self.get_dynamic_radius(t, total_t)
            radius = self.get_adaptive_radius(base_radius, amp_ref, phase_ref, amp_pred, phase_pred)
            low_freq_mask = self.build_batch_frequency_mask(h, w, radius, x_pred.device)

            amp_loss = torch.mean(low_freq_mask * torch.abs(torch.log1p(amp_pred) - torch.log1p(amp_ref)))
            phase_loss = torch.mean(low_freq_mask * torch.abs(self.wrap_phase(phase_pred - phase_ref)))
            loss = self.lambda_amp * amp_loss + self.lambda_phase * phase_loss

            grad = torch.autograd.grad(loss, x_pred, retain_graph=False, create_graph=False)[0]
            step_size = self.guidance_scale * guidance_multiplier
            x_guided = torch.clamp(x_pred - step_size * grad, 0.0, 1.0)

        return clf2diff(x_guided.detach()), loss.detach()

    def amplitude_phase_exchange_torch(self, x, x_0_t, freq_radius=None):
        if freq_radius is None:
            freq_radius = self.amplitude_cut_range
        x_t = self.get_noised_x(x, self.forward_noise_steps)
        t = (torch.ones(x.size(0)) * self.forward_noise_steps).to(x.device)
        at = self.compute_alpha(t.long())
        et = self.diffusion(x_t, t)
        if self.is_imagenet:
            et, _ = torch.split(et, 3, dim=1)
        x =  (x_t - et * (1 - at).sqrt()) / at.sqrt()
        # save_image(diff2clf(x), 'new_x_0_t.png')
        x = torch.clamp(diff2clf(x), 0.0, 1.0)
        x_0_t = torch.clamp(diff2clf(x_0_t), 0.0, 1.0)

        _, _, height, width = x.shape
        amplitude_ref, phase_ref = self.compute_fft_tensor(x)
        amplitude_pred, phase_pred = self.compute_fft_tensor(x_0_t)
        radius = self.get_adaptive_radius(
            freq_radius, amplitude_ref, phase_ref, amplitude_pred, phase_pred
        )
        low_freq_mask = self.build_batch_frequency_mask(height, width, radius, x.device).bool()

        amplitude = torch.where(low_freq_mask, amplitude_ref, amplitude_pred)
        phase = torch.where(low_freq_mask, phase_ref, phase_pred)
        fft = amplitude * torch.exp(1j * phase)
        new_x_0_t = torch.fft.ifft2(torch.fft.ifftshift(fft, dim=(-2, -1)), dim=(-2, -1)).abs()
        return clf2diff(torch.clamp(new_x_0_t.float(), 0.0, 1.0))

    def get_img_logits(self, x):

        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(x_diff,noised_x, self.attack_steps[i])

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        else:
            x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return x_clf,logits
