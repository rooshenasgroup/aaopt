import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
from typing import List
from inspect import isfunction

# --------------------------------------------

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDIMSampler():
    def __init__(self, model, device, n_steps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0., model_steps: int=1000, linear_start=0.0001, linear_end=0.02):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        :param n_steps: is the number of DDIM sampling steps, $S$
        :param ddim_discretize: specifies how to extract $\tau$ from $[1,2,\dots,T]$.
            It can be either `uniform` or `quad`.
        :param ddim_eta: is $\eta$ used to calculate $\sigma_{\tau_i}$. $\eta = 0$ makes the
            sampling process deterministic.
        """
        self.model = model
        self.device = device
        # Number of steps, $T$
        self.n_steps = model_steps
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.beta_0 = linear_start * self.n_steps
        self.beta_1 = linear_end * self.n_steps

        self.beta = torch.linspace(linear_start, linear_end, self.n_steps, dtype=torch.float64)
        alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)

        # Calculate $\tau$ to be uniformly distributed across $[1,2,\dots,T]$
        if ddim_discretize == 'uniform':
            c = self.n_steps // n_steps
            self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1
        # Calculate $\tau$ to be quadratically distributed across $[1,2,\dots,T]$
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps * .8), n_steps)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_bar).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alpha_bar).to(self.device)
        
        with torch.no_grad():
            # Get ${\color{lightgreen}\bar\alpha_t}$
            alpha_bar = self.alpha_bar

            # $\alpha_{\tau_i}$
            self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
            # $\sqrt{\alpha_{\tau_i}}$
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            # $\alpha_{\tau_{i-1}}$
            self.ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])

            # $$\sigma_{\tau_i} =
            # \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
            # \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$
            self.ddim_sigma = (ddim_eta *
                               ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)

            # $\sqrt{1 - \alpha_{\tau_i}}$
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise).float()
    
    @torch.no_grad()
    def get_diffusion(self, x_start, t):
        beta_t = extract_into_tensor(self.beta.to(x_start.device), t, x_start.shape)
        diffusion = torch.sqrt(beta_t)
        return diffusion
    
    def get_std(self, x, t):
        beta_0 = self.linear_start * self.n_steps
        beta_1 = self.linear_end * self.n_steps
        t = t / self.n_steps
        log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return std

    # @torch.no_grad()
    # def get_std(self, x_start, t):
    #     std = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    #     return std
    
    def sample_grad(self,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: torch.Tensor = None,
               uncond_scale: float = 1.,
               uncond_cond: torch.Tensor = None,
               skip_steps: int = 0,
               ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_{\tau_S}$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $i'$. We start sampling from $S - i'$.
            And `x_last` is then $x_{\tau_{S - i'}}$.
        """

        # Get device and batch size
        bs = shape[0]

        # Get $x_{\tau_S}$
        x = x_last if x_last is not None else torch.randn(shape, device=self.device)

        # Time steps to sample at $\tau_{S - i'}, \tau_{S - i' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in enumerate(time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, pred_x0, e_t = self.p_sample_grad(x, cond, ts, step, index=index,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale,
                                            uncond_cond=uncond_cond)

        # Return $x_0$
        return x
    
    def p_sample_grad(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int, *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1.,
                 uncond_cond: torch.Tensor = None):
        """
        ### Sample $x_{\tau_{i-1}}$

        :param x: is $x_{\tau_i}$ of shape `[batch_size, channels, height, width]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $\tau_i$ of shape `[batch_size]`
        :param step: is the step $\tau_i$ as an integer
        :param index: is index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        
        e_t = self.model(x, t, c)

        # Calculate $x_{\tau_{i - 1}}$ and predicted $x_0$
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x,
                                                      temperature=temperature,
                                                      repeat_noise=repeat_noise)

        #
        return x_prev, pred_x0, e_t

    @torch.no_grad()
    def sample(self,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: torch.Tensor = None,
               uncond_scale: float = 1.,
               uncond_cond: torch.Tensor = None,
               skip_steps: int = 0,
               ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_{\tau_S}$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $i'$. We start sampling from $S - i'$.
            And `x_last` is then $x_{\tau_{S - i'}}$.
        """

        # Get device and batch size
        bs = shape[0]

        # Get $x_{\tau_S}$
        x = x_last if x_last is not None else torch.randn(shape, device=self.device)

        # Time steps to sample at $\tau_{S - i'}, \tau_{S - i' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in enumerate(time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, pred_x0, e_t = self.p_sample(x, cond, ts, step, index=index,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale,
                                            uncond_cond=uncond_cond)

        # Return $x_0$
        return x

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int, *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1.,
                 uncond_cond: torch.Tensor = None):
        """
        ### Sample $x_{\tau_{i-1}}$

        :param x: is $x_{\tau_i}$ of shape `[batch_size, channels, height, width]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $\tau_i$ of shape `[batch_size]`
        :param step: is the step $\tau_i$ as an integer
        :param index: is index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        
        e_t = self.model(x, t, c)

        # Calculate $x_{\tau_{i - 1}}$ and predicted $x_0$
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x,
                                                      temperature=temperature,
                                                      repeat_noise=repeat_noise)

        #
        return x_prev, pred_x0, e_t

    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index: int, x: torch.Tensor, *,
                               temperature: float,
                               repeat_noise: bool):
        """
        ### Sample $x_{\tau_{i-1}}$ given $\epsilon_\theta(x_{\tau_i})$
        """

        # $\alpha_{\tau_i}$
        alpha = self.ddim_alpha[index]
        # $\alpha_{\tau_{i-1}}$
        alpha_prev = self.ddim_alpha_prev[index]
        # $\sigma_{\tau_i}$
        sigma = self.ddim_sigma[index]
        # $\sqrt{1 - \alpha_{\tau_i}}$
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        # Current prediction for $x_0$,
        # $$\frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}$$
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        # Direction pointing to $x_t$
        # $$\sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i})$$
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        # No noise is added, when $\eta = 0$
        if sigma == 0.:
            noise = 0.
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device)

        # Multiply noise by the temperature
        noise = noise * temperature
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev, pred_x0
    
# -------------------------------------------------------------------------------------
