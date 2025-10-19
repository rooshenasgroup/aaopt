import os
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils.transforms import *

def edm_sampler_multistep(
    init_x, net, diffusion_steps, num_steps=18, class_labels=None, randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1
):
    with torch.no_grad():
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=init_x.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        
        # Main sampling loop.
        n = torch.randn_like(init_x) * t_steps[diffusion_steps]
        x_next = (init_x + n).to(torch.float64)
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[diffusion_steps:-1], t_steps[diffusion_steps+1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

    return x_next.to(torch.float32)

def purify_x_edm_multistep(X, score, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(edm_sampler_multistep(X, score, int(args.forward_steps), args.total_steps)),0.0,1.0)
    
    return purif_X_re

def edm_sampler_one_shot(init_x, net, t, num_steps, class_labels=None):

    t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
    n = torch.randn_like(init_x) * t
    x_t = (init_x + n).to(torch.float64)
    denoised = net(x_t, t, class_labels)

    return denoised

def purify_x_edm_one_shot(X, score, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(edm_sampler_one_shot(X, score, args.forward_steps, args.total_steps)),0.0,1.0)
    
    return purif_X_re


class ImageModel(nn.Module):
    def __init__(self, init_x):
        super().__init__()
        self.img = nn.Parameter(init_x.clone())
        self.img.requires_grad = True

    def encode(self):
        return self.img
    

def adam_step(param, grad, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    m, v, t = state                     # tensors, all same device/dtype as param
    t = t + 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad.square()
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param_new = param - lr * m_hat / (v_hat.sqrt() + eps)
    return param_new, (m, v, t)

def sampler_opt_aaopt_1iter(
    init_x, net, sdiff_net, sdiff_sampler, init_t, num_steps, args, class_labels=None, randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7
):
    with torch.enable_grad():

        sample_img = init_x.clone().requires_grad_(True)
        state = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), 0
        
        starter_img = init_x.clone().detach()
        t_div = init_t / args.purify_iter
        
        for i in range(args.apprx_iter):
            if args.seq_t == 1:
                t = torch.tensor(init_t, dtype=torch.float64, device=init_x.device)
                init_t = init_t - t_div
            else:
                t = random.uniform(init_t - 0.1, init_t + 0.1)
                t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
            
            eps = torch.randn_like(init_x)
            sample_img_t = sample_img + eps * t
            
            denoised = net(sample_img_t, t, class_labels)
            
            if args.diff_reversed == 1:
                diff = starter_img - sample_img
            else:
                diff = sample_img - starter_img
            int_diff_t = (torch.ones((init_x.shape[0],)).to(init_x.device) * t * 1000).long()
            diff_epsilon = torch.randn_like(diff)
            diff_t = sdiff_sampler.q_sample(x_start=diff, t=int_diff_t, noise=diff_epsilon)
            score_diff_e_t = sdiff_net(diff_t, int_diff_t)
            
            eps_hat = (sample_img_t - denoised) / t
            
            if args.no_stop_grad == 1:
                loss = torch.mul((eps_hat - eps), sample_img).mean() + args.sdiff_cons * torch.mul((score_diff_e_t - diff_epsilon), sample_img).mean()
            else:
                loss = torch.mul((eps_hat - eps).detach(), sample_img).mean() + args.sdiff_cons * torch.mul((score_diff_e_t - diff_epsilon).detach(), sample_img).mean()
                                    
            grad = torch.autograd.grad(loss, sample_img, create_graph=True)[0]            
            sample_img, state = adam_step(sample_img, grad, state, args.lr)

    return sample_img


def sampler_opt_aaopt(
    init_x, net, sdiff_net, sdiff_sampler, init_t, num_steps, args, class_labels=None, randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7
):
    with torch.enable_grad():

        sample_img = init_x.clone().requires_grad_(True)
        state = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), 0
        
        starter_img = init_x.clone().detach()
        t_div = init_t / args.purify_iter
        
        for i in range(args.purify_iter):
            if args.seq_t == 1:
                t = torch.tensor(init_t, dtype=torch.float64, device=init_x.device)
                init_t = init_t - t_div
            else:
                t = random.uniform(init_t - 0.1, init_t + 0.1)
                t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
                
            eps = torch.randn_like(init_x)
            sample_img_t = sample_img + eps * t
    
            denoised = net(sample_img_t, t, class_labels)
            
            if args.diff_reversed == 1:
                diff = starter_img - sample_img
            else:
                diff = sample_img - starter_img
            int_diff_t = (torch.ones((init_x.shape[0],)).to(init_x.device) * t * 1000).long()
            diff_epsilon = torch.randn_like(diff)
            diff_t = sdiff_sampler.q_sample(x_start=diff, t=int_diff_t, noise=diff_epsilon)
            score_diff_e_t = sdiff_net(diff_t, int_diff_t)
            
            eps_hat = (sample_img_t - denoised) / t
    
            loss = torch.mul((eps_hat - eps).detach(), sample_img).mean() + args.sdiff_cons * torch.mul((score_diff_e_t - diff_epsilon).detach(), sample_img).mean()
                        
            if args.use_full_steps:
                grad = torch.autograd.grad(loss, sample_img, create_graph=True)[0]
            else:
                grad = torch.autograd.grad(loss, sample_img)[0]
            sample_img, state = adam_step(sample_img, grad, state, args.lr)

    return sample_img


def sampler_opt_x0_1iter(
    init_x, net, init_t, num_steps, args, class_labels=None, randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7
):
    with torch.enable_grad():

        sample_img = init_x.clone().requires_grad_(True)
        state = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), 0        
        
        for i in range(args.apprx_iter):
            t = random.uniform(init_t - 0.1, init_t + 0.1)
            t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
            eps = torch.randn_like(init_x)
            sample_img_t = sample_img + eps * t
            
            denoised = net(sample_img_t, t, class_labels)
            loss = F.mse_loss(denoised, sample_img)
                    
            grad = torch.autograd.grad(loss, sample_img, create_graph=True)[0]            
            sample_img, state = adam_step(sample_img, grad, state, args.lr)

    return sample_img

def sampler_opt_x0(
    init_x, net, init_t, num_steps, args, class_labels=None, randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7
):
    with torch.enable_grad():

        sample_img = init_x.clone().requires_grad_(True)
        state = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), 0
                    
        for i in range(args.purify_iter):
            t = random.uniform(init_t - 0.1, init_t + 0.1)
            t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
                
            eps = torch.randn_like(init_x)
            sample_img_t = sample_img + eps * t
            
            denoised = net(sample_img_t, t, class_labels)
            loss = F.mse_loss(denoised, sample_img)
            
            if args.use_full_steps:
                grad = torch.autograd.grad(loss, sample_img, create_graph=True)[0]
            else:
                grad = torch.autograd.grad(loss, sample_img)[0]
            sample_img, state = adam_step(sample_img, grad, state, args.lr)

    return sample_img


def sampler_opt_xt(
    init_x, net, t, num_steps, args, class_labels=None
):
    t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
    eps = torch.randn_like(init_x)
    x_t = init_x + eps * t

    model = ImageModel(x_t).to(init_x.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for i in range(args.purify_iter):

        sample_img = model.encode()
        denoised = net(sample_img, t, class_labels)
        eps_hat = (sample_img - denoised) / t
        
        loss = F.mse_loss(eps_hat, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()

    sample_img = model.encode().detach()
    denoised = net(sample_img, t, class_labels)

    return denoised

def purify_x_opt_aaopt(X, score, sdiff_net, sdiff_sampler, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(sampler_opt_aaopt(X, score, sdiff_net, sdiff_sampler, args.forward_steps, args.total_steps, args)),0.0,1.0)
    return purif_X_re

def purify_x_opt_aaopt_1iter(X, score, sdiff_net, sdiff_sampler, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(sampler_opt_aaopt_1iter(X, score, sdiff_net, sdiff_sampler, args.forward_steps, args.total_steps, args)),0.0,1.0)
    return purif_X_re
    
def purify_x_opt_x0(X, score, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(sampler_opt_x0(X, score, args.forward_steps, args.total_steps, args)),0.0,1.0)
    return purif_X_re

def purify_x_opt_x0_1iter(X, score, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(sampler_opt_x0_1iter(X, score, args.forward_steps, args.total_steps, args)),0.0,1.0)
    return purif_X_re

def purify_x_opt_xt(X, score, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(sampler_opt_xt(X, score, args.forward_steps, args.total_steps, args)),0.0,1.0)
    return purif_X_re