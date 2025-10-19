import os
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils.transforms import *

# class ImageModel(nn.Module):
#     def __init__(self, init_x):
#         super().__init__()
#         self.img = nn.Parameter(init_x.clone())
#         self.img.requires_grad = True

#     def encode(self):
#         return self.img
    
    
# 'adv_norm': 'l_inf' if args.att_lp_norm == -1 else 'l_2',
# 'adv_eps': args.att_eps,
# 'adv_eta': args.att_eps / 4.0,

def get_features(model, layer_name):
    features = {}

    def hook_fn(module, input, output):
        features[layer_name] = output

    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)

    return features


def pur_attack(x, y, score, network_clf, trans_to_clf, args, l2_scale=1, use_feat=False, feature_name="block3"):
    print("PUR ATTACK")
    
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    init_x = transform_diff(x)
    # init_t = args.forward_steps
    init_t = 0.25
    iter_len = 25
    # init_t = 0.1
    if use_feat:
        features = get_features(network_clf, feature_name)
        init_logits = network_clf(trans_to_clf(x))
        init_feat = features[feature_name].detach()
        print("init_feat: ", init_feat.shape)      
  
    if args.purify_method == "x0":
    
        sample_img = init_x.clone().detach().requires_grad_(True)
        # opt = torch.optim.Adam([sample_img], lr=args.lr)
        opt = torch.optim.Adam([sample_img], lr=0.1)
        # opt = torch.optim.Adadelta([sample_img], lr=0.1)

        # for i in range(args.purify_iter):
        for i in range(iter_len):
            # sample_img = model.encode()
            t = random.uniform(init_t - 0.1, init_t + 0.1)
            t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
            eps = torch.randn_like(init_x)
            
            # perturbed_img = sample_img + torch.randn_like(sample_img) * noise_scale
            
            sample_img_t = sample_img + eps * t
            denoised = score(sample_img_t, t, None)
            
            loss1 = F.mse_loss(denoised, sample_img)
            
            if not use_feat:
                logits = network_clf(trans_to_clf(transform_raw(denoised)))
                loss2 = F.cross_entropy(logits, y)
            else:
                logits = network_clf(trans_to_clf(transform_raw(denoised)))
                feat = features[feature_name]
                loss2 = F.mse_loss(feat, init_feat)
                
            # loss3 = F.mse_loss(denoised, sample_img_t)
            
            # logits = network_clf(trans_to_clf(transform_raw(denoised)))
            # feat = features[feature_name]
            # loss2 = F.cross_entropy(logits, y) + F.mse_loss(feat, init_feat)
            
            loss = loss1 - l2_scale * loss2
            # loss = (-1*loss1) - (l2_scale * loss2)
            
            if  i == 0 or i == iter_len - 1:
                print("Loss1: ", loss1.item(), "Loss2: ", l2_scale, " * ", loss2.item())
                print("Total loss: ", loss.item())
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # sample_img = torch.min(init_x + args.att_eps, torch.max(init_x - args.att_eps, sample_img.detach()))
            # sample_img.requires_grad_(True)
            # opt = torch.optim.Adam([sample_img], lr=args.lr)
            
            with torch.no_grad():
                sample_img.data = torch.clamp(
                    sample_img.data,
                    min=init_x - args.att_eps,
                    max=init_x + args.att_eps,
                )
            
        # sample_img = model.encode().detach()
        sample_img = sample_img.detach()
        x_out = torch.clamp(transform_raw(sample_img),0.0,1.0)

        return x_out

    elif args.purify_method == "xt":
        
        t = torch.tensor(init_t, dtype=torch.float64, device=init_x.device)
        eps = torch.randn_like(init_x)
        x_t = init_x + eps * t
        x_t.requires_grad_(True)
        opt = torch.optim.Adam([x_t], lr=0.1)

        # for i in range(args.purify_iter):
        for i in range(iter_len):
            denoised = score(x_t, t, None)
            eps_hat = (x_t - denoised) / t
            loss1 = F.mse_loss(eps_hat, eps)
            
            if not use_feat:
                logits = network_clf(trans_to_clf(transform_raw(denoised)))
                loss2 = F.cross_entropy(logits, y)
            else:
                logits = network_clf(trans_to_clf(transform_raw(denoised)))
                feat = features[feature_name]
                loss2 = F.mse_loss(feat, init_feat)
            
            if  i == 0 or i == iter_len - 1:
                print("Loss1: ", loss1.item(), "Loss2: ", l2_scale, " * ", loss2.item())
            
            loss = loss1 - l2_scale * loss2
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            # x_t = torch.min(init_x + args.att_eps, torch.max(init_x - args.att_eps, x_t.detach()))
            # x_t.requires_grad_(True)
            # opt = torch.optim.Adam([x_t], lr=args.lr)
            
            with torch.no_grad():
                x_t.data = torch.clamp(
                    x_t.data,
                    min=init_x - args.att_eps,
                    max=init_x + args.att_eps,
                )
            
        # sample_img = model.encode().detach()
        x_t = x_t.detach()
        x_out = torch.clamp(transform_raw(x_t),0.0,1.0)

        return x_out
    else:
        raise Exception("Invalid purify_method")




    
# def sampler_opt_x0(
#     init_x, net, init_t, num_steps, args, class_labels=None, randn_like=torch.randn_like,
#     sigma_min=0.002, sigma_max=80, rho=7
# ):

#     model = ImageModel(init_x).to(init_x.device)
#     opt = torch.optim.Adam(model.parameters(), lr=args.lr)

#     for i in range(args.purify_iter):
#         sample_img = model.encode()
#         t = random.uniform(init_t - 0.1, init_t + 0.1)
#         t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
#         eps = torch.randn_like(init_x)
#         sample_img_t = sample_img + eps * t
#         denoised = net(sample_img_t, t, class_labels)
#         # eps_hat = (sample_img_t - denoised) / t

#         # eps_1 = torch.randn_like(init_x)
#         # sample_img_t_1 = init_x + eps_1 * t
#         # denoised_1 = net(sample_img_t_1, t, class_labels)
#         # eps_hat_1 = (sample_img_t_1 - denoised) / t

#         # loss = F.mse_loss(denoised, sample_img) + args.loss_lambda * F.mse_loss(denoised, denoised_1) # SR
#         # loss = F.mse_loss(denoised, sample_img) + args.loss_lambda * F.mse_loss(sample_img, init_x) # MSE
#         loss = F.mse_loss(denoised, sample_img) # Diff
#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#     sample_img = model.encode().detach()

#     return sample_img

# def sampler_opt_xt(
#     init_x, net, t, num_steps, args, class_labels=None
# ):
#     t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
#     eps = torch.randn_like(init_x)
#     x_t = init_x + eps * t

#     model = ImageModel(x_t).to(init_x.device)
#     opt = torch.optim.Adam(model.parameters(), lr=args.lr)

#     for i in range(args.purify_iter):

#         sample_img = model.encode()
#         denoised = net(sample_img, t, class_labels)
#         eps_hat = (sample_img - denoised) / t
#         # loss = F.mse_loss(denoised, sample_img)
#         loss = F.mse_loss(eps_hat, eps)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#     sample_img = model.encode().detach()
#     denoised = net(sample_img, t, class_labels)

#     return denoised

# def purify_x_opt_x0(X, score, args):
#     transform_diff = raw_to_diff(args.dataset)
#     transform_raw = diff_to_raw(args.dataset)
#     X = transform_diff(X)
#     purif_X_re = torch.clamp(transform_raw(sampler_opt_x0(X, score, args.forward_steps, args.total_steps, args)),0.0,1.0)
#     return purif_X_re

# def purify_x_opt_xt(X, score, args):
    # transform_diff = raw_to_diff(args.dataset)
    # transform_raw = diff_to_raw(args.dataset)
    # X = transform_diff(X)
    # purif_X_re = torch.clamp(transform_raw(sampler_opt_xt(X, score, args.forward_steps, args.total_steps, args)),0.0,1.0)
    # return purif_X_re