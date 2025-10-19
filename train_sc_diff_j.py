import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

from utils import save_loss_plot_train_val
from unet_model import UNetModel
from ddim_sampler import DDIMSampler



class PairCleanAdvDataset(Dataset):
    def __init__(self, clean, advs):
        assert len(advs) >= 1, "ADV Tensors must be greater than 0"
        self.clean = clean
        self.advs = advs
        self.len_advs = len(advs)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        rand_adv_idx = torch.randint(0, self.len_advs, (1,)).item()
        return self.clean[idx], self.advs[rand_adv_idx][idx]

def get_train_test_dataloader(batch_size, device, args):
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform)
    elif args.dataset == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=False, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    all_train_images = torch.tensor(train_dataset.data).float().permute(0, 3, 1, 2) / 255.0
    all_test_images = torch.tensor(test_dataset.data).float().permute(0, 3, 1, 2) / 255.0
    
    print("Loading adversarial examples", flush=True)
    
    if args.dataset == 'CIFAR10':
        if args.clf_name == 'wideresnet-28-10':
            # NEW
            # Linf
            adv_train_dataset_eps2 = torch.load('./adv_data/adv_train_dataset_cif10_eps_2_norm_Linf.pt')
            adv_test_dataset_eps2 = torch.load('./adv_data/adv_test_dataset_cif10_eps_2_norm_Linf.pt')
            
            adv_train_dataset_eps8 = torch.load('./adv_data/adv_train_dataset_cif10_eps_8_norm_Linf.pt')
            adv_test_dataset_eps8 = torch.load('./adv_data/adv_test_dataset_cif10_eps_8_norm_Linf.pt')
            
            adv_train_dataset_eps4 = torch.load('./adv_data/adv_train_dataset_cif10_eps_4_norm_Linf.pt')
            adv_test_dataset_eps4 = torch.load('./adv_data/adv_test_dataset_cif10_eps_4_norm_Linf.pt')
            
            # L2
            adv_train_dataset_l2_eps_05 = torch.load('./adv_data/adv_train_dataset_cif10_eps_0.5_norm_L2.pt')
            adv_test_dataset_l2_eps_05 = torch.load('./adv_data/adv_test_dataset_cif10_eps_0.5_norm_L2.pt')
            
            adv_train_dataset_l2_eps_1 = torch.load('./adv_data/adv_train_dataset_cif10_eps_1.0_norm_L2.pt')
            adv_test_dataset_l2_eps_1 = torch.load('./adv_data/adv_test_dataset_cif10_eps_1.0_norm_L2.pt')
        
        elif args.clf_name == 'wideresnet-70-16':
            # Linf
            adv_train_dataset_eps2 = torch.load('./adv_data/adv_train_dataset_cif10_eps_2_norm_Linf_wideresnet-70-16.pt')
            adv_test_dataset_eps2 = torch.load('./adv_data/adv_test_dataset_cif10_eps_2_norm_Linf_wideresnet-70-16.pt')
            
            adv_train_dataset_eps8 = torch.load('./adv_data/adv_train_dataset_cif10_eps_8_norm_Linf_wideresnet-70-16.pt')
            adv_test_dataset_eps8 = torch.load('./adv_data/adv_test_dataset_cif10_eps_8_norm_Linf_wideresnet-70-16.pt')
            
            adv_train_dataset_eps4 = torch.load('./adv_data/adv_train_dataset_cif10_eps_4_norm_Linf_wideresnet-70-16.pt')
            adv_test_dataset_eps4 = torch.load('./adv_data/adv_test_dataset_cif10_eps_4_norm_Linf_wideresnet-70-16.pt')
            
            # L2
            adv_train_dataset_l2_eps_05 = torch.load('./adv_data/adv_train_dataset_cif10_eps_0.5_norm_L2_wideresnet-70-16.pt')
            adv_test_dataset_l2_eps_05 = torch.load('./adv_data/adv_test_dataset_cif10_eps_0.5_norm_L2_wideresnet-70-16.pt')
            
            adv_train_dataset_l2_eps_1 = torch.load('./adv_data/adv_train_dataset_cif10_eps_1.0_norm_L2_wideresnet-70-16.pt')
            adv_test_dataset_l2_eps_1 = torch.load('./adv_data/adv_test_dataset_cif10_eps_1.0_norm_L2_wideresnet-70-16.pt')
            
    elif args.dataset == 'CIFAR100':
        if args.clf_name == 'wideresnet-28-10':
            # Linf
            adv_train_dataset_eps2 = torch.load('./adv_data/adv_train_dataset_CIF100_eps_2_norm_Linf_wideresnet-28-10.pt')
            adv_test_dataset_eps2 = torch.load('./adv_data/adv_test_dataset_CIF100_eps_2_norm_Linf_wideresnet-28-10.pt')
            
            adv_train_dataset_eps8 = torch.load('./adv_data/adv_train_dataset_CIF100_eps_8_norm_Linf_wideresnet-28-10.pt')
            adv_test_dataset_eps8 = torch.load('./adv_data/adv_test_dataset_CIF100_eps_8_norm_Linf_wideresnet-28-10.pt')
            
            adv_train_dataset_eps4 = torch.load('./adv_data/adv_train_dataset_CIF100_eps_4_norm_Linf_wideresnet-28-10.pt')
            adv_test_dataset_eps4 = torch.load('./adv_data/adv_test_dataset_CIF100_eps_4_norm_Linf_wideresnet-28-10.pt')
            
            # L2
            adv_train_dataset_l2_eps_05 = torch.load('./adv_data/adv_train_dataset_CIF100_eps_0.5_norm_L2_wideresnet-28-10.pt')
            adv_test_dataset_l2_eps_05 = torch.load('./adv_data/adv_test_dataset_CIF100_eps_0.5_norm_L2_wideresnet-28-10.pt')
            
            adv_train_dataset_l2_eps_1 = torch.load('./adv_data/adv_train_dataset_CIF100_eps_1.0_norm_L2_wideresnet-28-10.pt')
            adv_test_dataset_l2_eps_1 = torch.load('./adv_data/adv_test_dataset_CIF100_eps_1.0_norm_L2_wideresnet-28-10.pt')
        else:
            raise NotImplementedError(f"Unknown classifier: {args.clf_name}")
    
    # train_dataset = PairCleanAdvDataset(all_train_images, [adv_train_dataset_eps2, adv_train_dataset_eps4, adv_train_dataset_eps8, adv_train_dataset_l2_eps_05, adv_train_dataset_l2_eps_1])
    # test_dataset = PairCleanAdvDataset(all_test_images, [adv_test_dataset_eps2, adv_test_dataset_eps4, adv_test_dataset_eps8, adv_test_dataset_l2_eps_05, adv_test_dataset_l2_eps_1])
    
    # Added clean images to adv list
    train_dataset = PairCleanAdvDataset(all_train_images, [all_train_images, adv_train_dataset_eps2, adv_train_dataset_eps4, adv_train_dataset_eps8, adv_train_dataset_l2_eps_05, adv_train_dataset_l2_eps_1])
    test_dataset = PairCleanAdvDataset(all_test_images, [all_test_images, adv_test_dataset_eps2, adv_test_dataset_eps4, adv_test_dataset_eps8, adv_test_dataset_l2_eps_05, adv_test_dataset_l2_eps_1])
    
    # # Added clean images to adv list
    # train_dataset = PairCleanAdvDataset(all_train_images, [all_train_images + torch.randn_like(all_train_images) * 0.001, adv_train_dataset_eps2, adv_train_dataset_eps4, adv_train_dataset_eps8, adv_train_dataset_l2_eps_05, adv_train_dataset_l2_eps_1])
    # test_dataset = PairCleanAdvDataset(all_test_images, [all_test_images + torch.randn_like(all_test_images) * 0.001, adv_test_dataset_eps2, adv_test_dataset_eps4, adv_test_dataset_eps8, adv_test_dataset_l2_eps_05, adv_test_dataset_l2_eps_1])
    
    for i in range(10):
        print("Diff: ", torch.abs(train_dataset[i][0] - train_dataset[i][1]).max(), flush=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_dataloader, test_dataloader

def cal_diff_loss(denoise_model, x_start, t, sampler, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = sampler.q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    return F.mse_loss(noise, predicted_noise)


def train_model(train_loader, sm_model, sampler, optimizer, device, diff_rev):
    losses = 0
    sm_model.train()
    start_time = time.time()
    scaler = lambda x: 2. * x - 1.
    rev_scaler = lambda x: (x + 1.) / 2.

    for batch_idx, (clean_images, adv_images) in enumerate(train_loader):
        clean_images = scaler(clean_images.to(device))
        adv_images = scaler(adv_images.to(device))
        

        with torch.enable_grad():
            t = torch.randint(0, sampler.n_steps, (clean_images.shape[0],), device=device).long()
            if diff_rev:
                difference = adv_images - clean_images
            else:
                difference = clean_images - adv_images
            
            loss = cal_diff_loss(sm_model, difference.detach(), t, sampler)
            losses += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    losses /= len(train_loader)
    print(f"TRAINING TIME TAKEN: {time.time() - start_time}", flush=True)
    print(f"Training loss: {losses}", flush=True)
    return losses

def evaluate(test_loader, sm_model, sampler, device, diff_rev):
    with torch.no_grad():
        sm_model.eval()
        start_time = time.time()
        losses = 0
        scaler = lambda x: 2. * x - 1.
        rev_scaler = lambda x: (x + 1.) / 2.
        
        for clean_images, adv_images in test_loader:
            clean_images = scaler(clean_images.to(device))
            adv_images = scaler(adv_images.to(device))
            
            
            t = torch.randint(0, sampler.n_steps, (clean_images.shape[0],), device=device).long()
            if diff_rev:
                difference = adv_images - clean_images
            else:
                difference = clean_images - adv_images
            
            loss = cal_diff_loss(sm_model, difference.detach(), t, sampler)
            losses += loss.item()

        losses /= len(test_loader)
        print(f"VALIDATION TIME TAKEN: {time.time() - start_time}", flush=True)
        print(f"Validation loss: {losses}", flush=True)
        return losses

def run(epochs, batch_size, lr, savefolder, unq_name, args):
    unq_name += args.dataset + '_' + args.clf_name + '_diff_rev_' + str(args.diff_rev)
    
    print(f'vars: {epochs}, {batch_size}, {lr}, {savefolder}, {unq_name}', flush=True)
    train_losses, val_losses = [], []

    savefolder += '/'
    save_paths = {
        'model': savefolder,
        'plot': savefolder
    }
    for p in save_paths.values():
        if not os.path.exists(p):
            os.makedirs(p)

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_properties(device)}", flush=True)
    
    score_model = UNetModel(in_channels=3, model_channels=64, 
                            out_channels=3, num_res_blocks=2, attention_resolutions=(16,), 
                            dropout=0.1, channel_mult=(1,2,2,2), num_heads=4)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)
    score_model = score_model.to(device)
    
    score_sampler = DDIMSampler(score_model, device, 50)
    
    train_dataloader, test_dataloader = get_train_test_dataloader(batch_size, device, args)
    
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}", flush=True)

        training_loss = train_model(train_dataloader, score_model, score_sampler, optimizer, device, args.diff_rev)
        validation_loss = evaluate(test_dataloader, score_model, score_sampler, device, args.diff_rev)
        train_losses.append(training_loss)
        val_losses.append(validation_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': score_model.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss,
        }, save_paths['model'] + unq_name)
        print('Model saved', flush=True)
    
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    save_loss_plot_train_val(train_losses, val_losses, 'Loss', ['Train', 'Val'], save_paths['plot'] + '_' + unq_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs to train [default: 3000]')
    parser.add_argument('--lr', type=float, default=0.00002,
                        help='learning rate [default: 0.00002]')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda num [default: 0]')
    parser.add_argument('--savefolder', type=str, default='./pretrained_models/cifar10_ebm_adv_',
                        help='folder name to save output [default: "./pretrained_models/cifar10_ebm_adv_"]')
    parser.add_argument('--unq-name', type=str, default='cifar10_score_DIFF_JOINT_',
                        help='identifier name for saving [default: "cifar10_score_DIFF_JOINT_"]')
    parser.add_argument('--clf-name', type=str, default='wideresnet-28-10',
                        help='identifier name for saving [default: "wideresnet-28-10"]')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset name [default: CIFAR10]')
    parser.add_argument('--diff-rev', type=int, default=1,
                        help='difference reversed [default: 1]')
    
    args = parser.parse_args()
    run(args.epochs, args.batch_size, args.lr, args.savefolder, args.unq_name, args)