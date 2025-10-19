import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from autoattack import AutoAttack

import sys
sys.path.append('./edm')
from utils import *



class PairTensorDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        # Ensure both tensors have the same length
        assert tensor1.size(0) == tensor2.size(0), "Tensors must have the same length"
        
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def __len__(self):
        return len(self.tensor1)

    def __getitem__(self, idx):
        item1 = self.tensor1[idx]
        item2 = self.tensor2[idx]
        return item1, item2
    
class PredModel(nn.Module):
    def __init__(self, classifier, trans_to_clf):
        super(PredModel, self).__init__()
        self.trans_to_clf = trans_to_clf
        self.classifier = classifier
        
    def forward(self, x):
        x = self.trans_to_clf(x)
        logits = self.classifier(x)
        return logits
    

def gen_adv(batch_size, pred_model, device, args, norm="Linf", eps=2/255, attacks_to_run=["apgd-ce"]):
    transform = transforms.Compose([transforms.transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=False, download=True, transform=transform)

    all_train_images, all_train_targets = torch.tensor(train_dataset.data).float(), torch.tensor(train_dataset.targets)
    all_test_images, all_test_targets = torch.tensor(test_dataset.data).float(), torch.tensor(test_dataset.targets)
    
    all_train_images, all_test_images = all_train_images.permute(0, 3, 1, 2) / all_train_images.max(), all_test_images.permute(0, 3, 1, 2) / all_test_images.max()
    print("all_train_images shape: ", all_train_images.shape, all_train_images.max(), all_train_images.min(), flush=True)
    
    print("Creating adversarial examples", flush=True)
    attack = AutoAttack(
                pred_model, norm=norm, eps=eps,
                attacks_to_run=attacks_to_run,
                version='custom',
                verbose=True,
                device=device
            )
    
    adv_train_dataset = attack.run_standard_evaluation(all_train_images, all_train_targets, bs=batch_size)
    adv_test_dataset = attack.run_standard_evaluation(all_test_images, all_test_targets, bs=batch_size)
    print("Adversarial examples created", flush=True)
    print("saving all adversarial examples", flush=True)
    
    if norm == 'Linf':
        torch.save(adv_train_dataset, f'./adv_data/adv_train_dataset_CIF100_eps_{int(eps*255)}_norm_{norm}_{args.clf_name}.pt')
        torch.save(adv_test_dataset, f'./adv_data/adv_test_dataset_CIF100_eps_{int(eps*255)}_norm_{norm}_{args.clf_name}.pt')
    else:
        torch.save(adv_train_dataset, f'./adv_data/adv_train_dataset_CIF100_eps_{eps}_norm_{norm}_{args.clf_name}.pt')
        torch.save(adv_test_dataset, f'./adv_data/adv_test_dataset_CIF100_eps_{eps}_norm_{norm}_{args.clf_name}.pt')
    
    paired_train_dataset = PairTensorDataset(all_train_images, adv_train_dataset)
    paired_test_dataset = PairTensorDataset(all_test_images, adv_test_dataset)
    
    print("saving samples from the pair dataset", flush=True)
    image_0, adv_image_0 = paired_test_dataset[0]
    image_0, adv_image_0 = image_0.numpy().transpose((1, 2, 0)), adv_image_0.numpy().transpose((1, 2, 0))
    image_0, adv_image_0 = (image_0 * 255).astype(np.uint8), (adv_image_0 * 255).astype(np.uint8)

    # Plot the image
    plt.imshow(image_0)
    plt.title(f"Label: {all_test_targets[1]}")
    plt.axis('off')
    plt.savefig('sample_image.png', bbox_inches='tight')
    
    plt.imshow(adv_image_0)
    plt.title(f"Label: {all_test_targets[1]}")
    plt.axis('off')
    plt.savefig('sample_image_adv.png', bbox_inches='tight')
    
    plt.close()
    
    return


def run(args):
    device = torch.device("cuda" )
    print("device: ", torch.cuda.get_device_properties(device), flush=True)
    
    clf = load_classifier(args.dataset, args.clf_name, device).to(device)
    clf.eval()
    trans_to_clf = get_transforms(args.dataset, args.clf_name)
    
    pred_model = PredModel(clf, trans_to_clf)
    
    print("norm: ", args.norm, flush=True)
    print("eps: ", args.eps, flush=True)
    
    if args.norm == "Linf":
        gen_adv(args.batch_size, pred_model, device, args, norm="Linf", eps=args.eps/255, attacks_to_run=["apgd-ce"])
    elif args.norm == "L2":
        gen_adv(args.batch_size, pred_model, device, args, norm="L2", eps=args.eps, attacks_to_run=["apgd-ce"])
    
    raise Exception("STOP")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch size for training [default: 512]')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda num [default: 0]')
    parser.add_argument('--clf-name', type=str, default='wideresnet-28-10',
                        help='classifier name [default: wideresnet-28-10]')
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        help='dataset name [default: CIFAR100]')
    parser.add_argument('--norm', type=str, default="Linf",
                        help="Linf or L2")
    parser.add_argument('--eps', type=float, default=2,
                        help="2, 4, 8 ... if Linf or 0.5, 1, ... if L2")
    args = parser.parse_args()

    run(args)


