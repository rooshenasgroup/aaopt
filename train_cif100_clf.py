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
from tqdm import tqdm
from clf_models.cifar10 import WideResNet


sys.path.append("/home/*2/*/")
from my_utils import save_loss_plot_train_val



# Model selection
def create_model(clf_type, device):
    if clf_type == 'wideresnet-28-10':
        # return DMWideResNet(num_classes=100, depth=28, width=10, activation_fn=Swish).to(device)
        return WideResNet(depth=28, widen_factor=10, dropRate=0.3, num_classes=100)
        
    elif clf_type == 'wideresnet-70-16':
        # return DMWideResNet(num_classes=100, depth=70, width=16, activation_fn=Swish).to(device)
        return WideResNet(depth=70, widen_factor=16, dropRate=0.3, num_classes=100)
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}. Use 'wrn-28-10' or 'wrn-70-16'.")
    
    
# Train
def train(trainloader, model, optimizer, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total = correct = 0
    running_loss = 0.0

    for images, labels in tqdm(trainloader, desc=f"Training Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch}] Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
    return avg_loss

# Evaluate
def evaluate(testloader, model, device):
    model.eval()
    correct = total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(testloader)
    print(f"Test Accuracy: {acc:.2f}%")
    return avg_loss


def run(epochs, batch_size, lr, savefolder, unq_name, args):
    unq_name += args.clf_name + '_'
    print(f'vars: {epochs}, {batch_size}, {lr}, {savefolder}, {unq_name}', flush=True)
    train_losses, val_losses = [], []

    savefolder += '/'
    save_paths = {
        'model': '/Data-HDD/*/models/' + savefolder,
        'plot': '/home/*2/*/plots/' + savefolder
    }
    
    for p in save_paths.values():
        if not os.path.exists(p):
            os.makedirs(p)

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_properties(device)}", flush=True)
    
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
    ])

    # Dataloaders
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    
    model = create_model(args.clf_name, device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, verbose=True)
    model = model.to(device)
        
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}", flush=True)

        training_loss = train(trainloader, model, optimizer, device, epoch)
        validation_loss = evaluate(testloader, model, device)
        scheduler.step()

        train_losses.append(training_loss)
        val_losses.append(validation_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
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

    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training [default: 128]')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate [default: 0.1]')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda num [default: 0]')
    parser.add_argument('--savefolder', type=str, default='cifar100',
                        help='folder name to save output [default: "cifar100"]')
    parser.add_argument('--unq-name', type=str, default='',
                        help='identifier name for saving [default: ""]')
    parser.add_argument('--clf-name', type=str, default='wideresnet-28-10',
                        help='identifier name for saving [default: "wideresnet-28-10"]')
    
    args = parser.parse_args()
    run(args.epochs, args.batch_size, args.lr, args.savefolder, args.unq_name, args)