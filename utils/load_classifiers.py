from robustbench import load_model
from torchvision import models
import torch
import os
from clf_models.mnist import *
from clf_models.cifar10 import *
from edm.torch_utils import distributed as dist
import sys
import open_clip
from contextlib import suppress
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms.transforms import Resize
import torchvision.transforms as transforms


# the following two functions are taken and updated from RobustCLIP
def load_open_clip(model_name: str = "ViT-L-14", pretrained: str = "openai", cache_dir: str = None, device="cpu"):
    try:
        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained='openai', cache_dir=cache_dir, device='cpu'
        )
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
        else:
            checkpoint = pretrained
        if 'vision_encoder_state_dict' in checkpoint.keys():  # tecoa checkpoint
            model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
        else:
            model.visual.load_state_dict(checkpoint)
    except Exception as e:
        # try loading whole model
        print(f'error: {e}', file=sys.stderr)
        print('retrying by loading whole model..', file=sys.stderr)
        torch.cuda.empty_cache()
        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir, device='cpu'
        )

    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer

def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=False, null_template=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    torch.Tensor of shape (D, C)
        D is the dimensionality of the text embeddings (model output size),
        and C is the number of classes.
    If null_template=True, returns a tensor of shape (D,) instead.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        
        if null_template:
            texts = [template.format(c="") for template in templates]
            for temp in texts:
                temp = tokenizer(temp).to(device)  # tokenize
                temp_embedding = model.encode_text(temp)
                temp_embedding = F.normalize(temp_embedding, dim=-1)
                # print("Temp embedding shape: ", temp_embedding.shape, flush=True)
                zeroshot_weights.append(temp_embedding)
        else:
            for classname in tqdm(classnames):
                if type(templates) == dict:
                    # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                    texts = templates[classname]
                elif type(templates) == list:
                    # generic prompts tht are specialized for each class by replacing {c} with the class name
                    texts = [template.format(c=classname) for template in templates]
                else:
                    raise ValueError("templates must be a list or a dict")
                            
                texts = tokenizer(texts).to(device)  # tokenize
                class_embeddings = model.encode_text(texts)
                # print("Class embeddings shape1: ", class_embeddings.shape, flush=True)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                # print("Class embedding shape2: ", class_embedding.shape, flush=True)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    
    if null_template:
        zeroshot_weights = zeroshot_weights.mean(dim=1)
        
    # print("Zeroshot weights shape: ", zeroshot_weights.shape, flush=True)
    return zeroshot_weights


class CLIPPredModel(nn.Module):
    def __init__(self, model, normalize, classifier):
        super(CLIPPredModel, self).__init__()
        self.model = model
        self.normalize = normalize
        self.classifier = classifier
        self.resize = Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None)
        
    def forward(self, data_unnorm):
        data_unnorm = self.resize(data_unnorm)
        data_norm = self.normalize(data_unnorm)
        features = self.model.encode_image(data_norm)
        features = F.normalize(features, dim=-1)

        logits = 100. * features @ self.classifier
        return logits
    
class TingImgNetPredModel(nn.Module):
    def __init__(self, classifier):
        super(TingImgNetPredModel, self).__init__()
        self.classifier = classifier
        self.resize = Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None)
        
    def forward(self, x):
        x = self.resize(x)
        logits = self.classifier(x)
        return logits


def update_state_dict(state_dict, idx_start=9):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove 'module.0.' (idx_start=9) or 'module.' (idx_start=7) of dataparallel 
        name = k[idx_start:]  
        new_state_dict[name]=v

    return new_state_dict

def load_classifier(dataset, classifier_name, device):

    if dataset == 'ImageNet':
        if classifier_name == 'resnet18':
            dist.print0(f'using imagenet resnet18...')
            model = models.resnet18(pretrained=True).eval()
        elif classifier_name == 'resnet50':
            dist.print0(f'using imagenet resnet50...')
            model = models.resnet50(pretrained=True).eval()
        elif classifier_name == 'resnet152':
            dist.print0(f'using imagenet resnet152...')
            model = models.resnet152(pretrained=True).eval()
        elif classifier_name == 'resnet101':
            dist.print0(f'using imagenet resnet101...')
            model = models.resnet101(pretrained=True).eval()
        elif classifier_name == 'wideresnet-50-2':
            dist.print0(f'using imagenet wideresnet-50-2...')
            model = models.wide_resnet50_2(pretrained=True).eval()
        # elif classifier_name == 'deit-s':
        #     dist.print0('using imagenet deit-s...')
        #     model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).eval()
        else:
            raise NotImplementedError(f'unknown {classifier_name}')

    
    elif dataset == 'MNIST':
        if classifier_name == 'mnist-lenet':
            dist.print0('using mnist lenet...')
            model = MNISTLeNet()
            
            model_path = os.path.join('pretrained_models', 'mnist-lenet.tar')
            dist.print0(f"=> loading mnist-lenet checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            dist.print0(f"=> loaded mnist-lenet checkpoint")
        
        elif classifier_name == 'mnist-lenet-raw-data':
            dist.print0('using mnist lenet raw data...')
            model = MNISTLeNet()
            
            model_path = os.path.join('pretrained_models', 'mnist-lenet-raw-data.tar')
            dist.print0(f"=> loading mnist-lenet-raw-data checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            dist.print0(f"=> loaded mnist-lenet-raw-data checkpoint")
        
    
    elif dataset == 'CIFAR10' or dataset == 'CIFAR10-C':
        if classifier_name == 'wideresnet-28-10-ckpt':
            dist.print0('using cifar10 wideresnet-28-10-ckpt...')
            # states_dict = torch.load(os.path.join('pretrained_models', 'cifar10_28_10.pt'), map_location=device)
            states_dict = torch.load(os.path.join('pretrained_models', 'cifar10-wrn-28-10.t7'), map_location=device)
            model = states_dict['net']
            
        elif classifier_name == 'wideresnet-28-10':
            dist.print0('using cifar10 wideresnet-28-10...')
            model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf', model_dir='./pretrained_models')  # pixel in [0, 1]
            # state_dict = torch.load("./pretrained_models/cifar10_28_10.pt", map_location=device)
            # model.load_state_dict(state_dict)

            # # Set the model to evaluation mode
            # model.eval()

            # print("Model loaded successfully!")

        elif classifier_name == 'wideresnet-70-16':
            dist.print0('using classifier from DiffAttack! ')
            dist.print0('using cifar10 wideresnet-70-16 (dm_wrn-70-16)...')
            from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet, Swish
            model = DMWideResNet(num_classes=10, depth=70, width=16, activation_fn=Swish)  # pixel in [0, 1]

            # model_path = os.path.join('pretrained_models', 'cifar10-wrn-70-16.pt')
            model_path = os.path.join('pretrained_models', 'weights-best.pt')
            dist.print0(f"=> loading wideresnet-70-16 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path)['model_state_dict']))
            model.eval()
            dist.print0(f"=> loaded wideresnet-70-16 checkpoint")
            
            # dist.print0('using cifar10 wideresnet-70-16 (dm_wrn-70-16) from DIFFATTACK')
            # from cifar10_resnet import WideResNet
            # model = WideResNet(depth=70, widen_factor=16, dropRate=0.0)
            
            # model_path = os.path.join('pretrained_models', 'weights-best.pt')
            # dist.print0(f"=> loading wideresnet-70-16 checkpoint '{model_path}'")
            # model.load_state_dict(torch.load(model_path)['model_state_dict'])
            # model.eval()
            # dist.print0(f"=> loaded wideresnet-70-16 checkpoint")

        elif classifier_name == 'wrn-70-16-dropout':
            dist.print0('using cifar10 wrn-70-16-dropout (standard wrn-70-16-dropout)...')
            model = WideResNet_70_16_dropout()  # pixel in [0, 1]

            model_path = os.path.join('pretrained_models', 'cifar10-wrn-70-16-dropout.pt')
            dist.print0(f"=> loading wrn-70-16-dropout checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            dist.print0(f"=> loaded wrn-70-16-dropout checkpoint")

        elif classifier_name == 'resnet-50':
            dist.print0('using cifar10 resnet-50...')
            model = ResNet50()  # pixel in [0, 1]

            model_path = os.path.join('pretrained_models', 'cifar10-resnet-50.pt')
            dist.print0(f"=> loading resnet-50 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            dist.print0(f"=> loaded resnet-50 checkpoint")
            
        elif classifier_name == 'clip':
            dist.print0('using cifar10 zeroshot clip...')
            clip_model, clip_transform, tokenizer = load_open_clip(model_name='ViT-L-14', pretrained='openai', cache_dir=None, device=device)
            clip_model.eval()

            # Load the classifier
            classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            templates = ["a photo of a {c}."] * len(classnames)
            zeroshot_weights = zero_shot_classifier(clip_model, tokenizer, classnames, templates, device)
            
            model = CLIPPredModel(clip_model, clip_transform.transforms[-1], zeroshot_weights).to(device)
            
        elif classifier_name == 'clip_fare2':
            dist.print0('using cifar10 zeroshot clip...')
            clip_model, clip_transform, tokenizer = load_open_clip(model_name='ViT-L-14', pretrained='/Data-HDD/*/models/fare2_clip_pur_coco_/fare_eps_2.pt', cache_dir=None, device=device)
            clip_model.eval()

            # Load the classifier
            classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            templates = ["a photo of a {c}."] * len(classnames)
            zeroshot_weights = zero_shot_classifier(clip_model, tokenizer, classnames, templates, device)
            
            model = CLIPPredModel(clip_model, clip_transform.transforms[-1], zeroshot_weights).to(device)
            
        elif classifier_name == 'clip_fare4':
            dist.print0('using cifar10 zeroshot clip...')
            clip_model, clip_transform, tokenizer = load_open_clip(model_name='ViT-L-14', pretrained='/Data-HDD/*/models/fare2_clip_pur_coco_/fare_eps_4.pt', cache_dir=None, device=device)
            clip_model.eval()

            # Load the classifier
            classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            templates = ["a photo of a {c}."] * len(classnames)
            zeroshot_weights = zero_shot_classifier(clip_model, tokenizer, classnames, templates, device)
            
            model = CLIPPredModel(clip_model, clip_transform.transforms[-1], zeroshot_weights).to(device)


        else:
            raise NotImplementedError(f'unknown {classifier_name}')

    elif dataset == 'CIFAR100' or dataset == 'CIFAR100-C':
        if classifier_name == 'wideresnet-28-10-ckpt':
            dist.print0('using cifar100 wideresnet-28-10-ckpt...')
            states_dict = torch.load(os.path.join('pretrained_models', 'cifar100-wrn-28-10.t7'), map_location=device)
            model = states_dict['net']
        
        elif classifier_name == 'wideresnet-70-16-ckpt':
            dist.print0('using cifar100 wideresnet-70-16-ckpt...')
            states_dict = torch.load(os.path.join('pretrained_models', 'cifar100-wrn-70-16.t7'), map_location=device)
            model = states_dict['net']
            
        elif classifier_name == 'wideresnet-28-10':
            dist.print0('using cifar100 wideresnet-28-10-ckpt...')
            model = WideResNet(depth=28, widen_factor=10, dropRate=0.3, num_classes=100)
            model.load_state_dict(torch.load(os.path.join('pretrained_models', 'cifar100/normwideresnet-28-10_'), map_location=device)['model_state_dict'])
            model.eval()
            
    elif dataset == 'TinyImageNet200':
        if classifier_name == 'resnet50':
            dist.print0('using tinyimagenet200 resnet50-ckpt...')
            resnet_model = models.resnet50(pretrained=False)
            resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 200)

            # checkpoint = torch.load("/Data-HDD/*/models/tinyimagenet/resnet50_sgd_normalize_0_last_epoch.pth", map_location=device)
            checkpoint = torch.load("/Data-HDD/*/models/tinyimagenet/resnet50_sgd_last_epoch.pth", map_location=device)
            resnet_model.load_state_dict(checkpoint['model_state_dict'])
            model = TingImgNetPredModel(resnet_model).to(device)
            model.eval()
        else:
            raise NotImplementedError(f'unknown {classifier_name}')
    
    else:
        raise NotImplementedError(f'unknown {classifier_name}')

    return model