import sys
sys.path.append('./clf_models')
sys.path.append('./edm')

import os
import argparse
import click
import logging
import pickle
import random
import time
from edm import dnnlib
from edm.torch_utils import distributed as dist
import numpy as np
import pandas as pd
import torch
from utils import *
from eval_bpda_eot import attack_and_purif_bpda_eot
from eval_aa import evaluate_autoattack
from defense import *
import tqdm
from unet_model import UNetModel
from ddim_sampler import DDIMSampler



def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--log', default='logs', help='Output path, including images and logs')
    parser.add_argument('--config', type=str, default='defalt.yml',  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--data_seed', nargs='+', help='Random seed for data subsets')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet200'])
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--clf_net', type=str, default='wideresnet-28-10')
    parser.add_argument('--subset_size', type=int, default=64, help='Size of the fixed subset')    
    
    # Purify
    parser.add_argument('--purify_iter', type=int, default=1, help='Number of iterations for purify')
    parser.add_argument('--apprx_iter', type=int, default=1, help='Number of iterations for approximating gradient in optimization for generating adversarial attacks')
    parser.add_argument('--purify_model', type=str, default='opt', choices=['opt', 've', 'vp', 'edm', 'edm_multi', 'ebm_one_shot', 'None'])
    parser.add_argument('--purify_method', type=str, default='aaopt', choices=['aaopt', 'x0', 'xt', 'None'])
    parser.add_argument('--total_steps', type=int, default=1, help='Number of total diffusion steps - not used (1 for now)')
    parser.add_argument('--forward_steps', type=float, default=1, help='Number of forward diffusion timesteps/Noise Scale of forward Process')
    
    # Optimization
    parser.add_argument('--loss_lambda', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--init', action= "store_true")

    # Attack
    parser.add_argument('--att_method', type=str, default='bpda_eot', choices=['fgsm', 'clf_pgd', 'bpda', 'bpda_eot', 'pgd_eot'])
    parser.add_argument('--att_lp_norm', type=int, default=-1, choices=[-1,1,2])
    parser.add_argument('--att_eps', type=float, default=8/255., help='8/255. for Linf, 0.5 for L2')
    parser.add_argument('--att_step', type=int, default=1, help='Step number of pgd attacks')
    parser.add_argument('--att_n_iter', type=int, default=50, help='Iteration number of adaptive attacks')
    parser.add_argument('--att_alpha', type=float, default=2/255., help='One-step attack pixel scale')
    parser.add_argument('--eot_defense_reps', type=int, default=150, help='Number of EOT for adaptive attacks')
    parser.add_argument('--eot_attack_reps', type=int, default=15, help='Number of EOT for defenses')

    # edm
    parser.add_argument('--network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
    parser.add_argument('--sigma_min', help='Lowest noise level  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
    parser.add_argument('--sigma_max', help='Highest noise level  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
    
    parser.add_argument('--sdiff_cons', type=float, default=0, help='score diff constant, if 0, will use baseline (no perturbation model)')
    parser.add_argument('--ex_grad', type=int, default=0, help='temp used for calc exact grad')
    parser.add_argument('--no_stop_grad', type=int, default=0, help='no stop grad on the diffusion models for adversarial evaluation')
    parser.add_argument('--diff_reversed', type=int, default=1, help='sign of difference perturbation model is trained on, 1: adv - clean, 0: clean - adv')
    parser.add_argument('--aa', type=int, default=0, help='use autoattack if 1')
    parser.add_argument('--seq_t', type=int, default=0, help='use sequential time if 1')
    parser.add_argument('--use_full_steps', type=int, default=0, help='use all steps in adv generation if 1 -> only for gradient based methods')
    parser.add_argument('--sdiff_net', type=str, default='./pretrained_models/cifar10_score_DIFF_JOINT_wideresnet-28-10_', help='score diff net path')
    parser.add_argument('--use_aa_rand', type=int, default=0, help='use rand version if 1 for AutoAttack')


    args = parser.parse_args()
    args.log = os.path.join(args.log, args.dataset, args.att_method, "l{}_{}x{}_it_{}_eot_{}_{}".format(
            args.att_lp_norm,
            args.subset_size,
            len(args.data_seed),
            args.att_n_iter,
            args.eot_defense_reps, 
            args.eot_attack_reps
            ),
            "model_{}_method_{}_total_{}_forward_{}_pur_{}".format(
            args.purify_model,
            args.purify_method,
            args.total_steps,
            args.forward_steps,
            args.purify_iter
            ),
            "lr_{}_init_{}_lambda_{}_seed_{}".format(
            args.lr,
            args.init,
            args.loss_lambda,
            args.seed
            ))
    if not os.path.exists(args.log):
        os.makedirs(args.log, exist_ok=True)
        
    # set logger  
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'seed_{}.txt'.format(args.seed)))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
        
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return args

def get_sdiff(args):
    sdiff_model = UNetModel(in_channels=3, model_channels=64, 
                    out_channels=3, num_res_blocks=2, attention_resolutions=(16,), 
                    dropout=0.1, channel_mult=(1,2,2,2), num_heads=4)
    
    if args.sdiff_net != '':
        print("loading perturbation model from provided path: ", args.sdiff_net, flush=True)
        sdiff_model.load_state_dict(torch.load(args.sdiff_net)['model_state_dict'])
    else:
        if args.dataset == 'CIFAR100':
            print(f"loading default cifar100 perturbation model for {args.clf_net}", flush=True)
            if args.clf_net == 'wideresnet-28-10':
                sdiff_model.load_state_dict(torch.load('/Data-HDD/*/models/cifar10_ebm_adv_/cifar100_score_DIFF_JOINT_CIFAR100_wideresnet-28-10_')['model_state_dict'])
            elif args.clf_net == 'wideresnet-70-16':
                raise NotImplementedError("wideresnet-70-16 not implemented for CIFAR100")
            else:
                raise NotImplementedError("Unknown classifier network for CIFAR100")
        
        elif args.dataset == 'CIFAR10':
            print(f"loading default cifar10 perturbation model for {args.clf_net}", flush=True)
            if args.clf_net == 'wideresnet-28-10':
                sdiff_model.load_state_dict(torch.load('/Data-HDD/*/models/cifar10_ebm_adv_/cifar10_score_DIFF_JOINT_wideresnet-28-10_')['model_state_dict'])
            elif args.clf_net == 'wideresnet-70-16':
                sdiff_model.load_state_dict(torch.load('/Data-HDD/*/models/cifar10_ebm_adv_/cifar10_score_DIFF_JOINT_wideresnet-70-16_')['model_state_dict'])
            else:
                raise NotImplementedError("Unknown classifier network for CIFAR10")
            
        else:
            raise NotImplementedError("Unknown dataset for sdiff model")
    sdiff_model.eval()
    sdiff_model.cuda()
    return sdiff_model


def main():
    args = parse_args_and_config()
    dist.init()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
        
    if dist.get_rank() == 0:
        logging.info("Using device: {}".format(device))
        logging.info("Writing log file to {}".format(args.log))
        logging.info("Using args: {}".format(args))
    
    # dataset and pre-trained model
    val_dataloader = preprocess_datasets(args.dataset, False, args.batch_size, args.data_seed, args.subset_size, dist=True)
    # val_dataloader = preprocess_datasets(args.dataset, True, args.batch_size, args.data_seed, args.subset_size, dist=True)
    
    clf = load_classifier(args.dataset, args.clf_net, args.device).to(args.device)
    clf.eval()
    trans_to_clf = get_transforms(args.dataset, args.clf_net)

    if args.dataset == 'CIFAR10':
        dist.print0(f"=> loading cifar10-diffusion checkpoint from '{args.network_pkl}'")
        with dnnlib.util.open_url(args.network_pkl, verbose=(dist.get_rank() == 0)) as f:
            score = pickle.load(f)['ema'].to(args.device)

    elif args.dataset == 'CIFAR100':
        dist.print0(f"=> loading cifar100-diffusion checkpoint from '{args.network_pkl}'")
        with dnnlib.util.open_url(args.network_pkl, verbose=(dist.get_rank() == 0)) as f:
            score = pickle.load(f)['ema'].to(args.device)
            
    if args.purify_model == 'opt' and args.purify_method == 'aaopt':
        sdiff_net = get_sdiff(args)
        sdiff_sampler = DDIMSampler(sdiff_net, device, 1)
        dist.print0("perturbation model loaded")
    else:
        sdiff_net = None
        sdiff_sampler = None
        print("Not using perturbation model", flush=True)
    
    torch.distributed.barrier()
    
    print("EPS: ", args.att_eps)
    print("SDIFF: ", args.sdiff_cons, flush=True)
    
    if args.aa == 1:
        print("Using AutoAttack ..... ", flush=True)
        x_adv, y_adv = evaluate_autoattack(args, val_dataloader, clf, trans_to_clf, score, sdiff_net, sdiff_sampler)
        # print(f'clean_acc: {clean_acc}, adv_acc: {adv_acc}')
        
        @torch.no_grad()
        def classifier_purif(args, dataloader, clf, trans_to_clf, score, diffusion):
            accuracy, aaccuracy, paccuracy, saccuracy = 0., 0., 0., 0.

            cnt = 0
            for i, (x_val, y_val) in enumerate(tqdm.tqdm(dataloader)):
                cnt += x_val.shape[0]
                x_val = x_val.to(args.device).to(torch.float32).detach()
                y_val = y_val.to(args.device).to(torch.long)
                y_val = y_val.view(-1,)

                x_val_repeat = x_val.repeat([args.eot_attack_reps, 1, 1, 1])
                
                perturbed_X = x_adv[i*args.batch_size:(i+1)*args.batch_size].detach()
                perturbed_X_repeat = perturbed_X.repeat([args.eot_attack_reps, 1, 1, 1])
                
                # plot x_val0 and x_adv0
                if i == 0:
                    plt.figure()
                    plt.imshow(x_val[0].permute(1,2,0).cpu().numpy())
                    plt.savefig(f'x_val.png')
                    plt.figure()
                    plt.imshow(perturbed_X[0].permute(1,2,0).cpu().numpy())
                    plt.savefig(f'x_adv.png')

                # purify natural and adversarial samples
                if args.purify_model == 'edm':
                    purif_X_re = purify_x_edm_one_shot(perturbed_X_repeat, score, args)
                    purif_X_no_attack_re = purify_x_edm_one_shot(x_val_repeat, score, args)
                elif args.purify_model == 'opt':
                    if args.purify_method == "aaopt":
                        purif_X_re = purify_x_opt_aaopt(perturbed_X_repeat, score, sdiff_net, sdiff_sampler, args)
                        purif_X_no_attack_re = purify_x_opt_aaopt(x_val_repeat, score, sdiff_net, sdiff_sampler, args)
                    elif args.purify_method == "x0":
                        purif_X_re = purify_x_opt_x0(perturbed_X_repeat, score, args)
                        purif_X_no_attack_re = purify_x_opt_x0(x_val_repeat, score, args)
                        # purif_X_re = purify_x_opt_x0_exact_sm(perturbed_X_repeat, score, args)
                        # purif_X_no_attack_re = purify_x_opt_x0_exact_sm(x_val_repeat, score, args)
                        # purif_X_re = purify_x_opt_xt_exact_sm(perturbed_X_repeat, score, args)
                        # purif_X_no_attack_re = purify_x_opt_xt_exact_sm(x_val_repeat, score, args)
                    elif args.purify_method == "xt":
                        purif_X_re = purify_x_opt_xt(perturbed_X_repeat, score, args)
                        purif_X_no_attack_re = purify_x_opt_xt(x_val_repeat, score, args)
                    
                
                with torch.no_grad():
                    # calculate standard acc (without purification)    
                    logit = clf(trans_to_clf(x_val.clone().detach()))
                    pred = logit.max(1, keepdim=True)[1].view(-1,).detach()
                    acc = (pred == y_val.clone().detach()).float().sum()
                    accuracy += acc.cpu().numpy()

                    # calculate robust loss and acc (without purification)
                    logit = clf(trans_to_clf(perturbed_X.clone().detach()))
                    apred = logit.max(1, keepdim=True)[1].view(-1,).detach()
                    aacc = (apred == y_val.clone().detach()).float().sum()
                    aaccuracy += aacc.cpu().numpy()
                    
                    # calculate standard loss and acc (with purification)
                    logit = clf(trans_to_clf(purif_X_no_attack_re.clone().detach()))
                    logit = logit.view(args.eot_attack_reps, args.batch_size, -1).mean(0)
                    spred = logit.max(1, keepdim=True)[1].view(-1,).detach()
                    sacc = (spred == y_val.clone().detach()).float().sum()
                    saccuracy += sacc.cpu().numpy()

                    # calculate robust loss and acc (with purification)
                    logit = clf(trans_to_clf(purif_X_re.clone().detach()))
                    logit = logit.view(args.eot_attack_reps, args.batch_size, -1).mean(0)
                    ppred = logit.max(1, keepdim=True)[1].view(-1,).detach()
                    pacc = (ppred == y_val.clone().detach()).float().sum()
                    paccuracy += pacc.cpu().numpy()

            return 100*accuracy, 100*aaccuracy, 100*saccuracy, 100*paccuracy, cnt
        
        std_acc, rob_acc, purif_std_acc, purif_rob_acc, cnt = classifier_purif(args, val_dataloader, clf, trans_to_clf, score, None)
        t = torch.tensor([std_acc, rob_acc, purif_std_acc, purif_rob_acc, cnt], dtype=torch.int, device='cuda')
    
        t = t.cpu().numpy()
        total_count = t[-1]
        t = t / total_count
        print("ACCS: ", t)
        
        print("Std Acc: ", t[0])
        print("Rob Acc: ", t[1])
        print("Purif Std Acc: ", t[2])
        print("Purif Rob Acc: ", t[3])
        return

    start_time = time.time()
    class_path, ims_adv\
        = attack_and_purif_bpda_eot(args, None, val_dataloader, clf, trans_to_clf, score, sdiff_net, sdiff_sampler, None)
    init_acc = float(class_path[0, :].sum())
    robust_acc = float(class_path[-1, :].sum())
    end_time = time.time()
    print(f'Rank: {dist.get_rank()}, x_adv shape: {ims_adv.shape}')
    t = torch.tensor([init_acc, robust_acc, class_path.shape[1]], dtype=torch.float64, device='cuda')
    
    torch.distributed.barrier()
    torch.distributed.all_reduce(t)
    t = t.cpu().numpy()
    total_count = int(t[-1])
    t = t / total_count
    
    if dist.get_rank() == 0:
        logging.info("Rank:0")
        logging.info('standard accuracy: {}'.format(init_acc / class_path.shape[1]))
        logging.info('robust accuracy: {}'.format(robust_acc / class_path.shape[1]))
        
        logging.info("All gpus")
        logging.info('standard accuracy: {}'.format(t[0]))
        logging.info('robust accuracy: {}'.format(t[1]))

        logging.info("time elapsed: {:.2f}s".format(end_time-start_time))
        
        df=pd.DataFrame()
        new_row ={
                "dataset":args.dataset, 
                "att_method":args.att_method,
                "att_lp_norm":args.att_lp_norm,
                "subset_size":args.subset_size,
                "subset_number":len(args.data_seed),
                "att_step":args.att_step,
                "att_n_iter":args.att_n_iter,
                "att_eot_defense":args.eot_defense_reps,
                "att_eot_attack":args.eot_attack_reps,
                "purify_type":args.purify_model,
                "purify_method":args.purify_method,
                "total_steps":args.total_steps,
                "forward_steps":args.forward_steps,
                "purify_iter":args.purify_iter,
                "lr":args.lr,
                "init":args.init,
                "loss_lambda":args.loss_lambda,
                'seed':args.seed,
                "std_acc":t[0],
                "rob_acc":t[1]
        }
        df = df.append(new_row, ignore_index=True)
        df.to_csv(os.path.join("results","{}_{}_l{}_{}x{}_iter_{}_eot_{}_{}_model_{}_method_{}_total_{}_forward_{}_pur_{}_seed_{}.csv").format(
                args.dataset, 
                args.att_method,
                args.att_lp_norm,
                args.subset_size,
                len(args.data_seed),
                args.att_n_iter,
                args.eot_defense_reps, 
                args.eot_attack_reps,
                args.purify_model,
                args.purify_method,
                args.total_steps,
                args.forward_steps,
                args.purify_iter,
                args.seed     
        ))
    
    
    
if __name__ == '__main__':
    main()