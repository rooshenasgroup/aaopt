import torch
from eval_bpda_eot import Purify_Model
from autoattack import AutoAttack as AutoAttackLib

class EOTWrapper(torch.nn.Module):
    def __init__(self, model, n_eot, eot_transform):
        super(EOTWrapper, self).__init__()
        self.model = model
        self.n_eot = n_eot  # Number of transformations for EOT
        self.eot_transform = eot_transform  # Function to apply random transformations

    def forward(self, x):
        batch_size = x.size(0)
        # Stack n_eot transformed versions of the input
        x_eot = torch.stack([self.eot_transform(x) for _ in range(self.n_eot)], dim=0)  # Shape: [n_eot, batch_size, ...]
        # Reshape for model input: [n_eot * batch_size, ...]
        x_eot = x_eot.view(-1, *x.size()[1:])
        # Get model predictions
        logits = self.model(x_eot)  # Shape: [n_eot * batch_size, num_classes]
        # Reshape and average over EOT samples
        logits = logits.view(self.n_eot, batch_size, -1).mean(dim=0)  # Shape: [batch_size, num_classes]
        return logits
    

def evaluate_autoattack(args, dataloader, clf, trans_to_clf, scorenet, sdiff_net, sdiff_sampler):
    """
    Evaluate model robustness using AutoAttack.

    Args:
        args: Configuration object with device, dataset, etc.
        dataloader: DataLoader for evaluation data.
        clf: Base classifier model.
        trans_to_clf: Transformation to classifier input space.
        scorenet: Score network for purification.
    """
    # Initialize model
    base_model = Purify_Model(clf, trans_to_clf, scorenet, sdiff_net, sdiff_sampler, args).to(args.device)
    base_model.eval()
    
    eot_transform = lambda x: x
    # Wrap the model with EOT
    model = EOTWrapper(base_model, args.eot_attack_reps, eot_transform).to(args.device)

    # Configure AutoAttack
    print(f"Running AutoAttack... at {args.att_eps}", flush=True)
    # norm = 'Linf' if args.att_lp_norm == -1 else 'L2',
    # adversary = AutoAttackLib(model, norm=norm, eps=args.att_eps, version='custom', verbose=True, attacks_to_run=['apgd-ce', 'apgd-t', 'fab-t', 'square'])
    # # 'standard' includes APGD-CE, APGD-T, FAB, Square attacks
    
    norm = 'Linf' if args.att_lp_norm == -1 else 'L2'
    if args.use_aa_rand == 1:
        adversary = AutoAttackLib(model, norm=norm, eps=args.att_eps, version='rand', verbose=True)
    else:
        adversary = AutoAttackLib(model, norm=norm, eps=args.att_eps, version='custom', verbose=True, attacks_to_run=['apgd-ce', 'apgd-t', 'fab-t', 'square'])
    

    # Evaluation metrics
    total_samples = 0
    clean_correct = 0
    all_x = []
    all_y = []

    # Collect all data for AutoAttack
    for x_val, y_val in dataloader:
        x_val = x_val.to(args.device, torch.float32)
        y_val = y_val.to(args.device, torch.long).view(-1)
        all_x.append(x_val)
        all_y.append(y_val)
        total_samples += x_val.size(0)

    # Concatenate all batches
    x_all = torch.cat(all_x, dim=0)
    y_all = torch.cat(all_y, dim=0)
    
    del all_x, all_y
    torch.cuda.empty_cache()

    # # Clean accuracy on full set
    # with torch.no_grad():
    #     logits_clean = model(x_all)
    #     pred_clean = logits_clean.argmax(dim=1)
    #     clean_correct = pred_clean.eq(y_all).sum().item()
    #     clean_acc = clean_correct / total_samples
    #     print(f"Initial Clean Accuracy (before attack): {clean_acc:.4f}", flush=True)
        
    # Run AutoAttack with return_labels=True
    x_adv, y_adv = adversary.run_standard_evaluation(
        x_all, y_all, bs=args.batch_size, return_labels=True
    )

    # Compute adversarial accuracy from y_adv
    adv_correct = y_adv.eq(y_all).sum().item()
    adv_acc = adv_correct / total_samples
    
    return x_adv, y_adv
    
    # print(f"Final Clean Accuracy: {clean_acc:.4f}")
    # print(f"Final Robust Accuracy: {adv_acc:.4f}")
    # return clean_acc, adv_acc    
    
    # # Final results
    # clean_acc = clean_correct / total_samples
    # adv_acc = adv_correct / total_samples
    # print(f"Final Clean Accuracy: {clean_acc:.4f}")
    # print(f"Final Robust Accuracy: {adv_acc:.4f}")
    # return clean_acc, adv_acc