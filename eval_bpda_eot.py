from attacks import *
from utils import *
from defense import *
import tqdm

class Purify_Model(nn.Module):
    def __init__(self, clf, trans_to_clf, scorenet, sdiff_net, sdiff_sampler, args):
        super().__init__()
        self.args = args
        self.classifier = clf
        self.scorenet = scorenet
        self.trans_to_clf = trans_to_clf
        self.sdiff_net = sdiff_net
        self.sdiff_sampler = sdiff_sampler

    def purify_and_classify(self, x):
        if self.args.purify_model == "edm":
            purif_x = purify_x_edm_one_shot(x, self.scorenet, self.args)
        elif self.args.purify_model == "edm_multi":
            purif_x = purify_x_edm_multistep(x, self.scorenet, self.args)
        elif self.args.purify_model == "opt":
            if self.args.purify_method == "aaopt":
                purif_x = purify_x_opt_aaopt(x, self.scorenet, self.sdiff_net, self.sdiff_sampler, self.args)
            elif self.args.purify_method == "x0":
                purif_x = purify_x_opt_x0(x, self.scorenet, self.args)
            elif self.args.purify_method == "xt":
                purif_x = purify_x_opt_xt(x, self.scorenet, self.args)
            else:
                raise NotImplementedError
        elif self.args.purify_model == "None":
            purif_x = x
        logit = self.classifier(self.trans_to_clf(purif_x))
        return logit
    
    def purify(self, x):
        if self.args.purify_model == "edm":
            purif_x = purify_x_edm_one_shot(x, self.scorenet, self.args)
        elif self.args.purify_model == "edm_multi":
            purif_x = purify_x_edm_multistep(x, self.scorenet, self.args)
        elif self.args.purify_model == "opt":
            if self.args.purify_method == "aaopt":
                purif_x = purify_x_opt_aaopt(x, self.scorenet, self.sdiff_net, self.sdiff_sampler, self.args)
            elif self.args.purify_method == "x0":
                purif_x = purify_x_opt_x0(x, self.scorenet, self.args)
            elif self.args.purify_method == "xt":
                purif_x = purify_x_opt_xt(x, self.scorenet, self.args)
            else:
                raise NotImplementedError
        elif self.args.purify_model == "None":
            purif_x = x
        return purif_x
    
    def alternative(self, x):
        if self.args.purify_model == "edm":
            purif_x = purify_x_edm_one_shot(x, self.scorenet, self.args)
        elif self.args.purify_model == "edm_multi":
            # purif_x = purify_x_edm_multistep(x, self.scorenet, self.args)
            purif_x = purify_x_edm_one_shot(x, self.scorenet, self.args)
        elif self.args.purify_model == "opt":
            if self.args.ex_grad == 1:
                if self.args.purify_method == "aaopt":
                    purif_x = purify_x_opt_aaopt_1iter(x, self.scorenet, self.sdiff_net, self.sdiff_sampler, self.args)
                elif self.args.purify_method == "x0":
                    purif_x = purify_x_opt_x0_1iter(x, self.scorenet, self.args)
                elif self.args.purify_method == "xt":
                    # 1 iter not available for xt yet
                    purif_x = purify_x_edm_one_shot(x, self.scorenet, self.args)
                else:
                    raise NotImplementedError
            else:
                purif_x = purify_x_edm_one_shot(x, self.scorenet, self.args)
        elif self.args.purify_model == "None":
            purif_x = x
        logit = self.classifier(self.trans_to_clf(purif_x))
        return logit
    
    def classify(self, x):
        logit = self.classifier(self.trans_to_clf(x))
        return logit
    
    def forward(self, x):
        with torch.enable_grad():
            # logit = self.purify_and_classify(x)
            if self.args.use_full_steps:
                logit = self.purify_and_classify(x)
            else:
                logit = self.alternative(x)
        return logit

def attack_and_purif_bpda_eot(args, config, dataloader, clf, trans_to_clf, score, sdiff_net, sdiff_sampler, diffusion):

    eval_model =  Purify_Model(clf, trans_to_clf, score, sdiff_net, sdiff_sampler, args)
    adversary_edm = BPDA_EOT_Attack(eval_model, args)
    
    class_path = torch.zeros([args.att_n_iter + 2, 0]).bool()
    ims_adv = torch.zeros(0)

    for i, (x_val, y_val) in enumerate(tqdm.tqdm(dataloader)):
        x_val = x_val.to(args.device).to(torch.float32)
        y_val = y_val.to(args.device).to(torch.long)
        y_val = y_val.view(-1,)
        class_batch, ims_adv_batch = adversary_edm.attack_batch(x_val, y_val)
        class_path = torch.cat((class_path, class_batch), dim=1)
        ims_adv = torch.cat((ims_adv, ims_adv_batch), dim=0)
        print(f'finished {i+1}-th batch in attack_all')

    return class_path, ims_adv