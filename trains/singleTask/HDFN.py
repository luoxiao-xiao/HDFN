import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss


logger = logging.getLogger('MMSA')


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class HDFN():
    """Trainer class for the HDFN model.

    Implements the joint optimization objective:
        L_total = L_MSA + L_dp + 0.1*(L_recon + L_s + 0.5*(L_ort + L_consist))
                  + 0.2*L_nce + 0.2*L_wc
    """

    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine    = nn.CosineEmbeddingLoss()
        self.metrics   = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE       = MSE()
        self.sim_loss  = HingeLoss()

    def do_train(self, model, dataloader, return_epoch_results=False):
        params    = model[0].parameters()
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      verbose=True, patience=self.args.patience)
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = [model[0]]
        model = net

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss  = 0.0
            left_epochs = self.args.update_epochs

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio  = batch_data['audio'].to(self.args.device)
                    text   = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model[0](text, audio, vision)

                    # L_MSA: primary sentiment prediction loss
                    loss_msa = self.criterion(output['output_logit'], labels)

                    # L_dp: dynamic perspective prediction loss
                    loss_dp = (self.criterion(output['dynamic_loss_l'], labels) +
                               self.criterion(output['dynamic_loss_v'], labels) +
                               self.criterion(output['dynamic_loss_a'], labels))

                    # L_nce: contrastive loss
                    loss_nce = output['nce_loss']

                    # L_recon: reconstruction loss
                    loss_recon = (self.MSE(output['recon_l'], output['origin_l']) +
                                  self.MSE(output['recon_v'], output['origin_v']) +
                                  self.MSE(output['recon_a'], output['origin_a']))

                    # L_s: specific-feature consistency after reconstruction
                    loss_s = (self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r']) +
                              self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r']) +
                              self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r']))

                    # L_ort: orthogonality loss between specific and shared features
                    if self.args.dataset_name == 'mosi':
                        num = 50
                    elif self.args.dataset_name == 'mosei':
                        num = 10
                    loss_ort = (
                        self.cosine(output['s_l'].reshape(-1, num),
                                    output['c_l'].reshape(-1, num),
                                    torch.tensor([-1]).cuda()) +
                        self.cosine(output['s_v'].reshape(-1, num),
                                    output['c_v'].reshape(-1, num),
                                    torch.tensor([-1]).cuda()) +
                        self.cosine(output['s_a'].reshape(-1, num),
                                    output['c_a'].reshape(-1, num),
                                    torch.tensor([-1]).cuda())
                    )

                    # L_consist: shared-feature cross-modal consistency
                    c_l, c_v, c_a = output['c_l'], output['c_v'], output['c_a']
                    loss_consist = (self.MSE(c_l, c_v) +
                                    self.MSE(c_l, c_a) +
                                    self.MSE(c_v, c_a))

                    # L_wc: weight-constraint loss (weights must sum to 1)
                    loss_wc = (torch.mean((output['w_l'].sum(dim=-1) - 1) ** 2) +
                               torch.mean((output['w_v'].sum(dim=-1) - 1) ** 2) +
                               torch.mean((output['w_a'].sum(dim=-1) - 1) ** 2))

                    # Joint loss
                    combined_loss = (
                        loss_msa + loss_dp +
                        0.1 * (loss_recon + loss_s + 0.5 * (loss_ort + loss_consist)) +
                        0.2 * loss_nce + 0.2 * loss_wc
                    )

                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(list(model[0].parameters()),
                                                  self.args.grad_clip)

                    train_loss += combined_loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs

                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) "
                f"[{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )

            val_results  = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'],  mode="TEST")
            cur_valid    = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            torch.save(model[0].state_dict(),
                       './pt/' + str(self.args.dataset_name) + '_' + str(epochs) + '.pth')

            isBetter = (
                cur_valid <= (best_valid - 1e-6)
                if min_or_max == 'min'
                else cur_valid >= (best_valid + 1e-6)
            )
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model[0].state_dict(),
                           './pt/HDFN' + str(self.args.dataset_name) + '.pth')

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                epoch_results['test'].append(
                    self.do_test(model, dataloader['test'], mode="TEST")
                )

            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        if return_sample_results:
            ids, sample_results, all_labels = [], [], []
            features = {"Feature_t": [], "Feature_a": [],
                        "Feature_v": [], "Feature_f": []}

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio  = batch_data['audio'].to(self.args.device)
                    text   = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision)
                    loss   = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss    = eval_loss / len(dataloader)
        pred, true   = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"]      = ids
            eval_results["SResults"] = sample_results
            for k in features:
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels']   = all_labels

        return eval_results
