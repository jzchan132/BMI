import os
import types
from typing import Optional, List

import torch
import torch.nn.functional as F
import torchmetrics
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import pickle
from torch import nn
from torch.nn.init import xavier_normal_
from transformers import get_scheduler, AutoModelForSequenceClassification

from base_model import BaseModel


class LanguageModel(BaseModel):
    def __init__(self,
                 arch: str, num_classes: int, dataset: str,
                 optimizer: DictConfig, scheduler: DictConfig, annealing: DictConfig,
                 freeze_epochs=-1, oracle=None, attn_reg=False,
                 # LM attention
                 output_attentions=True, output_predropout_attentions=True,
                 attention_probs_dropout_prob=0.1, attn_reg_layers='last',
                 loss_type: str = "kldiv", multitask: bool = False,
                 # kd related
                 kd: bool = False, temperature: float = 1.0, alpha: float = 1.0,
                 kd_logit_weight: float = 1.0, kd_attn_weight: float = 0.0, kd_expl_weight: float = 0.0,
                 heads_regularize: bool = False, kd_attn_layers: List[int] = [],
                 # save output
                 save_outputs: bool = False, exp_id: Optional[str] = None,
                 **kwargs):
        if oracle or attn_reg:
            assert not (oracle and attn_reg)
            assert (oracle or attn_reg and attn_reg_layers)
            assert loss_type != 'kd'

        super().__init__()

        self.save_hyperparameters()

        self.p = types.SimpleNamespace()
        self.p.arch = arch
        self.p.dataset = dataset
        self.p.optimizer = optimizer
        self.p.scheduler = scheduler
        self.p.annealing = annealing
        self.p.freeze_epochs = freeze_epochs
        self.p.output_attentions = output_attentions
        # self.p.output_predropout_attentions = output_predropout_attentions
        self.p.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.p.multitask = multitask
        self.p.loss_type = loss_type

        self.metrics = torch.nn.ModuleDict({
            'acc': torchmetrics.Accuracy(),
            'macro_f1': torchmetrics.F1(num_classes=num_classes, average='macro'),
            'micro_f1': torchmetrics.F1(num_classes=num_classes, average='micro'),
        })
        if dataset in ['sst2', 'stf']:
            self.metrics['binary_f1'] = torchmetrics.F1(num_classes=num_classes, average='micro', ignore_index=0)

        self.oracle = oracle
        self.attn_reg = attn_reg
        self.attn_reg_layers = attn_reg_layers
        self.heads_regularize = heads_regularize
        self.kd_attn_layers = torch.LongTensor(kd_attn_layers)

        self.kd = kd
        
        assert temperature >= 1
        self.temperature = temperature

        assert 1 >= alpha >= 0
        self.alpha = alpha

        self.kd_logit_weight = kd_logit_weight
        self.kd_attn_weight = kd_attn_weight
        self.kd_expl_weight = kd_expl_weight

        if save_outputs:
            assert exp_id is not None
        self.save_outputs = save_outputs
        self.exp_id = exp_id

        assert num_classes >= 2
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            arch,
            num_labels=num_classes if dataset != 'cose_inhouse' else 1,
            output_attentions=self.p.output_attentions,
            output_hidden_states=self.p.multitask,
            # output_predropout_attentions=self.p.output_predropout_attentions,
            attention_probs_dropout_prob=self.p.attention_probs_dropout_prob,
        )
        if self.p.multitask:
            self.dropout = nn.Dropout(self.classifier.config.hidden_dropout_prob)
            self.proj = nn.Linear(self.classifier.config.hidden_size, 1)
            xavier_normal_(self.proj.weight)
            self.proj.bias.data.zero_()

        self.output_atten = True

    def calc_loss(self, logits, targets):
        assert len(logits) == len(targets)
        return F.cross_entropy(logits, targets)

    def calc_multitask_expl_loss(self, logits, targets, mask):
        out = F.binary_cross_entropy_with_logits(
            logits.squeeze(2), targets, reduction='none')
        out = torch.masked_select(out, mask.bool())

        return out.mean()

    def calc_attn_loss(self, attns, expls,
                       attn_mask=None, all_label_expls=None, targets=None):
        """
        attns (bsz, #layers, #heads, #tokens, #tokens)
        expls (bsz, #tokens)
        attn_mask (bsz, #tokens)
        """

        layers = attns.shape[1]
        heads = attns.shape[2]
        if self.heads_regularize:
            cls_attns = attns[:, :, :, 0, :]
            expl_distr = F.softmax(expls, dim=1).unsqueeze(dim=1).repeat(1, layers, 1).unsqueeze(dim=2).repeat(1, 1,
                                                                                                               heads, 1)

        else:
            attns_avg = torch.mean(attns, dim=2)
            cls_attns = attns_avg[:, :, 0, :]

            add = (1 - attn_mask) * (-10000)
            # (bsz, #layers, #tokens)
            add_unsq = add.unsqueeze(1).repeat(1, layers, 1)
            cls_attns += add_unsq
            expls += add
            # (bsz, #layers, #tokens)
            expl_distr = F.softmax(expls, dim=1).unsqueeze(dim=1).repeat(1, layers, 1)

        if self.p.loss_type == 'kldiv':
            attn_distr = F.log_softmax(cls_attns, dim=-1)
            assert attn_distr.shape == expl_distr.shape
            attn_loss = F.kl_div(attn_distr, expl_distr, reduction='batchmean', log_target=False)

        elif self.p.loss_type == 'mse':
            attn_distr = F.softmax(cls_attns, dim=-1)
            assert attn_distr.shape == expl_distr.shape
            attn_loss = F.mse_loss(attn_distr, expl_distr)

        elif self.p.loss_type in ['contrastive_kldiv', 'contrastive_mse']:
            assert layers == 1, 'Layers > 1 not supported for contrastive_kldiv!'
            num_labels = all_label_expls.shape[1]
            add_unsq = add.unsqueeze(1).repeat(1, num_labels, 1)
            all_label_expls += add_unsq
            all_label_expl_distr = F.softmax(all_label_expls, dim=2)

            if self.p.loss_type == 'contrastive_kldiv':
                attn_distr = F.log_softmax(cls_attns, dim=-1)
                attn_distr = attn_distr.repeat(1, num_labels, 1)
                pairwise_loss = -F.kl_div(attn_distr, all_label_expl_distr, reduction='none', log_target=False).sum(-1)
            elif self.p.loss_type == 'contrastive_mse':
                attn_distr = F.softmax(cls_attns, dim=-1)
                attn_distr = attn_distr.repeat(1, num_labels, 1)
                pairwise_loss = -F.mse_loss(attn_distr, all_label_expl_distr, reduction='none').sum(-1)

            attn_loss = F.cross_entropy(pairwise_loss, targets)

        return attn_loss

    def calc_kd_loss(self, student, teacher):
        if self.p.loss_type == 'kldiv':
            student = student / self.temperature
            student = F.log_softmax(student, dim=-1)

            teacher = teacher / self.temperature
            teacher = F.softmax(teacher, dim=-1)
            
            return self.temperature**2 * F.kl_div(student, teacher, reduction='batchmean', log_target=False)

        elif self.p.loss_type == 'mse':
            return F.mse_loss(student, teacher)

    #####
    # metric
    ####

    def get_step_metrics(self, preds, targets):
        res = {}
        for key, label in self.metrics.items():
            res.update({key: label(preds, targets) * 100})
        return res

    def get_epoch_metrics(self):
        res = {}
        for key, label in self.metrics.items():
            res.update({key: label.compute() * 100})
            label.reset()
        return res

    ######
    # step
    #####

    def forward(self, input, attention_mask, embed=False):
        if embed:
            outputs = self.classifier(
                inputs_embeds = input,
                attention_mask=attention_mask
            )
        else:
            outputs = self.classifier(
                input_ids=input,
                attention_mask=attention_mask
            )
        logits = outputs.logits
        # (bsz, #layers, #heads, #tokens, #tokens)
        attns = outputs['attentions']
        return logits, attns

    def run_step(self, batch, split):
        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        targets = batch['label']
        # override func param `split`
        split: str = batch['split']
        expls: Optional[torch.Tensor] = batch['explanation']
        all_lbl_expls: Optional[torch.Tensor] = batch['label_explanation']
        teacher_logits: Optional[torch.Tensor] = batch['teacher_logits']
        teacher_attns: Optional[torch.Tensor] = batch['teacher_attns']

        aux_loss = 0.0
        ret_dict = {}
        if (
                self.oracle == 'all'
                or self.oracle == 'train' and split == 'train'
        ):
            if self.save_outputs:
                logits, attns = self(input_ids, expls)
            else:
                logits = self(input_ids, expls)
        elif self.attn_reg:
            assert expls is not None
            logits, attns = self(input_ids, attn_mask)
            attn_loss = self.calc_attn_loss(attns, expls, attn_mask, all_lbl_expls, targets)
            aux_loss = self.loss_annealing.get_weight(self.global_step) * attn_loss

            self.log(f'{split}_attn_loss_step', attn_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
            self.log(f'{split}_attn_reg_wt', self.loss_annealing.get_weight(self.global_step))

            ret_dict['attn_loss'] = attn_loss

        elif self.p.multitask:
            assert expls is not None
            logits, expl_logits = self(input_ids, attn_mask)
            expl_loss = self.calc_multitask_expl_loss(expl_logits, expls, attn_mask)
            aux_loss = self.loss_annealing.get_weight(self.global_step) * expl_loss

            self.log(f'{split}_multitask_loss_step', expl_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
            self.log(f'{split}_multitask_reg_wt', self.loss_annealing.get_weight(self.global_step))

            ret_dict['expl_loss'] = expl_loss

        elif self.kd:

            logits, attns = self(input_ids, attn_mask)
            kd_logit_wt_loss, kd_attn_wt_loss, kd_expl_wt_loss = 0.0, 0.0, 0.0

            if self.kd_logit_weight > 0:
                assert teacher_logits != None
                kd_logit_loss = self.calc_kd_loss(logits, teacher_logits)
                kd_logit_wt_loss = self.kd_logit_weight * kd_logit_loss
                self.log(f'{split}_kd_logit_loss_step', kd_logit_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
                self.log(f'{split}_kd_logit_wt_loss_step', kd_logit_wt_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
                ret_dict['kd_logit_loss'] = kd_logit_loss
                ret_dict['kd_logit_wt_loss'] = kd_logit_wt_loss
            
            if self.kd_attn_weight > 0:
                assert teacher_attns != None
                if len(self.kd_attn_layers) > 0:
                    teacher_attns = teacher_attns[:, self.kd_attn_layers.to(self.device), :]
                kd_attn_loss = self.calc_kd_loss(attns, teacher_attns)
                kd_attn_wt_loss = self.kd_attn_weight * kd_attn_loss
                self.log(f'{split}_kd_attn_loss_step', kd_attn_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
                self.log(f'{split}_kd_attn_wt_loss_step', kd_attn_wt_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
                ret_dict['kd_attn_loss'] = kd_attn_loss
                ret_dict['kd_attn_wt_loss'] = kd_attn_wt_loss

            if self.kd_expl_weight > 0:
                raise NotImplementedError

            aux_loss = kd_logit_wt_loss + kd_attn_wt_loss + kd_expl_wt_loss
            
        elif self.save_outputs:
            logits, attns = self(input_ids, attn_mask)
        else:
            logits = self(input_ids, attn_mask)

        preds = torch.argmax(logits, dim=1)

        task_loss = self.calc_loss(logits, targets)
        if self.kd:
            loss = self.alpha * task_loss + (1 - self.alpha) * aux_loss
        else:
            loss = task_loss + aux_loss

        metrics = self.get_step_metrics(preds, targets)

        self.log(f'{split}_loss_step', loss.item(), prog_bar=True, sync_dist=(split != 'train'))
        self.log(f'{split}_task_loss_step', task_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
        self.log(f'{split}_acc_step', metrics['acc'], prog_bar=True, sync_dist=(split != 'train'))
        self.log(f'{split}_macro_f1_step', metrics['macro_f1'], prog_bar=True, sync_dist=(split != 'train'))
        self.log(f'{split}_micro_f1_step', metrics['micro_f1'], prog_bar=True, sync_dist=(split != 'train'))
        if self.p.dataset in ['sst2', 'stf']:
            self.log(f'{split}_binary_f1_step', metrics['binary_f1'], prog_bar=True, sync_dist=(split != 'train'))

        ret_dict.update({
            'preds': preds, 'targets': targets,
            'loss': loss, 'task_loss': task_loss,
            'split': split
        })

        if self.save_outputs:
            ret_dict["logits"] = logits
            ret_dict["attns"] = attns

        return ret_dict

    def aggregate_epoch(self, outputs, split):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        task_loss = torch.stack([x['task_loss'] for x in outputs]).mean()

        if self.attn_reg:
            attn_loss = torch.stack([x['attn_loss'] for x in outputs]).mean()
            self.log(f'{split}_attn_loss_epoch', attn_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
        elif self.p.multitask:
            expl_loss = torch.stack([x['expl_loss'] for x in outputs]).mean()
            self.log(f'{split}_multitask_loss_epoch', expl_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
        elif self.kd:
            if self.kd_logit_weight > 0:
                kd_logit_loss = torch.stack([x['kd_logit_loss'] for x in outputs]).mean()
                self.log(f'{split}_kd_logit_loss_epoch', kd_logit_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
            if self.kd_attn_weight > 0:
                kd_attn_loss = torch.stack([x['kd_attn_loss'] for x in outputs]).mean()
                self.log(f'{split}_kd_attn_loss_epoch', kd_attn_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
            if self.kd_expl_weight > 0:
                kd_expl_loss = torch.stack([x['kd_expl_loss'] for x in outputs]).mean()
                self.log(f'{split}_kd_expl_loss_epoch', kd_attn_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
        if self.save_outputs:
            out_dir = f'{get_original_cwd()}/../save/{self.exp_id}/model_outputs'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            split = outputs[0]['split']
            for key in ['logits', 'preds', 'attns']:
                out_data = torch.cat([x[key] for x in outputs]).cpu().detach()
                out_file = os.path.join(out_dir, f'{split}_{key}.pkl')
                pickle.dump(out_data, open(out_file, 'wb'))

        metrics = self.get_epoch_metrics()
        self.log(f'{split}_loss_epoch', loss.item(), prog_bar=True, sync_dist=(split != 'train'))
        self.log(f'{split}_task_loss_epoch', task_loss.item(), prog_bar=True, sync_dist=(split != 'train'))
        self.log(f'{split}_acc_epoch', metrics['acc'], prog_bar=True, sync_dist=(split != 'train'))
        self.log(f'{split}_macro_f1_epoch', metrics['macro_f1'], prog_bar=True, sync_dist=(split != 'train'))
        self.log(f'{split}_micro_f1_epoch', metrics['micro_f1'], prog_bar=True, sync_dist=(split != 'train'))
        if self.p.dataset in ['sst2', 'stf']:
            self.log(f'{split}_binary_f1_epoch', metrics['binary_f1'], prog_bar=True, sync_dist=(split != 'train'))

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.p.optimizer.weight_decay,
            },
            {
                'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]

        if self.p.multitask:
            optimizer_parameters += [
                {
                    'params': [p for n, p in self.proj.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.p.optimizer.weight_decay,
                },
                {
                    'params': [p for n, p in self.proj.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                },
            ]

        optimizer = instantiate(
            self.p.optimizer, params=optimizer_parameters,
            _convert_="partial"
        )

        # configure the loss annealing for attention regularization loss
        self.loss_annealing = instantiate(
            self.p.annealing,
            num_training_steps=self.total_steps
        )

        if self.p.scheduler.lr_scheduler == 'linear_with_warmup':
            if self.p.scheduler.warmup_updates > 1.0:
                warmup_steps = int(self.p.scheduler.warmup_updates)
            else:
                warmup_steps = int(self.total_steps *
                                   self.p.scheduler.warmup_updates)
            print(
                f'\nTotal steps: {self.total_steps} with warmup steps: {warmup_steps}\n')

            scheduler = get_scheduler(
                "linear", optimizer=optimizer,
                num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)

            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        elif self.p.lr_scheduler == 'fixed':
            return [optimizer]
        else:
            raise NotImplementedError
