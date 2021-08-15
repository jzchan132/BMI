import os
import types
from pathlib import Path
from typing import Optional

import pickle as pickle
from hydra.utils import get_original_cwd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 dataset: str, data_path: str,
                 train_batch_size: int = 1, eval_batch_size: int = 1, train_shuffle: bool = True,
                 num_workers: int = 0,
                 attr_method: Optional[str] = None,
                 use_explanations: bool = False, expl_path: Optional[str] = None,
                 load_label_expls: bool = False, label_expl_path: Optional[str] = None,
                 teacher_exp_id: int = None, load_teacher_logits: bool = False, load_teacher_attns: bool = False,
                 ss_mode: Optional[str] = None, ss_split_path: Optional[str] = None,
                 noisy_splits: Optional[str] = None
                 ):
        super().__init__()
        if use_explanations:
            assert expl_path
            assert attr_method

        self.p = types.SimpleNamespace()
        self.p.dataset = dataset
        # ${data_dir}/${.dataset}/${model.arch}/
        self.p.data_path = data_path
        self.p.train_batch_size = train_batch_size
        self.p.eval_batch_size = eval_batch_size
        self.p.train_shuffle = train_shuffle
        self.p.num_workers = num_workers

        self.p.attr_method = attr_method

        self.p.use_explanations = use_explanations
        self.p.expl_path = expl_path  # SLM-X/explanation_file

        self.p.load_label_expls = load_label_expls
        self.p.label_expl_path = label_expl_path  # SLM-X/explanation_file

        self.p.teacher_exp_id = teacher_exp_id
        self.p.load_teacher_logits = load_teacher_logits
        self.p.load_teacher_attns = load_teacher_attns

        self.p.ss_mode = ss_mode
        self.p.ss_split_path = ss_split_path

        self.p.noisy_splits = noisy_splits

    def load_dataset(self, split):
        dataset = {}
        data_path = os.path.join(self.p.data_path, split)
        assert Path(data_path).exists()
        for key in tqdm(['item_idx', 'input_ids', 'attention_mask', 'label'], desc=f'Loading {split} set'):
            with open(os.path.join(data_path, f'{key}.pkl'), 'rb') as f:
                dataset[key] = pickle.load(f)

        if self.p.use_explanations:
            if self.p.noisy_splits is not None and split in self.p.noisy_splits.split(','):
                expl_path = self.p.expl_path.replace('/expls_', '/noisy_expls_')
            else:
                expl_path = self.p.expl_path
            print(f'Using {expl_path} for split: {split}\n')

            expl_path = os.path.join(
                data_path, self.p.attr_method, self.p.expl_path)
            assert Path(expl_path).exists()
            dataset['explanation'] = pickle.load(open(expl_path, 'rb'))

        if self.p.load_label_expls:
            label_expl_path = os.path.join(
                data_path, self.p.attr_method, self.p.label_expl_path)
            assert Path(label_expl_path).exists()
            dataset['label_explanation'] = pickle.load(open(label_expl_path, 'rb'))

        if self.p.load_teacher_logits:
            teacher_logits_path = os.path.join(
                get_original_cwd(), f'../save/{self.p.teacher_exp_id}/model_outputs/train_logits.pkl')
            assert Path(teacher_logits_path).exists(), f"path {teacher_logits_path}"
            dataset['teacher_logits'] = pickle.load(open(teacher_logits_path, 'rb'))

        if self.p.load_teacher_attns:
            teacher_attns_path = os.path.join(
                get_original_cwd(), f'../save/{self.p.teacher_exp_id}/model_outputs/train_attns.pkl')
            assert Path(teacher_attns_path).exists(), f"path {teacher_attns_path}"
            dataset['teacher_attns'] = pickle.load(open(teacher_attns_path, 'rb'))

        if self.p.ss_mode is not None:
            with open(self.p.ss_split_path, 'rb') as f:
                ss_split = pickle.load(f)

            if self.p.ss_mode == 'labeled':
                ss_indices = ss_split[0]
            elif self.p.ss_mode == 'unlabeled':
                ss_indices = ss_split[1]
            elif self.p.ss_mode == 'all':
                ss_indices = ss_split[0] + ss_split[1]

            for key, val in dataset.items():
                dataset[key] = [val[i] for i in ss_indices]

        return dataset

    def setup(self, splits=['all']):
        self.data = {}
        if splits == ['all']:
            splits = ['train', 'dev', 'test']

        for split in splits:
            self.data[split] = TextClassificationDataset(
                self.load_dataset(split), split,
                self.p.use_explanations, self.p.load_label_expls, self.p.load_teacher_logits, self.p.load_teacher_attns)

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.data['train'],
            batch_size=self.p.train_batch_size,
            num_workers=self.p.num_workers,
            collate_fn=self.data['train'].collater,
            shuffle=shuffle,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data['dev'],
            batch_size=self.p.eval_batch_size,
            num_workers=self.p.num_workers,
            collate_fn=self.data['dev'].collater,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.data['test'],
            batch_size=self.p.eval_batch_size,
            num_workers=self.p.num_workers,
            collate_fn=self.data['test'].collater,
            pin_memory=True
        )


class TextClassificationDataset(Dataset):
    def __init__(self, dataset, split,
                 use_explanations: bool, load_label_expls: bool, load_teacher_logits: bool, load_teacher_attns: bool):
        self.data = dataset
        self.split = split
        self.use_explanations = use_explanations
        self.load_label_expls = load_label_expls
        self.load_teacher_logits = load_teacher_logits
        self.load_teacher_attns = load_teacher_attns

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.data['input_ids'][idx])
        attention_mask = torch.LongTensor(self.data['attention_mask'][idx])
        label = torch.LongTensor([self.data['label'][idx]])
        item_idx = torch.LongTensor([self.data['item_idx'][idx]])

        explanation = torch.stack(
            [x.to('cpu') for x in self.data['explanation'][idx]], dim=0) \
            if self.use_explanations else None

        all_label_expls = torch.stack(
            [x.to('cpu') for x in self.data['label_explanation'][idx]], dim=0) \
            if self.load_label_expls else None

        teacher_logits = torch.stack(
            [x.to('cpu') for x in self.data['teacher_logits'][idx]], dim=0) \
            if self.load_teacher_logits else None

        teacher_attns = torch.stack(
            [x.to('cpu') for x in self.data['teacher_attns'][idx]], dim=0) \
            if self.load_teacher_attns else None

        return (
            input_ids, attention_mask, label, item_idx,
            explanation, all_label_expls, teacher_logits, teacher_attns
        )

    def collater(self, items):
        batch = {
            'input_ids': torch.stack([x[0] for x in items], dim=0),
            'attention_mask': torch.stack([x[1] for x in items], dim=0),
            'label': torch.cat([x[2] for x in items]),
            'item_idx': torch.cat([x[3] for x in items]),
            # when evaluate_ckpt=true, split always test
            'split': self.split,
            'explanation': None,
            'label_explanation': None,
            'teacher_logits': None,
            'teacher_attns': None,
        }

        if self.use_explanations:
            batch['explanation'] = torch.stack([x[4] for x in items], dim=0).float()

        if self.load_label_expls:
            batch['label_explanation'] = torch.stack([x[5] for x in items], dim=0).float()

        if self.load_teacher_logits:
            batch['teacher_logits'] = torch.stack([x[6] for x in items], dim=0).float()

        if self.load_teacher_attns:
            batch['teacher_attns'] = torch.stack([x[7] for x in items], dim=0).float()

        return batch
