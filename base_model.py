from typing import Optional

import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # update in `setup`
        self.total_steps = None

    def forward(self, **kwargs):
        raise NotImplementedError

    def calc_loss(self, preds, targets):
        raise NotImplementedError

    def calc_acc(self, preds, targets):
        raise NotImplementedError

    def run_step(self, batch, split):
        raise NotImplementedError

    def aggregate_epoch(self, outputs, split):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # # freeze encoder for initial few epochs based on p.freeze_epochs
        # if self.current_epoch < self.p.freeze_epochs:
        # 	freeze_net(self.text_encoder)
        # else:
        # 	unfreeze_net(self.text_encoder)

        return self.run_step(batch, 'train')

    def training_epoch_end(self, outputs):
        self.aggregate_epoch(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self.run_step(batch, 'valid')

    def validation_epoch_end(self, outputs):
        self.aggregate_epoch(outputs, 'valid')

    def test_step(self, batch, batch_idx):
        return self.run_step(batch, 'test')

    def test_epoch_end(self, outputs):
        self.aggregate_epoch(outputs, 'test')

    def setup(self, stage: Optional[str] = None):
        """calculate total steps"""
        if stage == 'fit':
            # Get train dataloader
            train_loader = self.trainer.datamodule.train_dataloader(
                shuffle=self.trainer.datamodule.p.train_shuffle
            )
            ngpus = len(self.trainer.gpus)
            # ngpus = self.trainer.gpus
            # Calculate total steps
            effective_batch_size = (self.trainer.datamodule.p.train_batch_size *
                                    max(1, ngpus) * self.trainer.accumulate_grad_batches)
            self.total_steps = int(
                (len(train_loader.dataset) // effective_batch_size) * float(self.trainer.max_epochs))

    def configure_optimizers(self):
        raise NotImplementedError
