
import lightning as L  # type: ignore
from lightning.pytorch.callbacks import ModelCheckpoint # type: ignore
from lightning.pytorch.loggers import TensorBoardLogger # type: ignore
import torch
from utils.io import create_chkpt
from utils.helpers import get_model_device
from utils.visualize import Visualizer
import os
import json

class LightningTrainer(L.LightningModule):
    def __init__(self,
                 model,
                 loss,
                 optimizer,
                 lr_scheduler=None,
                 use_torch_compile=False,
                 torch_compile_kwargs={'mode': 'max-autotune', 'backend': 'inductor'},
                 dataloader=None,
                 progress_bar_log_iter_interval=50):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss', 'optimizer', 'lr_scheduler', 'dataloader'])

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader

        if use_torch_compile:
            self.model = torch.compile(self.model, **torch_compile_kwargs)

    def training_step(self, batch, batch_idx):
        samples = batch[0]
        loss_out = self._calculate_loss(samples)
        loss = loss_out['loss']
        self.log_dict(loss_out.get('to_log', {}), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.lr_scheduler:
            return [self.optimizer], [self.lr_scheduler]
        return self.optimizer

    def _calculate_loss(self, samples):
        if self.loss.mode == 'post_forward':
            model_out = self.model(samples)
            inputs = {'data': samples, 'is_train': self.model.training, **model_out}
            return self.loss(**inputs)
        elif self.loss.mode == 'pre_forward':
            inputs = {'model': self.model, 'data': samples, 'is_train': self.model.training}
            return self.loss(**inputs)
        elif self.loss.mode == 'optimizes_internally':
            return self.loss(samples, self.model, self.optimizer)
        else:
            raise ValueError(f"Unknown loss function mode: {self.loss.mode}")

    def on_train_epoch_end(self):
        if self.loss.schedulers:
            self.loss.step_schedulers()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step % self.hparams.progress_bar_log_iter_interval == 0:
            pass

    def train_dataloader(self):
        return self.dataloader

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        self.chkpt_viz = kwargs.pop('chkpt_viz', False)
        super().__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        if self.chkpt_viz:
            self._save_visualization(self.dirpath)

    def _save_visualization(self, dir):
        visualizer = Visualizer(
            vae_model=self.trainer.model.model,
            dataset=self.trainer.datamodule.train_dataloader().dataset,
            is_plot=False,
            save_dir=dir
        )
        visualizer.plot_all_latent_traversals()
        visualizer.plot_random_reconstructions()

def get_trainer(
    model,
    loss,
    optimizer,
    lr_scheduler,
    dataloader,
    train_id,
    determinism_kwargs,
    use_torch_compile,
    torch_compile_kwargs,
    is_progress_bar,
    progress_bar_log_iter_interval,
    chkpt_every_n_steps,
    chkpt_step_type,
    chkpt_save_path,
    chkpt_save_dir,
    chkpt_save_master_dir,
    chkpt_viz,
    max_steps,
    step_unit
):
    lightning_model = LightningTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_torch_compile=use_torch_compile,
        torch_compile_kwargs=torch_compile_kwargs,
        dataloader=dataloader,
        progress_bar_log_iter_interval=progress_bar_log_iter_interval
    )

    callbacks = []
    if chkpt_save_dir or chkpt_save_master_dir or chkpt_save_path:
        checkpoint_callback = CustomModelCheckpoint(
            dirpath=chkpt_save_dir or chkpt_save_master_dir,
            filename='{epoch}-{step}',
            save_top_k=-1,
            every_n_train_steps=chkpt_every_n_steps if chkpt_step_type == 'iter' else None,
            every_n_epochs=chkpt_every_n_steps if chkpt_step_type == 'epoch' else None,
            save_last=True,
            chkpt_viz=chkpt_viz
        )
        callbacks.append(checkpoint_callback)

    logger = TensorBoardLogger(save_dir=os.path.join(chkpt_save_dir or chkpt_save_master_dir or ".", "logs"), name=train_id)

    trainer = L.Trainer(
        max_epochs=max_steps if step_unit == 'epoch' else -1,
        max_steps=max_steps if step_unit == 'iter' else -1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=is_progress_bar,
        log_every_n_steps=progress_bar_log_iter_interval,
        deterministic=determinism_kwargs.get('deterministic', False) if determinism_kwargs else False,
    )

    return trainer, lightning_model
