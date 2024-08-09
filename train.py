# IMPORT PACKAGES
from medmnist_dataset import get_medmnist_dataset, DataFlag
from data_module import DataModule2D
from dino_model import DINO_Model

import math
import torch
import wandb
import pickle
from typing import Literal, Union, List
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor


## HYPERPARAMETERS:
MODEL_NR:int = 2
SAVE_TOP_CKPTS:int = 3
WANDB_PROJECT:str = "PATHMNIST"
MODEL_TYPE:Literal["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16"] = "dino_vits8"
DATA_FLAG:DataFlag = DataFlag.PATHMNIST
IMG_SIZE:Literal[28,64,128,224]=128
TRAINABLE_LAYERS:Union[int, Literal["all"]] = "all"
EPOCHS:int = 15
BATCH_SIZE:int = 32
MAX_LR:float = 3e-4
LR_ANNEAL_STRATEGY:Literal["linear", "cos"] = "cos"
INIT_DIV_FACTOR:int = 50
FINAL_DIV_FACTOR:int = 10000
PCT_START:float = 0.2
SELECTED_TRAIN_TRANSFORMS:List[str] = ["ToImage", "DoDtype","Normalize", "RandomResizedCrop"]
BCKB_DROPOUT:float = 0.12
NUM_WORKERS:int = 4


def train_model():
    # add a checkpoint callback that saves the model with the lowest validation loss
    checkpoint_name = f"{MODEL_TYPE}_{MODEL_NR}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="ckpt/",
        filename=checkpoint_name,
        save_top_k=SAVE_TOP_CKPTS,
        monitor="val_loss",
        mode="min",
        enable_version_counter=True
    )

    # Logger:
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=f"{MODEL_TYPE}_{MODEL_NR}", job_type='train')
    wandb_logger.log_hyperparams({
        "model_nr": MODEL_NR,
        "save_top_ckpts": SAVE_TOP_CKPTS,
        "model_type": MODEL_TYPE,
        "data_flag": DATA_FLAG.value,
        "img_size": IMG_SIZE,
        "trainable_layers": TRAINABLE_LAYERS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_lr": MAX_LR,
        "lr_anneal_strategy": LR_ANNEAL_STRATEGY,
        "init_div_factor": INIT_DIV_FACTOR,
        "final_div_factor": FINAL_DIV_FACTOR,
        "pct_start": PCT_START,
        "selected_train_transforms": SELECTED_TRAIN_TRANSFORMS,
        "backbone_dropout": BCKB_DROPOUT,
        "num_workers": NUM_WORKERS
    })

    # Cleaning cache:
    torch.cuda.empty_cache()
    
    torch.set_float32_matmul_precision('medium')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(accelerator="gpu", devices=3, strategy="ddp",
                         precision=32, max_epochs=EPOCHS,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=wandb_logger,
                         log_every_n_steps=20
                        )

    dm = DataModule2D(
        data_flag=DataFlag.PATHMNIST,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        selected_train_transforms=SELECTED_TRAIN_TRANSFORMS
    )
    dm.prepare_data()
    dm.setup()
    total_train_steps = dm.get_epoch_train_steps() * EPOCHS

    model = DINO_Model(
        model_type=MODEL_TYPE,
        dataset_name=DATA_FLAG,
        trainable_layers=TRAINABLE_LAYERS,
        backbone_dropout=BCKB_DROPOUT,
        max_lr=MAX_LR,
        lr_anneal_strategy=LR_ANNEAL_STRATEGY,
        init_div_factor=INIT_DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR,
        total_steps=total_train_steps,
        pct_start=PCT_START
    )

    trainer.fit(model, dm)

    # Free up memory
    del trainer
    del model
    del dm
    
    #Finishing run
    wandb.finish()


if __name__ == "__main__":
    train_model()
    
