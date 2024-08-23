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
MODEL_NR:int = 4
SAVE_TOP_CKPTS:int = 1
WANDB_PROJECT:str = "ORGANAMNIST"
MODEL_TYPE:Literal["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16"] = "dino_vits8"
DATA_FLAG:DataFlag = DataFlag.ORGANAMNIST
IMG_SIZE:Literal[28,64,128,224] = 224
TRAINABLE_LAYERS:Union[int, Literal["all"]] = "all"
USE_CLASS_WEIGHTS:bool = True
EPOCHS:int = 20
BATCH_SIZE:int = 31
MAX_LR:float = 3e-5

LR_SCHEDULE:Literal["fixed", "plateau", "one_cycle"] = "one_cycle"
## If LR_SCHEDULE == "plateau":
PATIENCE:int = 5
## If LR_SCHEDULE == "fixed":
MILESTONES:List[int] = [3, 6, 8, 9]
GAMMA:float = 0.1
## If LR_SCHEDULE == "one_cycle":
LR_ANNEAL_STRATEGY:Literal["linear", "cos"] = "cos"
INIT_DIV_FACTOR:int = 50
FINAL_DIV_FACTOR:int = 10000
PCT_START:float = 0.30

ACCUMULATE_GRAD_BATCHES:int = 3
DEVICES:int = 4
SELECTED_TRAIN_TRANSFORMS:List[str] = ["ToImage","DoDtype","RandomAdjustSharpness", "ColorJitter", "RandomHorizontalFlip", "RandomResizedCrop", "Normalize"]
BCKB_DROPOUT:float = 0.12
NUM_WORKERS:int = 4


LR_PARAMS={
    "patience": PATIENCE,
    "milestones": MILESTONES,
    "gamma": GAMMA,
    "lr_anneal_strategy": LR_ANNEAL_STRATEGY,
    "init_div_factor": INIT_DIV_FACTOR,
    "final_div_factor": FINAL_DIV_FACTOR,
    "pct_start": PCT_START
}


def save_dm_dict():
    dm_dict = {
    "data_flag": DATA_FLAG,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,  
    "num_workers": NUM_WORKERS,
    "selected_train_transforms": SELECTED_TRAIN_TRANSFORMS,
    "mmap_mode": "r" if IMG_SIZE == 224 else None
    }

    with open(f"/home/dzban112/MedMNIST/dm_cache/{DATA_FLAG.value}_{MODEL_TYPE}_{MODEL_NR}_dm_hyp.pkl", 'wb') as file:
        pickle.dump(dm_dict, file)
    

def train_model():
    # add a checkpoint callback that saves the model with the lowest validation loss
    checkpoint_name = f"{DATA_FLAG.value}_{MODEL_TYPE}_{MODEL_NR}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="ckpt/",
        filename=checkpoint_name,
        save_top_k=SAVE_TOP_CKPTS,
        monitor="val_loss",
        mode="min",
        enable_version_counter=True
    )

    # Logger:
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=f"{WANDB_PROJECT}_{MODEL_TYPE}_{MODEL_NR}", job_type='train')
    wandb_logger.log_hyperparams({
        "model_nr": MODEL_NR,
        "save_top_ckpts": SAVE_TOP_CKPTS,
        "model_type": MODEL_TYPE,
        "data_flag": DATA_FLAG.value,
        "img_size": IMG_SIZE,
        "trainable_layers": TRAINABLE_LAYERS,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_lr": MAX_LR,
        "lr_schedule": LR_SCHEDULE,
        "patience": PATIENCE if LR_SCHEDULE == "plateau" else None,
        "milestones": MILESTONES if LR_SCHEDULE == "fixed" else None,
        "gamma": GAMMA if LR_SCHEDULE == "fixed" else None,
        "lr_anneal_strategy": LR_ANNEAL_STRATEGY if LR_SCHEDULE == "one_cycle" else None,
        "init_div_factor": INIT_DIV_FACTOR if LR_SCHEDULE == "one_cycle" else None,
        "final_div_factor": FINAL_DIV_FACTOR if LR_SCHEDULE == "one_cycle" else None,
        "pct_start": PCT_START if LR_SCHEDULE == "one_cycle" else None,
        "accumulate_grad": ACCUMULATE_GRAD_BATCHES,
        "devices": DEVICES,
        "selected_train_transforms": SELECTED_TRAIN_TRANSFORMS,
        "backbone_dropout": BCKB_DROPOUT,
        "num_workers": NUM_WORKERS
    })

    # Cleaning cache:
    torch.cuda.empty_cache()
    
    torch.set_float32_matmul_precision('medium')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(accelerator="gpu", devices=DEVICES, strategy="ddp",
                         precision=32, max_epochs=EPOCHS,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=wandb_logger,
                         log_every_n_steps=20,
                         accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES
                        )

    dm = DataModule2D(
        data_flag=DATA_FLAG,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        selected_train_transforms=SELECTED_TRAIN_TRANSFORMS,
        mmap_mode = "r" if IMG_SIZE==224 else None
    )

    save_dm_dict()

    model = DINO_Model(
        model_type=MODEL_TYPE,
        dataset_name=DATA_FLAG,
        trainable_layers=TRAINABLE_LAYERS,
        backbone_dropout=BCKB_DROPOUT,
        max_lr=MAX_LR,
        lr_schedule=LR_SCHEDULE,
        lr_params=LR_PARAMS,
        use_class_weights=USE_CLASS_WEIGHTS
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
    
