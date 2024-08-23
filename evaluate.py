import argparse
from typing import List
import torch
import pytorch_lightning as pl
import pandas as pd
import pickle
import os

from medmnist_dataset import DataFlag
from data_module import DataModule2D
from dino_model import DINO_Model


def validate_model(model, dm, **kwargs):
    # Unpack kwargs
    DATA_FLAG = kwargs.get("DATA_FLAG")
    MODEL_TYPE = kwargs.get("MODEL_TYPE")
    MODEL_NR = kwargs.get("MODEL_NR")
    VERSION = kwargs.get("VERSION")
    
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=32, logger=False)
    val_results = trainer.validate(model, dm)
    metrics = {
        "ID": f"{DATA_FLAG}_{MODEL_TYPE}_{MODEL_NR}" if VERSION==None else f"{DATA_FLAG}_{MODEL_TYPE}_{MODEL_NR}_{VERSION}",
        "Val_AUROC": round(val_results[0]["Val_MulticlassAUROC"], 3),
        "Val_ACC": round(val_results[0]["Val_MulticlassAccuracy"], 3)
    }
    return metrics


def test_model(model, dm, **kwargs):
    # Unpack kwargs
    DATA_FLAG = kwargs.get("DATA_FLAG")
    MODEL_TYPE = kwargs.get("MODEL_TYPE")
    MODEL_NR = kwargs.get("MODEL_NR")
    VERSION = kwargs.get("VERSION")
    
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=32, logger=False)
    test_results = trainer.test(model, dm)
    metrics = {
        "ID": f"{DATA_FLAG}_{MODEL_TYPE}_{MODEL_NR}" if VERSION==None else f"{DATA_FLAG}_{MODEL_TYPE}_{MODEL_NR}_{VERSION}",
        "Test_AUROC": round(test_results[0]["Test_MulticlassAUROC"], 3),
        "Test_ACC": round(test_results[0]["Test_MulticlassAccuracy"], 3),
        "Test_loss": round(test_results[0]["test_loss"], 3)
    }
    return metrics


def save_results(val_metrics, test_metrics):
    assert val_metrics["ID"] == test_metrics["ID"]
    new_row = pd.DataFrame({
        "ID": [val_metrics["ID"]],
        "Val_ACC": [val_metrics["Val_ACC"]],
        "Val_AUROC": [val_metrics["Val_AUROC"]],
        "Test_ACC": [test_metrics["Test_ACC"]],
        "Test_AUROC": [test_metrics["Test_AUROC"]],
        "Test_loss": [test_metrics["Test_loss"]]
    })
    
    results_path = "/home/dzban112/MedMNIST/results/"
    results_file = os.path.join(results_path, "results_table.csv") 
    if not os.path.exists(results_file):
        results_table = new_row
        results_table.to_csv(results_file, index=False)
    else:
        results_table = pd.read_csv(results_file)
        # Appends the new row to the existing DataFrame
        results_table = pd.concat([results_table, new_row], ignore_index=True)
        results_table.to_csv(results_file, index=False)
    

def model_evaluation(**kwargs):
    """
    Evaluates model on the test set.
    Arguments:
    - version: Optional; specific version of the checkpoint to load.
    - **kwargs: Any additional keyword arguments required for setting up paths and data.
    """
    # Unpack kwargs
    DATA_FLAG = kwargs.get("DATA_FLAG")
    MODEL_TYPE = kwargs.get("MODEL_TYPE")
    MODEL_NR = kwargs.get("MODEL_NR")
    VERSION = kwargs.get("VERSION")
    CHECKPOINTS_PATH = kwargs.get("CHECKPOINTS_PATH")

    # DataModule setup:
    dm_hyp_file = f"/home/dzban112/MedMNIST/dm_cache/{DATA_FLAG}_{MODEL_TYPE}_{MODEL_NR}_dm_hyp.pkl"
    with open(dm_hyp_file, 'rb') as file:
        dm_dict = pickle.load(file)
    
    dm = DataModule2D(**dm_dict)
        
    if VERSION is None:
        ckpt_path = CHECKPOINTS_PATH + f"{DATA_FLAG}_{MODEL_TYPE}_{MODEL_NR}.ckpt"
    else:
        ckpt_path = CHECKPOINTS_PATH + f"{DATA_FLAG}_{MODEL_TYPE}_{MODEL_NR}-{VERSION}.ckpt"
    
    # Model setup:
    model = DINO_Model.load_from_checkpoint(ckpt_path, strict=False)

    # Eval
    val_metrics = validate_model(model, dm, **kwargs)
    test_metrics = test_model(model, dm, **kwargs)

    # Save results
    save_results(val_metrics, test_metrics)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a DINO model on the MedMNIST dataset.")
    
    parser.add_argument("--data_flag", type=str, required=True, help="The data flag.")
    parser.add_argument("--model_type", type=str, required=True, help="The model type.")
    parser.add_argument("--model_nr", type=int, required=True, help="The model number.")
    parser.add_argument("--checkpoints_path", type=str, default="/home/dzban112/MedMNIST/ckpt/", help="The path to the model checkpoints.")
    parser.add_argument("--version", type=str, default=None, help="The specific version of the checkpoint to load (optional).")
    
    args = parser.parse_args()
    
    # Pack the arguments into a dictionary
    params = {
        "DATA_FLAG": args.data_flag,
        "MODEL_TYPE": args.model_type,
        "MODEL_NR": args.model_nr,
        "VERSION": args.version,
        "CHECKPOINTS_PATH": args.checkpoints_path
    }
    
    # Perform the evaluation
    model_evaluation(**params)
