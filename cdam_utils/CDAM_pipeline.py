# Imports
import torch
import pickle
import os
import numpy as np
from types import MethodType
from typing import Literal, Tuple, Dict
from dino_model import DINO_Model
from medmnist_dataset import get_medmnist_dataset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
from cdam_utils.att_cdam_utils import get_maps
from cdam_utils.plot_utils import plot_CDAM2

current_directory = os.getcwd()

model_map = {
    "pathmnist": {"task": "classification", "model_nr": 3},
    "bloodmnist": {"task": "classification", "model_nr": 1},
    "organamnist": {"task": "classification", "model_nr": 4}
}


# MAIN


def load_fitted_factors(dataset_name:str):
    with open(current_directory+f"/fitted_factors/{dataset_name}.pkl", 'rb') as f:
        fitted_factors = pickle.load(f)
    return fitted_factors

                                     
def apply_transforms(image, device, mean, std):
    image = F.pil_to_tensor(image)
    image = F.convert_image_dtype(image, dtype=torch.float32)
    image = F.normalize(image, mean=mean, std=std)
    image = image.unsqueeze(0).to(device)
    return image


def cdam_pipeline(DATASET,
                  IDX:int,
                  MODEL_BCKB: Literal["dino_vits8", "dino_vitb8", 
                  "dino_vits16", "dino_vitb16", "vit_b_16", "vit_l_16", "dinov2_vits14_reg", "dinov2_vitb14_reg"],
                  CKPT_VERSION: int,
                  UPPER_QUANTILE: float,
                 )->Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Function running CDAM pipeline.
    """
    # Translating arguments:
    ckpt_versions = {1:"",
                     2:"-v1",
                     3:"-v2"}
    patch_sizes = {"dino_vits8":8,
                   "dino_vitb8":8,
                   "dino_vits16":16,
                   "dino_vitb16":16,
                   "vit_b_16":16,
                   "dinov2_vits14_reg":14,
                   "dinov2_vitb14_reg":14
                  }
    PATCH_SIZE = patch_sizes[MODEL_BCKB]

    MODEL_NR = model_map[DATASET.info['python_class'].lower()]["model_nr"]
    
    # Loading model and registering hooks:
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    model = DINO_Model.load_from_checkpoint(current_directory + f"/ckpt/{DATASET.info['python_class'].lower()}_{MODEL_BCKB}_{MODEL_NR}{ckpt_versions[CKPT_VERSION]}.ckpt", strict=False).to(device).eval()

    ## Creating hooks:
    activation = {}
    def get_activation(name):
        """
        Function to extract activations before the last MHSA layer.
        """
        def hook(model, input, output):
            activation[name] = output[0].detach()
        return hook
    
    grad = {}
    def get_gradient(name):
        """
        Function to extract gradients.
        """
        def hook(model, input, output):
            grad[name] = output
        return hook
    
    ## Registering hooks:
    ### We store the: 
    #### i) normalized activations entering the last MHSA layer.
    #### ii) gradients wrt the normalized inputs to the final attention layer.
    ### Both are required to compute CDAM score.
    ### We don't need to register hook on MHSA to extract attention weights in case of DINO, because DINO backbone has it already implemented.
    if MODEL_BCKB in ["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16", "dinov2_vits14_reg", "dinov2_vitb14_reg"]:
        final_block_norm1 = model.backbone.blocks[-1].norm1
        
    activation_hook = final_block_norm1.register_forward_hook(
        get_activation("last_att_in"))
    grad_hook = final_block_norm1.register_full_backward_hook(
        get_gradient("last_att_in"))

    # Taking mean and std from fitted factors
    MEAN, STD = load_fitted_factors(DATASET.info["python_class"].lower())

    # Loading image from repository:
    original_img, label = DATASET[IDX]
    img = original_img.copy()
    img = apply_transforms(img, device, MEAN, STD)
    
    # Model inference:
    model = model.to(device)
    attention_map, CDAM_ALL, model_output = get_maps(model, img, grad, activation, class2idx=DATASET.info['label'], patch_size=PATCH_SIZE, upper_quantile=UPPER_QUANTILE)

    return (original_img, attention_map, CDAM_ALL, model_output)


import random
from functools import wraps

def random_id(DATASET)->int:
    n = len(DATASET)
    random_idx = random.randint(0, n)
    return random_idx

def auto_random_idx(min_idx, max_idx):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            random_idx = random.randint(min_idx, max_idx)
            return func(*args, IDX=random_idx, **kwargs)
        return wrapper
    return decorator


#@auto_random_idx(min_idx=0, max_idx=100)
def call_CDAM(DATASET, IDX:int, MODEL_BCKB:str, CKPT_VERSION:int, UPPER_QUANTILE:float):
    original_img, attention_map, CDAM_maps, model_output = cdam_pipeline(DATASET=DATASET,
                                                                         IDX=IDX,
                                                                         MODEL_BCKB=MODEL_BCKB,
                                                                         CKPT_VERSION=CKPT_VERSION,
                                                                         UPPER_QUANTILE=UPPER_QUANTILE
                                                                        )
    groundTruth = DATASET[IDX][1]
    print(f"Ground truth label: {DATASET.info['label'][str(groundTruth[0])]}. Predicted: {DATASET.info['label'][str(np.argmax(model_output))]}.")
    #plot_CDAM(original_img, attention_map, CDAM_maps, model_output)
    plot_CDAM2(original_img, attention_map, CDAM_maps, model_output)
    return original_img, attention_map, CDAM_maps, model_output