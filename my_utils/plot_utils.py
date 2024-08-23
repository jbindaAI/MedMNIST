# Module containing some useful plotting functions.

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from my_utils.att_cdam_utils import get_cmap
import pandas as pd
import random
import pickle
import os


def plot_res_class(original_img, maps, model_output, save_name:Optional[str]=None):
    """Using matplotlib, plot the original image and the relevance maps"""
    maps2plot = []
    target_names = []
    maps2plot.append(maps[0])
    target_names.append("Attention Map")
    for key in maps[1].keys():
        maps2plot.append(maps[1][key])
        target_names.append("CDAM"+ "\nMalignant class")

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), layout='constrained')
    ## Original:
    axs[0].imshow(original_img[:,:,0], cmap='gray')
    axs[0].set_title("Original")
    axs[0].tick_params(axis='both', 
                       which='both', 
                       bottom=False, 
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )
    ## Attention map:
    axs[1].imshow(maps2plot[0], cmap=get_cmap(maps2plot[0]))
    axs[1].set_title(target_names[0])
    axs[1].tick_params(axis='both', 
                       which='both', 
                       bottom=False, 
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )
    ## CDAM Map:
    cdam = axs[2].imshow(maps2plot[1], cmap=get_cmap(maps2plot[1]))
    axs[2].set_title(target_names[1])
    axs[2].tick_params(axis='both', 
                       which='both', 
                       bottom=False, 
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )
    fig.colorbar(cdam, ax=axs[2], shrink=0.6)
    plt.suptitle(f"Probability of malignant class: {round(model_output, 2)}")

    if save_name:
        import os
        if not os.path.exists("relevance_maps"):
            os.makedirs("relevance_maps")
        plt.savefig(f"relevance_maps/{save_name}", format="png", transparent=True, bbox_inches='tight')

    return None

    
def plot_CDAM(original_img, attention_map, cdam_maps, preds, save_name:Optional[str]=None):
    """Using matplotlib, plot the original image and the relevance maps"""
    maps2plot=[]
    target_names=[]
    for key in cdam_maps.keys():
        maps2plot.append(cdam_maps[key])
        target_names.append(key)

    for map_, title in zip(maps2plot, target_names):
        # Biomarker Regression:
        fig, axs = plt.subplots(1, 3, figsize=(10, 4), layout='constrained')

        ## Original img:
        cmap = "grey" if original_img.mode == "L" else None
        axs[0].imshow(original_img, cmap=cmap)
        axs[0].set_title("Original")
        axs[0].tick_params(axis='both', 
                           which='both', 
                           bottom=False, 
                           left=False,
                           labelbottom=False,
                           labelleft=False
                          )
        
        ## Attention map:
        axs[1].imshow(attention_map, cmap=get_cmap(attention_map))
        axs[1].set_title("Attention Map")
        axs[1].tick_params(axis='both', 
                           which='both', 
                           bottom=False, 
                           left=False,
                           labelbottom=False,
                           labelleft=False
                          )
                      
        ## CDAM Map:
        cdam = axs[2].imshow(map_, cmap=get_cmap(map_))
        axs[2].set_title(title)
        axs[2].tick_params(axis='both', 
                           which='both', 
                           bottom=False, 
                           left=False,
                           labelbottom=False,
                           labelleft=False
                          )
        fig.colorbar(cdam, ax=axs[2], shrink=0.6)
        
        plt.suptitle("CDAM maps")
        plt.show()
    
        if save_name:
            if not os.path.exists("relevance_maps"):
                os.makedirs("relevance_maps")
            plt.savefig(f"relevance_maps/{save_name}""_"+title, format="png", transparent=True, bbox_inches='tight')
    return None


def plot_ori_att_reg(original_img, attention_map):
    fig, axs = plt.subplots(1, 2, figsize=(5, 10))
    axs[0].imshow(original_img[:,:,0], cmap='gray')
    axs[0].set_title("Original")
    axs[0].tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )
    axs[1].imshow(attention_map, cmap=get_cmap(attention_map))
    axs[1].set_title("Attention Map")
    axs[1].tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )

    return None