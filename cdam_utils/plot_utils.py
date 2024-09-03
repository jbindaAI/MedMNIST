# Module containing some useful plotting functions.

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from cdam_utils.att_cdam_utils import get_cmap
from scipy.stats import skew
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


def plot_CDAM2(original_img, attention_map, cdam_maps, preds, save_name: Optional[str] = None):
    """Using matplotlib, plot the original image and the relevance maps"""
    keys = cdam_maps["CDAM_maps_raw_upsampled"].keys()

    for key in keys:
        # Create a grid of subplots: 1st row with 5 plots, 2nd row with 1 plot spanning all columns
        fig, axs = plt.subplots(2, 5, figsize=(15, 8), layout='constrained')

        # Original img:
        cmap = "grey" if original_img.mode == "L" else None
        axs[0][0].imshow(original_img, cmap=cmap)
        axs[0][0].set_title("Original")
        axs[0][0].axis('off')  # Hide axis

        # Attention map:
        axs[0][1].imshow(attention_map, cmap=get_cmap(attention_map))
        axs[0][1].set_title("Attention Map")
        axs[0][1].axis('off')  # Hide axis

        # Raw CDAM Map:
        map_ = cdam_maps["CDAM_maps_raw_upsampled"][key]
        cdam1 = axs[0][2].imshow(map_, cmap=get_cmap(map_))
        axs[0][2].set_title("Raw CDAM map")
        axs[0][2].axis('off')  # Hide axis

        # Calculate and display the sum of pixel values for the Raw CDAM map
        raw_sum = map_.sum()
        axs[0][2].text(0.5, -0.05, f"Sum: {raw_sum:.2f}", ha='center', va='top', transform=axs[0][2].transAxes, fontsize=10, color='black')

        # Clipped CDAM Map:
        map_ = cdam_maps["CDAM_maps_clipped"][key]
        cdam2 = axs[0][3].imshow(map_, cmap=get_cmap(map_))
        axs[0][3].set_title("Clipped CDAM map")
        axs[0][3].axis('off')  # Hide axis

        # Calculate and display the sum of pixel values for the Clipped CDAM map
        clipped_sum = map_.sum()
        axs[0][3].text(0.5, -0.05, f"Sum: {clipped_sum:.2f}", ha='center', va='top', transform=axs[0][3].transAxes, fontsize=10, color='black')

        # Normalized CDAM Map:
        map_ = cdam_maps["CDAM_maps_normalized"][key]
        cdam3 = axs[0][4].imshow(map_, cmap=get_cmap(map_))
        axs[0][4].set_title("Normalized CDAM map")
        axs[0][4].axis('off')  # Hide axis

        # Calculate and display the sum of pixel values for the Normalized CDAM map
        normalized_sum = map_.sum()
        axs[0][4].text(0.5, -0.05, f"Sum: {normalized_sum:.2f}", ha='center', va='top', transform=axs[0][4].transAxes, fontsize=10, color='black')

        # Optionally, add colorbars to each image
        fig.colorbar(cdam1, ax=axs[0][2], shrink=0.6)
        fig.colorbar(cdam2, ax=axs[0][3], shrink=0.6)
        fig.colorbar(cdam3, ax=axs[0][4], shrink=0.6)

        # Merge all columns in the second row for a wider histogram
        ax_hist = fig.add_subplot(2, 1, 2)

        # CDAM histogram
        pixel_values = cdam_maps["CDAM_score_raw"][key]
        sns.histplot(pixel_values, kde=True, bins=100, color="skyblue", edgecolor="black", ax=ax_hist)
        ax_hist.set_title("Raw CDAM histogram")
        ax_hist.set_xlabel("CDAM Values")
        ax_hist.set_ylabel("Frequency")

        # Hide unused subplots in the first row
        for i in range(5):
            axs[1][i].axis('off')

        skewness = skew(pixel_values)
        ax_hist.text(0.5, -0.15, f"Skewness: {skewness:.2f}", ha='center', va='top', transform=ax_hist.transAxes, fontsize=11, color='black')

        plt.suptitle(f"CDAM maps for {key}", y=0.99, fontsize=16)
        plt.show()

        if save_name:
            if not os.path.exists("relevance_maps"):
                os.makedirs("relevance_maps")
            plt.savefig(f"relevance_maps/{save_name}_{key}.png", format="png", transparent=True, bbox_inches='tight')

    return None


def plot_CDAM3(original_img, attention_map, cdam_maps, preds, save_name: Optional[str] = None):
    """Using matplotlib, plot the original image and the relevance maps"""
    keys = cdam_maps["CDAM_maps_raw_upsampled"].keys()

    for key in keys:
        # Create a grid of subplots: 1st row with 5 plots, 2nd row with 1 plot spanning all columns
        fig, axs = plt.subplots(2, 5, figsize=(15, 8), layout='constrained')

        # Original img:
        cmap = "grey" if original_img.mode == "L" else None
        axs[0][0].imshow(original_img, cmap=cmap)
        axs[0][0].set_title("Original")
        axs[0][0].axis('off')  # Hide axis

        # Attention map:
        axs[0][1].imshow(attention_map, cmap=get_cmap(attention_map))
        axs[0][1].set_title("Attention Map")
        axs[0][1].axis('off')  # Hide axis

        # Raw CDAM Map:
        map_ = cdam_maps["CDAM_maps_raw_upsampled"][key]
        cdam1 = axs[0][2].imshow(map_, cmap=get_cmap(map_))
        axs[0][2].set_title("Raw CDAM map")
        axs[0][2].axis('off')  # Hide axis
        
        # Calculate and display the sum of pixel values for the Raw CDAM map
        raw_sum = map_.sum()
        axs[0][2].text(0.5, -0.05, f"Sum: {raw_sum:.2f}", ha='center', va='top', transform=axs[0][2].transAxes, fontsize=10, color='black')

        # Clipped CDAM Map:
        map_ = cdam_maps["CDAM_maps_clipped"][key]
        cdam2 = axs[0][3].imshow(map_, cmap=get_cmap(map_))
        axs[0][3].set_title("Clipped CDAM map")
        axs[0][3].axis('off')  # Hide axis

        # Normalized CDAM Map:
        map_ = cdam_maps["CDAM_maps_normalized"][key]
        cdam3 = axs[0][4].imshow(map_, cmap=get_cmap(map_))
        axs[0][4].set_title("Normalized CDAM map")
        axs[0][4].axis('off')  # Hide axis

        # Optionally, add colorbars to each image
        fig.colorbar(cdam1, ax=axs[0][2], shrink=0.6)
        fig.colorbar(cdam2, ax=axs[0][3], shrink=0.6)
        fig.colorbar(cdam3, ax=axs[0][4], shrink=0.6)

        # Merge all columns in the second row for a wider histogram
        ax_hist = fig.add_subplot(2, 1, 2)

        # Commented-out histogram
        pixel_values = cdam_maps["CDAM_score_raw"][key]
        sns.histplot(pixel_values, kde=True, bins=100, color="skyblue", edgecolor="black", ax=ax_hist)
        ax_hist.set_title("Raw CDAM histogram")
        ax_hist.set_xlabel("Pixel Values")
        ax_hist.set_ylabel("Frequency")

        # Hide unused subplots in the first row
        for i in range(5):
            axs[1][i].axis('off')

        plt.suptitle(f"CDAM maps for {key}")
        plt.show()

        if save_name:
            if not os.path.exists("relevance_maps"):
                os.makedirs("relevance_maps")
            plt.savefig(f"relevance_maps/{save_name}_{key}.png", format="png", transparent=True, bbox_inches='tight')

    return None