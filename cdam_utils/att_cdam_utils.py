import matplotlib.colors as clr
import cmasher as cmr
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle

# Globals
PATCH_SIZE=8

# Plotting utils
mycmap = clr.LinearSegmentedColormap.from_list(
    "Random gradient 1030",
    (
        (0.000, (0.000, 0.890, 1.000)),
        (0.370, (0.263, 0.443, 0.671)),
        (0.500, (0.000, 0.000, 0.000)),
        (0.630, (0.545, 0.353, 0.267)),
        (1.000, (1.000, 0.651, 0.000)),
    ),
)


def get_cmap(heatmap):
    """Return a diverging colormap, such that 0 is at the center(black)"""
    if heatmap.min() > 0 and heatmap.max() > 0:
        bottom = 0.5
        top = 1.0
    elif heatmap.min() < 0 and heatmap.max() < 0:
        bottom = 0.0
        top = 0.5
    else:
        bottom = 0.5 - abs((heatmap.min() / abs(heatmap).max()) / 2)
        top = 0.5 + abs((heatmap.max() / abs(heatmap).max()) / 2)
    return cmr.get_sub_cmap(mycmap, bottom, top)


# Obtaining maps
## Obtaining attention map
def get_attention_map(model, sample_img, head=None, return_raw=False):
    """This returns the attentions when CLS token is used as query in the last attention layer, averaged over all attention heads"""
    if model.model_type in ["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16"]:
        attentions = model.backbone.get_last_selfattention(sample_img)
        nh = attentions.shape[1]  # number of heads
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1) # Getting attention weights when CLS token is a query.
        
    elif model.model_type in ["dinov2_vits14_reg", "dinov2_vitb14_reg"]:
        attentions = model.backbone.get_last_selfattention(sample_img)
        nh = attentions.shape[1]  # number of heads
        attentions = attentions[0, :, 0, 5:].reshape(nh, -1) # Getting attention weights when CLS token is a query. Avoiding register tokens.

    w_featmap = sample_img.shape[-2] // PATCH_SIZE
    h_featmap = sample_img.shape[-1] // PATCH_SIZE

    if return_raw:
        return torch.mean(attentions, dim=0).squeeze().detach().cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = torch.nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest")[0]
    if head == None:
        mean_attention = torch.mean(attentions, dim=0).squeeze().detach().cpu().numpy()
        return mean_attention
    else:
        return attentions[head].squeeze().detach().cpu().numpy()


## Obtaining CDAM map
def get_CDAM(class_score, model_bckb, activation, grad, upper_quantile:float):
    """The class_score is the activation of a neuron in the prediction vector"""
    class_score.backward(retain_graph=True)
    if model_bckb in ["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16", "vit_b_16", "vit_l_16"]:
        # Token 0 is CLS and others are image patch tokens
        tokens = activation["last_att_in"][1:]
        grads = grad["last_att_in"][0][0, 1:]
        
    elif model_bckb in ["dinov2_vits14_reg", "dinov2_vitb14_reg"]:
        # Token 0 is CLS, then tokens 1-4 are registers. Tokens above 5 are "true" tokens.
        tokens = activation["last_att_in"][5:]
        grads = grad["last_att_in"][0][0, 5:]

    attention_scores = torch.tensor(
        [torch.dot(tokens[i], grads[i]) for i in range(tokens.shape[0])]
    )

    res = {}
    raw = attention_scores.detach().clone()
    res['raw'] = raw.numpy()
    # Z-score Normalization
    normalized = (attention_scores-attention_scores.mean())/attention_scores.std()
    res['normalized'] = normalized
    # Clipping
    clipped = torch.clamp(
            attention_scores,
            min=torch.quantile(attention_scores, 1.0-upper_quantile),
            max=torch.quantile(attention_scores, upper_quantile),
        )
    res['clipped'] = clipped

    # Reshaping into image shape
    w = int(np.sqrt(attention_scores.squeeze().shape[0]))
    res['raw_upsampled'] = torch.nn.functional.interpolate(raw.reshape(w,w).unsqueeze(0).unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest").squeeze().numpy()
    res['normalized_upsampled'] = torch.nn.functional.interpolate(normalized.reshape(w,w).unsqueeze(0).unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest").squeeze().numpy()
    res['clipped_upsampled'] = torch.nn.functional.interpolate(clipped.reshape(w,w).unsqueeze(0).unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest").squeeze().numpy()
    return res


## Wrapper to obtain both Attention map and CDAM map.
def get_maps(model, img, grad, activation, class2idx, patch_size, upper_quantile:float):
    """
    Wrapper function to get the attention map and the concept map for a given image and target class.
    """
    # Update global variable PATCH_SIZE:
    global PATCH_SIZE
    PATCH_SIZE=patch_size

    pred = model(img)
    attention_map = get_attention_map(model, img, return_raw=False)

    CDAM_maps_clipped = {}
    CDAM_maps_raw_upsampled = {}
    CDAM_maps_normalized = {}
    CDAM_score_raw = {} # not upsampled
    
    softmax = torch.nn.Softmax(dim=1)
    for class_, classname in class2idx.items():
        model.zero_grad()
        res = get_CDAM(
            class_score=pred[0][int(class_)],
            model_bckb=model.model_type,
            activation=activation,
            grad=grad,
            upper_quantile=upper_quantile
        )
        key = classname.split("(")[0]
        CDAM_maps_raw_upsampled[key]=res["raw_upsampled"]
        CDAM_maps_clipped[key]=res["clipped_upsampled"]
        CDAM_maps_normalized[key]=res["normalized_upsampled"]
        CDAM_score_raw[key]=res['raw']

    CDAM_ALL = {"CDAM_maps_clipped":CDAM_maps_clipped,
                "CDAM_maps_raw_upsampled":CDAM_maps_raw_upsampled,
                "CDAM_maps_normalized":CDAM_maps_normalized,
                "CDAM_score_raw":CDAM_score_raw}
    probs = softmax(pred).cpu().detach().squeeze().numpy()
    return attention_map,  CDAM_ALL , probs