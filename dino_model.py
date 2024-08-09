from pytorch_lightning import LightningModule
from typing import Literal, Union
import torch
from torch import nn
from medmnist_dataset import DataFlag
from DINO_ViT_source.vision_transformer import instantiate_model, weights_map

from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy, MultilabelAccuracy, MulticlassAUROC, BinaryAUROC, MultilabelAUROC


class OrdinalRegressionLoss(nn.Module):
    """
    Loss for ordinal regression task. UNDER CONSTRUCTION!
    """
    def __init__(self):
        super(OrdinalRegressionLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # outputs: shape [batch_size, num_classes-1]
        # targets: shape [batch_size]
        targets = targets.unsqueeze(1)  # shape [batch_size, 1]
        targets = (targets >= torch.arange(outputs.size(1)).float()).float().to(outputs.device)
        return self.bce_loss(outputs, targets)


# Dictionary mapping Datasets to proper criterions and input channels, out logits params:
SETUP_DICT = {
    DataFlag.PATHMNIST: {"criterion": nn.CrossEntropyLoss(), 
                         "metrics": [MulticlassAccuracy(num_classes=9), MulticlassAUROC(num_classes=9)], 
                         "in_chans": 3, 
                         "out_logits": 9},
    DataFlag.OCTMNIST: {"criterion": nn.CrossEntropyLoss(), 
                        "metrics": [MulticlassAccuracy(num_classes=4), MulticlassAUROC(num_classes=4)], 
                        "in_chans": 1, 
                        "out_logits": 4},
    DataFlag.PNEUMONIAMNIST: {"criterion": nn.BCELoss, 
                              "metrics": [BinaryAccuracy, BinaryAUROC], 
                              "in_chans": 1, 
                              "out_logits": 1},
    DataFlag.CHESTMNIST: {"criterion": nn.BCELoss, 
                          "metrics": [MultilabelAccuracy(num_labels=14), MultilabelAUROC(num_labels=14)], 
                          "in_chans": 1, 
                          "out_logits": 14},
    DataFlag.DERMAMNIST: {"criterion": nn.CrossEntropyLoss, 
                          "metrics": [MulticlassAccuracy(num_classes=7), MulticlassAUROC(num_classes=7)], 
                          "in_chans": 3, 
                          "out_logits": 7},
    DataFlag.RETINAMNIST: {"criterion": OrdinalRegressionLoss, 
                           "metrics": [MulticlassAccuracy(num_classes=5), MulticlassAUROC(num_classes=5)], 
                           "in_chans": 3, 
                           "out_logits": 5},
    DataFlag.BREASTMNIST: {"criterion": nn.BCELoss, 
                           "metrics": [BinaryAccuracy, BinaryAUROC], 
                           "in_chans": 1, 
                           "out_logits": 1},
    DataFlag.BLOODMNIST: {"criterion": nn.CrossEntropyLoss, 
                          "metrics": [MulticlassAccuracy(num_classes=8), MulticlassAUROC(num_classes=8)], 
                          "in_chans": 3, 
                          "out_logits": 8},
    DataFlag.TISSUEMNIST: {"criterion": nn.CrossEntropyLoss, 
                           "metrics": [MulticlassAccuracy(num_classes=8), MulticlassAUROC(num_classes=8)], 
                           "in_chans": 1, 
                           "out_logits": 8},
    DataFlag.ORGANAMNIST: {"criterion": nn.CrossEntropyLoss, 
                           "metrics": [MulticlassAccuracy(num_classes=11), MulticlassAUROC(num_classes=11)], 
                           "in_chans": 1, 
                           "out_logits": 11},
    DataFlag.ORGANCNMIST: {"criterion": nn.CrossEntropyLoss, 
                           "metrics": [MulticlassAccuracy(num_classes=11), MulticlassAUROC(num_classes=11)], 
                           "in_chans": 1, 
                           "out_logits": 11},
    DataFlag.ORGANSNMIST: {"criterion": nn.CrossEntropyLoss, 
                           "metrics": [MulticlassAccuracy(num_classes=11), MulticlassAUROC(num_classes=11)], 
                           "in_chans": 1, 
                           "out_logits": 11}
}



def set_encoder_dropout_p(module, dropout_p):
    if isinstance(module, nn.Dropout):
        # Sets dropout probability for dropout layers within encoder blocks.
        module.p = dropout_p


class DINO_Model(LightningModule):
    __doc__ = """Vision Transformer based on DINO backbone with adjustable number of input channels."""
    def __init__(
        self,
        model_type:Literal["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16"],
        dataset_name:DataFlag=DataFlag.PATHMNIST,
        trainable_layers:Union[int, Literal["all"]]="all",
        backbone_dropout:float=0.0,
        max_lr:float=1e-3,
        lr_anneal_strategy:Literal["linear", "cos"]="cos",
        init_div_factor:int=100,
        final_div_factor:int=10000,
        total_steps:int=100,
        pct_start:float=0.10
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_type=model_type
        self.in_chans=SETUP_DICT[dataset_name]["in_chans"]
        self.out_logits=SETUP_DICT[dataset_name]["out_logits"]
        self.trainable_layers=trainable_layers
        self.backbone_dropout=backbone_dropout
        self.max_lr=max_lr
        self.lr_anneal_strategy=lr_anneal_strategy
        self.init_div_factor=init_div_factor
        self.final_div_factor=final_div_factor
        self.total_steps=total_steps
        self.pct_start=pct_start

        self.criterion = SETUP_DICT[dataset_name]["criterion"]
        self.metrics = SETUP_DICT[dataset_name]["metrics"]
        

        self.backbone=self._backbone_setting()
        self.mlp_head=nn.Sequential(nn.Linear(self.backbone.embed_dim, self.out_logits))

    def _backbone_setting(self, in_chans=None, model_type=None, backbone_dropout=None, trainable_layers=None):
        
        in_chans = in_chans if in_chans is not None else self.in_chans
        model_type = model_type if model_type is not None else self.model_type
        backbone_dropout = backbone_dropout if backbone_dropout is not None else self.backbone_dropout
        trainable_layers = trainable_layers if trainable_layers is not None else self.trainable_layers
        
        ## Backbone setting:
        if model_type in ["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16"]:
            state_dict = torch.load(f"DINO_ViT_source/pretrained/{weights_map[model_type]}")
            if in_chans != 3:
                # Adjusting embedding weights with "weight inflation" strategy:
                emb_weights = state_dict["patch_embed.proj.weight"]
                emb_weights = emb_weights.sum(1, keepdim=True) # Reducing color channels to one channel - gray channel.
                emb_weights = emb_weights.repeat(1, in_chans, 1, 1)/in_chans # Inflating weights.
                state_dict["patch_embed.proj.weight"] = emb_weights
            
            # Instantiating model:
            backbone = instantiate_model(model_type, in_chans=in_chans)
            backbone.load_state_dict(state_dict, strict=True)

            # changing dropout values in the backbone:
            if backbone_dropout > 0.0:
                backbone.apply(lambda module: set_encoder_dropout_p(module, dropout_p=backbone_dropout))

            # Freezing layers (if needed):
            if trainable_layers != "all":
                all_layers = len(list(self.backbone.parameters()))
                for i, p in enumerate(self.backbone.parameters()):
                    if i < (all_layers - trainable_layers):
                        p.requires_grad = False
        else:
            raise Exception("Provided model name is not recognized.")
        
        return backbone

    def setup(self, stage=None):
        # Move criterion and metrics to the appropriate device
        device = self.device
        self.criterion = self.criterion.to(device)
        self.metrics = [metric.to(device) for metric in self.metrics]

    def _compute_metrics(self, predictions, targets):
        results = {}
        for metric in self.metrics:
            metric_value = metric(predictions, targets)
            results[metric.__class__.__name__] = metric_value
        return results

    def forward(self, imgs):
        x = self.backbone(imgs)
        x = self.mlp_head(x)
        return x
        
    def common_step(self, batch):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = self.criterion(logits, labels.squeeze())
        return [logits, loss, labels.squeeze()]

    def training_step(self, batch, batch_idx):
        _, loss, _ = self.common_step(batch)
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss, labels = self.common_step(batch)
        results = self._compute_metrics(predictions=logits, targets=labels)
        for res in results.keys():
            self.log("Val_"+res, results[res], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits, loss, labels = self.common_step(batch)
        for res in results.keys():
            self.log("Test_"+res, results[res], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           div_factor=self.init_div_factor,
                                                           final_div_factor=self.final_div_factor,
                                                           max_lr=self.max_lr,
                                                           cycle_momentum=True,
                                                           pct_start=self.pct_start,
                                                           anneal_strategy=self.lr_anneal_strategy,
                                                           total_steps=self.total_steps
                                                          )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler]
