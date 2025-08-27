import torch.nn as nn
import torch.nn.functional as F


class ClsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        loss = 0
        cls_loss = F.cross_entropy(preds, labels)
        loss += cls_loss
        
        loss_dict = {
            "loss/cls_loss": cls_loss.item(),
        }
        return loss, loss_dict
