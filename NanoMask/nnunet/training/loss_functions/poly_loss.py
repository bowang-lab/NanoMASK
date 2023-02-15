from typing import Optional
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, SoftDiceLossSquared
from nnunet.utilities.nd_softmax import softmax_helper

# https://github.com/yiyixuxu/polyloss-pytorch/blob/master/PolyLoss.py
def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 softmax: bool = True,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'none',
                 epsilon: float = 1.0,
                 ) -> None:
        super(Poly1CrossEntropyLoss, self).__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes, 
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)


# poly CE loss and poly focal loss taken from: https://github.com/abhuse/polyloss-pytorch/blob/main/polyloss.py
# class Poly1CrossEntropyLoss(nn.Module):
#     def __init__(self,
#                  num_classes: int,
#                  epsilon: float = 1.0,
#                  reduction: str = "none",
#                  weight: Tensor = None):
#         """
#         Create instance of Poly1CrossEntropyLoss
#         :param num_classes:
#         :param epsilon:
#         :param reduction: one of none|sum|mean, apply reduction to final loss tensor
#         :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
#         """
#         super(Poly1CrossEntropyLoss, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.reduction = reduction
#         self.weight = weight
#         return

#     def forward(self, logits, labels):
#         """
#         Forward pass
#         :param logits: tensor of shape [N, num_classes]
#         :param labels: tensor of shape [N]
#         :return: poly cross-entropy loss
#         """
        # one_hot_labels = F.one_hot(labels.long(), self.num_classes).transpose(1, -1).squeeze_(-1)
        # one_hot_labels.to(device=logits.device, dtype=logits.dtype)
            
        # pt = torch.sum(one_hot_labels * F.softmax(logits, dim=1), dim=1)
        # if labels.shape[1] == 1:
            # labels = labels[:, 0]
        # CE = F.cross_entropy(input=logits,
        #                      target=labels.long(),
        #                      reduction='none',
        #                      weight=self.weight)
        # poly1 = CE + self.epsilon * (1 - pt)
        
        # if self.reduction == "mean":
        #     poly1 = poly1.mean()
        # elif self.reduction == "sum":
        #     poly1 = poly1.sum()
        # return poly1
    
    
        # if logits.dim() > 2:
        #     # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
        #     logits = logits.view(logits.size(0), logits.size(1), -1)
        #     logits = logits.permute(0, 2, 1).contiguous()
        #     logits = logits.view(-1, logits.size(-1))
        # labels = torch.squeeze(labels, 1)
        # labels = labels.view(-1, 1)
    
        # idx = labels.cpu().long()

        # one_hot_key = torch.FloatTensor(labels.size(0), self.num_classes).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logits.device:
        #     one_hot_key = one_hot_key.to(logits.device)
            
        # pt = torch.sum(one_hot_key * F.softmax(logits, dim=-1), dim=-1)
        
        # labels = labels[:, 0]
        # CE = F.cross_entropy(input=logits,
        #                      target=labels.long(),
        #                      reduction='none',
        #                      weight=self.weight)
        
        # poly1 = CE + self.epsilon * (1 - pt)
        # if self.reduction == "mean":
        #     poly1 = poly1.mean()
        # elif self.reduction == "sum":
        #     poly1 = poly1.sum()
            
        # print(poly1.shape)
        # return poly1


class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: Tensor = None,
                 pos_weight: Tensor = None,
                 label_is_onehot: bool = False):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise 
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                # labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)
                labels = F.one_hot(labels.long(), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1


# DC+Poly CE loss and DC+Poly Focal loss modified from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/focal_loss.py
class DC_and_Poly1CrossEntropy_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, poly1_ce_kwargs, aggregate="sum", square_dice=False, weight_poly1_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for Poly1_CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param poly1_ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_poly1_ce:
        :param weight_dice:
        """
        super(DC_and_Poly1CrossEntropy_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            poly1_ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_poly1_ce = weight_poly1_ce
        self.aggregate = aggregate
        self.poly1_ce = Poly1CrossEntropyLoss(**poly1_ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        poly1_ce_loss = self.poly1_ce(net_output, target[:, 0].long()) if self.weight_poly1_ce != 0 else 0
        if self.ignore_label is not None:
            poly1_ce_loss *= mask[:, 0]
            poly1_ce_loss = poly1_ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_poly1_ce * poly1_ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class DC_and_Poly1Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, poly1_focal_kwargs, aggregate="sum", square_dice=False, weight_poly1_focal=1, weight_dice=1,
                 log_dice=False):
        """
        CAREFUL. Weights for Poly1_Focal and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param poly1_ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_poly1_ce:
        :param weight_dice:
        """
        super(DC_and_Poly1Focal_loss, self).__init__()
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_poly1_focal = weight_poly1_focal
        self.aggregate = aggregate
        self.poly1_focal = Poly1FocalLoss(apply_nonlin=softmax_helper, **poly1_focal_kwargs)

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        poly1_focal_loss = self.poly1_focal(net_output, target[:, 0].long()) if self.weight_poly1_focal != 0 else 0

        if self.aggregate == "sum":
            result = self.weight_poly1_ce * poly1_focal_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result