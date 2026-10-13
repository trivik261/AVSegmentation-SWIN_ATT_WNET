"""Weighted auxiliary loss for the separate Attention-U-Net DS model."""


def supervised_loss(logits, auxiliary_logits, target, criterion, auxiliary_weight=0.4):
    if auxiliary_weight < 0:
        raise ValueError("Auxiliary weight must be nonnegative")
    loss = criterion(logits, target)
    if auxiliary_logits is not None:
        loss = loss + auxiliary_weight * criterion(auxiliary_logits, target)
    return loss
