"""Channel-balanced reconstruction loss on masked observed entries."""
import torch
import torch.nn.functional as F


def masked_loss(prediction, target, mask, channel_types, pos_weight=None, binary_loss="bce"):
    losses = []
    for channel, kind in enumerate(channel_types):
        select = mask[..., channel].bool()
        if not select.any():
            continue
        p, y = prediction[..., channel][select], target[..., channel][select]
        if kind == "binary" and binary_loss == "bce":
            weight = None if pos_weight is None else torch.as_tensor(pos_weight, device=p.device, dtype=p.dtype)
            losses.append(F.binary_cross_entropy_with_logits(p, y, pos_weight=weight))
        else:
            losses.append(F.mse_loss(p, y))
    if not losses:
        raise ValueError("No observed masked entries in batch")
    return torch.stack(losses).mean()
