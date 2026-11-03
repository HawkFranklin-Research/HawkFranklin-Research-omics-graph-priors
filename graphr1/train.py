"""Masked autoencoder training with fixed validation masks."""
import copy
import logging
import time
import numpy as np
import pandas as pd
import torch
from .losses import masked_loss
from .masking import fixed_mask


def initial_history(model, data, graph, train_idx, val_idx, cfg, seed, device):
    """Evaluate frozen initialization using the same validation masks."""
    model.eval()
    losses = {}
    with torch.no_grad():
        for name, indices, mode, offset in (("train", train_idx, "entry", 0), ("entry", val_idx, "entry", 1), ("node", val_idx, "node", 2)):
            mask = fixed_mask(data.m[indices], cfg["entry_rate"] if mode == "entry" else cfg["node_rate"], seed+offset, mode)
            values = []
            for start in range(0, len(indices), cfg["batch_size"]):
                idx, hidden = indices[start:start+cfg["batch_size"]], mask[start:start+cfg["batch_size"]]
                if not hidden.any(): continue
                target = torch.as_tensor(data.x[idx], device=device)
                pred = model(data.make_batch(idx, graph, hidden).to(device))["recon"].reshape(target.shape)
                values.append(float(masked_loss(pred, target, torch.as_tensor(hidden, device=device), cfg["channel_types"], binary_loss=cfg.get("binary_loss", "bce")).cpu()))
            if not values: raise ValueError("Validation mask has no observed entries")
            losses[name] = float(np.mean(values))
    return pd.DataFrame([dict(epoch=0, train_loss=losses["train"], val_loss_EM=losses["entry"], val_loss_NM=losses["node"], lr=0., time_s=0., trained=False)])


def fit(model, data, graph, train_idx, val_idx, cfg, seed, device="cpu"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(seed)
    channel_types = cfg["channel_types"]
    binary_channels = np.array(channel_types) == "binary"
    positives = data.x[train_idx][:, :, binary_channels]
    observed = data.m[train_idx][:, :, binary_channels]
    pos = ((positives > 0) & (observed > 0)).sum()
    neg = ((positives == 0) & (observed > 0)).sum()
    pos_weight = min(50., neg / max(pos, 1))
    masks = {mode: fixed_mask(data.m[val_idx], cfg["entry_rate"] if mode == "entry" else cfg["node_rate"], seed + (1 if mode == "entry" else 2), mode) for mode in ("entry", "node")}
    best, best_epoch, best_state, history = float("inf"), 0, None, []
    for epoch in range(cfg["epochs"]):
        started = time.monotonic()
        model.train()
        losses = []
        for idx in np.array_split(rng.permutation(train_idx), max(1, int(np.ceil(len(train_idx) / cfg["batch_size"])))):
            mask = fixed_mask(data.m[idx], cfg["entry_rate"], int(rng.integers(2**31)), "entry") | fixed_mask(data.m[idx], cfg["node_rate"], int(rng.integers(2**31)), "node")
            if not mask.any():
                continue
            batch = data.make_batch(idx, graph, mask).to(device)
            target = torch.as_tensor(data.x[idx], device=device)
            mask_tensor = torch.as_tensor(mask, device=device)
            prediction = model(batch)["recon"].reshape(target.shape)
            loss = masked_loss(prediction, target, mask_tensor, channel_types, pos_weight, cfg.get("binary_loss", "bce"))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = {}
        with torch.no_grad():
            for mode, mask in masks.items():
                per_batch = []
                for start in range(0, len(val_idx), cfg["batch_size"]):
                    idx, m = val_idx[start:start+cfg["batch_size"]], mask[start:start+cfg["batch_size"]]
                    if not m.any():
                        continue
                    batch = data.make_batch(idx, graph, m).to(device)
                    target = torch.as_tensor(data.x[idx], device=device)
                    pred = model(batch)["recon"].reshape(target.shape)
                    per_batch.append(float(masked_loss(pred, target, torch.as_tensor(m, device=device), channel_types, pos_weight, cfg.get("binary_loss", "bce")).cpu()))
                if not per_batch:
                    raise ValueError("Validation mask has no observed entries")
                val_losses[mode] = float(np.mean(per_batch))
        history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), val_loss_EM=val_losses["entry"], val_loss_NM=val_losses["node"], lr=optimizer.param_groups[0]["lr"], time_s=time.monotonic()-started))
        logging.getLogger(__name__).info("epoch=%d train=%.6f val_em=%.6f val_nm=%.6f", epoch + 1, history[-1]["train_loss"], val_losses["entry"], val_losses["node"])
        if val_losses["entry"] < best:
            best, best_epoch = val_losses["entry"], epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if cfg.get("patience") and epoch + 1 - best_epoch >= cfg["patience"]:
            break
    if best_state is None:
        raise ValueError("Training produced no usable checkpoint")
    model.load_state_dict(best_state)
    return best_state, pd.DataFrame(history)
