import numpy as np
import torch


def cc_weighted(pred_up, fdm_up, region_map):
    B = pred_up.size(0)
    cost_sum = 0.
    for b in range(B):
        pm = pred_up[b, 0]
        gm = fdm_up[b, 0]
        wm = region_map[b, 0]

        pm_w = pm * wm
        gm_w = gm * wm
        pm_mean = pm_w.mean()
        gm_mean = gm_w.mean()
        pm_c = pm_w - pm_mean
        gm_c = gm_w - gm_mean
        num = (pm_c * gm_c).sum()
        den = torch.sqrt((pm_c ** 2).sum() * (gm_c ** 2).sum() + 1e-8)
        c = num / den
        cost_i = 1. - c
        cost_sum += cost_i
    return cost_sum / B


def nss_weighted(pred_up, fix_720, region_map):
    B = pred_up.size(0)
    vals = []
    for b in range(B):
        pm = pred_up[b, 0]
        fm = fix_720[b, 0]
        wm = region_map[b, 0]

        pm_w = pm * wm
        fix_sum = fm.sum().item()
        if fix_sum < 1:
            vals.append(0.)
            continue
        mean_ = pm_w.mean()
        std_ = pm_w.std(unbiased=False) + 1e-8
        pmZ = (pm_w - mean_) / std_
        val = (pmZ * fm).sum().item() / (fix_sum + 1e-8)
        vals.append(val)
    return float(np.mean(vals))


def kl_weighted(pred_up, fdm_up, region_map):
    B = pred_up.size(0)
    kl_sum = 0.
    for b in range(B):
        pm = pred_up[b, 0].clone()
        gm = fdm_up[b, 0].clone()
        wm = region_map[b, 0]

        pm_w = pm * wm
        gm_w = gm * wm
        psum = pm_w.sum().item()
        gsum = gm_w.sum().item()
        if psum < 1e-8 or gsum < 1e-8:
            continue
        pm_w /= psum
        gm_w /= gsum
        ratio = (gm_w + 1e-8) / (pm_w + 1e-8)
        kl_val = (gm_w * torch.log(ratio)).sum().item()
        kl_sum += kl_val
    return kl_sum / B


def cc_torch(pred_up, fdm_up):
    B = pred_up.size(0)
    cost_sum = 0.
    for b in range(B):
        pm = pred_up[b, 0]
        gm = fdm_up[b, 0]
        pm_mean = pm.mean()
        gm_mean = gm.mean()
        pm_c = pm - pm_mean
        gm_c = gm - gm_mean
        num = (pm_c * gm_c).sum()
        den = torch.sqrt((pm_c ** 2).sum() * (gm_c ** 2).sum() + 1e-8)
        c = num / den
        cost_i = 1. - c
        cost_sum += cost_i
    return cost_sum / B


def nss_torch(pred_up, fix_720):
    B = pred_up.size(0)
    vals = []
    for b in range(B):
        pm = pred_up[b, 0]
        fm = fix_720[b, 0]
        s = fm.sum().item()
        if s < 1:
            vals.append(0.)
            continue
        mn = pm.mean()
        st = pm.std(unbiased=False) + 1e-8
        pmZ = (pm - mn) / st
        val = (pmZ * fm).sum().item() / (s + 1e-8)
        vals.append(val)
    return float(np.mean(vals))


def kl_torch(pred_up, fdm_up):
    B = pred_up.size(0)
    kl_sum = 0.
    for b in range(B):
        pm = pred_up[b, 0].clone()
        gm = fdm_up[b, 0].clone()
        p_sum = pm.sum().item()
        g_sum = gm.sum().item()
        if p_sum < 1e-8 or g_sum < 1e-8:
            continue
        pm /= p_sum
        gm /= g_sum
        ratio = (gm + 1e-8) / (pm + 1e-8)
        kl_val = (gm * torch.log(ratio)).sum().item()
        kl_sum += kl_val
    return kl_sum / B


def total_variation_loss(pred_40x60):
    dx = torch.abs(pred_40x60[:, :, 1:, :] - pred_40x60[:, :, :-1, :])
    dy = torch.abs(pred_40x60[:, :, :, 1:] - pred_40x60[:, :, :, :-1])
    tv_val = dx.mean() + dy.mean()
    return tv_val
