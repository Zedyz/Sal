import numpy as np
import torch
import torch.nn.functional as F


def auc_judd(pred_map, fix_map):
    pm = pred_map.view(-1).cpu().numpy()
    fm = fix_map.view(-1).cpu().numpy().astype(np.int32)
    sidx = np.argsort(-pm)
    sorted_lbl = fm[sidx]
    ctp = np.cumsum(sorted_lbl)
    totp = ctp[-1]
    totn = len(fm) - totp
    if totp < 1 or totn < 1:
        return 0.5
    tpr = ctp / (totp + 1e-8)
    idx2 = np.arange(1, len(pm) + 1)
    fpr = (idx2 - ctp) / (totn + 1e-8)
    tpr = np.concatenate(([0], tpr))
    fpr = np.concatenate(([0], fpr))
    return float(np.trapz(tpr, fpr))


def shuffled_auc(pred_2d, fix_2d, neg_fixations):
    pm_np = pred_2d.cpu().numpy()
    pos_coords = (fix_2d > 0.5).nonzero(as_tuple=False)
    if pos_coords.size(0) < 1:
        return 0.5
    pos_vals = [pm_np[r, c] for (r, c) in pos_coords]
    neg_vals = []
    for (ny, nx) in neg_fixations:
        if 0 <= ny < pm_np.shape[0] and 0 <= nx < pm_np.shape[1]:
            neg_vals.append(pm_np[ny, nx])

    if len(pos_vals) < 1 or len(neg_vals) < 1:
        return 0.5

    pos_vals = np.array(pos_vals)
    neg_vals = np.array(neg_vals)
    labels = np.concatenate((np.ones_like(pos_vals), np.zeros_like(neg_vals)))
    scores = np.concatenate((pos_vals, neg_vals))
    sidx = np.argsort(-scores)
    sorted_lbl = labels[sidx]
    ctp = np.cumsum(sorted_lbl)
    totp = ctp[-1]
    totn = len(labels) - totp
    if totp < 1 or totn < 1:
        return 0.5
    tpr = ctp / (totp + 1e-8)
    idx2 = np.arange(1, len(labels) + 1)
    fpr = (idx2 - ctp) / (totn + 1e-8)
    tpr = np.concatenate(([0], tpr))
    fpr = np.concatenate(([0], fpr))
    return float(np.trapz(tpr, fpr))


def safe_pred_map(pred):
    # remove NaNs or Inf
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e5, neginf=0.0)
    return torch.clamp(pred, min=0.0)


def kl_div_loss(pred_map, gt_map, eps=1e-8):
    B, C, H, W = pred_map.shape
    pf = pred_map.view(B, -1)
    gf = gt_map.view(B, -1)
    psum = pf.sum(dim=1, keepdim=True).clamp(min=eps)
    gsum = gf.sum(dim=1, keepdim=True).clamp(min=eps)
    p = gf / gsum
    q = pf / psum
    kl_each = (p * torch.log((p + eps) / (q + eps))).sum(dim=1)
    return kl_each.mean()


def nss_bounded_loss(pred_map, fix_map):
    """
    cost = 1 - sigmoid(NSS)
    """
    B, _, H, W = pred_map.shape
    raw_nss = []
    eps = 1e-8
    for i in range(B):
        pm = pred_map[i, 0]
        fm = fix_map[i, 0]
        fc = fm.sum().item()
        if fc < 1:
            raw_nss.append(0.0)
            continue
        m = pm.mean().item()
        s = pm.std(unbiased=False).item() + eps
        pmn = (pm - m) / s
        val = float((pmn * fm).sum().item() / fc)
        raw_nss.append(val)

    if len(raw_nss) == 0:
        return pred_map.new_tensor(0.0), 0.0

    nss_t = torch.tensor(raw_nss, device=pred_map.device)
    cost = 1.0 - torch.sigmoid(nss_t)
    return cost.mean(), float(nss_t.mean().item())


def compute_cc(pred_map, fdm_map):
    pm = pred_map.view(-1)
    gm = fdm_map.view(-1)
    pm_m = pm.mean()
    gm_m = gm.mean()
    pm_c = pm - pm_m
    gm_c = gm - gm_m
    num = (pm_c * gm_c).sum().item()
    den = (pm_c.square().sum().item() * gm_c.square().sum().item()) ** 0.5
    if den < 1e-10:
        return 0.0
    return float(num / den)


def cc_loss_as_cost(pred_map, fdm_map):
    return 1.0 - compute_cc(pred_map, fdm_map)


def ranking_loss(pred_map, fix_map, margin=1.0, neg_samples=10):
    B, C, H, W = pred_map.shape
    import random
    losses = []
    for b in range(B):
        pm = pred_map[b, 0]
        fm = fix_map[b, 0]
        fix_coords = (fm > 0.5).nonzero(as_tuple=False)
        if fix_coords.size(0) < 1:
            continue
        neg_coords = (fm < 0.5).nonzero(as_tuple=False)
        for pc in fix_coords:
            pos_val = pm[pc[0], pc[1]]
            for _ in range(neg_samples):
                idx = random.randint(0, neg_coords.size(0) - 1)
                nc = neg_coords[idx]
                neg_val = pm[nc[0], nc[1]]
                hinge = F.relu(margin + neg_val - pos_val)
                losses.append(hinge)
    if len(losses) == 0:
        return pred_map.new_tensor(0.0)
    return torch.stack(losses).mean()


def total_variation_loss(sal):
    dx = sal[:, :, 1:, :] - sal[:, :, :-1, :]
    dy = sal[:, :, :, 1:] - sal[:, :, :, :-1]
    return (dx ** 2).mean() + (dy ** 2).mean()


def combined_loss(pred_map, fdm_map, fix_map, alpha=1.0, beta=0.0, gamma_rank=0.0, tv_weight=0.0):
    """
    L = KL + alpha*(1 - sigmoid(NSS)) + beta*(1-CC) + gamma_rank*Ranking + tv_weight*TV
    """
    Lkl = kl_div_loss(pred_map, fdm_map)
    Lnss_cost, raw_nss = nss_bounded_loss(pred_map, fix_map)

    B = pred_map.size(0)
    cost_cc = 0.0
    if beta > 0:
        cc_acc = 0.0
        for i in range(B):
            cc_acc += cc_loss_as_cost(pred_map[i, 0], fdm_map[i, 0])
        cost_cc = cc_acc / B

    cst_rank = ranking_loss(pred_map, fix_map, margin=1.0, neg_samples=10) if gamma_rank > 0 else 0.0
    cst_tv = total_variation_loss(pred_map) if tv_weight > 0 else 0.0

    total = Lkl + alpha * Lnss_cost + beta * cost_cc + gamma_rank * cst_rank + tv_weight * cst_tv
    return total, float(Lkl.item()), float(raw_nss), float(cost_cc), float(cst_rank), float(cst_tv)


def single_pass_evaluate(model, loader, all_train_fix, alpha=1.0, beta=0.0,
                         gamma=0.0, tv_weight=0.0, device=None):
    """
    Evaluate model on a dataloader. Returns dict of metrics: loss, nss, cc, kl, aucj, sauc
    """
    import torch
    model.eval()
    sum_loss = 0.0
    sum_nss = 0.0
    sum_cc = 0.0
    sum_kl = 0.0
    sum_aucj = 0.0
    sum_sauc = 0.0
    count = 0

    with torch.no_grad():
        for images, fdms, eyemaps, mask, names in loader:
            images = images.to(device)
            fdms = fdms.to(device)
            eyemaps = eyemaps.to(device)

            pm = safe_pred_map(model(images))
            cL, lkl, nssv, ccc, rnk, tvv = combined_loss(
                pm, fdms, eyemaps,
                alpha=alpha, beta=beta, gamma_rank=gamma, tv_weight=tv_weight
            )
            sum_loss += float(cL.item())

            B2 = pm.size(0)
            for i in range(B2):
                pm2d = pm[i, 0].cpu()
                fm2d = fdms[i, 0].cpu()
                ey2d = eyemaps[i, 0].cpu()

                # NSS
                fc = ey2d.sum().item()
                if fc > 0:
                    mm_ = pm2d.mean().item()
                    ss_ = pm2d.std(unbiased=False).item() + 1e-8
                    pmn_ = (pm2d - mm_) / ss_
                    sum_nss += float((pmn_ * ey2d).sum().item() / fc)
                else:
                    sum_nss += 0.0

                # CC
                pm_m = pm2d.mean().item()
                fm_m = fm2d.mean().item()
                pm_c = pm2d - pm_m
                fm_c = fm2d - fm_m
                den = (pm_c.square().sum().item() * fm_c.square().sum().item()) ** 0.5
                if den < 1e-10:
                    sum_cc += 0.0
                else:
                    num = (pm_c * fm_c).sum().item()
                    sum_cc += float(num / den)

                # KL
                kl_single = kl_div_loss(
                    pm2d.unsqueeze(0).unsqueeze(0),
                    fm2d.unsqueeze(0).unsqueeze(0)
                )
                sum_kl += float(kl_single.item())

                # AUC-J
                aj_ = auc_judd(pm2d, ey2d)
                sum_aucj += aj_

                # sAUC
                this_fname = names[i]
                pos_coords = (ey2d > 0.5).nonzero(as_tuple=False)
                Kpos = pos_coords.size(0)
                neg_cands = all_train_fix[all_train_fix[:, 0] != this_fname]
                import numpy as np
                if Kpos < 1 or len(neg_cands) < Kpos:
                    sauc_val = 0.5
                else:
                    idxz = np.random.choice(len(neg_cands), size=Kpos, replace=False)
                    neg_s = neg_cands[idxz]
                    neg_yx = neg_s[:, 1:].astype(np.int32)
                    sauc_val = shuffled_auc(pm2d, ey2d, neg_yx)
                sum_sauc += sauc_val

            count += B2

    if count < 1:
        return {"loss": 0.0, "nss": 0.0, "cc": 0.0, "kl": 0.0, "aucj": 0.5, "sauc": 0.5}

    ret = {}
    ret["loss"] = sum_loss / len(loader)
    ret["nss"] = sum_nss / count
    ret["cc"] = sum_cc / count
    ret["kl"] = sum_kl / count
    ret["aucj"] = sum_aucj / count
    ret["sauc"] = sum_sauc / count
    return ret
