import os
import random
import math
import csv
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.io as tvio
from tqdm import tqdm
from transformers import Dinov2Config, Dinov2Model
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class SalDataset(Dataset):

    def __init__(
            self,
            orig_dir,
            fdm_dir,
            eyemap_dir,
            file_list=None,
            is_train=False,
            scale_range=(0.95, 1.05),
            brightness_range=(0.9, 1.1)
    ):
        self.orig_dir = orig_dir
        self.fdm_dir = fdm_dir
        self.eyemap_dir = eyemap_dir
        self.files = file_list if file_list else []
        self.is_train = is_train
        self.scale_range = scale_range
        self.brightness_range = brightness_range

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        base, ext = os.path.splitext(fname)
        ext = ext.lower()

        img_path = os.path.join(self.orig_dir, fname)
        image = tvio.read_image(img_path).float() / 255.0

        fdm_path = os.path.join(self.fdm_dir, base + ext)
        if not os.path.exists(fdm_path):
            alt_ext = ('.png' if ext != '.png' else '.jpg')
            fdm_path = os.path.join(self.fdm_dir, base + alt_ext)
        fdm = tvio.read_image(fdm_path)

        if fdm.shape[0] > 1:
            fdm = fdm[:1]
        fdm = fdm.float() / 255.0

        eye_path = os.path.join(self.eyemap_dir, base + ext)
        if not os.path.exists(eye_path):
            alt_ext = ('.png' if ext != '.png' else '.jpg')
            eye_path = os.path.join(self.eyemap_dir, base + alt_ext)
        eye = tvio.read_image(eye_path)
        if eye.shape[0] > 1:
            eye = eye[:1]
        eye = (eye > 127).float()

        return image, fdm, eye, fname


def collate_fn(batch, patch_size=14):
    bsz = len(batch)
    heights = [b[0].shape[1] for b in batch]
    widths = [b[0].shape[2] for b in batch]
    Hmax = max(heights)
    Wmax = max(widths)
    if Hmax % patch_size != 0:
        Hmax = ((Hmax // patch_size) + 1) * patch_size
    if Wmax % patch_size != 0:
        Wmax = ((Wmax // patch_size) + 1) * patch_size

    imgs = torch.zeros(bsz, 3, Hmax, Wmax, dtype=torch.float32)
    fdms = torch.zeros(bsz, 1, Hmax, Wmax, dtype=torch.float32)
    eyes = torch.zeros(bsz, 1, Hmax, Wmax, dtype=torch.float32)
    mask = torch.zeros(bsz, 1, Hmax, Wmax, dtype=torch.float32)
    names = []

    for i, (im, fd, ey, nm) in enumerate(batch):
        C, H, W = im.shape
        imgs[i, :, :H, :W] = im
        fdms[i, :, :H, :W] = fd
        eyes[i, :, :H, :W] = ey
        mask[i, :, :H, :W] = 1.0
        names.append(nm)

    return imgs, fdms, eyes, mask, names


def auc_judd(pred_2d, fix_2d):
    pm = pred_2d.view(-1).numpy()
    fm = fix_2d.view(-1).numpy().astype(np.int32)
    sidx = np.argsort(-pm)  # descending
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
    pm_np = pred_2d.numpy()
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
    B, C, H, W = pred_map.shape
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
    nss_t = pred_map.new_tensor(raw_nss)
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


def combined_loss(pred_map, fdm_map, fix_map,
                  alpha=1.0, beta=0.0,
                  gamma_rank=0.0,
                  tv_weight=0.0):
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


def single_pass_evaluate(
        model,
        loader,
        all_train_fix,
        alpha=1.0,
        beta=0.0,
        gamma_rank=0.0,
        tv_weight=0.0,
        device=None
):

    model.eval()
    sum_loss = 0.0
    sum_nss = 0.0
    sum_cc = 0.0
    sum_kl = 0.0
    sum_aucj = 0.0
    sum_sauc = 0.0
    count = 0

    with torch.no_grad():
        for images, fdms, eyemaps, mask, names in tqdm(loader, desc="[Eval]", leave=False):
            images = images.to(device)
            fdms = fdms.to(device)
            eyemaps = eyemaps.to(device)
            mask = mask.to(device)

            pm = safe_pred_map(model(images))

            pm_ = pm * mask
            fdm_ = fdms * mask
            eye_ = eyemaps * mask

            cL, lkl, nssv, ccc, rnk, tvv = combined_loss(
                pm_, fdm_, eye_,
                alpha=alpha, beta=beta, gamma_rank=gamma_rank, tv_weight=tv_weight
            )
            sum_loss += float(cL.item())

            B2 = pm.size(0)
            for i in range(B2):
                pm2d = pm[i, 0] * mask[i, 0]
                fm2d = fdms[i, 0] * mask[i, 0]
                ey2d = eyemaps[i, 0] * mask[i, 0]

                pm2d_cpu = pm2d.cpu()
                fm2d_cpu = fm2d.cpu()
                ey2d_cpu = ey2d.cpu()

                fc = ey2d_cpu.sum().item()
                if fc > 0:
                    m = pm2d_cpu.mean().item()
                    s = pm2d_cpu.std(unbiased=False).item() + 1e-8
                    pmn = (pm2d_cpu - m) / s
                    sum_nss += float((pmn * ey2d_cpu).sum().item() / fc)
                else:
                    sum_nss += 0.0

                cc_val = compute_cc(pm2d.unsqueeze(0).unsqueeze(0),
                                    fm2d.unsqueeze(0).unsqueeze(0))
                sum_cc += cc_val

                kl_single = kl_div_loss(
                    pm2d.unsqueeze(0).unsqueeze(0),
                    fm2d.unsqueeze(0).unsqueeze(0)
                )
                sum_kl += float(kl_single.item())

                aucj_val = 0.5
                if fc > 0:
                    aucj_val = auc_judd(pm2d_cpu, (ey2d_cpu > 0.5))
                sum_aucj += aucj_val

                pos_coords = (ey2d_cpu > 0.5).nonzero(as_tuple=False)
                Kpos = pos_coords.size(0)
                sauc_val = 0.5
                if Kpos > 0:
                    this_fname = names[i]
                    neg_cands = all_train_fix[all_train_fix[:, 0] != this_fname]
                    if len(neg_cands) >= Kpos:
                        idxz = np.random.choice(len(neg_cands), size=Kpos, replace=False)
                        neg_s = neg_cands[idxz]
                        neg_yx = neg_s[:, 1:].astype(np.int32)
                        valid_neg = []
                        for (ny, nx) in neg_yx:
                            if 0 <= ny < pm2d.shape[0] and 0 <= nx < pm2d.shape[1]:
                                if mask[i, 0, ny, nx] > 0.5:
                                    valid_neg.append((ny, nx))
                        if len(valid_neg) > 0:
                            sauc_val = shuffled_auc(pm2d_cpu, (ey2d_cpu > 0.5), valid_neg)
                sum_sauc += sauc_val

            count += B2

    if count < 1:
        return {
            "loss": 0.0, "nss": 0.0, "cc": 0.0,
            "kl": 0.0, "aucj": 0.5, "sauc": 0.5
        }

    ret = {}
    ret["loss"] = sum_loss / len(loader)
    ret["nss"] = sum_nss / count
    ret["cc"] = sum_cc / count
    ret["kl"] = sum_kl / count
    ret["aucj"] = sum_aucj / count
    ret["sauc"] = sum_sauc / count
    return ret


class LocalWindowAttn(nn.Module):
    def __init__(self, dim, nhead, window_size=7):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=False)

    def forward(self, x_bhwf):
        B, Hf, Wf, D = x_bhwf.shape
        ws = self.window_size
        pad_h = (ws - (Hf % ws)) % ws
        pad_w = (ws - (Wf % ws)) % ws
        x_padded = F.pad(x_bhwf, (0, 0, 0, pad_w, 0, pad_h))
        Hp = Hf + pad_h
        Wp = Wf + pad_w

        x_reshaped = x_padded.view(
            B, Hp // ws, ws,
               Wp // ws, ws,
            D
        )
        x_reshaped = x_reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
        b_, wh, ww, ws1, ws2, d_ = x_reshaped.shape
        x_reshaped = x_reshaped.view(b_, wh * ww, ws1 * ws2, d_)

        total_windows = b_ * (wh * ww)
        x_2 = x_reshaped.view(total_windows, ws1 * ws2, d_)
        x_2t = x_2.permute(1, 0, 2)
        attn_out, _ = self.attn(x_2t, x_2t, x_2t)
        attn_out = attn_out.permute(1, 0, 2).contiguous()

        attn_out = attn_out.view(b_, wh * ww, ws1, ws2, d_)
        attn_out = attn_out.view(b_, wh, ww, ws1, ws2, d_)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).contiguous()
        attn_out = attn_out.view(b_, Hp, Wp, d_)
        attn_out = attn_out[:, :Hf, :Wf, :]
        return attn_out


class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead, window_size=7, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.local_attn = LocalWindowAttn(dim, nhead, window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x_bhwf):
        B, H, W, D = x_bhwf.shape
        x_attn = self.local_attn(x_bhwf)
        x_res = x_bhwf + x_attn
        x_res = x_res.view(B * H * W, D)
        x_res = self.norm1(x_res)
        x_res = x_res.view(B, H, W, D)

        ff_in = x_res.view(B * H * W, D)
        ff_out = self.ff(ff_in)
        x_res2 = ff_in + ff_out
        x_res2 = self.norm2(x_res2)
        return x_res2.view(B, H, W, D)


class Decoder(nn.Module):
    def __init__(self, dim, depth=2, nhead=4, ff_dim=2048, window_size=7):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, nhead, window_size, ff_dim) for _ in range(depth)
        ])
        self.ln_final = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, 1)

    def forward(self, x_bhwf):
        B, H, W, D = x_bhwf.shape
        for blk in self.blocks:
            x_bhwf = blk(x_bhwf)
        x_2d = x_bhwf.view(B * H * W, D)
        x_2d = self.ln_final(x_2d)
        sal_flat = self.proj_out(x_2d)
        sal_map = sal_flat.view(B, H, W, 1).permute(0, 3, 1, 2)
        return sal_map


class ViT(nn.Module):
    def __init__(self,
                 model_name="facebook/dinov2-base",
                 decoder_depth=2,
                 decoder_heads=4,
                 decoder_ff=2048,
                 window_size=7):
        super().__init__()
        config = Dinov2Config.from_pretrained(model_name)
        self.backbone = Dinov2Model.from_pretrained(model_name, config=config)
        self.hidden_dim = self.backbone.config.hidden_size
        self.decoder = Decoder(
            dim=self.hidden_dim,
            depth=decoder_depth,
            nhead=decoder_heads,
            ff_dim=decoder_ff,
            window_size=window_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_input = 2.0 * x - 1.0
        out = self.backbone(pixel_values=x_input)
        feats = out.last_hidden_state
        patch_tokens = feats[:, 1:, :]
        Hf = H // 14
        Wf = W // 14
        patch_2d = patch_tokens.view(B, Hf, Wf, self.hidden_dim)
        sal_low = self.decoder(patch_2d)  # => [B,1,Hf,Wf]
        sal_map = F.interpolate(sal_low, size=(H, W), mode='bilinear', align_corners=False)
        return sal_map


if __name__ == "__main__":

    alpha_fixed = 1.0
    beta_fixed = 0.05
    gamma_rank_fixed = 0.02
    tv_weight_fixed = 0.002

    batch_size = 4
    lr_backbone = 1e-5
    lr_new = 3e-5
    weight_decay = 1e-2
    num_epochs = 100
    patience = 30

    checkpoint_path = "checkpoint_noconv1.pth"
    best_model_path = "best_checkpoint_noconv1.pth"
    log_file = "training_log_noconv1.csv"

    orig_dir = "images"
    fdm_dir = "fdm_720x1280"
    eyemap_dir = "eye_fix_720x1280"
    train_json = "../dataset/scanpaths_train.json"
    test_json = "../dataset/scanpaths_test.json"

    with open(train_json, 'r') as f:
        train_data = json.load(f)
    train_names = sorted({d["name"] for d in train_data if "name" in d})

    with open(test_json, 'r') as f:
        test_data = json.load(f)
    val_names = sorted({d["name"] for d in test_data if "name" in d})

    print(f"Train size: {len(train_names)}, Val size: {len(val_names)}")


    train_ds = SalDataset(
        orig_dir=orig_dir,
        fdm_dir=fdm_dir,  # read FDM from dataset
        eyemap_dir=eyemap_dir,
        file_list=train_names,
        is_train=True
    )
    val_ds = SalDataset(
        orig_dir=orig_dir,
        fdm_dir=fdm_dir,
        eyemap_dir=eyemap_dir,
        file_list=val_names,
        is_train=False
    )

    print("Collecting negative fixations from entire train set for sAUC...")
    train_fixations_dict = {}
    for i in range(len(train_ds)):
        _, _, eye, fname = train_ds[i]
        coords = eye[0].nonzero(as_tuple=False)
        train_fixations_dict.setdefault(fname, [])
        for c in coords:
            train_fixations_dict[fname].append((int(c[0]), int(c[1])))

    all_train_fix = []

    for fn, coords in train_fixations_dict.items():
        for (y, x) in coords:
            all_train_fix.append((fn, y, x))
    all_train_fix = np.array(all_train_fix, dtype=object)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViT(
        model_name="facebook/dinov2-base",
        decoder_depth=2,
        decoder_heads=4,
        decoder_ff=2048,
        window_size=7
    ).to(device)


    def param_groups_backbone_vs_new(m, lr_backbone, lr_new):
        b_params = []
        n_params = []
        for name, p in m.named_parameters():
            if "backbone" in name:
                b_params.append(p)
            else:
                n_params.append(p)
        return [
            {"params": b_params, "lr": lr_backbone},
            {"params": n_params, "lr": lr_new}
        ]


    param_groups = param_groups_backbone_vs_new(model, lr_backbone, lr_new)
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    start_epoch = 1
    best_score = -999.0
    no_improve = 0

    # Logging
    if not os.path.isfile(log_file):
        with open(log_file, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([
                "epoch", "alpha", "beta", "gamma", "tv_weight",
                "train_loss",
                "val_loss", "val_nss", "val_cc", "val_kl", "val_aucj", "val_sauc"
            ])

    if os.path.isfile(checkpoint_path):
        print(f"[INFO] Resuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_score" in ckpt:
            best_score = ckpt["best_score"]
        print(f"[INFO] Resumed from epoch {start_epoch - 1}, best_score={best_score:.3f}")

    alpha_val = alpha_fixed
    beta_val = beta_fixed
    gamma_val = gamma_rank_fixed
    tv_val = tv_weight_fixed

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for images, fdms, eyemaps, mask, names in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]",
                                                       leave=False):
            images = images.to(device)
            fdms = fdms.to(device)
            eyemaps = eyemaps.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            pm = safe_pred_map(model(images))

            pm_ = pm * mask
            fdm_ = fdms * mask
            eye_ = eyemaps * mask

            cL, lkl, nssv, ccc, rnk, tvv = combined_loss(
                pm_, fdm_, eye_,
                alpha=alpha_val, beta=beta_val,
                gamma_rank=gamma_val, tv_weight=tv_val
            )
            cL.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(cL.item())

        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()

        val_stats = single_pass_evaluate(
            model, val_loader, all_train_fix,
            alpha=alpha_val, beta=beta_val, gamma_rank=gamma_val, tv_weight=tv_val,
            device=device
        )

        print(
            f"\nEpoch {epoch}/{num_epochs} => alpha={alpha_val}, beta={beta_val}, gamma={gamma_val}, tv={tv_val}\n"
            f"Train => loss={avg_train_loss:.4f}\n"
            f"Val   => loss={val_stats['loss']:.4f}, NSS={val_stats['nss']:.3f}, "
            f"CC={val_stats['cc']:.3f}, KL={val_stats['kl']:.3f}, "
            f"AUC-J={val_stats['aucj']:.3f}, sAUC={val_stats['sauc']:.3f}"
        )

        with open(log_file, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([
                epoch, f"{alpha_val:.2f}", f"{beta_val}", f"{gamma_val}", f"{tv_val}",
                f"{avg_train_loss:.4f}",
                f"{val_stats['loss']:.4f}", f"{val_stats['nss']:.3f}",
                f"{val_stats['cc']:.3f}", f"{val_stats['kl']:.3f}",
                f"{val_stats['aucj']:.3f}", f"{val_stats['sauc']:.3f}"
            ])

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score
        }, checkpoint_path)
        print(f"[INFO] Checkpoint => {checkpoint_path}")

        current_score = val_stats["nss"] + val_stats["cc"]
        if current_score > best_score:
            best_score = current_score
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] New best => NSS+CC={best_score:.3f} => {best_model_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[INFO] No improvement for {no_improve} epochs => stopping early.")
                break
