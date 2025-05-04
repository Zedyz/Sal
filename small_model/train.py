import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from saldataset import SalDataset
from torch.optim import AdamW
import csv
from tqdm import tqdm

from config import ALPHA_FACE, ALPHA_TEXT, ALPHA_BANNER, train_json, val_json, EPOCHS, DEVICE, IN_H, IN_W, GT_H, GT_W, \
    LAMBDA_NSS, LAMBDA_KL, LAMBDA_TV, BATCH_SIZE, LR, WEIGHT_DECAY
from loss import cc_weighted, nss_weighted, kl_weighted, total_variation_loss, cc_torch, nss_torch, \
    kl_torch

from metrics import cc, nss, kldiv, auc_judd, auc_shuff
from model import Model


def build_region_map(face_720, text_720, bann_720):
    region_map = torch.ones_like(face_720)
    region_map += ALPHA_FACE * face_720
    region_map += ALPHA_TEXT * text_720
    region_map += ALPHA_BANNER * bann_720
    return region_map


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(train_json, 'r') as f:
    data_train = json.load(f)
train_files = sorted({d["name"] for d in data_train if "name" in d})

with open(val_json, 'r') as f:
    data_val = json.load(f)
val_files = sorted({d["name"] for d in data_val if "name" in d})

print(f"Train= {len(train_files)}, Val= {len(val_files)}")
root = "."

train_ds = SalDataset(root, train_files)
val_ds = SalDataset(root, val_files)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = Model().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_metric = -9999.
csv_file = open("train_fullres_tv.csv", "w", newline='')
writer = csv.writer(csv_file)
writer.writerow([
    "epoch",
    "train_cost", "train_cc", "train_nss", "train_kl",
    "val_cost", "val_cc", "val_nss", "val_kl", "val_aucj", "val_sauc",
    "metric=(cc+nss-kl)"
])

for epoch in range(EPOCHS):
    model.train()
    train_sum = 0.
    train_count = 0

    train_cc_sum = 0.
    train_nss_sum = 0.
    train_kl_sum = 0.
    train_count_cc = 0

    loop_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
    for batch in loop_train:
        # fetch
        img_full = batch["image_full"].to(DEVICE)
        face_720 = batch["face_full"].to(DEVICE)
        text_720 = batch["text_full"].to(DEVICE)
        bann_720 = batch["banner_full"].to(DEVICE)
        fdm_full = batch["fdm_full"].to(DEVICE)
        eye_full = batch["eye_full"].to(DEVICE)

        img_320x480 = F.interpolate(img_full, (IN_H, IN_W), mode='bilinear', align_corners=False)
        pred_40x60 = model(img_320x480)
        pred_up = F.interpolate(pred_40x60, (GT_H, GT_W), mode='bilinear', align_corners=False)

        region_map = build_region_map(face_720, text_720, bann_720)

        cost_cc = cc_weighted(pred_up, fdm_full, region_map)
        n_val = nss_weighted(pred_up, eye_full, region_map)
        cost_nss = (1.0 - n_val)
        kl_val = kl_weighted(pred_up, fdm_full, region_map)

        cost_tv = total_variation_loss(pred_40x60)

        cost = cost_cc + LAMBDA_NSS * cost_nss + LAMBDA_KL * kl_val + LAMBDA_TV * cost_tv

        optimizer.zero_grad()
        cost.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        B = img_full.size(0)
        train_sum += cost.item() * B
        train_count += B

        cc_val = (1.0 - cost_cc.item())
        train_cc_sum += cc_val * B
        train_nss_sum += n_val * B
        train_kl_sum += kl_val * B
        train_count_cc += B

    train_cost = train_sum / train_count
    train_cc = train_cc_sum / train_count_cc
    train_nss = train_nss_sum / train_count_cc
    train_kl = train_kl_sum / train_count_cc

    model.eval()
    val_sum = 0.
    val_count = 0

    cc_sum = 0.
    nss_sum = 0.
    kl_sum = 0.
    aucj_sum = 0.
    sauc_sum = 0.
    sample_count = 0

    loop_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
    with torch.no_grad():
        for batch in loop_val:
            img_full = batch["image_full"].to(DEVICE)
            fdm_full = batch["fdm_full"].to(DEVICE)
            eye_full = batch["eye_full"].to(DEVICE)

            # forward => unweighted
            img_320x480 = F.interpolate(img_full, (IN_H, IN_W), mode='bilinear', align_corners=False)
            pred_40x60 = model(img_320x480)
            pred_up = F.interpolate(pred_40x60, (GT_H, GT_W), mode='bilinear', align_corners=False)

            c_val = cc_torch(pred_up, fdm_full)
            n_val = nss_torch(pred_up, eye_full)
            kl_v = kl_torch(pred_up, fdm_full)
            cost_val = c_val + LAMBDA_NSS * (1.0 - n_val) + LAMBDA_KL * kl_v

            B = img_full.size(0)
            val_sum += cost_val.item() * B
            val_count += B

            pred_np = pred_up.squeeze(1).cpu().numpy()
            fdm_np = fdm_full.squeeze(1).cpu().numpy()
            eye_np = eye_full.squeeze(1).cpu().numpy()

            for b in range(B):
                smap = pred_np[b]
                mxv = smap.max()
                if mxv > 1e-8:
                    smap = smap / mxv
                fdmgt = fdm_np[b]
                eyegt = eye_np[b]

                c_ = cc(smap, fdmgt)
                nss_ = nss(smap, eyegt)
                kl_ = kldiv(smap, fdmgt)
                aucj_ = auc_judd(smap, eyegt)

                if B > 1:
                    other_idx = random.choice([xx for xx in range(B) if xx != b])
                    other_map = eye_np[other_idx]
                else:
                    other_map = np.zeros_like(eyegt)

                sauc_ = auc_shuff(smap, eyegt, other_map)

                cc_sum += c_
                nss_sum += nss_
                kl_sum += kl_
                aucj_sum += aucj_
                sauc_sum += sauc_
                sample_count += 1

    val_cost = val_sum / val_count
    val_cc = cc_sum / sample_count
    val_nss = nss_sum / sample_count
    val_kl = kl_sum / sample_count
    val_aucj = aucj_sum / sample_count
    val_sauc = sauc_sum / sample_count

    metric = (val_cc + val_nss) - val_kl

    print(f"[Epoch {epoch}/{EPOCHS}] "
          f"train_cost={train_cost:.4f}, train_cc={train_cc:.4f}, train_nss={train_nss:.4f}, train_kl={train_kl:.4f} || "
          f"val_cost={val_cost:.4f}, cc={val_cc:.4f}, nss={val_nss:.4f}, kl={val_kl:.4f}, "
          f"aucj={val_aucj:.4f}, sauc={val_sauc:.4f}, metric={metric:.4f}")

    writer.writerow([
        epoch,
        train_cost, train_cc, train_nss, train_kl,
        val_cost, val_cc, val_nss, val_kl, val_aucj, val_sauc,
        metric
    ])
    csv_file.flush()

    if metric > best_metric:
        best_metric = metric
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
            "val_cc": val_cc,
            "val_nss": val_nss,
            "val_kl": val_kl,
            "val_aucj": val_aucj,
            "val_sauc": val_sauc
        }
        torch.save(ckpt, "best_fullres_tv.pth")
        print(f"** Saved best => metric={best_metric:.4f}")

csv_file.close()
final_ckp = {
    "epoch": EPOCHS,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "best_metric": best_metric
}
torch.save(final_ckp, "final_fullres_tv.pth")
print("Done => final_fullres_tv.pth")
