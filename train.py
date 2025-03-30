import os
import csv
import json
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SaliencyDataset, collate_fn
from models import build_saliency_model
from losses_and_metrics import single_pass_evaluate, combined_loss, safe_pred_map

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def fix_random_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def param_groups_backbone_vs_new(model, lr_backbone, lr_new):
    """
    Splits model parameters into:
      - backbone (lower LR)
      - new layers (higher LR)
    """
    b_params, n_params = [], []
    for name, p in model.named_parameters():
        if "backbone" in name:
            b_params.append(p)
        else:
            n_params.append(p)
    return [
        {"params": b_params, "lr": lr_backbone},
        {"params": n_params, "lr": lr_new},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dino",
                        help="Which saliency model to use: dino | convnext | purevit, etc.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    args = parser.parse_args()

    fix_random_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha_fixed = 1.0
    beta_fixed = 0.05
    gamma_rank_fixed = 0.01
    tv_weight_fixed = 0.001

    batch_size = 2
    lr_backbone = 1e-5
    lr_new = 3e-5
    weight_decay = 1e-2
    num_epochs = args.epochs
    patience = 10

    checkpoint_path = f"checkpoint_{args.model}.pth"
    best_model_path = f"best_checkpoint_{args.model}.pth"
    log_file = f"training_log_{args.model}.csv"

    orig_dir = "dataset/orig_websaliency_all"
    fdm_dir = "dataset/fdm_websaliency"
    eyemap_dir = "dataset/eyemaps_websaliency"
    train_json = "dataset/scanpaths_train.json"
    test_json = "dataset/scanpaths_test.json"

    with open(train_json, 'r') as f:
        train_data = json.load(f)
    train_names = sorted({d["name"] for d in train_data if "name" in d})

    with open(test_json, 'r') as f:
        test_data = json.load(f)
    val_names = sorted({d["name"] for d in test_data if "name" in d})

    print(f"Model={args.model}, Train size={len(train_names)}, Val size={len(val_names)}")

    # Build dataset
    train_ds = SaliencyDataset(
        orig_dir, fdm_dir, eyemap_dir,
        file_list=train_names,
        is_train=True
    )
    val_ds = SaliencyDataset(
        orig_dir, fdm_dir, eyemap_dir,
        file_list=val_names,
        is_train=False
    )

    print("collecting negative fixations from entire train set for sAUC")
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
    import numpy as np
    all_train_fix = np.array(all_train_fix, dtype=object)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, patch_size=14)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, patch_size=14)
    )

    model = build_saliency_model(args.model).to(device)

    param_groups = param_groups_backbone_vs_new(model, lr_backbone, lr_new)
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    start_epoch = 1
    best_score = -999.0
    no_improve = 0

    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_score" in ckpt:
            best_score = ckpt["best_score"]
        print(f"[INFO] Resumed from epoch={start_epoch - 1}, best_score={best_score:.3f}")

    # CSV header if needed
    if not os.path.isfile(log_file):
        with open(log_file, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([
                "epoch", "alpha", "beta", "gamma", "tv_weight",
                "train_avg_loss",
                "train_loss", "train_nss", "train_cc", "train_kl", "train_aucj", "train_sauc",
                "val_loss", "val_nss", "val_cc", "val_kl", "val_aucj", "val_sauc"
            ])

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0

        batch_iter = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{num_epochs}", leave=False)
        for images, fdms, eyemaps, mask, names in batch_iter:
            images = images.to(device)
            fdms = fdms.to(device)
            eyemaps = eyemaps.to(device)

            optimizer.zero_grad()
            pm = safe_pred_map(model(images))

            cL, lkl, nssv, ccc, rnk, tvv = combined_loss(
                pm, fdms, eyemaps,
                alpha=alpha_fixed,
                beta=beta_fixed,
                gamma_rank=gamma_rank_fixed,
                tv_weight=tv_weight_fixed
            )
            cL.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(cL.item())

        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()

        train_stats = single_pass_evaluate(
            model, train_loader, all_train_fix,
            alpha=alpha_fixed, beta=beta_fixed,
            gamma=gamma_rank_fixed, tv_weight=tv_weight_fixed,
            device=device
        )
        val_stats = single_pass_evaluate(
            model, val_loader, all_train_fix,
            alpha=alpha_fixed, beta=beta_fixed,
            gamma=gamma_rank_fixed, tv_weight=tv_weight_fixed,
            device=device
        )

        print(
            f"\nEpoch {epoch}/{num_epochs} | model={args.model} "
            f"| alpha={alpha_fixed}, beta={beta_fixed}, gamma={gamma_rank_fixed}, tv={tv_weight_fixed}"
        )
        print(
            f"  [Train] avg_loss={avg_train_loss:.4f} "
            f"=> single_pass: loss={train_stats['loss']:.4f}, NSS={train_stats['nss']:.3f}, "
            f"CC={train_stats['cc']:.3f}, KL={train_stats['kl']:.3f}, AUC-J={train_stats['aucj']:.3f}, "
            f"sAUC={train_stats['sauc']:.3f}"
        )
        print(
            f"  [Val]   loss={val_stats['loss']:.4f}, NSS={val_stats['nss']:.3f}, "
            f"CC={val_stats['cc']:.3f}, KL={val_stats['kl']:.3f}, AUC-J={val_stats['aucj']:.3f}, "
            f"sAUC={val_stats['sauc']:.3f}"
        )

        with open(log_file, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([
                epoch, f"{alpha_fixed}", f"{beta_fixed}", f"{gamma_rank_fixed}", f"{tv_weight_fixed}",
                f"{avg_train_loss:.4f}",

                f"{train_stats['loss']:.4f}", f"{train_stats['nss']:.3f}",
                f"{train_stats['cc']:.3f}", f"{train_stats['kl']:.3f}",
                f"{train_stats['aucj']:.3f}", f"{train_stats['sauc']:.3f}",

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


if __name__ == "__main__":
    main()
