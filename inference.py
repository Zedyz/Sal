import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode
import torchvision.io as tvio

from dataset import SaliencyDataset, collate_fn
from losses_and_metrics import safe_pred_map
from models import build_saliency_model

# Example checkpoints & training logs (update paths as needed)
model_checkpoints = {
    "dino": "checkpoints/best_checkpoint_dino.pth",
    "convnext": "checkpoints/best_checkpoint_convnext.pth",
    "purevit": "checkpoints/best_checkpoint_purevit.pth",
    "swin": "checkpoints/best_checkpoint_swin.pth",
}

training_logs = {
    "dino": "training_runs/training_log_dino.csv",
    "convnext": "training_runs/training_log_convnext.csv",
    "purevit": "training_runs/training_log_purevit.csv",
    "swin": "training_runs/training_log_swin.csv",
}


def load_model(model_name, ckpt_path, device="cuda"):
    model = build_saliency_model(model_name).to(device)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        loaded_dict = ckpt["model_state"]
    else:
        loaded_dict = ckpt

    model_dict = model.state_dict()
    remove_keys = []
    for k in loaded_dict.keys():
        if k not in model_dict:
            remove_keys.append(k)
    for k in remove_keys:
        print(f"Removing unexpected key for '{model_name}': {k}")
        del loaded_dict[k]

    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    return model


def find_best_epoch_nss_cc(csv_path):
    """
    Returns (best_epoch, best_nss, best_cc) for the maximum (val_nss + val_cc).
    """
    df = pd.read_csv(csv_path)
    if ("val_nss" not in df.columns) or ("val_cc" not in df.columns):
        return None, None, None

    combined = df["val_nss"] + df["val_cc"]
    best_idx = combined.idxmax()
    best_epoch = df.loc[best_idx, "epoch"]
    best_nss = df.loc[best_idx, "val_nss"]
    best_cc = df.loc[best_idx, "val_cc"]
    return best_epoch, best_nss, best_cc


def test_multiple_images_with_gt(
    device="cuda",
    test_json="dataset/scanpaths_test.json",
    orig_dir="dataset/orig_websaliency_all",
    fdm_dir="dataset/fdm_websaliency",
    eyemap_dir="dataset/eyemaps_websaliency",
    num_images=5
):
    """
    Creates a single figure with 2*nimages rows and 4 columns.
    For each image i:
        - Row (2*i)   => single wide subplot spanning all 4 columns for ground-truth
        - Row (2*i+1) => 4 subplots, each for a model prediction (dino, convnext, purevit, swin)
    Titles show best epoch + individual NSS/CC from training logs.
    """

    # 1) Gather test filenames
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    test_names = sorted({d["name"] for d in test_data if "name" in d})

    # 2) Create dataset & loader
    test_ds = SaliencyDataset(
        orig_dir=orig_dir,
        fdm_dir=fdm_dir,
        eyemap_dir=eyemap_dir,
        file_list=test_names,
        is_train=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=lambda b: collate_fn(b, patch_size=14)
    )

    # 3) Load up to num_images
    images_list = []
    for idx, batch in enumerate(test_loader):
        if idx >= num_images:
            break
        images_list.append(batch)
    nimages = len(images_list)
    print(f"Loaded {nimages} test images into memory.")

    # 4) Load models
    model_names = ["dino", "convnext", "purevit", "swin"]
    loaded_models = {}
    for mname in model_names:
        loaded_models[mname] = load_model(mname, model_checkpoints[mname], device)

    # 5) Parse logs to get best epoch + (NSS, CC)
    best_info = {}
    for mname in model_names:
        csv_path = training_logs.get(mname, None)
        if csv_path and os.path.isfile(csv_path):
            ep, nss, cc = find_best_epoch_nss_cc(csv_path)
            best_info[mname] = (ep, nss, cc)
        else:
            best_info[mname] = (None, None, None)

    # 6) Create a figure with 2*nimages rows, 4 columns
    # Make the figure fairly large
    fig = plt.figure(figsize=(30, 10 * nimages))

    # We define height_ratios so the GT row is slightly smaller (2.2) than the predictions row (3).
    from matplotlib.gridspec import GridSpec
    height_ratios = []
    for _ in range(nimages):
        height_ratios.extend([2.2, 3])  # GT is 2.2 units tall, models row is 3 units

    # Create the GridSpec with some vertical space
    gs = GridSpec(
        nrows=2*nimages, ncols=4,
        height_ratios=height_ratios,
        figure=fig,
        wspace=0,   # no horizontal gap
        hspace=0.2  # add some vertical gap
    )

    title_fontsize = 20

    # 7) Fill figure
    for i, (imgs, fdms, eyemaps, mask, names) in enumerate(images_list):
        image_name = names[0]
        imgs = imgs.to(device)
        fdms = fdms.to(device)

        # Convert original image to NumPy
        orig_img = imgs[0].cpu().permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)

        # A) GT subplot => row=2*i, spanning col=0..3
        ax_gt = fig.add_subplot(gs[2*i, 0:4])
        gt_fdm = fdms[0, 0].cpu().numpy()
        ax_gt.imshow(orig_img)
        ax_gt.imshow(gt_fdm, cmap='jet', alpha=0.4)
        ax_gt.set_title(f"[{image_name}] Ground Truth", fontsize=title_fontsize)
        ax_gt.axis("off")

        # B) 4 model predictions => row=2*i+1, col=0..3
        for j, mname in enumerate(model_names):
            model = loaded_models[mname]
            ep, best_nss, best_cc = best_info[mname]

            # Inference
            with torch.no_grad():
                pred_raw = model(imgs)
                pred_map = safe_pred_map(pred_raw)

            pred_np = pred_map[0, 0].cpu().numpy()
            pred_vis = pred_np / (pred_np.max() + 1e-8)

            ax_pred = fig.add_subplot(gs[2*i+1, j])
            ax_pred.imshow(orig_img)
            ax_pred.imshow(pred_vis, cmap='jet', alpha=0.4)

            # Title includes epoch, NSS, CC
            if ep is not None:
                ax_pred.set_title(
                    f"{mname}\n(Ep={ep}, NSS={best_nss:.3f}, CC={best_cc:.3f})",
                    fontsize=title_fontsize
                )
            else:
                ax_pred.set_title(f"{mname}\n(No log info)", fontsize=title_fontsize)
            ax_pred.axis("off")

    # Adjust subplot margins
    # Keep no left/right margin, but allow top/bottom margin to avoid cutting off text
    fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.05, hspace=0.2)

    # Finally show the figure
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="multi",
                        help="Which inference task to run: 'multi' or 'single'.")
    parser.add_argument("--single_image_path", type=str, default="example_input.jpg",
                        help="Path to a single image if running `--task single`.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_json", type=str, default="dataset/scanpaths_test.json")
    parser.add_argument("--orig_dir", type=str, default="dataset/orig_websaliency_all")
    parser.add_argument("--fdm_dir", type=str, default="dataset/fdm_websaliency")
    parser.add_argument("--eyemap_dir", type=str, default="dataset/eyemaps_websaliency")
    parser.add_argument("--num_images", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.task == "multi":
        test_multiple_images_with_gt(
            device=device,
            test_json=args.test_json,
            orig_dir=args.orig_dir,
            fdm_dir=args.fdm_dir,
            eyemap_dir=args.eyemap_dir,
            num_images=args.num_images
        )
    else:
        pass


if __name__ == "__main__":
    test_multiple_images_with_gt()
