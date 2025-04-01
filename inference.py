import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torchvision.io as tvio
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode
from dataset import SaliencyDataset, collate_fn
from losses_and_metrics import safe_pred_map
from models import build_saliency_model

model_checkpoints = {
    "dino": "best_checkpoint_dino.pth",
    "convnext": "best_checkpoint_convnext.pth",
    "purevit": "best_checkpoint_purevit.pth",
    "swin": "best_checkpoint_swin.pth",
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


def test_multiple_images_with_gt(
        device="cuda",
        test_json="dataset/scanpaths_test.json",
        orig_dir="dataset/orig_websaliency_all",
        fdm_dir="dataset/fdm_websaliency",
        eyemap_dir="dataset/eyemaps_websaliency",
        num_images=5
):
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    test_names = sorted({d["name"] for d in test_data if "name" in d})

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

    images_list = []
    for idx, batch in enumerate(test_loader):
        if idx >= num_images:
            break
        images_list.append(batch)

    print(f"Loaded {len(images_list)} test images into memory.")
    print("Will display figures for: 4 models × the chosen images.")

    for model_name, ckpt_path in model_checkpoints.items():
        model = load_model(model_name, ckpt_path, device)

        print(f"Running model '{model_name}' on {len(images_list)} images")

        for (imgs, fdms, eyemaps, mask, names) in images_list:
            image_name = names[0]

            imgs = imgs.to(device)
            fdms = fdms.to(device)

            with torch.no_grad():
                pred_raw = model(imgs)
                pred_map = safe_pred_map(pred_raw)

            orig_img = imgs[0].cpu().permute(1, 2, 0).numpy()
            orig_img = np.clip(orig_img, 0, 1)
            gt_fdm = fdms[0, 0].cpu().numpy()
            pred_np = pred_map[0, 0].cpu().numpy()
            pred_vis = pred_np / (pred_np.max() + 1e-8)

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Model: {model_name} | Image: {image_name}", fontsize=15)
            axs[0].imshow(orig_img)
            axs[0].imshow(gt_fdm, cmap='jet', alpha=0.4)
            axs[0].set_title("Ground Truth Saliency")
            axs[0].axis("off")
            axs[1].imshow(orig_img)
            axs[1].imshow(pred_vis, cmap='jet', alpha=0.4)
            axs[1].set_title("Predicted Saliency")
            axs[1].axis("off")
            display(fig)
            plt.close(fig)


def test_single_image(
        image_path="example_input.jpg",
        device="cuda"
):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = tvio.read_image(image_path, mode=ImageReadMode.RGB).float() / 255.0

    _, H, W = img.shape
    img_batch = img.unsqueeze(0).to(device)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Single Image: {os.path.basename(image_path)} (No Ground Truth)", fontsize=16)

    for idx, (model_name, ckpt_path) in enumerate(model_checkpoints.items()):
        model = load_model(model_name, ckpt_path, device)
        with torch.no_grad():
            pred_raw = model(img_batch)
            pred_map = safe_pred_map(pred_raw)

        pred_np = pred_map[0, 0].cpu().numpy()
        pred_vis = pred_np / (pred_np.max() + 1e-8)

        orig_img_np = img.cpu().permute(1, 2, 0).numpy()
        orig_img_np = np.clip(orig_img_np, 0, 1)

        axs[idx].imshow(orig_img_np)
        axs[idx].imshow(pred_vis, cmap='jet', alpha=0.4)
        axs[idx].set_title(f"{model_name} Prediction")
        axs[idx].axis("off")

    display(fig)
    plt.close(fig)


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
        test_single_image(
            image_path=args.single_image_path,
            device=device
        )


test_single_image("uia1.png", device="cuda")
