import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CKPT_PATH, DEVICE, TEST_JSON_PATH, ROOT_FOLDER, OUT_FOLDER, IN_H, IN_W, GT_H, GT_W
from model import Model
from saldataset import SalDataset

model = Model()
ckp = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(ckp["model"], strict=True)
model = model.eval().to(DEVICE)

with open(TEST_JSON_PATH, "r") as f:
    data_test = json.load(f)
test_files = sorted({d["name"] for d in data_test if "name" in d})
print(f"Inference on {len(test_files)} images from: {TEST_JSON_PATH}")

ds_test = SalDataset(ROOT_FOLDER, test_files)
dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)

os.makedirs(OUT_FOLDER, exist_ok=True)

with torch.no_grad():
    for batch in tqdm(dl_test, desc="Inference"):
        fname = batch["stem"][0]
        img_full = batch["image_full"].to(DEVICE)

        img_320x480 = F.interpolate(img_full, size=(IN_H, IN_W), mode='bilinear', align_corners=False)
        pred_40x60 = model(img_320x480)

        pred_up = F.interpolate(pred_40x60, size=(GT_H, GT_W), mode='bilinear', align_corners=False)
        sal_map = pred_up[0, 0].cpu().numpy()
        sal_map -= sal_map.min()
        sal_map /= (sal_map.max() + 1e-8)
        smap_255 = (sal_map * 255).astype(np.uint8)

        outpath = os.path.join(OUT_FOLDER, os.path.splitext(fname)[0] + "_sal.png")
        cv2.imwrite(outpath, smap_255)

print(f"Done: {OUT_FOLDER}")
