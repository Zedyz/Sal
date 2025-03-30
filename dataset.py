import os
import random
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from torch.utils.data import Dataset


class SaliencyDataset(Dataset):
    def __init__(self,
                 orig_dir,
                 fdm_dir,
                 eyemap_dir,
                 file_list=None,
                 is_train=False,
                 scale_range=(0.95, 1.05),
                 brightness_range=(0.9, 1.1)):
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
            alt_ext = '.png' if ext != '.png' else '.jpg'
            fdm_path = os.path.join(self.fdm_dir, base + alt_ext)
        fdm = tvio.read_image(fdm_path)
        if fdm.shape[0] > 1:
            fdm = fdm[:1]  # keep just one channel
        fdm = fdm.float() / 255.0

        eye_path = os.path.join(self.eyemap_dir, base + ext)
        if not os.path.exists(eye_path):
            alt_ext = '.png' if ext != '.png' else '.jpg'
            eye_path = os.path.join(self.eyemap_dir, base + alt_ext)
        eye = tvio.read_image(eye_path)
        if eye.shape[0] > 1:
            eye = eye[:1]
        eye = (eye > 127).float()

        bluh = False
        if self.is_train and bluh == True:
            scale_factor = random.uniform(*self.scale_range)
            new_h = int(image.shape[1] * scale_factor)
            new_w = int(image.shape[2] * scale_factor)

            image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                                  mode='bilinear', align_corners=False)[0]
            fdm = F.interpolate(fdm.unsqueeze(0), size=(new_h, new_w),
                                mode='bilinear', align_corners=False)[0]
            eye = F.interpolate(eye.unsqueeze(0), size=(new_h, new_w),
                                mode='bilinear', align_corners=False)[0]

            bright_factor = random.uniform(*self.brightness_range)
            image = torch.clamp(image * bright_factor, 0.0, 1.0)

        return image, fdm, eye, fname


def collate_fn(batch, patch_size=14):
    """
    Collate => pad each image/fdm/eye to multiples of patch_size in H and W,
    then stack them. Return (imgs, fdms, eyes, mask, names).
    """
    bsz = len(batch)
    heights = [b[0].shape[1] for b in batch]
    widths = [b[0].shape[2] for b in batch]

    Hmax = max(heights)
    Wmax = max(widths)

    # Round up to multiple-of-patch_size
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
        _, H, W = im.shape
        imgs[i, :, :H, :W] = im
        fdms[i, :, :H, :W] = fd
        eyes[i, :, :H, :W] = ey
        mask[i, :, :H, :W] = 1.0
        names.append(nm)

    return imgs, fdms, eyes, mask, names
