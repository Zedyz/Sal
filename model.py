import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import IN_H, IN_W


class Encoder(nn.Module):
    def __init__(self, in_chans=3):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224',
                                          pretrained=True,
                                          in_chans=in_chans)
        self.backbone.head = nn.Identity()

        self.backbone.patch_embed.img_size = (IN_H, IN_W)
        self.backbone.patch_embed.grid_size = (IN_H // 16, IN_W // 16)
        self.backbone.patch_embed.num_patches = (IN_H // 16) * (IN_W // 16)

        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = self.backbone.patch_embed(x)
        old_pe = self.backbone.pos_embed
        new_pe = interpolate_pos_embed(self.backbone.patch_embed, old_pe, H, W).to(x.device)

        cls_tok = new_pe[:, 0:1, :]
        patch_pe = new_pe[:, 1:, :]
        x_ = x_ + patch_pe
        cls_tokens = cls_tok.expand(B, -1, -1)
        x_ = torch.cat([cls_tokens, x_], dim=1)

        x_ = self.backbone.pos_drop(x_)
        for blk in self.backbone.blocks:
            x_ = blk(x_)
        x_ = self.backbone.norm(x_)
        return x_[:, 1:, :]


class Decoder(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.sal_fc = nn.Linear(hidden_dim, 1)

    def forward(self, tokens):
        B, N, d = tokens.shape
        x_ = self.proj(tokens)
        x_4d = x_.view(B, 20, 30, -1).permute(0, 3, 1, 2)
        up = F.interpolate(x_4d, size=(40, 60), mode='bilinear', align_corners=False)
        up_4d = up.permute(0, 2, 3, 1)
        out = self.sal_fc(up_4d)
        out = out.permute(0, 3, 1, 2)
        return torch.sigmoid(out)


class ComponentSaliencyNet(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()
        self.face_decoder = Decoder(embed_dim, hidden_dim)
        self.text_decoder = Decoder(embed_dim, hidden_dim)
        self.banner_decoder = Decoder(embed_dim, hidden_dim)
        self.base_decoder = Decoder(embed_dim, hidden_dim)

        self.fuse_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

    def forward(self, tokens):
        face_map = self.face_decoder(tokens)
        text_map = self.text_decoder(tokens)
        banner_map = self.banner_decoder(tokens)
        base_map = self.base_decoder(tokens)

        cat_ = torch.cat([face_map, text_map, banner_map, base_map], dim=1)
        B, C, H, W = cat_.shape
        cat_flat = cat_.view(B, C, H * W).permute(0, 2, 1)
        fused = self.fuse_mlp(cat_flat)
        fused_map = fused.view(B, 1, H, W)
        fused_map = torch.sigmoid(fused_map)
        return fused_map, (face_map, text_map, banner_map, base_map)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_chans=3)
        self.csn = ComponentSaliencyNet(embed_dim=768, hidden_dim=128)

    def forward(self, rgb_320x480):
        tokens = self.encoder(rgb_320x480)
        final_map, comps = self.csn(tokens)
        return final_map


def interpolate_pos_embed(patch_embed, old_pos_embed, H, W):
    cls_tok = old_pos_embed[:, 0:1, :]
    patch_tok = old_pos_embed[:, 1:, :]
    c = patch_tok.shape[2]
    old_size = int(math.sqrt(patch_tok.shape[1]))
    old_2d = patch_tok.view(1, old_size, old_size, c).permute(0, 3, 1, 2)

    new_h = H // 16
    new_w = W // 16
    new_2d = F.interpolate(old_2d, size=(new_h, new_w), mode='bicubic', align_corners=False)
    new_2d = new_2d.permute(0, 2, 3, 1).reshape(1, new_h * new_w, c)
    return torch.cat([cls_tok, new_2d], dim=1)
