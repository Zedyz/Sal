import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


# dinov2

class DinoV2Saliency(nn.Module):
    """
    A saliency model that uses DINOv2-based ViT-B/14 as the backbone,
    plus a lightweight conv head for upsampling to the original resolution.
    """

    def __init__(self):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        hidden_dim = self.backbone.config.hidden_size  # 768 for ViT-B/14
        self.conv_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # [0,1] → [-1,1] (dinov2 expects [-1,1])
        x_input = 2.0 * x - 1.0

        out = self.backbone(pixel_values=x_input)
        feats = out.last_hidden_state  # [B, N, hidden_dim]
        patch_tokens = feats[:, 1:, :]  # skip CLS

        # Reshape patch tokens into a 2D feature map
        Hf = H // 14
        Wf = W // 14
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, -1, Hf, Wf)

        # Run the conv head => low-res saliency
        sal_low = self.conv_head(patch_tokens)
        # Upsample to original size
        sal_map = F.interpolate(sal_low, (H, W), mode='bilinear', align_corners=False)
        return sal_map


# conv + vit

def set_bn_eval(module):
    """
    Sets BatchNorm / LayerNorm modules to eval mode and
    disables gradient. Typically used to 'freeze' them.
    """
    if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        module.eval()
        module.requires_grad_(False)


def freeze_convnext_stages(model, stage_idx=3):
    """
    Freeze certain portions (stages) of a convnext_tiny backbone.
    stage_idx in [0..3], or -1 => unfreeze all.
    If stage_idx=3 => freeze entire backbone of convnext_tiny.
    """
    # Unfreeze everything first
    for p in model.backbone.model.features.parameters():
        p.requires_grad = True

    if stage_idx < 0:
        return

    total_len = len(model.backbone.model.features)
    block_split = (stage_idx + 1) * total_len // 4
    for i in range(block_split):
        for p in model.backbone.model.features[i].parameters():
            p.requires_grad = False


class ConvNeXtBackboneFinal(nn.Module):
    """
    Return ONLY the final feature map from a convnext_tiny => shape [B,768,Hf,Wf].
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.features = self.model.features

    def forward(self, x):
        out = x
        for blk in self.features:
            out = blk(out)
        return out


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic depth (drop path).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    B = x.shape[0]
    rand_tensor = keep_prob + torch.rand(B, 1, 1, device=x.device)
    mask = (rand_tensor.floor() == 1)
    return x * mask / keep_prob


class StochasticDepthTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Extended TransformerEncoderLayer with drop_path (stochastic depth).
    """

    def __init__(self, *args, drop_path_rate=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_path_rate = drop_path_rate

    def forward(self, src, *args, **kwargs):
        x = super().forward(src, *args, **kwargs)
        x = drop_path(x, self.drop_path_rate, self.training)
        return x


class ConvNeXtViT(nn.Module):
    """
    ConvNeXt -> final feature map => flatten => [CLS]+pos => Transformer => aggregator => upsample
    """

    def __init__(
            self,
            backbone_out_ch=768,
            embed_dim=768,
            depth=4,
            nhead=8,
            mlp_ratio=4.0,
            drop_rate=0.1,
            drop_path_rate=0.1
    ):
        super().__init__()
        self.backbone = ConvNeXtBackboneFinal()
        self.backbone.apply(set_bn_eval)

        if backbone_out_ch != embed_dim:
            self.proj = nn.Conv2d(backbone_out_ch, embed_dim, kernel_size=1)
        else:
            self.proj = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None
        self.embed_dim = embed_dim

        enc_layer = StochasticDepthTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=int(mlp_ratio * embed_dim),
            batch_first=True,
            dropout=drop_rate,
            drop_path_rate=drop_path_rate
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.head_pre = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.head_final = nn.Conv2d(embed_dim, 1, kernel_size=1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, _, H, W = x.shape
        feat = self.backbone(x)  # => [B, 768, Hf, Wf]

        # optional projection if embed_dim != backbone_out_ch
        if self.proj is not None:
            feat = self.proj(feat)

        Hf, Wf = feat.shape[2], feat.shape[3]
        tokens = feat.flatten(2).transpose(1, 2)
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)
        total_tokens = tokens.shape[1]

        # Lazy init pos_embed if needed
        if (self.pos_embed is None) or (self.pos_embed.shape[1] < total_tokens):
            pe = torch.zeros(1, total_tokens, self.embed_dim)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = nn.Parameter(pe.to(tokens.device))

        tokens = tokens + self.pos_embed[:, :total_tokens, :]

        encoded = self.transformer(tokens)
        patch_tokens = encoded[:, 1:, :]
        out2d = patch_tokens.transpose(1, 2).view(B, -1, Hf, Wf)

        xh = self.head_pre(out2d)
        sal_low = self.head_final(xh)
        sal_map = F.interpolate(sal_low, (H, W), mode='bilinear', align_corners=False)
        return sal_map


# vit, no bb

class PureVisionTransformer(nn.Module):
    """
    A slightly bigger pure ViT for saliency:
      - embed_dim=576
      - depth=10
      - patch_size=16
      => aggregator => upsample => final saliency
    """

    def __init__(
            self,
            img_size=384,
            patch_size=16,
            in_chans=3,
            embed_dim=576,
            depth=10,
            nhead=8,
            mlp_ratio=4.0,
            drop_rate=0.1,
            drop_path_rate=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # [CLS] token + pos_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_h = self.img_size // self.patch_size
        self.num_patches = num_h * num_h
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        enc_layer = StochasticDepthTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            dropout=drop_rate,
            drop_path_rate=drop_path_rate
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.head_pre = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.head_final = nn.Conv2d(embed_dim, 1, kernel_size=1)

        # init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        # Resize => (img_size, img_size)
        x_rsz = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        # Patchify => [B, embed_dim, Hf, Wf]
        feat = self.patch_embed(x_rsz)
        Hf, Wf = feat.shape[2], feat.shape[3]

        # Flatten => tokens => [B, N, embed_dim]
        tokens = feat.flatten(2).transpose(1, 2)

        # [CLS] + pos_embed
        cls_tok = self.cls_token.expand(B, -1, -1)
        x_tok = torch.cat([cls_tok, tokens], dim=1)
        x_tok = x_tok + self.pos_embed[:, :(Hf * Wf + 1), :]

        encoded = self.transformer(x_tok)  # => [B, N+1, embed_dim]
        patch_tokens = encoded[:, 1:, :]  # skip CLS

        out2d = patch_tokens.transpose(1, 2).view(B, -1, Hf, Wf)
        xh = self.head_pre(out2d)
        sal_low = self.head_final(xh)
        # upsample back to original input size
        sal_map = F.interpolate(sal_low, (H, W), mode='bilinear', align_corners=False)
        return sal_map


# swin

class SwinBackbone(nn.Module):
    """
    A Swin Tiny backbone that outputs features at multiple stages.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import swin_t, Swin_T_Weights
        self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.swin.eval()
        self.features = self.swin.features

    def forward(self, x):
        feats = []
        out = x
        for stage in self.features:
            out = stage(out)
            feats.append(out)
        # f1= feats[1] => (B, H/4, W/4, 96)
        # f2= feats[3] => (B, H/8, W/8, 192)
        # f3= feats[5] => (B, H/16, W/16, 384)
        # f4= feats[7] => (B, H/32, W/32, 768)
        f1 = feats[1].permute(0, 3, 1, 2).contiguous()
        f2 = feats[3].permute(0, 3, 1, 2).contiguous()
        f3 = feats[5].permute(0, 3, 1, 2).contiguous()
        f4 = feats[7].permute(0, 3, 1, 2).contiguous()
        return f1, f2, f3, f4


def freeze_stage_params_full(m, stage_idx=3):
    """
    For HybridSwinViT, freeze certain blocks of the Swin Tiny backbone.
    stage_idx in [0..3], or -1 => unfreeze all.
    """
    for p in m.backbone.swin.parameters():
        p.requires_grad = True
    # stage map => which indices in self.swin.features to freeze
    stage_map = {
        0: [0, 1],
        1: [0, 1, 2, 3],
        2: [0, 1, 2, 3, 4, 5],
        3: [0, 1, 2, 3, 4, 5, 6, 7],
    }
    if stage_idx < 0:
        return
    freeze_list = stage_map.get(stage_idx, [])
    for i in freeze_list:
        for p in m.backbone.swin.features[i].parameters():
            p.requires_grad = False


class HybridSwinViT(nn.Module):
    """
    Combines multiple scales from a Swin Tiny backbone, merges them into
    a single embedding plane, adds a Transformer block, optional center prior,
    and outputs a saliency map.
    """

    def __init__(
            self,
            embed_dim=960,
            dropout=0.25,
            drop_path_rate=0.15,
            num_transformer_layers=8,
            nhead=16,
            ff_dim=4096,
            center_prior=True
    ):
        super().__init__()
        self.backbone = SwinBackbone()
        self.backbone.apply(set_bn_eval)

        self.proj1 = nn.Conv2d(96, embed_dim, 1)
        self.proj2 = nn.Conv2d(192, embed_dim, 1)
        self.proj3 = nn.Conv2d(384, embed_dim, 1)
        self.proj4 = nn.Conv2d(768, embed_dim, 1)

        # pos_enc => final scale=32x32
        self.pos_enc = nn.Parameter(torch.zeros(1, embed_dim, 32, 32))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        enc_layer = StochasticDepthTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        self.use_center = center_prior
        if center_prior:
            self.center_bias = nn.Parameter(torch.zeros(1, 1, 32, 32))
            self.center_scale = nn.Parameter(torch.tensor(0.01))

        self.head_pre = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.head_final = nn.utils.spectral_norm(nn.Conv2d(embed_dim, 1, 1))

        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

    def forward(self, x):
        B, _, H, W = x.shape
        f1, f2, f3, f4 = self.backbone(x)

        p1 = self.proj1(f1)  # => [B,embed_dim,Hf,Wf]
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)

        # unify all at the largest (f4) scale => stride=32
        Hf, Wf = p4.shape[2], p4.shape[3]
        p1_up = F.interpolate(p1, (Hf, Wf), mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, (Hf, Wf), mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, (Hf, Wf), mode='bilinear', align_corners=False)
        fused = p1_up + p2_up + p3_up + p4

        # Add 2D positional embedding
        pos_rsz = F.interpolate(self.pos_enc, (Hf, Wf), mode='bilinear', align_corners=False)
        fused = fused + pos_rsz

        # Flatten => add CLS => transform => remove CLS => upsample
        fused = fused.flatten(2).transpose(1, 2)  # => (B,N,embed_dim)
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, fused], dim=1)

        encoded = self.transformer(tokens)
        patch_tokens = encoded[:, 1:, :]
        patch_tokens = patch_tokens.transpose(1, 2).view(B, -1, Hf, Wf)

        xh = self.head_pre(patch_tokens)
        sal_low = self.head_final(xh)

        if self.use_center:
            cb = F.interpolate(self.center_bias, (Hf, Wf), mode='bilinear', align_corners=False)
            sal_low = sal_low + cb * self.center_scale

        sal_map = F.interpolate(sal_low, (H, W), mode='bilinear', align_corners=False)
        return sal_map


def build_saliency_model(model_name: str):
    """
    Factory function to create whichever model you want, by name:
      "dino"       -> DinoV2Saliency
      "convnext"   -> ConvNeXtViT
      "purevit"    -> PureVisionTransformer
      "swin" -> HybridSwinViT
    """
    model_name = model_name.lower()
    if model_name == "dino":
        return DinoV2Saliency()
    elif model_name == "convnext":
        return ConvNeXtViT()
    elif model_name == "purevit":
        return PureVisionTransformer()
    elif model_name == "swin":
        return HybridSwinViT(
            embed_dim=960,
            dropout=0.25,
            drop_path_rate=0.15,
            num_transformer_layers=8,
            nhead=16,
            ff_dim=4096,
            center_prior=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")
