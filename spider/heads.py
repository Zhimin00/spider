import torch
import torch.nn as nn
import torch.nn.functional as F
from spider.roma import TransformerDecoder, Block, MemEffAttention, ConvRefiner, Decoder, CosKernel, GP, cls_to_flow_refine
from einops import rearrange
import pdb
import spider.utils.path_to_dust3r
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa
import time
inf = float('inf')

def post_process(d, desc_mode, desc_conf_mode, mlp=False):
    if not mlp:
        fmap = d.permute(0, 2, 3, 1)
    else:
        fmap = d
    desc = reg_desc(fmap[..., :-1], desc_mode)
    desc_conf = reg_dense_conf(fmap[..., -1], desc_conf_mode)
    return desc, desc_conf

class ResidualBottleneck(nn.Module):
    """Residual bottleneck with gentle reduction."""
    def __init__(self, in_dim, out_dim, mid_dim=None):
        super().__init__()
        if mid_dim is None:
            # keep mid large enough to avoid information bottleneck
            mid_dim = max(out_dim * 2, min(in_dim, 512))
        self.conv1 = nn.Conv2d(in_dim, mid_dim, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_dim)
        self.conv2 = nn.Conv2d(mid_dim, out_dim, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        self.relu  = nn.ReLU(inplace=True)

        self.short = nn.Identity() if in_dim == out_dim else nn.Conv2d(in_dim, out_dim, 1, bias=False)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.short(x))


class FusionGate(nn.Module):
    """
    Attention gate to fuse upsampled coarse (x_up) and fine (f):
      alpha = sigmoid(G([x_up, f]))  (spatial gate)
      out   = alpha * x_up + (1 - alpha) * f
      then a 3x3 refine conv
    Optionally, can condition the gate on an external confidence map.
    """
    def __init__(self, ch, cond_conf=False):
        super().__init__()
        in_ch = ch * 2 + (1 if cond_conf else 0)
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1, bias=True),
        )
        self.refine = nn.Conv2d(ch, ch, 3, padding=1)

        # initialize last conv for stable start (bias to prefer equal mix ~0.5)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x_up, f, conf=None):
        if conf is not None:
            x = torch.cat([x_up, f, conf], dim=1)
        else:
            x = torch.cat([x_up, f], dim=1)
        alpha = torch.sigmoid(self.gate(x))
        y = alpha * x_up + (1.0 - alpha) * f
        y = self.refine(y)
        return y, alpha


class MultiscaleFeatureRefiner(nn.Module):
    def __init__(self, desc_dim, desc_mode, desc_conf_mode, patch_size):
        super().__init__()
        self.desc_dim = desc_dim
        self.desc_mode = desc_mode
        self.desc_conf_mode = desc_conf_mode
        self.patch_size = patch_size

        in_dims = (1792, 512, 256, 128, 64)
        proj_dims = (256, 128, 128, 64, 64)
        hidden_dim = 128
        assert len(in_dims) == 5 and len(proj_dims) == 5

        # Per-scale projection (Residual bottleneck) to controlled widths, then unify to hidden_dim
        self.proj_blocks = nn.ModuleList([
            ResidualBottleneck(in_dims[i], proj_dims[i]) for i in range(5)
        ])
        self.to_hidden = nn.ModuleList([
            nn.Conv2d(proj_dims[i], hidden_dim, 1, bias=False) for i in range(5)
        ])

        # Attention-gated fusion from coarse->fine
        self.fuse16_8 = FusionGate(hidden_dim, cond_conf=False)
        self.fuse8_4  = FusionGate(hidden_dim, cond_conf=False)
        self.fuse4_2  = FusionGate(hidden_dim, cond_conf=False)
        self.fuse2_1  = FusionGate(hidden_dim, cond_conf=False)

        self.head_desc = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, desc_dim, 1)
        )
        self.head_conf = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1)
        )

    def _proj_to_hidden(self, f, idx):
        f = self.proj_blocks[idx](f)
        f = self.to_hidden[idx](f)
        return f
    

    def forward(self, cnn_feats, true_shape, upsample = False, low_desc = None, low_certainty = None):  # dict: {"16": f16, "8": f8, "4": f4, "2": f2, "1": f1]
        pyramid = {}
        H, W = true_shape
        scales = [1, 2, 4, 8, 16]
        N_Hs = [H // s if s != 16 else H // self.patch_size for s in scales]
        N_Ws = [W // s if s != 16 else W // self.patch_size for s in scales]

        # feat_pyramid = {}
        # for i, s in enumerate(scales):
        #     nh, nw = N_Hs[i], N_Ws[i]
        #     feat = rearrange(cnn_feats[i], 'b (nh nw) c -> b c nh nw', nh=nh, nw=nw)
        #     feat_pyramid[s] = feat  ##  b, c, nh, nw
        #     del feat

        # 1) Per-scale projection → unify to hidden_dim
        feat16 = rearrange(cnn_feats[-1], 'b (nh nw) c -> b c nh nw', nh=H // self.patch_size, nw=W // self.patch_size)
        P16 = self._proj_to_hidden(feat16, 0)  # (B,hidden,H//16,W//16)
        feat8 = rearrange(cnn_feats[-2], 'b (nh nw) c -> b c nh nw', nh=H//8, nw=W//8)
        P8  = self._proj_to_hidden(feat8,  1)
        feat4 = rearrange(cnn_feats[-3], 'b (nh nw) c -> b c nh nw', nh=H//4, nw=W//4)
        P4  = self._proj_to_hidden(feat4,  2)
        feat2 = rearrange(cnn_feats[1], 'b (nh nw) c -> b c nh nw', nh=H//2, nw=W//2)
        P2  = self._proj_to_hidden(feat2,  3)
        feat1 = rearrange(cnn_feats[0], 'b (nh nw) c -> b c nh nw', nh=H//1, nw=W//1)
        P1  = self._proj_to_hidden(feat1,  4)
        del cnn_feats
        # torch.cuda.empty_cache()
        
        # 2) Progressive refinement with attention-gated fusion
        x = F.interpolate(P16, size=P8.shape[-2:], mode='bilinear', align_corners=False)
        x, _ = self.fuse16_8(x, P8)
        pyramid['8']=rearrange(x, 'b c nh nw -> b (nh nw) c' , nh=H//8, nw=W//8)      # save fused scale-8
        del P16, P8

        x = F.interpolate(x, size=P4.shape[-2:], mode='bilinear', align_corners=False)
        x, _ = self.fuse8_4(x, P4)
        pyramid['4']=rearrange(x, 'b c nh nw -> b (nh nw) c' , nh=H//4, nw=W//4)      # save fused scale-4
        del P4

        x = F.interpolate(x, size=P2.shape[-2:], mode='bilinear', align_corners=False)
        x, _ = self.fuse4_2(x, P2)
        pyramid['2']=rearrange(x, 'b c nh nw -> b (nh nw) c' , nh=H//2, nw=W//2)        # save fused scale-2
        del P2

        x = F.interpolate(x, size=P1.shape[-2:], mode='bilinear', align_corners=False)
        x, _ = self.fuse2_1(x, P1)
        pyramid['1']=rearrange(x, 'b c nh nw -> b (nh nw) c' , nh=H//1, nw=W//1)       # save fused scale-1
        del P1
        # 3) Heads (full-res)
        desc = self.head_desc(x)                 # (B,24,H,W)
        desc_conf = self.head_conf(x)  # (B,1,H,W)
        desc = reg_desc(desc.movedim(1, -1), self.desc_mode)
        desc_conf = reg_dense_conf(desc_conf[:, 0], self.desc_conf_mode)
        return {'desc': desc, 'desc_conf': desc_conf, # [B, H, W, D], [B, H, W]
                }  


def reg_dense_conf(x, mode= ('exp', 0, inf)):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')

def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc

class WarpHead(nn.Module):
    def __init__(
            self,
            decoder,
            patch_size
    ):
        super().__init__()
        self.decoder = decoder
        self.patch_size = patch_size

    def forward(self, cnn_feats1, cnn_feats2, true_shape1, true_shape2, upsample = False, scale_factor = 1, finest_corresps=None):
        feat1_pyramid = {}
        H1, W1 = true_shape1[-2:]
        if upsample:
            scales = [1, 2, 4, 8]
        else:
            scales = [1, 2, 4, 8, 16]
        N_Hs1 = [H1 // s if s != 16 else H1 // self.patch_size for s in scales]
        N_Ws1 = [W1 // s if s != 16 else W1 // self.patch_size for s in scales]
        
        for i, s in enumerate(scales):
            nh, nw = N_Hs1[i], N_Ws1[i]
            feat = rearrange(cnn_feats1[i], 'b (nh nw) c -> b nh nw c', nh=nh, nw=nw)
            feat1_pyramid[s] = feat.permute(0, 3, 1, 2).contiguous()  ## b, c, nh, nw
            del feat
        feat2_pyramid = {}
        H2, W2 = true_shape2[-2:]
        N_Hs2 = [H2 // s if s != 16 else H2 // self.patch_size for s in scales]
        N_Ws2 = [W2 // s if s != 16 else W2 // self.patch_size for s in scales]
        for i, s in enumerate(scales):
            nh, nw = N_Hs2[i], N_Ws2[i]
            feat = rearrange(cnn_feats2[i], 'b (nh nw) c -> b nh nw c', nh=nh, nw=nw)
            feat2_pyramid[s] = feat.permute(0, 3, 1, 2).contiguous()
            del feat
        corresps = self.decoder(feat1_pyramid, 
                                feat2_pyramid, 
                                upsample = upsample, 
                                **(finest_corresps if finest_corresps else {}),
                                scale_factor=scale_factor)
        return corresps


## warp or linear
def head_factory(head_type, net):
    patch_size = net.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
            patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
        assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
        patch_size = patch_size[0]
    if head_type == 'warp':     
        gp_dim = 512
        feat_dim = 512
        decoder_dim = gp_dim + feat_dim
        cls_to_coord_res = 64
        coordinate_decoder = TransformerDecoder(nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]), 
        decoder_dim, 
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        pos_enc = False,)

        dw = True
        hidden_blocks = 8
        kernel_size = 5
        displacement_emb = "linear"
        disable_local_corr_grad = True

        conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512+128+(2*7+1)**2,
                2 * 512+128+(2*7+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius = 7,
                corr_in_other = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "8": ConvRefiner(
                2 * 512+64+(2*3+1)**2,
                2 * 512+64+(2*3+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius = 3,
                corr_in_other = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "4": ConvRefiner(
                2 * 256+32+(2*2+1)**2,
                2 * 256+32+(2*2+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius = 2,
                corr_in_other = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "2": ConvRefiner(
                2 * 64+16,
                128+16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks = hidden_blocks,
                displacement_emb = displacement_emb,
                displacement_emb_dim = 6,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
        }
        )
        kernel_temperature = 0.2
        learn_temperature = False
        no_cov = True
        kernel = CosKernel
        only_attention = False
        basis = "fourier"
        gp16 = GP(
            kernel,
            T=kernel_temperature,
            learn_temperature=learn_temperature,
            only_attention=only_attention,
            gp_dim=gp_dim,
            basis=basis,
            no_cov=no_cov,
        )
        gps = nn.ModuleDict({"16": gp16})
        proj16 = nn.Sequential(nn.Conv2d(1024+768, 512, 1, 1), nn.BatchNorm2d(512))
        proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
        proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
        proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
        proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
        # proj8 = nn.Sequential(nn.Conv2d(128, 512, 1, 1), nn.BatchNorm2d(512))
        # proj4 = nn.Sequential(nn.Conv2d(128, 256, 1, 1), nn.BatchNorm2d(256))
        # proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
        # proj1 = nn.Sequential(nn.Conv2d(128, 9, 1, 1), nn.BatchNorm2d(9))
        proj = nn.ModuleDict({
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
            })
        displacement_dropout_p = 0.0
        gm_warp_dropout_p = 0.0
        decoder = Decoder(coordinate_decoder, 
                        gps,
                        proj, 
                        conv_refiner, 
                        detach=True,#True, 
                        scales=["16", "8", "4", "2", "1"], 
                        displacement_dropout_p = displacement_dropout_p,
                        gm_warp_dropout_p = gm_warp_dropout_p)
        return WarpHead(decoder, patch_size)
    elif head_type == 'msfr':
        return MultiscaleFeatureRefiner(net.local_feat_dim, net.desc_mode, net.desc_conf_mode, patch_size)
    
    else:
        raise NotImplementedError(
            f"unexpected {head_type=}")


