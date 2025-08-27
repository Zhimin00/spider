import torch
import torch.nn as nn
import torch.nn.functional as F
from spider.roma import TransformerDecoder, Block, MemEffAttention, ConvRefiner, Decoder, CosKernel, GP
from einops import rearrange
import pdb
import spider.utils.path_to_dust3r
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa
inf = float('inf')

def post_process(d, desc_mode, desc_conf_mode, mlp=False):
    if not mlp:
        fmap = d.permute(0, 2, 3, 1)
    else:
        fmap = d
    desc = reg_desc(fmap[..., :-1], desc_mode)
    desc_conf = reg_dense_conf(fmap[..., -1], desc_conf_mode)
    return desc, desc_conf

class FM_MLP(nn.Module):
    def __init__(self, desc_dim, desc_mode, desc_conf_mode, patch_size, hidden_dim_factor = 4, detach=False):
        super().__init__()
        self.desc_dim = desc_dim
        self.desc_mode = desc_mode
        self.desc_conf_mode = desc_conf_mode
        self.patch_size = patch_size
        self.detach = detach
        
        
        self.head_local_features = Mlp(in_features=1024 + 768,
                            hidden_features=int(hidden_dim_factor * (1024 + 768)),
                            out_features=(self.desc_dim + 1)*self.patch_size ** 2)

    def forward(self, cnn_feats, true_shape, upsample = False, low_desc = None, low_certainty = None):  # dict: {"16": f16, "8": f8, "4": f4, "2": f2, "1": f1]
        H, W = true_shape
        if upsample:
            scales = [1, 2, 4, 8]
        else:
            scales = [1, 2, 4, 8, 16]
        N_Hs = [H // s if s != 16 else H // self.patch_size for s in scales]
        N_Ws = [W // s if s != 16 else W // self.patch_size for s in scales]

        feat_pyramid = {}
        for i, s in enumerate(scales):
            nh, nw = N_Hs[i], N_Ws[i]
            feat = rearrange(cnn_feats[i], 'b (nh nw) c -> b nh nw c', nh=nh, nw=nw)
            # feat_pyramid[s] = feat.permute(0, 3, 1, 2).contiguous()  ## b, c, nh, nw
            feat_pyramid[s] = feat  ##  b, c, nh, nw
            del feat
        local_features = self.head_local_features(feat_pyramid[16])  # B,H//16,W//16,D
        local_features = F.pixel_shuffle(local_features.permute(0, 3, 1, 2), self.patch_size)  # B,d,H,W
        desc, desc_conf = post_process(local_features, self.desc_mode, self.desc_conf_mode)
        return {'desc': desc, 'desc_conf': desc_conf, # [B, H, W, D], [B, H, W]
                }  


class FM_conv(nn.Module):
    def __init__(self, desc_dim, desc_mode, desc_conf_mode, patch_size, detach = False):
        super().__init__()
        self.desc_dim = desc_dim
        self.desc_mode = desc_mode
        self.desc_conf_mode = desc_conf_mode
        self.patch_size = patch_size
        self.detach = detach

        self.proj16 = nn.Sequential(nn.Conv2d(1024+768, 1792, 1, 1), nn.BatchNorm2d(1792))

        self.pred16 = self._make_block(1792, 1792, (self.desc_dim + 1) * self.patch_size ** 2)
        
    def _make_block(self, in_dim, hidden_dim, out_dim, bn_momentum=0.01):
        return nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(hidden_dim, momentum = bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0),
        )

    def forward(self, cnn_feats, true_shape, upsample = False, low_desc = None, low_certainty = None):  # dict: {"16": f16, "8": f8, "4": f4, "2": f2, "1": f1]
        H1, W1 = true_shape[-2:]
        if upsample:
            scales = [1, 2, 4, 8]
        else:
            scales = [1, 2, 4, 8, 16]
        N_Hs = [H1 // s if s != 16 else H1 // self.patch_size for s in scales]
        N_Ws = [W1 // s if s != 16 else W1 // self.patch_size for s in scales]
        feat_pyramid = {}
        for i, s in enumerate(scales):
            nh, nw = N_Hs[i], N_Ws[i]
            feat = rearrange(cnn_feats[i], 'b (nh nw) c -> b c nh nw', nh=nh, nw=nw)
            # feat_pyramid[s] = feat.permute(0, 3, 1, 2).contiguous()  ## b, c, nh, nw
            feat_pyramid[s] = feat  ##  b, c, nh, nw
            del feat
        
        f16 = self.proj16(feat_pyramid[16]) #b, c, h//16, w//16
        d = self.pred16(f16) #b, (D+1)*256, h//16, w//16
        d = F.pixel_shuffle(d, self.patch_size) #b, D+1, h, w
        desc, desc_conf = post_process(d, self.desc_mode, self.desc_conf_mode)

        return {'desc': desc, 'desc_conf': desc_conf, # [B, H, W, D], [B, H, W]
                }  

class MultiScaleFM_conv(nn.Module):
    def __init__(self, desc_dim, desc_mode, desc_conf_mode, patch_size, detach = False):
        super().__init__()
        self.desc_dim = desc_dim
        self.desc_mode = desc_mode
        self.desc_conf_mode = desc_conf_mode
        self.patch_size = patch_size
        self.detach = detach

        self.proj16 = nn.Sequential(nn.Conv2d(1024+768, 512, 1, 1), nn.BatchNorm2d(512))
        self.proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
        self.proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
        self.proj2 = nn.Sequential(nn.Conv2d(128, 256, 1, 1), nn.BatchNorm2d(256))
        self.proj1 = nn.Sequential(nn.Conv2d(64, 256, 1, 1), nn.BatchNorm2d(256))

        self.up8 = nn.Sequential(nn.Conv2d(512+512, 512, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(nn.Conv2d(512+256, 256, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(256+256, 256, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.Conv2d(256+256, 256, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pred16 = self._make_block(512, 512, (self.desc_dim + 1) * self.patch_size ** 2)
        self.pred8 = self._make_block(512, 512, (self.desc_dim + 1) * 8 ** 2)
        self.pred4 = self._make_block(256, 256, (self.desc_dim + 1) * 4 ** 2)
        self.pred2 = self._make_block(256, 256, (self.desc_dim + 1) * 2 ** 2)
        self.pred1 = self._make_block(256, 256, self.desc_dim + 1)
        
    def _make_block(self, in_dim, hidden_dim, out_dim, kernel_size=3, bn_momentum=0.01):
        return nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, groups=1, bias=True),
            nn.BatchNorm2d(hidden_dim, momentum = bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0),
        )

    def forward(self, cnn_feats, true_shape, upsample = False, low_desc = None, low_certainty = None):  # dict: {"16": f16, "8": f8, "4": f4, "2": f2, "1": f1]
        H1, W1 = true_shape[-2:]
        if upsample:
            scales = [1, 2, 4, 8]
        else:
            scales = [1, 2, 4, 8, 16]
        N_Hs = [H1 // s if s != 16 else H1 // self.patch_size for s in scales]
        N_Ws = [W1 // s if s != 16 else W1 // self.patch_size for s in scales]
        feat_pyramid = {}
        for i, s in enumerate(scales):
            nh, nw = N_Hs[i], N_Ws[i]
            feat = rearrange(cnn_feats[i], 'b (nh nw) c -> b c nh nw', nh=nh, nw=nw)
            # feat_pyramid[s] = feat.permute(0, 3, 1, 2).contiguous()  ## b, c, nh, nw
            feat_pyramid[s] = feat  ##  b, c, nh, nw
            del feat
        
        f16 = self.proj16(feat_pyramid[16]) #b, c, h//16, w//16
        f8 = self.proj8(feat_pyramid[8])
        f4 = self.proj4(feat_pyramid[4])
        f2 = self.proj2(feat_pyramid[2])
        f1 = self.proj1(feat_pyramid[1])

        pred16 = self.pred16(f16) #b, (D+1)*256, h//16, w//16
        pred16 = F.pixel_shuffle(pred16, self.patch_size) #b, D+1, h, w
        desc_16, desc_conf_16 = post_process(pred16, self.desc_mode, self.desc_conf_mode)

        f16_up = F.interpolate(f16, size=(N_Hs[3], N_Ws[3]), mode="bilinear")
        f8_up = self.up8(torch.cat([f8, f16_up], dim=1))
        pred8 = self.pred8(f8_up) #b, (D+1)*64, h//8, w//8
        pred8 = F.pixel_shuffle(pred8, 8) #b, D+1, h, w
        desc_8, desc_conf_8 = post_process(pred8, self.desc_mode, self.desc_conf_mode)

        f8_up = F.interpolate(f8_up, size=(N_Hs[2], N_Ws[2]), mode="bilinear")
        f4_up = self.up4(torch.cat([f4, f8_up], dim=1))
        pred4 = self.pred4(f4_up) #b, (D+1)*16, h//4, w//4
        pred4 = F.pixel_shuffle(pred4, 4) #b, D+1, h, w
        desc_4, desc_conf_4 = post_process(pred4, self.desc_mode, self.desc_conf_mode)

        f4_up = F.interpolate(f4_up, size=(N_Hs[1], N_Ws[1]), mode="bilinear")
        f2_up = self.up2(torch.cat([f2, f4_up], dim=1))
        pred2 = self.pred2(f2_up) #b, (D+1)*4, h//2, w//2
        pred2 = F.pixel_shuffle(pred2, 2) #b, D+1, h, w
        desc_2, desc_conf_2 = post_process(pred2, self.desc_mode, self.desc_conf_mode)

        f2_up = F.interpolate(f4_up, size=(N_Hs[0], N_Ws[0]), mode="bilinear")
        f1_up = self.up1(torch.cat([f1, f2_up], dim=1))
       
        pred1 = self.pred2(f1_up) #b, D+1, h//1, w//1
        desc, desc_conf = post_process(pred1, self.desc_mode, self.desc_conf_mode)

        return {'desc': desc, 'desc_conf': desc_conf, # [B, H, W, D], [B, H, W]
                'desc_16': desc_16, 'desc_conf_16': desc_conf_16,
                'desc_8': desc_8, 'desc_conf_8': desc_conf_8,
                'desc_4': desc_4, 'desc_conf_4': desc_conf_4,
                'desc_2': desc_2, 'desc_conf_2': desc_conf_2,
                }  


class MultiScaleFM(nn.Module):
    def __init__(self, desc_dim, desc_mode, desc_conf_mode, patch_size, detach = False):
        super().__init__()
        self.desc_dim = desc_dim
        self.desc_mode = desc_mode
        self.desc_conf_mode = desc_conf_mode
        self.patch_size = patch_size
        self.detach = detach

        self.proj16 = nn.Sequential(nn.Conv2d(1024+768, 512, 1, 1), nn.BatchNorm2d(512))
        self.proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
        self.proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
        self.proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
        self.proj1 = nn.Sequential(nn.Conv2d(64, 32, 1, 1), nn.BatchNorm2d(32))

        self.init_desc = self._make_block(512, 512, self.desc_dim + 1)

        self.refine16 = self._make_block(512 + self.desc_dim + 1, 512 + self.desc_dim + 1, self.desc_dim + 1)
        self.refine8 = self._make_block(512 + self.desc_dim + 1, 512 + self.desc_dim + 1, self.desc_dim + 1)
        self.refine4 = self._make_block(256 + self.desc_dim + 1, 256 + self.desc_dim + 1, self.desc_dim + 1)
        self.refine2 = self._make_block(64 + self.desc_dim + 1, 64 + self.desc_dim + 1, self.desc_dim + 1)
        self.refine1 = self._make_block(32 + self.desc_dim + 1, 32 + self.desc_dim + 1, self.desc_dim + 1)
        
        
    def _make_block(self, in_dim, hidden_dim, out_dim, bn_momentum=0.01):
        return nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=5, stride=1, padding=2, groups=in_dim, bias=True),
            nn.BatchNorm2d(hidden_dim, momentum = bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2, groups=hidden_dim, bias=True),
            nn.BatchNorm2d(hidden_dim, momentum = bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2, groups=hidden_dim, bias=True),
            nn.BatchNorm2d(hidden_dim, momentum = bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2, groups=hidden_dim, bias=True),
            nn.BatchNorm2d(hidden_dim, momentum = bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),

            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0),
        )

    def forward(self, cnn_feats, true_shape, upsample = False, low_desc = None, low_certainty = None):  # dict: {"16": f16, "8": f8, "4": f4, "2": f2, "1": f1]
        H1, W1 = true_shape[-2:]
        if upsample:
            scales = [1, 2, 4, 8]
        else:
            scales = [1, 2, 4, 8, 16]
        N_Hs = [H1 // s if s != 16 else H1 // self.patch_size for s in scales]
        N_Ws = [W1 // s if s != 16 else W1 // self.patch_size for s in scales]
        feat_pyramid = {}
        for i, s in enumerate(scales):
            nh, nw = N_Hs[i], N_Ws[i]
            feat = rearrange(cnn_feats[i], 'b (nh nw) c -> b c nh nw', nh=nh, nw=nw)
            # feat_pyramid[s] = feat.permute(0, 3, 1, 2).contiguous()  ## b, c, nh, nw
            feat_pyramid[s] = feat  ##  b, c, nh, nw
            del feat
        
        if upsample:
            d = torch.cat([low_desc, low_certainty.unsqueeze(-1)], dim=-1)
            d = d.permute(0, 3, 1, 2)
            desc_16 = desc_conf_16 = None
        else:
            f16 = self.proj16(feat_pyramid[16])
            d = self.init_desc(f16)
            d = self.refine16(torch.cat([d, f16], dim=1)) + d
            desc_16, desc_conf_16 = post_process(d, self.desc_mode, self.desc_conf_mode)
            
        d = F.interpolate(
                d,
                size=(N_Hs[3], N_Ws[3]),
                mode="bilinear",
            ) # [B, D+1, H//8, W//8]
        
        if self.detach:
            d = d.detach()

        d = self.refine8(torch.cat([d, self.proj8(feat_pyramid[8])], dim=1)) + d # [B, D+1, H//8, W//8]
        desc_8, desc_conf_8 = post_process(d, self.desc_mode, self.desc_conf_mode)

        d =  F.interpolate(
                    d,
                    size=(N_Hs[2], N_Ws[2]),
                    mode="bilinear",
                )  # [B, D+1, H//4, W//4]
        if self.detach:
            d = d.detach()
        
        d = self.refine4(torch.cat([d, self.proj4(feat_pyramid[4])], dim=1)) + d # [B, D+1, H//4, W//4]
        desc_4, desc_conf_4 = post_process(d, self.desc_mode, self.desc_conf_mode)

        d =  F.interpolate(
                    d,
                    size=(N_Hs[1], N_Ws[1]),
                    mode="bilinear",
                )  # [B, D+1, H//2, W//2]
        if self.detach:
            d = d.detach()

        d = self.refine2(torch.cat([d, self.proj2(feat_pyramid[2])], dim=1)) + d # [B, D+1, H//2, W//2]
        desc_2, desc_conf_2 = post_process(d, self.desc_mode, self.desc_conf_mode)
        
        d =  F.interpolate(
                    d,
                    size=(N_Hs[0], N_Ws[0]),
                    mode="bilinear",
                )  # [B, D+1, H//1, W//1]
        if self.detach:
            d = d.detach()

        d = self.refine1(torch.cat([d, self.proj1(feat_pyramid[1])], dim=1)) + d # [B, D+1, H, W]
        
        desc, desc_conf = post_process(d, self.desc_mode, self.desc_conf_mode)

        return {'desc': desc, 'desc_conf': desc_conf, # [B, H, W, D], [B, H, W]
                'desc_16': desc_16, 'desc_conf_16': desc_conf_16,
                'desc_8': desc_8, 'desc_conf_8': desc_conf_8,
                'desc_4': desc_4, 'desc_conf_4': desc_conf_4,
                'desc_2': desc_2, 'desc_conf_2': desc_conf_2,
                }  


class MultiScaleFM_MLP(nn.Module):
    def __init__(self, desc_dim, desc_mode, desc_conf_mode, patch_size, hidden_dim_factor = 4, detach=False):
        super().__init__()
        self.desc_dim = desc_dim
        self.desc_mode = desc_mode
        self.desc_conf_mode = desc_conf_mode
        self.patch_size = patch_size
        self.detach = detach
        
        
        self.init_desc = Mlp(in_features=1024 + 768,
                            hidden_features=int(hidden_dim_factor * (1024 + 768)),
                            out_features=(self.desc_dim + 1)*4)
        # self.refine16 = Mlp(in_features=1024 + 768 + self.desc_dim + 1, 
        #                     hidden_features=int(hidden_dim_factor * (1024 + 768 + self.desc_dim + 1)),
        #                     out_features=(self.desc_dim + 1)*4)
        self.refine8 = Mlp(in_features=512 + self.desc_dim + 1,
                            hidden_features=int(hidden_dim_factor * (512 + self.desc_dim + 1)),
                            out_features=(self.desc_dim + 1)*4)
        self.refine4 = Mlp(in_features=256 + self.desc_dim + 1,
                            hidden_features=int(hidden_dim_factor * (256 + self.desc_dim + 1)),
                            out_features=(self.desc_dim + 1)*4)
        self.refine2 = Mlp(in_features=128 + self.desc_dim + 1,
                            hidden_features=int(hidden_dim_factor * (128 + self.desc_dim + 1)),
                            out_features=(self.desc_dim + 1)*4)
        self.refine1 = Mlp(in_features=64 + self.desc_dim + 1,
                            hidden_features=int(hidden_dim_factor * (64 + self.desc_dim + 1)),
                            out_features=(self.desc_dim + 1))

    def forward(self, cnn_feats, true_shape, upsample = False, low_desc = None, low_certainty = None):  # dict: {"16": f16, "8": f8, "4": f4, "2": f2, "1": f1]
        H, W = true_shape
        if upsample:
            scales = [1, 2, 4, 8]
        else:
            scales = [1, 2, 4, 8, 16]
        N_Hs = [H // s if s != 16 else H // self.patch_size for s in scales]
        N_Ws = [W // s if s != 16 else W // self.patch_size for s in scales]

        feat_pyramid = {}
        for i, s in enumerate(scales):
            nh, nw = N_Hs[i], N_Ws[i]
            feat = rearrange(cnn_feats[i], 'b (nh nw) c -> b nh nw c', nh=nh, nw=nw)
            # feat_pyramid[s] = feat.permute(0, 3, 1, 2).contiguous()  ## b, c, nh, nw
            feat_pyramid[s] = feat  ##  b, c, nh, nw
            del feat

        if upsample:
            d = torch.cat([low_desc, low_certainty.unsqueeze(-1)], dim=-1) #B, H//8, W//8, D
            d = F.interpolate(d.permute(0, 3, 1, 2), size = (N_Hs[3], N_Ws[3]), mode='bilinear') #B, H//8, W//8, D
            d = d.permute(0, 2, 3, 1)
            desc_8 = desc_conf_8 = None
            # desc_8, desc_conf_8 = post_process(d, self.desc_mode, self.desc_conf_mode)
        else:
            d = self.init_desc(feat_pyramid[16]) #B, H//16, W//16, D*4
            d = F.pixel_shuffle(d.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)  # B,H//8,W//8, D
            desc_8, desc_conf_8 = post_process(d, self.desc_mode, self.desc_conf_mode, mlp=True)
            # d = torch.cat([desc_8, desc_conf_8.unsqueeze(-1)], dim=-1)
        if self.detach:
            d = d.detach()
        d = self.refine8(torch.cat([d, feat_pyramid[8]], dim=-1)) # B,H//8,W//8, D*4
        d = F.pixel_shuffle(d.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1) # B, H//4, W//4, D
        desc_4, desc_conf_4 = post_process(d, self.desc_mode, self.desc_conf_mode, mlp=True)
        # d = torch.cat([desc_4, desc_conf_4.unsqueeze(-1)], dim=-1)
        if self.detach:
            d = d.detach()

        d = self.refine4(torch.cat([d, feat_pyramid[4]], dim=-1)) # B,H//4,W//4, D*4
        d = F.pixel_shuffle(d.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1) # B, H//2, W//2, D
        desc_2, desc_conf_2 = post_process(d, self.desc_mode, self.desc_conf_mode, mlp=True)
        # d = torch.cat([desc_2, desc_conf_2.unsqueeze(-1)], dim=-1)
        if self.detach:
            d = d.detach()

        d = self.refine2(torch.cat([d, feat_pyramid[2]], dim=-1)) # B,H//2,W//2, D*4
        d = F.pixel_shuffle(d.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1) # B, H//1, W//1, D

        d = self.refine1(torch.cat([d, feat_pyramid[1]], dim=-1)) # B,H//1,W//1, D
        desc_before, desc_conf_before = d[..., :-1], d[..., -1]
        desc, desc_conf = post_process(d, self.desc_mode, self.desc_conf_mode, mlp=True)
        return {'desc': desc, 'desc_conf': desc_conf, # [B, H, W, D], [B, H, W]
                'desc_8': desc_8, 'desc_conf_8': desc_conf_8,
                'desc_4': desc_4, 'desc_conf_4': desc_conf_4,
                'desc_2': desc_2, 'desc_conf_2': desc_conf_2,
                'desc_before': desc_before, 'desc_conf_before': desc_conf_before,
                }  


class MultiScaleFM_MLP_brute(nn.Module):
    def __init__(self, desc_dim, desc_mode, desc_conf_mode, patch_size, hidden_dim_factor = 4, detach=False):
        super().__init__()
        self.desc_dim = desc_dim
        self.desc_mode = desc_mode
        self.desc_conf_mode = desc_conf_mode
        self.patch_size = patch_size
        self.detach = detach
        
        
        self.init_desc = Mlp(in_features=1024 + 768,
                            hidden_features=int(hidden_dim_factor * (1024 + 768)),
                            out_features=(self.desc_dim + 1)* self.patch_size**2)
        self.refine8 = Mlp(in_features=512 + self.desc_dim + 1,
                            hidden_features=int(hidden_dim_factor * (512 + self.desc_dim + 1)),
                            out_features=(self.desc_dim + 1)* 8**2)
        self.refine4 = Mlp(in_features=256 + self.desc_dim + 1,
                            hidden_features=int(hidden_dim_factor * (256 + self.desc_dim + 1)),
                            out_features=(self.desc_dim + 1)* 4**2)
        self.refine2 = Mlp(in_features=128 + self.desc_dim + 1,
                            hidden_features=int(hidden_dim_factor * (128 + self.desc_dim + 1)),
                            out_features=(self.desc_dim + 1)* 2**2)
        self.refine1 = Mlp(in_features=64 + self.desc_dim + 1,
                            hidden_features=int(hidden_dim_factor * (64 + self.desc_dim + 1)),
                            out_features=(self.desc_dim + 1)* 1**2)

    def forward(self, cnn_feats, true_shape, upsample = False, desc = None, certainty = None):  # dict: {"16": f16, "8": f8, "4": f4, "2": f2, "1": f1]
        H, W = true_shape
        if upsample:
            scales = [1, 2, 4, 8]
        else:
            scales = [1, 2, 4, 8, 16]
        N_Hs = [H // s if s != 16 else H // self.patch_size for s in scales]
        N_Ws = [W // s if s != 16 else W // self.patch_size for s in scales]

        feat_pyramid = {}
        for i, s in enumerate(scales):
            nh, nw = N_Hs[i], N_Ws[i]
            feat = rearrange(cnn_feats[i], 'b (nh nw) c -> b nh nw c', nh=nh, nw=nw)
            # feat_pyramid[s] = feat.permute(0, 3, 1, 2).contiguous()  ## b, c, nh, nw
            feat_pyramid[s] = feat  ##  b, c, nh, nw
            del feat

        if upsample:
            d = torch.cat([desc, certainty.unsqueeze(-1)], dim=-1) #B, H//8, W//8, D
            d = F.interpolate(d.permute(0, 3, 1, 2), size = (N_Hs[3], N_Ws[3]), mode='bilinear') #B, H//8, W//8, D
            desc_8 = desc_conf_8 = None
        else:
            d = self.init_desc(feat_pyramid[16]) #B, H//16, W//16, D * 4
            d = F.pixel_shuffle(d.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)  # B,H//8,W//8, D
            desc_8, desc_conf_8 = post_process(d, self.desc_mode, self.desc_conf_mode, mlp=True)
            d = torch.cat([desc_8, desc_conf_8.unsqueeze(-1)], dim=-1)
        if self.detach:
            d = d.detach()

        d = self.refine8(torch.cat([d, feat_pyramid[8]], dim=-1)) # B,H//8,W//8, D*4
        d = F.pixel_shuffle(d.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1) # B, H//4, W//4, D
        desc_4, desc_conf_4 = post_process(d, self.desc_mode, self.desc_conf_mode, mlp=True)
        d = torch.cat([desc_4, desc_conf_4.unsqueeze(-1)], dim=-1)
        
        if self.detach:
            d = d.detach()

        d = self.refine4(torch.cat([d, feat_pyramid[4]], dim=-1)) # B,H//4,W//4, D*4
        d = F.pixel_shuffle(d.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1) # B, H//2, W//2, D
        desc_2, desc_conf_2 = post_process(d, self.desc_mode, self.desc_conf_mode, mlp=True)
        d = torch.cat([desc_2, desc_conf_2.unsqueeze(-1)], dim=-1)

        if self.detach:
            d = d.detach()

        d = self.refine2(torch.cat([d, feat_pyramid[2]], dim=-1)) # B,H//2,W//2, D*4
        d = F.pixel_shuffle(d.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1) # B, H//1, W//1, D

        d = self.refine1(torch.cat([d, feat_pyramid[1]], dim=-1)) # B,H//1,W//1, D
        
        desc, desc_conf = post_process(d, self.desc_mode, self.desc_conf_mode, mlp=True)
        return {'desc': desc, 'desc_conf': desc_conf, # [B, H, W, D], [B, H, W]
                'desc_8': desc_8, 'desc_conf_8': desc_conf_8,
                'desc_4': desc_4, 'desc_conf_4': desc_conf_4,
                'desc_2': desc_2, 'desc_conf_2': desc_conf_2,
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
                        detach=True, 
                        scales=["16", "8", "4", "2", "1"], 
                        displacement_dropout_p = displacement_dropout_p,
                        gm_warp_dropout_p = gm_warp_dropout_p)
        return WarpHead(decoder, patch_size)
    
    elif head_type == 'fm':
        return MultiScaleFM(net.local_feat_dim, net.desc_mode, net.desc_conf_mode, patch_size, net.detach)
    elif head_type == 'fm_mlp':
        return MultiScaleFM_MLP(net.local_feat_dim, net.desc_mode, net.desc_conf_mode, patch_size, detach = net.detach)
    elif head_type == 'fm_conv':
        return FM_conv(net.local_feat_dim, net.desc_mode, net.desc_conf_mode, patch_size, detach = net.detach)
    elif head_type == 'fm_ms_conv':
        return MultiScaleFM_conv(net.local_feat_dim, net.desc_mode, net.desc_conf_mode, patch_size, detach = net.detach)
    elif head_type == 'mlp':
        return FM_MLP(net.local_feat_dim, net.desc_mode, net.desc_conf_mode, patch_size, detach = net.detach)
    else:
        raise NotImplementedError(
            f"unexpected {head_type=}")


