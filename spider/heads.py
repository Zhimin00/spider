import torch
import torch.nn as nn
import torch.nn.functional as F
from spider.roma import TransformerDecoder, Block, MemEffAttention, ConvRefiner, Decoder, CosKernel, GP
from einops import rearrange
import pdb

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
        N_Hs1 = [H1 // 1, H1 // 2, H1 // 4, H1 // 8, H1 // self.patch_size]
        N_Ws1 = [W1 // 1, W1 // 2, W1 // 4, W1 // 8, W1 // self.patch_size]
        for i, s in enumerate([1, 2, 4, 8, 16]):
            nh, nw = N_Hs1[i], N_Ws1[i]
            feat = rearrange(cnn_feats1[i], 'b (nh nw) c -> b nh nw c', nh=nh, nw=nw)
            feat1_pyramid[s] = feat.permute(0, 3, 1, 2).contiguous()  ## b, c, nh, nw
            del feat
        feat2_pyramid = {}
        H2, W2 = true_shape2[-2:]
        N_Hs2 = [H2 // 1, H2 // 2, H2 // 4, H2 // 8, H2 // self.patch_size]
        N_Ws2 = [W2 // 1, W2 // 2, W2 // 4, W2 // 8, W2 // self.patch_size]
        for i, s in enumerate([1, 2, 4, 8, 16]):
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
    if head_type == 'warp':
        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
       

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
    
    else:
        raise NotImplementedError(
            f"unexpected {head_type=}")


