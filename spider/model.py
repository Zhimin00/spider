# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# SPIDER model class
# --------------------------------------------------------

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import spider.utils.path_to_dust3r #noqa
from dust3r.utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from dust3r.patch_embed import get_patch_embed
from dust3r.heads import head_factory as dust3r_head_factory
# from dust3r.model import CroCoNet
import dust3r.utils.path_to_croco  # noqa: F401

from models.croco import CroCoNet  # noqa
from models.blocks import Mlp


import os
import torchvision.models as tvm
from spider.utils.misc import interleave_list, transpose_to_landscape_warp
from spider.heads import head_factory

import pdb

inf = float('inf')

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)

class VGG19_all(nn.Module): #scale 8,4,2,1
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])#40

    def forward(self, x, **kwargs):
        feats = []
        scale = 1
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
                scale = scale*2
            x = layer(x) ## [B, C, H, W]  at scale 1, 2, 4, 8
        return [feat.permute(0, 2, 3, 1).flatten(1, 2).float() for feat in feats] ## [B, H*W, C] at scale 1,2,4,8
    

class SPIDER (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output warp directly
    """

    def __init__(self,
                 head_type='warp',
                 freeze='backbone',
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 vgg_pretrained = True,
                 landscape_only = True,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(head_type, landscape_only, **croco_kwargs)
        self.cnn = VGG19_all(pretrained=vgg_pretrained)
        self.cnn_feature_dims = [64, 128, 256, 512]
        self.set_freeze(freeze)
        

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(SPIDER, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_size = patch_size
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
            'backbone': [self.mask_token, self.enc_norm, self.decoder_embed, self.dec_norm, self.patch_embed.norm, self.patch_embed.proj, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, head_type, landscape_only, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.head_type = head_type
        # allocate heads
        self.downstream_head = head_factory(head_type, self)
        # magic wrapper
        self.head1 = transpose_to_landscape_warp(self.downstream_head, activate=landscape_only)

    def cnn_embed(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape
        ns = [W * H, (W//2) * (H//2), (W//4) * (H//4), (W//8) * (H//8)]
        # allocate result
        feats = [img.new_zeros((B, ns[idx], self.cnn_feature_dims[idx])) for idx in range(len(self.cnn_feature_dims))]
        feat1, feat2, feat4, feat8 = feats
        
        feat1[is_landscape], feat2[is_landscape], feat4[is_landscape], feat8[is_landscape] = self.cnn(img[is_landscape])
        feat1[is_portrait], feat2[is_portrait], feat4[is_portrait], feat8[is_portrait] = self.cnn(img[is_portrait].swapaxes(-1, -2))
        cnn_feats = [feat1, feat2, feat4, feat8]
        return cnn_feats

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        cnn_feats = self.cnn_embed(image, true_shape=true_shape)
        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, cnn_feats

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, cnn_feats = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
            cnn_feats, cnn_feats2 = zip(*[feat.chunk(2, dim=0) for feat in cnn_feats])
            cnn_feats = list(cnn_feats)
            cnn_feats2 = list(cnn_feats2)
        else:
            out, pos, cnn_feats = self._encode_image(img1, true_shape1)
            out2, pos2, cnn_feats = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2, cnn_feats, cnn_feats2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2, cnn_feats1, cnn_feats2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
            cnn_feats1, cnn_feats2 = interleave_list(cnn_feats1, cnn_feats2)
        else:
            feat1, feat2, pos1, pos2, cnn_feats1, cnn_feats2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2), (cnn_feats1, cnn_feats2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, cnn_feats1, cnn_feats2, shape1, shape2):
        B, S, D = cnn_feats1[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(cnn_feats1, cnn_feats2, shape1, shape2)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2), (cnn_feats1, cnn_feats2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        enc_output1, dec_output1 = dec1[0], dec1[-1]
        enc_output2, dec_output2 = dec2[0], dec2[-1]
        feat16_1 = torch.cat([enc_output1, dec_output1], dim=-1)
        feat16_2 = torch.cat([enc_output2, dec_output2], dim=-1)
        cnn_feats1.append(feat16_1)
        cnn_feats2.append(feat16_2)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(enabled=False):
                corresps = self._downstream_head(1, cnn_feats1, cnn_feats2, shape1, shape2)
        return corresps
    
class SPIDER_POINTMAP (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output pointmaps for two images separately
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='backbone',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        dec_dim, enc_dim = self.decoder_embed.weight.shape
        self.pose_embed = Mlp(12, 4*dec_dim, dec_dim)
        self.pose_mlp = Mlp(dec_dim)
        self.mlp = Mlp( in_features=dec_dim, hidden_features=int(dec_dim * 4))
        self.norm = nn.LayerNorm(dec_dim)


        # dust3r specific initialization
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)
        

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(SPIDER, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_size = patch_size
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
            'backbone': [self.mask_token, self.enc_norm, self.decoder_embed, self.dec_norm, self.patch_embed.norm, self.patch_embed.proj, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type1 = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = dust3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = dust3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)

        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, relpose1 = None, relpose2 = None):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2) #B, N, C
            if relpose1 is not None:
                x1 = torch.cat((self.pose_mlp1(relpose1), f1), dim = 1) #B, N+1, C
                f1 = f1 + self.mlp(self.norm(x1))[:, 1:] #B, N, C

            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            if relpose2 is not None:
                x2 = torch.cat((self.pose_mlp2(relpose2), f1), dim = 1)
                f2 = f2 + self.mlp(self.norm(x2))[:, 1:]
                # pdb.set_trace()

            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)
    
    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        relpose1 = view1.get('known_pose')
        relpose2 = view2.get('known_pose')
        if relpose1 is not None:
            # pdb.set_trace()
            pose_emb1 = self.pose_embed(relpose1[:, :3].flatten(1)).unsqueeze(1)
        else:
            pose_emb1 = None
        if relpose2 is not None:
            pose_emb2 = self.pose_embed(relpose2[:, :3].flatten(1)).unsqueeze(1)
        else:
            pose_emb2 = None
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, relpose1 = pose_emb1, relpose2 = pose_emb2)
        
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
