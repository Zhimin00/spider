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
from spider.utils.misc import interleave_list, transpose_to_landscape_warp, transpose_to_landscape_fm, transpose_to_landscape_fmwarp
from spider.heads import head_factory
from spider.blocks import BlockInject, Block_embed
from mast3r.catmlp_dpt_head import mast3r_head_factory
import pdb

inf = float('inf')

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model
    if "ManyAR_PatchEmbed" in args:
        args = args.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    elif "ManyAR_DINOv3" in args:
        args = args.replace("ManyAR_DINOv3", "PatchEmbedDINOv3")
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
        self.landscape_only = landscape_only
        

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
            # 'backbone+desc': [self.mask_token, self.enc_norm, self.decoder_embed, self.dec_norm, self.patch_embed.norm, self.patch_embed.proj, self.enc_blocks, self.dec_blocks, self.dec_blocks2,
            #                   self.downstream_head.head_local_features1, self.downstream_head.head_local_features2],
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
        if self.landscape_only:
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
        else:
            cnn_feats = self.cnn(img)
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
            out2, pos2, cnn_feats2 = self._encode_image(img2, true_shape2)
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

    def _encode_image_upsample(self, image, true_shape):
        cnn_feats = self.cnn_embed(image, true_shape=true_shape)
        return cnn_feats

    def _encode_image_pairs_upsample(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            cnn_feats = self._encode_image_upsample(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            cnn_feats, cnn_feats2 = zip(*[feat.chunk(2, dim=0) for feat in cnn_feats])
            cnn_feats = list(cnn_feats)
            cnn_feats2 = list(cnn_feats2)
        else:
            cnn_feats = self._encode_image_upsample(img1, true_shape1)
            cnn_feats2 = self._encode_image_upsample(img2, true_shape2)
        return cnn_feats, cnn_feats2
    
    def _encode_symmetrized_upsample(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1[::2], img2[::2], shape1[::2], shape2[::2])
            cnn_feats1, cnn_feats2 = interleave_list(cnn_feats1, cnn_feats2)
        else:
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1, img2, shape1, shape2)

        return (shape1, shape2), (cnn_feats1, cnn_feats2)
    
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

    def _downstream_head(self, head_num, cnn_feats1, cnn_feats2, shape1, shape2, upsample=False,finest_corresps=None):
        B, S, D = cnn_feats1[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(cnn_feats1, cnn_feats2, shape1, shape2, upsample=upsample, finest_corresps=finest_corresps)

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
    
    def match(self, view1, view2, finest_corresps=None):
        # encode the two images --> B,S,D
        assert finest_corresps is not None
        (shape1, shape2), (cnn_feats1, cnn_feats2) = self._encode_symmetrized_upsample(view1, view2)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(enabled=False):
                corresps = self._downstream_head(1, cnn_feats1, cnn_feats2, shape1, shape2, upsample=True, finest_corresps=finest_corresps)
        return corresps

class SPIDER_MAST3R (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output warp directly
    """

    def __init__(self,
                 head_type='warp',
                 freeze='backbone',
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 vgg_pretrained = True,
                 landscape_only = True,
                 fm_head_type = 'desc',
                 local_feat_dim = 24,
                 desc_mode=('norm'),
                 desc_conf_mode=('exp', 0, inf),
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.desc_conf_mode = desc_conf_mode
        self.desc_mode = desc_mode
        self.local_feat_dim = local_feat_dim

        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(head_type, fm_head_type, landscape_only, **croco_kwargs)
        self.cnn = VGG19_all(pretrained=vgg_pretrained)
        self.cnn_feature_dims = [64, 128, 256, 512]
        self.set_freeze(freeze)
        self.landscape_only = landscape_only
        

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

    def set_downstream_head(self, head_type, fm_head_type, landscape_only, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.head_type = head_type
        # allocate heads
        self.downstream_head = head_factory(head_type, self)
        # magic wrapper
        self.head1 = transpose_to_landscape_warp(self.downstream_head, activate=landscape_only)
        
        self.head_type = head_type
        # allocate heads
        self.downstream_head1 = head_factory(fm_head_type,  self)
        self.downstream_head2 = head_factory(fm_head_type,  self)
        # magic wrapper
        self.head2 = transpose_to_landscape_fm(self.downstream_head1, activate=landscape_only)
        self.head3 = transpose_to_landscape_fm(self.downstream_head2, activate=landscape_only)


    def cnn_embed(self, img, true_shape):
        B, C, H, W = img.shape
        if self.landscape_only:
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
        else:
            cnn_feats = self.cnn(img)
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
            out2, pos2, cnn_feats2 = self._encode_image(img2, true_shape2)
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

    def _encode_image_upsample(self, image, true_shape):
        cnn_feats = self.cnn_embed(image, true_shape=true_shape)
        return cnn_feats

    def _encode_image_pairs_upsample(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            cnn_feats = self._encode_image_upsample(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            cnn_feats, cnn_feats2 = zip(*[feat.chunk(2, dim=0) for feat in cnn_feats])
            cnn_feats = list(cnn_feats)
            cnn_feats2 = list(cnn_feats2)
        else:
            cnn_feats = self._encode_image_upsample(img1, true_shape1)
            cnn_feats2 = self._encode_image_upsample(img2, true_shape2)
        return cnn_feats, cnn_feats2
    
    def _encode_symmetrized_upsample(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1[::2], img2[::2], shape1[::2], shape2[::2])
            cnn_feats1, cnn_feats2 = interleave_list(cnn_feats1, cnn_feats2)
        else:
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1, img2, shape1, shape2)

        return (shape1, shape2), (cnn_feats1, cnn_feats2)
    
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

    def _downstream_head(self, head_num, cnn_feats1, cnn_feats2, shape1, shape2, upsample=False,finest_corresps=None):
        B, S, D = cnn_feats1[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(cnn_feats1, cnn_feats2, shape1, shape2, upsample=upsample, finest_corresps=finest_corresps)
    
    def _downstream_fmhead(self, head_num, cnn_feats, shape, upsample=False, low_desc = None, low_certainty = None):
        B, S, D = cnn_feats[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(cnn_feats, shape, upsample=upsample, low_desc=low_desc, low_certainty=low_certainty)

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
                res1 = self._downstream_fmhead(2, cnn_feats1, shape1, upsample = False, low_desc = None, low_certainty = None)
                res2 = self._downstream_fmhead(3, cnn_feats2, shape2, upsample = False, low_desc = None, low_certainty = None)
        return corresps, res1, res2
    

class SPIDER_FM (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output warp directly
    """

    def __init__(self,
                 detach=False,
                 head_type='fm',
                 freeze='backbone',
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 vgg_pretrained = True,
                 landscape_only = True,
                 local_feat_dim = 24,
                 desc_mode=('norm'),
                 desc_conf_mode=('exp', 0, inf),
                 **croco_kwargs):
        self.detach = detach
        self.desc_conf_mode = desc_conf_mode
        self.desc_mode = desc_mode
        self.local_feat_dim = local_feat_dim
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(head_type, landscape_only, **croco_kwargs)
        self.cnn = VGG19_all(pretrained=vgg_pretrained)
        self.cnn_feature_dims = [64, 128, 256, 512]
        self.set_freeze(freeze)
        self.landscape_only = landscape_only
        
        

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
        self.downstream_head1 = head_factory(head_type,  self)
        self.downstream_head2 = head_factory(head_type,  self)
        # magic wrapper
        self.head1 = transpose_to_landscape_fm(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape_fm(self.downstream_head2, activate=landscape_only)


    def cnn_embed(self, img, true_shape):
        B, C, H, W = img.shape
        if self.landscape_only:
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
        else:
            cnn_feats = self.cnn(img)
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
            out2, pos2, cnn_feats2 = self._encode_image(img2, true_shape2)
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

    def _encode_image_upsample(self, image, true_shape):
        cnn_feats = self.cnn_embed(image, true_shape=true_shape)
        return cnn_feats

    def _encode_image_pairs_upsample(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            cnn_feats = self._encode_image_upsample(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            cnn_feats, cnn_feats2 = zip(*[feat.chunk(2, dim=0) for feat in cnn_feats])
            cnn_feats = list(cnn_feats)
            cnn_feats2 = list(cnn_feats2)
        else:
            cnn_feats = self._encode_image_upsample(img1, true_shape1)
            cnn_feats2 = self._encode_image_upsample(img2, true_shape2)
        return cnn_feats, cnn_feats2
    
    def _encode_symmetrized_upsample(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1[::2], img2[::2], shape1[::2], shape2[::2])
            cnn_feats1, cnn_feats2 = interleave_list(cnn_feats1, cnn_feats2)
        else:
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1, img2, shape1, shape2)

        return (shape1, shape2), (cnn_feats1, cnn_feats2)
    
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

    def _downstream_head(self, head_num, cnn_feats, shape, upsample=False, low_desc = None, low_certainty = None):
        B, S, D = cnn_feats[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(cnn_feats, shape, upsample=upsample, low_desc=low_desc, low_certainty=low_certainty)

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
                res1 = self._downstream_head(1, cnn_feats1, shape1, upsample = False, low_desc = None, low_certainty = None)
                res2 = self._downstream_head(2, cnn_feats2, shape2, upsample = False, low_desc = None, low_certainty = None)

        return res1, res2

class SPIDER_FMwarp (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output warp directly
    """
    def __init__(self,
                 detach=True,
                 head_type='descwarp',
                 freeze='backbone+desc',
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 landscape_only = True,
                 local_feat_dim = 24,
                 desc_mode=('norm'),
                 desc_conf_mode=('exp', 0, inf),
                 **croco_kwargs):
        self.detach = detach
        self.desc_conf_mode = desc_conf_mode
        self.desc_mode = desc_mode
        self.local_feat_dim = local_feat_dim
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(head_type, landscape_only, **croco_kwargs)
        self.set_freeze(freeze)
        self.landscape_only = landscape_only
        
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
            'backbone+desc': [self.mask_token, self.enc_norm, self.decoder_embed, self.dec_norm, self.patch_embed.norm, self.patch_embed.proj, self.enc_blocks, self.dec_blocks, self.dec_blocks2,
                              self.downstream_head.head_local_features1, self.downstream_head.head_local_features2],
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
        self.head1 = transpose_to_landscape_fmwarp(self.downstream_head, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos= self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos= self._encode_image(img1, true_shape1)
            out2, pos2= self._encode_image(img2, true_shape2)
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
            feat1, feat2, pos1, pos2= self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)
    
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

    def _downstream_head(self, head_num, feat16_1, feat16_2, shape1, shape2):
        B, S, D = feat16_1.shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(feat16_1, feat16_2, shape1, shape2)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2)= self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        enc_output1, dec_output1 = dec1[0], dec1[-1]
        enc_output2, dec_output2 = dec2[0], dec2[-1]
        feat16_1 = torch.cat([enc_output1, dec_output1], dim=-1)
        feat16_2 = torch.cat([enc_output2, dec_output2], dim=-1)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(enabled=False):
                corresps = self._downstream_head(1, feat16_1, feat16_2, shape1, shape2)

        return corresps
    
    
class RelPoseEmbedGenerator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = Mlp(12, 4*embed_dim, embed_dim)

    def forward(self, relpose):
        # relpose: [B, 6]
        return self.fc(relpose) #[B, embed_dim]

class PoseFiLM_var(nn.Module):
    def __init__(self, pose_dim, hidden, channel_list):
        super().__init__()
        self.mlp = Mlp(pose_dim, 4*hidden, hidden)

        self.gammas = nn.ModuleList()
        self.betas  = nn.ModuleList()
        for c in channel_list:
            self.gammas.append(nn.Linear(hidden, c))
            self.betas.append(nn.Linear(hidden, c))
        
        for g, b in zip(self.gammas, self.betas):
            nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)
            nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)

    def forward(self, p):                     # p: (B, 12)
        h = self.mlp(p)                      # (B, hidden)
        gamma_list, beta_list = [], []
        for g_layer, b_layer in zip(self.gammas, self.betas):
            gamma_list.append(g_layer(h))    # (B, c_k)
            beta_list.append(b_layer(h))     # (B, c_k)
        return gamma_list, beta_list

class PoseTokenAttn(nn.Module):
    def __init__(self, pose_dim, channel_list,
                 n_head=8, n_layer=1):
        super().__init__()
        self.pose_embed = nn.ModuleList([
            Mlp(pose_dim, C*4, C) for C in channel_list
        ])

        self.attns1 = nn.ModuleList()
        for C in channel_list:
            block = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=C,
                                           nhead=n_head,
                                           batch_first=True),
                num_layers=n_layer
            )
            self.attns1.append(block)

        self.attns2 = nn.ModuleList()
        for C in channel_list:
            block = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=C,
                                           nhead=n_head,
                                           batch_first=True),
                num_layers=n_layer
            )
            self.attns2.append(block)

    def forward(self, dec_output, relpose1, relpose2):
        outputs_with_relpose = []
        for (f1, f2), proj_pose, block1, block2 in zip(dec_output, self.pose_embed, self.attns1, self.attns2):
            t1 = proj_pose(relpose1[:, :3].flatten(1)).unsqueeze(1)
            f1 = torch.cat([t1, f1], dim=1)
            f1 = block1(f1)

            t2 = proj_pose(relpose2[:, :3].flatten(1)).unsqueeze(1)
            f2 = torch.cat([t2, f2], dim=1)
            f2 = block2(f2)
            outputs_with_relpose.append((f1[:, 1:], f2[:, 1:]))
        return zip (*outputs_with_relpose)




class SPIDER_POINTMAP (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output pointmaps for two images separately
    """

    def __init__(self,
                #  attn_mode='embed',
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
        # print(self.enc_embed_dim)
        # print(self.dec_embed_dim)
        
        # channel_list = [self.enc_embed_dim] + [self.dec_embed_dim] * self.dec_depth      # e.g. [1024, 768, 768, ...]
        ### Attention
        # self.attn = PoseTokenAttn(12, channel_list)
        # self.attn_mode = attn_mode
        self.pose_embed_enc = Mlp(12, 4*self.enc_embed_dim, self.enc_embed_dim)
        self.pose_embed_dec = Mlp(12, 4*self.dec_embed_dim, self.dec_embed_dim)
        self.attn_enc1 = Block_embed(self.enc_embed_dim)
        self.attn_enc2 = deepcopy(self.attn_enc1)
        self.attns1 = nn.ModuleList([Block_embed(self.dec_embed_dim) for i in range(self.dec_depth)])
        self.attns2 = deepcopy(self.attns1)
        # self.dec_cls = ('_cls' in self.attn_mode)
        # self.dec_num_cls = 0
        # if self.dec_cls:
        #     self.cls_token1 = nn.Parameter(torch.zeros((self.dec_embed_dim,)))
        #     self.cls_token2 = nn.Parameter(torch.zeros((self.dec_embed_dim,)))
        #     self.dec_num_cls = 1
        # self.attns1 = nn.ModuleList([Block_embed(self.dec_embed_dim) for i in range(self.dec_depth)])
        # self.attns2 = deepcopy(self.attns1)
        # self.init_attn_blocks()
        # DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
        ### FiLM
        # self.pose_film1 = PoseFiLM_var(12, self.enc_embed_dim, channel_list)
        # self.pose_film2 = PoseFiLM_var(12, self.enc_embed_dim, channel_list)

        ### add embed
        # self.relpose_embed_generators = nn.ModuleList([RelPoseEmbedGenerator(embed_dim=self.enc_embed_dim)] +
            # [RelPoseEmbedGenerator(embed_dim=self.dec_embed_dim) for _ in range(self.dec_depth)])
        # self.relpose_embed_generators2 = nn.ModuleList([RelPoseEmbedGenerator(embed_dim=self.enc_embed_dim)] +
        #     [RelPoseEmbedGenerator(embed_dim=self.dec_embed_dim) for _ in range(self.dec_depth)])

        # dec_dim, enc_dim = self.decoder_embed.weight.shape
        # self.attn_enc_block = Block(enc_dim, 12, mlp_ratio=4, qkv_bias=True)
        # dec_depth = len(self.dec_blocks)
        # self.attn_dec_blocks = nn.ModuleList([
        #     Block(dec_dim, 12, mlp_ratio=4, qkv_bias=True)
        #     for i in range(dec_depth)])
        
        # self.pose_embed_enc = Mlp(12, 4*enc_dim, enc_dim)
        # self.pose_embed_dec = Mlp(12, 4*dec_dim, dec_dim)
        # self.attn1 = nn.MultiheadAttention(embed_dim=dec_dim, num_heads=8, batch_first=True)
        # self.norm1 = nn.LayerNorm(dec_dim)
        # self.norm_final1 = nn.LayerNorm(dec_dim)

        # self.attn2 = nn.MultiheadAttention(embed_dim=dec_dim, num_heads=8, batch_first=True)
        # self.norm2 = nn.LayerNorm(dec_dim)
        # self.norm_final2 = nn.LayerNorm(dec_dim)

        # dust3r specific initialization
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)
        
    def init_attn_blocks(self):
        for i in range(len(self.attns1)):
            for blocks in [self.attns1, self.attns2]:
                block = blocks[i]
                block.init(self.dec_embed_dim)

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

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2) #B, N, C
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return final_output

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def _decoder_with_relpose(self, decoder_output, relpose1=None, relpose2=None):
        # relpose_embeds1 = [generator(relpose1[:, :3].flatten(1)).unsqueeze(1) for generator in self.relpose_embed_generators]
        # relpose_embeds2 = [generator(relpose2[:, :3].flatten(1)).unsqueeze(1) for generator in self.relpose_embed_generators]

        # outputs_with_relpose = []
        # for i, (f1, f2) in enumerate(decoder_output):
        #     f1 = f1 + relpose_embeds1[i]
        #     f2 = f2 + relpose_embeds2[i]
        #     outputs_with_relpose.append((f1, f2))
        # return zip(*outputs_with_relpose)

        # gamma_list1, beta_list1 = self.pose_film1(relpose1[:, :3].flatten(1))     # 13 x (B, c_k)
        # gamma_list2, beta_list2 = self.pose_film2(relpose2[:, :3].flatten(1))
        # outputs_with_relpose = []
        # for i, (f1, f2) in enumerate(decoder_output):            # feat: (B, N, c_k)
        #     g1 = gamma_list1[i].unsqueeze(1)              # (B,1,c_k)
        #     b1 = beta_list1[i].unsqueeze(1)
        #     g2 = gamma_list2[i].unsqueeze(1)              # (B,1,c_k)
        #     b2 = beta_list2[i].unsqueeze(1)
        #     f1 = f1 * (1 + g1) + b1
        #     f2 = f2 * (1 + g2) + b2
        #     outputs_with_relpose.append((f1, f2))
        # return zip (*outputs_with_relpose)

        # outputs_with_relpose = []
        # f1, f2 = decoder_output[0]
        # outputs_with_relpose.append((f1, f2))
        # cls1 = self.cls_token1[None,None].expand(len(f1),1,-1).clone()
        # cls2 = self.cls_token2[None,None].expand(len(f2),1,-1).clone()
        # pose_emb1 = self.pose_embed(relpose1[:,:3].flatten(1)).unsqueeze(1)
        # pose_emb2 = self.pose_embed(relpose2[:,:3].flatten(1)).unsqueeze(1)
        # cls1 = cls1 + pose_emb1
        # cls2 = cls2 + pose_emb2
        # new_pos1 = torch.cat((-pos1.new_ones(len(cls1), 1, 2), pos1), dim=1)
        # new_pos2 = torch.cat((-pos2.new_ones(len(cls2), 1, 2), pos2), dim=1)
        # for i, (f_dec, blk1, blk2) in enumerate(zip(decoder_output[1:], self.attns1, self.attns2)):
        #     f1, f2 = f_dec
        #     f1 = torch.cat((cls1, f1), dim=1)
        #     f2 = torch.cat((cls2, f2), dim=1)
        #     f1, _ = blk1(f1, f2, new_pos1, new_pos2, relpose=pose_emb1, num_cls=self.dec_num_cls)
        #     f2, _ = blk2(f2, f1, new_pos2, new_pos1, relpose=pose_emb2, num_cls=self.dec_num_cls)
        #     outputs_with_relpose.append((f1[:,1:], f2[:,1:]))
        # return zip(*outputs_with_relpose)

        # outputs_with_relpose = []
        # if relpose1 is not None:
        #     pose_emb1 = self.pose_embed(relpose1[:,:3].flatten(1)).unsqueeze(1)
        # else:
        #     pose_emb1 = None
        # if relpose2 is not None:
        #     pose_emb2 = self.pose_embed(relpose2[:,:3].flatten(1)).unsqueeze(1)
        # else:
        #     pose_emb2 = None

        # f1, f2 = decoder_output[0]
        # outputs_with_relpose.append((f1, f2))
        # for i, (f_dec, blk1, blk2) in enumerate(zip(decoder_output[1:], self.attns1, self.attns2)):
        #     f1, f2 = f_dec
        #     if pose_emb1 is not None:
        #         f1 = f1 + pose_emb1
        #     if pose_emb2 is not None:
        #         f2 = f2 + pose_emb2
        #     f1 = blk1(f1)
        #     f2 = blk2(f2)
        #     outputs_with_relpose.append((f1, f2))
        # return zip(*outputs_with_relpose)
    
        outputs_with_relpose = []
        if relpose1 is not None:
            pose_emb_enc1 = self.pose_embed_enc(relpose1[:,:3].flatten(1)).unsqueeze(1)
            pose_emb_dec1 = self.pose_embed_dec(relpose1[:,:3].flatten(1)).unsqueeze(1)
        else:
            pose_emb_enc1 = None
            pose_emb_dec1 = None
        if relpose2 is not None:
            pose_emb_enc2 = self.pose_embed_enc(relpose2[:,:3].flatten(1)).unsqueeze(1)
            pose_emb_dec2 = self.pose_embed_dec(relpose2[:,:3].flatten(1)).unsqueeze(1)
        else:
            pose_emb_enc2 = None
            pose_emb_dec2 = None
        f1, f2 = decoder_output[0]
        if pose_emb_enc1 is not None:
            f1 = f1 + pose_emb_enc1
        if pose_emb_enc2 is not None:
            f2 = f2 + pose_emb_enc2
        f1 = self.attn_enc1(f1)
        f2 = self.attn_enc2(f2)
        outputs_with_relpose.append((f1, f2))
        for i, (f_dec, blk1, blk2) in enumerate(zip(decoder_output[1:], self.attns1, self.attns2)):
            f1, f2 = f_dec
            if pose_emb_dec1 is not None:
                f1 = f1 + pose_emb_dec1
            if pose_emb_dec2 is not None:
                f2 = f2 + pose_emb_dec2
            f1 = blk1(f1)
            f2 = blk2(f2)
            outputs_with_relpose.append((f1, f2))
        return zip(*outputs_with_relpose)
        
    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
        
        relpose1 = view1.get('known_pose')
        relpose2 = view2.get('known_pose')
        # pdb.set_trace()
        decoder_output = self._decoder(feat1, pos1, feat2, pos2)
        dec1, dec2 = self._decoder_with_relpose(decoder_output, relpose1, relpose2)
        
        #     # dec1, dec2 = self.attn(decoder_output, relpose1, relpose2)
        #     print('adding relpose')
            
        # else:
        #     final_output = self._decoder(feat1, pos1, feat2, pos2)
        #     dec1, dec2 = zip(*final_output)
        
        # f1 = dec1[-1]
        # f2 = dec1[-1]
        # # combine all ref images into object-centric representation
        # relpose1 = view1.get('known_pose')
        # relpose2 = view2.get('known_pose')
        # if relpose1 is not None and relpose2 is not None:
        #     # pdb.set_trace()
        #     # img1 side
        #     cls1 = self.cls_token1[None, None].expand(len(f1),1,-1).clone() #C -> B, 1, C
        #     pose_emb1 = cls1 + self.pose_embed(relpose1[:, :3].flatten(1)).unsqueeze(1) #B, 12 -> B, 1, C
        #     x1 = torch.cat((pose_emb1, f1), dim = 1) #B, N+1, C
        #     x1 = self.norm1(x1)
        #     attn_out1, _ = self.attn1(x1, x1, x1)
        #     x1 = self.norm_final1(x1 + attn_out1)
        #     f1 = x1[:, 1:]
        #     dec1 += (f1,)
        #     # img2 side
        #     cls2 = self.cls_token2[None, None].expand(len(f2),1,-1).clone() #C -> B, 1, C
        #     pose_emb2 = cls2 + self.pose_embed(relpose2[:, :3].flatten(1)).unsqueeze(1) #B, 12 -> B, 1, C
        #     x2 = torch.cat((pose_emb2, f2), dim = 1) #B, N+1, C
        #     x2 = self.norm2(x2)
        #     attn_out2, _ = self.attn2(x2, x2, x2)
        #     x2 = self.norm_final2(x2 + attn_out2)
        #     f2 = x2[:, 1:]
        #     dec2 += (f2,)
        #     # pdb.set_trace()
        #     print('adding relpose')
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2


class SPIDER_twoheads (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output warp directly
    """

    def __init__(self,
                 detach=False,
                 head_type1='warp',
                 head_type2='fm',
                 freeze='backbone',
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 vgg_pretrained = True,
                 landscape_only = True,
                 local_feat_dim = 24,
                 desc_mode=('norm'),
                 desc_conf_mode=('exp', 0, inf),
                 **croco_kwargs):
        self.detach = detach
        self.desc_conf_mode = desc_conf_mode
        self.desc_mode = desc_mode
        self.local_feat_dim = local_feat_dim
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(head_type1, head_type2, landscape_only, **croco_kwargs)
        self.cnn = VGG19_all(pretrained=vgg_pretrained)
        self.cnn_feature_dims = [64, 128, 256, 512]
        self.set_freeze(freeze)
        self.landscape_only = landscape_only
        
        

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
            'backbone+fm': [self.mask_token, self.enc_norm, self.decoder_embed, self.dec_norm, self.patch_embed.norm, self.patch_embed.proj, self.enc_blocks, self.dec_blocks, self.dec_blocks2, self.downstream_head1.init_desc, self.downstream_head2.init_desc],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, head_type1, head_type2, landscape_only, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.head_type1 = head_type1
        self.head_type2 = head_type2
        # allocate heads
        self.downstream_headwarp = head_factory(head_type1, self)
        # magic wrapper
        self.headwarp = transpose_to_landscape_warp(self.downstream_headwarp, activate=landscape_only)
        self.downstream_head1 = head_factory(head_type2,  self)
        self.downstream_head2 = head_factory(head_type2,  self)
        # magic wrapper
        self.head1 = transpose_to_landscape_fm(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape_fm(self.downstream_head2, activate=landscape_only)
        
    def cnn_embed(self, img, true_shape):
        B, C, H, W = img.shape
        if self.landscape_only:
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
        else:
            cnn_feats = self.cnn(img)
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
            out2, pos2, cnn_feats2 = self._encode_image(img2, true_shape2)
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

    def _encode_image_upsample(self, image, true_shape):
        cnn_feats = self.cnn_embed(image, true_shape=true_shape)
        return cnn_feats

    def _encode_image_pairs_upsample(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            cnn_feats = self._encode_image_upsample(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            cnn_feats, cnn_feats2 = zip(*[feat.chunk(2, dim=0) for feat in cnn_feats])
            cnn_feats = list(cnn_feats)
            cnn_feats2 = list(cnn_feats2)
        else:
            cnn_feats = self._encode_image_upsample(img1, true_shape1)
            cnn_feats2 = self._encode_image_upsample(img2, true_shape2)
        return cnn_feats, cnn_feats2
    
    def _encode_symmetrized_upsample(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1[::2], img2[::2], shape1[::2], shape2[::2])
            cnn_feats1, cnn_feats2 = interleave_list(cnn_feats1, cnn_feats2)
        else:
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1, img2, shape1, shape2)

        return (shape1, shape2), (cnn_feats1, cnn_feats2)
    
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

    def _downstream_head(self, head_num, cnn_feats, shape, upsample=False, low_desc = None, low_certainty = None):
        B, S, D = cnn_feats[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(cnn_feats, shape, upsample=upsample, low_desc=low_desc, low_certainty=low_certainty)
    def _downstream_headwarp(self, head_num, cnn_feats1, cnn_feats2, shape1, shape2, upsample=False,finest_corresps=None):
        B, S, D = cnn_feats1[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(cnn_feats1, cnn_feats2, shape1, shape2, upsample=upsample, finest_corresps=finest_corresps)
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
                corresps = self._downstream_headwarp('warp', cnn_feats1, cnn_feats2, shape1, shape2)
                res1 = self._downstream_head(1, cnn_feats1, shape1, upsample = False, low_desc = None, low_certainty = None)
                res2 = self._downstream_head(2, cnn_feats2, shape2, upsample = False, low_desc = None, low_certainty = None)
        
        return corresps, res1, res2


class DINOv3_SPIDER (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output warp directly
    """

    def __init__(self,desc_mode=('norm'), 
                 two_confs=False, 
                 desc_conf_mode=None,
                 output_mode='pts3d',
                 head_type1='warp',
                 head_type2='cat',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='ManyAR_DINOv3',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 vgg_pretrained = True,
                 **croco_kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.enc_blocks=None
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type1, head_type2, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.cnn = VGG19_all(pretrained=vgg_pretrained)
        self.cnn_feature_dims = [64, 128, 256, 512]
        self.set_freeze(freeze)
        self.landscape_only = landscape_only
        
        

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
            'encoder': [self.patch_embed],
            'backbone': [self.patch_embed, self.enc_norm, self.decoder_embed, self.dec_norm, self.dec_blocks, self.dec_blocks2],
            }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type1, head_type2, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type1 = head_type1
        self.head_type2 = head_type2
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head = head_factory(head_type1, self)
        self.downstream_head1 = mast3r_head_factory(head_type2, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type2, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.headwarp = transpose_to_landscape_warp(self.downstream_head, activate=landscape_only)
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        
    def cnn_embed(self, img, true_shape):
        B, C, H, W = img.shape
        if self.landscape_only:
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
        else:
            cnn_feats = self.cnn(img)
            return cnn_feats

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        cnn_feats = self.cnn_embed(image, true_shape=true_shape)
        # add positional embedding without cls token
        assert self.enc_pos_embed is None

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
            out2, pos2, cnn_feats2 = self._encode_image(img2, true_shape2)
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

    def _encode_image_upsample(self, image, true_shape):
        cnn_feats = self.cnn_embed(image, true_shape=true_shape)
        return cnn_feats

    def _encode_image_pairs_upsample(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            cnn_feats = self._encode_image_upsample(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            cnn_feats, cnn_feats2 = zip(*[feat.chunk(2, dim=0) for feat in cnn_feats])
            cnn_feats = list(cnn_feats)
            cnn_feats2 = list(cnn_feats2)
        else:
            cnn_feats = self._encode_image_upsample(img1, true_shape1)
            cnn_feats2 = self._encode_image_upsample(img2, true_shape2)
        return cnn_feats, cnn_feats2
    
    def _encode_symmetrized_upsample(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1[::2], img2[::2], shape1[::2], shape2[::2])
            cnn_feats1, cnn_feats2 = interleave_list(cnn_feats1, cnn_feats2)
        else:
            cnn_feats1, cnn_feats2 = self._encode_image_pairs_upsample(img1, img2, shape1, shape2)

        return (shape1, shape2), (cnn_feats1, cnn_feats2)
    
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

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def _downstream_headwarp(self, head_num, cnn_feats1, cnn_feats2, shape1, shape2, upsample=False,finest_corresps=None):
        B, S, D = cnn_feats1[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(cnn_feats1, cnn_feats2, shape1, shape2, upsample=upsample, finest_corresps=finest_corresps)
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
                corresps = self._downstream_headwarp('warp', cnn_feats1, cnn_feats2, shape1, shape2)
                res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
                res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
        res2['pts3d_in_other_view'] = res2.pop('pts3d')
        return corresps, res1, res2