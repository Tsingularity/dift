import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import gc
from PIL import Image, ImageDraw
from torchvision.transforms import PILToTensor
import os
import open_clip
from torchvision import transforms
import copy
import json
from tqdm.notebook import tqdm
import time
import datetime
import math


def interpolate_pos_encoding(clip_model, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
    resolution images.
    Source:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
    """

    num_patches = embeddings.shape[1] - 1
    pos_embedding = clip_model.positional_embedding.unsqueeze(0)
    num_positions = pos_embedding.shape[1] - 1
    if num_patches == num_positions and height == width:
        return clip_model.positional_embedding
    class_pos_embed = pos_embedding[:, 0]
    patch_pos_embed = pos_embedding[:, 1:]
    dim = embeddings.shape[-1]
    h0 = height // clip_model.patch_size[0]
    w0 = width // clip_model.patch_size[1]
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    output = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return output


class CLIPFeaturizer:
    def __init__(self):
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        visual_model = clip_model.visual
        visual_model.output_tokens = True
        self.clip_model = visual_model.eval().cuda()


    @torch.no_grad()
    def forward(self,
                x, # single image, [1,c,h,w]
                block_index):
        batch_size = 1
        clip_model = self.clip_model
        if clip_model.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], clip_model.grid_size[0], clip_model.patch_size[0], clip_model.grid_size[1], clip_model.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], clip_model.grid_size[0] * clip_model.grid_size[1], -1)
            x = clip_model.patchnorm_pre_ln(x)
            x = clip_model.conv1(x)
        else:
            x = clip_model.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat(
            [clip_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if(x.shape[1] > clip_model.positional_embedding.shape[0]):
            dim = int(math.sqrt(x.shape[1]) * clip_model.patch_size[0])
            x = x + interpolate_pos_encoding(clip_model, x, dim, dim).to(x.dtype)
        else:
            x = x + clip_model.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = clip_model.patch_dropout(x)
        x = clip_model.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        num_channel = x.size(2)
        ft_size = int((x.shape[0]-1) ** 0.5)

        for i, r in enumerate(clip_model.transformer.resblocks):
            x = r(x)

            if i == block_index:
                tokens = x.permute(1, 0, 2)  # LND -> NLD
                tokens = tokens[:, 1:]
                tokens = tokens.transpose(1, 2).contiguous().view(batch_size, num_channel, ft_size, ft_size) # NCHW

                return tokens