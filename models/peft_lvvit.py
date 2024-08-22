import jittor as jt
from jittor import nn
from .lv_vit.lvvit import LV_ViT

from .peft_modules import *

class LV_ViT_Tuner(nn.Module):
    """ All instance variables in this class will be optimized.
    """
    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()
        
        n_layers = len(vit_model.blocks)
        emb_dim = vit_model.embed_dim
        dtype = vit_model.pos_embed.dtype
        
        use_adaptformer = cfg.adaptformer
        adapter_dim = cfg.adapter_dim
        partial = cfg.partial
        
        if partial is None:
            _start, _end = 0, n_layers
        elif isinstance(partial, int):
            _start, _end = n_layers - partial, n_layers
        elif isinstance(partial, list):
            _start, _end = partial[0], partial[1]
        
        if use_adaptformer:
            adaptformer_list = nn.ModuleList([
                *[None] * (_start),
                *[AdaptFormer(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            adaptformer_list = nn.ModuleList([None] * n_layers)
            
        self.adaptformer_list = adaptformer_list

import jittor as jt
from jittor import nn
from .lv_vit.lvvit import LV_ViT

class Peft_LV_ViT(nn.Module):
    def __init__(self, vit_model: LV_ViT):
        super().__init__()

        self.lvvit = vit_model
        self.out_dim = self.lvvit.embed_dim

        self.forward_embeddings = self.lvvit.forward_embeddings
        self.cls_token = self.lvvit.cls_token
        self.pos_embed = self.lvvit.pos_embed
        self.pos_drop = self.lvvit.pos_drop
        self.blocks = self.lvvit.blocks
        self.norm = self.lvvit.norm

    @property
    def dtype(self):
        return self.lvvit.pos_embed.dtype

    def execute(self, x, tuner=None, head=None):
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose(1, 2)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        n_layers = len(self.blocks)
        for i in range(n_layers):
            block = self.blocks[i]
            if tuner is not None:
                adaptformer = tuner.adaptformer_list[i]
            else:
                adaptformer = None
            x = x + block.drop_path(block.attn(block.norm1(x)))/block.skip_lam

            identity = x
            x = block.mlp(block.norm2(x))
            if adaptformer is not None:
                x = x + adaptformer(identity)
            x = identity + block.drop_path(x)/block.skip_lam
        x = self.norm(x)

        x = x[:, 0]

        if head is None:
            return x
        else:
            return head(x)

