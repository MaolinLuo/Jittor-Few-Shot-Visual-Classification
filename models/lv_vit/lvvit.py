import jittor as jt
from jittor import nn
import numpy as np
from jittor.init import trunc_normal_

from .layers import *  # Ensure that the layers imported here are adapted to Jittor

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'LV_ViT_Tiny': _cfg(),
    'LV_ViT': _cfg(),
    'LV_ViT_Medium': _cfg(crop_pct=1.0),
    'LV_ViT_Large': _cfg(crop_pct=1.0),
}

def get_block(block_type, **kargs):
    if block_type=='mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type=='ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type=='tr':
        # transformer block
        return Block(**kargs)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_dpr(drop_path_rate, depth, drop_path_decay='linear'):
    if drop_path_decay == 'linear':
        # linear dpr decay
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay == 'fix':
        # use fixed dpr
        dpr = [drop_path_rate] * depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate) == depth
        dpr = drop_path_rate
    return dpr

class LV_ViT(nn.Module):
    """ Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', hybrid_backbone=None, norm_layer=nn.LayerNorm, p_emb='4_2', head_dim=None,
                 skip_lam=1.0, order=None, mix_token=False, return_dense=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes == 0 else num_classes
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if p_emb == '4_2':
                patch_embed_fn = PatchEmbed4_2
            elif p_emb == '4_2_128':
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        # Position embedding and dropout
        self.cls_token = nn.Parameter(jt.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(jt.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Building the transformer blocks
        if order is None:
            dpr = get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList([
                Block(dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(depth)])
        else:
            dpr = get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([
                get_block(order[i], dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(len(order))])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() # We don't use this head!!!
        
        self.return_dense = return_dense
        self.mix_token = mix_token
        
        if return_dense:
            self.aux_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        # Weights Initialization
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_embeddings(self,x):
        x = self.patch_embed(x)
        return x
    
    def forward_tokens(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
    def forward_features(self,x):
        # simple forward to obtain feature map (without mixtoken)
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        return x
    
    def execute(self, x):
        x = self.forward_embeddings(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        x_cls = x[:,0]

        return x_cls


def vit(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb=1, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model


def lvvit(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model


def lvvit_t(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=240, depth=12, num_heads=4, mlp_ratio=3.,
        p_emb='4_2',skip_lam=1., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Tiny']
    return model


def lvvit_s(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model


def lvvit_m(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Medium']
    return model


def lvvit_l(pretrained=False, **kwargs):
    order = ['tr']*24 # this will override depth, can also be set as None
    model = LV_ViT(patch_size=16, embed_dim=768,depth=24, num_heads=12, mlp_ratio=3.,
        p_emb='4_2_128',skip_lam=3., return_dense=True,mix_token=True, order=order, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Large']
    return model
