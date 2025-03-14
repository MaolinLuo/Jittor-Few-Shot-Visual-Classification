from jittor import nn
import math

from .peft_modules import *


class ViT_Tuner(nn.Module):
    """ All instance variables in this class will be optimized.
    """
    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()

        n_layers = len(vit_model.transformer.resblocks)
        emb_dim = vit_model.positional_embedding.shape[1]
        seq_len = vit_model.positional_embedding.shape[0]
        patch_size = vit_model.conv1.kernel_size
        dtype = vit_model.conv1.weight.dtype

        blocks = vit_model.transformer.resblocks
        attn_in_dim = blocks[0].attn.in_proj_bias.shape[0]
        attn_out_dim = blocks[0].attn.out_proj.bias.shape[0]
        mlp_in_dim = blocks[0].mlp.c_fc.bias.shape[0]
        mlp_out_dim = blocks[0].mlp.c_proj.bias.shape[0]

        use_full_tuning = cfg.full_tuning
        use_bias_tuning = cfg.bias_tuning
        use_ln_tuning = cfg.ln_tuning
        use_vpt_shallow = cfg.vpt_shallow
        use_vpt_deep = cfg.vpt_deep
        use_adapter = cfg.adapter
        use_adaptformer = cfg.adaptformer
        use_lora = cfg.lora
        use_ssf_attn = cfg.ssf_attn
        use_ssf_mlp = cfg.ssf_mlp
        use_ssf_ln = cfg.ssf_ln
        partial = cfg.partial
        vpt_len = cfg.vpt_len
        adapter_dim = cfg.adapter_dim

        if partial is None:
            partial = n_layers
        
        if (use_vpt_shallow or use_vpt_deep) and (vpt_len is None):
            vpt_len = 10
            print("Visual prompt length set to {}".format(vpt_len))
        
        if (use_adapter or use_adaptformer or use_lora) and (adapter_dim is None):
            adapter_dim = 2 ** max(0, int(math.log2(num_classes / (n_layers * 2))))
            # adapter_dim = max(1, num_classes // (n_layers * 2))
            print("Adapter bottle dimension set to {}".format(adapter_dim))

        if use_full_tuning:
            block_tuned = blocks[n_layers - partial: n_layers]
        else:
            block_tuned = None

        if use_bias_tuning:
            bias_tuned = nn.ParameterList([
                param for name, param in blocks.named_parameters()
                if name.endswith("bias")
            ])
        else:
            bias_tuned = None
        
        if use_ln_tuning:
            ln_tuned = nn.ModuleList([
                mod for name, mod in blocks.named_modules()
                if isinstance(mod, nn.LayerNorm)
            ])
        else:
            ln_tuned = None

        assert int(use_vpt_shallow) + int(use_vpt_deep) < 2
        if use_vpt_shallow:
            vpt_list = nn.ModuleList([
                VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype),
                *[nn.Identity()] * (n_layers - 1)
            ])
        elif use_vpt_deep:
            vpt_list = nn.ModuleList([
                *[nn.Identity()] * (n_layers - partial),
                *[VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype) for _ in range(partial)]
            ])
        else:
            vpt_list = nn.ModuleList([nn.Identity()] * n_layers)
        
        if use_adapter:
            adapter_list = nn.ModuleList([
                *[nn.Identity()] * (n_layers - partial),
                *[Adapter(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype) for _ in range(partial)]
            ])
        else:
            adapter_list = nn.ModuleList([nn.Identity()] * n_layers)

        if use_adaptformer:
            adaptformer_list = nn.ModuleList([
                *[nn.Identity()] * (n_layers - partial),
                *[AdaptFormer(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype) for _ in range(partial)]
            ])
        else:
            adaptformer_list = nn.ModuleList([nn.Identity()] * n_layers)

        if use_lora:
            lora_list = nn.ModuleList([
                *[nn.Identity()] * (n_layers - partial),
                *[nn.ModuleDict({
                    "q": LoRA(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype),
                    "v": LoRA(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            lora_list = nn.ModuleList([nn.Identity()] * n_layers)

        if use_ssf_attn:
            ssf_attn_list = nn.ModuleList([
                *[nn.Identity()] * (n_layers - partial),
                *[nn.ModuleDict({
                    "attn_in": SSF(attn_in_dim, dtype=dtype),
                    "attn_out": SSF(attn_out_dim, dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            ssf_attn_list = nn.ModuleList([nn.Identity()] * n_layers)

        if use_ssf_mlp:
            ssf_mlp_list = nn.ModuleList([
                *[nn.Identity()] * (n_layers - partial),
                *[nn.ModuleDict({
                    "mlp_in": SSF(mlp_in_dim, dtype=dtype),
                    "mlp_out": SSF(mlp_out_dim, dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            ssf_mlp_list = nn.ModuleList([nn.Identity()] * n_layers)
        
        if use_ssf_ln:
            ssf_ln_list = nn.ModuleList([
                *[nn.Identity()] * (n_layers - partial),
                *[nn.ModuleDict({
                    "ln_1": SSF(emb_dim, dtype=dtype),
                    "ln_2": SSF(emb_dim, dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            ssf_ln_list = nn.ModuleList([nn.Identity()] * n_layers)
        
        # To be optimized
        self.block_tuned = block_tuned
        self.bias_tuned = bias_tuned
        self.ln_tuned = ln_tuned
        self.vpt_list = vpt_list
        self.adapter_list = adapter_list
        self.adaptformer_list = adaptformer_list
        self.lora_list = lora_list
        self.ssf_attn_list = ssf_attn_list
        self.ssf_mlp_list = ssf_mlp_list
        self.ssf_ln_list = ssf_ln_list


class Peft_ViT(nn.Module):
    def __init__(self, vit_model):
        super().__init__()

        self.patch_embedding = vit_model.conv1
        self.class_embedding = vit_model.class_embedding
        self.positional_embedding = vit_model.positional_embedding
        self.ln_pre = vit_model.ln_pre
        self.blocks = vit_model.transformer.resblocks
        self.ln_post = vit_model.ln_post
        self.proj = vit_model.proj  # not used
        self.out_dim = self.ln_post.bias.shape[0]
        # self.out_dim = self.proj.shape[1]

    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    def execute(self, x, tuner=None, head=None):
        x = x.to(self.dtype)
        x = self.patch_embedding(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        class_embedding_expanded = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, self.class_embedding.shape[0])
        x = jt.concat([class_embedding_expanded, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]

        n_layers = len(self.blocks)

        for i in range(n_layers):
            block = self.blocks[i]

            if tuner is not None:
                vpt = tuner.vpt_list[i]
                adapter = tuner.adapter_list[i]
                adaptformer = tuner.adaptformer_list[i]
                lora = tuner.lora_list[i]
                ssf_attn = tuner.ssf_attn_list[i]
                ssf_mlp = tuner.ssf_mlp_list[i]
                ssf_ln = tuner.ssf_ln_list[i]
            else:
                vpt = adapter = adaptformer = lora = ssf_attn = ssf_mlp = ssf_ln = None

            if not isinstance(vpt, nn.Identity):
                x = vpt(x)

            _seq_len_after_vpt = x.shape[1]

            x = x.permute(1, 0, 2)  # NLD -> LND

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2

            _attn_in_proj_weight = _attn.in_proj_weight
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj = _mlp.c_fc
            _mlp_act = _mlp.gelu
            _mlp_out_proj = _mlp.c_proj

            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads
            

            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            identity = x  # deep copy

            x = _ln_1(x)
            if not isinstance(ssf_ln, nn.Identity):
                x = ssf_ln["ln_1"](x)

            qkv = nn.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            if not isinstance(ssf_attn, nn.Identity):
                qkv = ssf_attn["attn_in"](qkv)

            q, k, v = qkv.chunk(3, dim=-1)

            if not isinstance(lora, nn.Identity):
                q = q + lora["q"](x)
                v = v + lora["v"](x)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
            # scaled_dot_product_attention:
            q = q / math.sqrt(_head_dim)
            attn = jt.matmul(q, k.transpose(-2, -1))
            attn = nn.softmax(attn, dim=-1)
            x = jt.matmul(attn, v)

            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            
            x = nn.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            if not isinstance(ssf_attn, nn.Identity):
                x = ssf_attn["attn_out"](x)

            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)

            x = x + identity

            ##########################
            ## Feed-Forward Network ##
            ##########################
            identity = x  # deep copy

            x = _ln_2(x)
            if not isinstance(ssf_ln, nn.Identity):
                x = ssf_ln["ln_2"](x)

            x = _mlp_in_proj(x)
            if not isinstance(ssf_mlp, nn.Identity):
                x = ssf_mlp["mlp_in"](x)
            
            x = _mlp_act(x)

            x = _mlp_out_proj(x)
            if not isinstance(ssf_mlp, nn.Identity):
                x = ssf_mlp["mlp_out"](x)
            
            if not isinstance(adapter, nn.Identity):
                x = x + adapter(x)
            
            if not isinstance(adaptformer, nn.Identity):
                x = x + adaptformer(identity)
            
            x = x + identity
            
            x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 0, :]
        x = self.ln_post(x)

        if head is None:
            return x
        else:
            return head(x)
