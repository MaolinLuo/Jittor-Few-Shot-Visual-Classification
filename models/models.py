import jittor as jt
from jittor import init
from jittor import nn
from tqdm import tqdm

from .clip_text import CLIP_Text
from .peft_vit import Peft_ViT, ViT_Tuner
from .peft_lvvit import Peft_LV_ViT, LV_ViT_Tuner

from .classifiers import *

class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model, num_prompts_per_class):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.num_prompts_per_class= num_prompts_per_class
        self.dtype = clip_model.dtype
    
    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = jt.split(text, 1000)
            text_features = jt.concat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image.astype(self.dtype))

    @jt.no_grad()
    def init_text_features(self, prompts):
        text_features = self.encode_text(prompts)
        text_features = jt.normalize(text_features, dim=-1)
        self.text_features = text_features

    def execute(self, image):
        image_features = self.encode_image(image)
        image_features = jt.normalize(image_features, dim=-1)
        logit = nn.linear(image_features, self.text_features)
        
        start_index = 0
        logit_per_class = []
        for num_prompts in self.num_prompts_per_class:
            end_index = start_index + num_prompts
            logit_class = jt.max(logit[:, start_index:end_index], dim=1)
            logit_per_class.append(logit_class)
            start_index = end_index
        
        logit = jt.stack(logit_per_class).t()
        logit = self.logit_scale * logit
        return logit


class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes, prompts, num_prompts_per_class):
        super().__init__()

        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = Peft_ViT(clip_model.visual)
        self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        head_weight = self.init_head_text_feat(prompts, num_prompts_per_class)
        self.head = eval(cfg.classifier)(feat_dim, num_classes, head_weight, dtype, **cfg)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = jt.split(text, 1000)
            text_features = jt.concat([self.text_encoder(x) for x in text_split])
        return text_features
    
    @jt.no_grad()
    def init_head_text_feat(self, prompts, num_prompts_per_class):
        print("Initialize head with text features")
        text_features = self.text_encoder(prompts)
        text_features = jt.normalize(text_features, dim=-1)

        text_features = text_features @ self.image_encoder.proj.t()
        
        start_index = 0
        logit_per_class = []
        for num_prompts in num_prompts_per_class:
            end_index = start_index + num_prompts
            logit_class = jt.max(text_features[start_index:end_index, :], dim=0)
            logit_per_class.append(logit_class)
            start_index = end_index
        
        text_features = jt.stack(logit_per_class)
        text_features = jt.normalize(text_features, dim=-1)

        return text_features
    
    def execute(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        return self.image_encoder(image, tuner, head)


class PeftModelFromViT(nn.Module):
    def __init__(self, cfg, vit_model, num_classes, train_init_loader):
        super().__init__()

        self.num_classes = num_classes

        if cfg.backbone.startswith("IN1K-ViT"):
            self.image_encoder = Peft_ViT(vit_model)
            self.tuner = ViT_Tuner(cfg, vit_model, num_classes)
        elif cfg.backbone.startswith("IN1K-LV-ViT"):
            self.image_encoder = Peft_LV_ViT(vit_model)
            self.tuner = LV_ViT_Tuner(cfg, vit_model, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        if cfg.train:
            head_weight = self.init_head_class_mean(train_init_loader)
        else:
            head_weight = None
        jt.sync_all()
        jt.gc()
        self.head = eval(cfg.classifier)(feat_dim, num_classes, head_weight, dtype, **cfg)
    
    @jt.no_grad()
    def init_head_class_mean(self, train_init_loader):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            feature = self.execute(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = jt.concat(all_features, dim=0)
        all_labels = jt.concat(all_labels, dim=0)

        sorted_index = all_labels.argsort()[0]
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, _, label_counts = jt.unique(all_labels, return_inverse=True, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = jt.concat(class_means, dim=0)
        class_means = jt.normalize(class_means, dim=-1)

        return class_means

    def execute(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        return self.image_encoder(image, tuner, head)
