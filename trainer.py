import os
import time
import datetime
import math
import json
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import jittor as jt
from jittor import transform
from jittor.dataset import DataLoader

from jclip import clip

import datasets
from models import *
from models.lv_vit.lvvit import *

from utils.meter import AverageMeter
from utils.evaluator import Evaluator


def load_clip(backbone_name):
    # JCLIP's default precision is fp32
    backbone_name = backbone_name.lstrip("CLIP-")
    model, preprocess = clip.load(backbone_name)
    
    return model

def load_vit(backbone_name):
    if backbone_name.startswith("IN1K-LV-ViT"):
        model = lvvit_l(img_size=512)
        model.load_parameters(jt.load('lvvit_l-150M-512-86.4.pkl'))
        
    return model

class Trainer:
    def __init__(self, cfg):

        jt.flags.use_cuda = 1

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.animal_idxs, self.caltech_idxs, self.food_idxs, self.dog_idxs)

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        elif cfg.backbone.startswith("IN1K-LV-ViT"):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        transform_train = transform.Compose([
            transform.RandomResizedCrop(resolution),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.ImageNormalize(mean, std),
        ])

        transform_plain = transform.Compose([
            clip.Resize(resolution),
            transform.CenterCrop(resolution),
            transform.ToTensor(),
            transform.ImageNormalize(mean, std),
        ])

        if cfg.validate_ensemble:
            if cfg.ensemble_mode == "fivecrop":
                transform_validate = transform.Compose([
                    clip.Resize(resolution + expand),
                    transform.FiveCrop(resolution),
                    transform.Lambda(lambda crops: np.stack([transform.ToTensor()(crop) for crop in crops])),
                    transform.ImageNormalize(mean, std),
                ])
            elif cfg.ensemble_mode == "tencrop":
                transform_validate = transform.Compose([
                    clip.Resize(resolution + expand),
                    transform.TenCrop(resolution),
                    transform.Lambda(lambda crops: np.stack([transform.ToTensor()(crop) for crop in crops])),
                    transform.ImageNormalize(mean, std),
                ])
        else:
            transform_validate = transform.Compose([
                clip.Resize(resolution * 8 // 7),
                transform.CenterCrop(resolution),
                transform.Lambda(lambda crop: np.stack([transform.ToTensor()(crop)])),
                transform.ImageNormalize(mean, std),
            ])

        # getattr(datasets, cfg.dataset).split_dataset()
        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, 
                                                       transform=transform_train, 
                                                       classes_path=cfg.classes_path,
                                                       train_path=cfg.train_path)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, 
                                                            transform=transform_plain, 
                                                            classes_path=cfg.classes_path,
                                                            train_path=cfg.train_path)
        train_validate_dataset = getattr(datasets, cfg.dataset)(root, train=True, 
                                                                transform=transform_validate, 
                                                                classes_path=cfg.classes_path,
                                                                train_path=cfg.train_path)
        validate_dataset = getattr(datasets, cfg.dataset)(root, train=False, 
                                                          transform=transform_validate, 
                                                          classes_path=cfg.classes_path,
                                                          validate_path=cfg.validate_path)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, 
                                                      transform=transform_validate, 
                                                      test_path=cfg.test_path, 
                                                      classes_path=cfg.classes_path)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        if len(self.classnames) >= 374:   
            self.animal_idxs = np.array([i for i in range(0, 52)])
            self.caltech_idxs = np.array([i for i in range(52, 143)])
            self.food_idxs = np.array([i for i in range(143, 244)])
            self.dog_idxs = np.array([i for i in range(244, 374)])
        else:
            self.animal_idxs = self.caltech_idxs = self.food_idxs = self.dog_idxs = np.array([i for i in range(0, len(self.classnames))])

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=32, shuffle=False)

        self.train_validate_loader = DataLoader(train_validate_dataset,
            batch_size=8, shuffle=False)

        self.validate_loader = DataLoader(validate_dataset,
            batch_size=8, shuffle=False)
        
        self.test_loader = DataLoader(test_dataset,
            batch_size=8, shuffle=False)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading JCLIP (backbone: {cfg.backbone})")
            clip_model = load_clip(cfg.backbone)
            prompts, num_prompts_per_class = self.get_tokenized_prompts(classnames)
            self.model = ZeroShotCLIP(clip_model, num_prompts_per_class)
            self.model.init_text_features(prompts)
            self.tuner = None
            self.head = None

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading JCLIP (backbone: {cfg.backbone})")
            clip_model = load_clip(cfg.backbone)
            prompts, num_prompts_per_class = self.get_tokenized_prompts(classnames)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes, prompts, num_prompts_per_class)
            self.tuner = self.model.tuner
            self.head = self.model.head
        
        elif cfg.backbone.startswith("IN1K-LV-ViT"):
            print(f"Loading LV-ViT (backbone: {cfg.backbone})")
            vit_model = load_vit(cfg.backbone)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes, self.train_init_loader)
            self.tuner = self.model.tuner
            self.head = self.model.head
            
        else:
            raise ValueError(f"Unsupported backbone type: {cfg.backbone}")

        if not cfg.zero_shot and cfg.train:
            self.build_optimizer()
            self.build_criterion()

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning on gradients in the tuner")
        self.tuner.requires_grad_(True)
        print("Turning on gradients in the head")
        self.head.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = jt.optim.SGD([{"params": self.tuner.parameters()},
                                   {"params": self.head.parameters()}],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = jt.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)

    def build_criterion(self):
        cfg = self.cfg

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()

    def get_tokenized_prompts(self, classnames):
        with open("utils/descriptors.json") as f:
            descriptors = json.load(f)
        with open("utils/templates.json") as f:
            templates = json.load(f)
        
        template_default = "a photo of a {}."
        template_animal = templates["Animal"]
        template_caltech = templates["Caltech-101"]
        template_food = templates["Food-101"]
        template_dog = templates["Thu-dog"]
        template_car = templates["Stanford-Cars"]
        
        def generate_prompts(template_list, class_name, descriptor_key):
            generated_prompts = []
            for tp in template_list:
                prompt = tp.format(class_name)
                if descriptor_key in descriptors:
                    for dc in descriptors[descriptor_key]:
                        generated_prompts.append(prompt + " " + dc)
                else:
                    generated_prompts.append(prompt)
            return generated_prompts, len(generated_prompts)

        prompts = []
        num_prompts_per_class = []
        for c in classnames:
            class_name = c.replace("_", " ").strip()
            
            if "Animal" in class_name:
                class_name = class_name.replace("Animal", "").strip()
                class_prompts, num_prompts = generate_prompts(template_animal, class_name, c)
            elif "Caltech-101" in class_name:
                class_name = class_name.replace("Caltech-101", "").strip()
                class_prompts, num_prompts = generate_prompts(template_caltech, class_name, c)
            elif "Food-101" in class_name:
                class_name = class_name.replace("Food-101", "").strip()
                class_prompts, num_prompts = generate_prompts(template_food, class_name, c)
            elif "Thu-dog" in class_name:
                class_name = class_name.replace("Thu-dog", "").strip()
                class_prompts, num_prompts = generate_prompts(template_dog, class_name, c)
            elif "Stanford-Cars" in class_name:
                class_name = class_name.replace("Stanford-Cars", "").strip()
                class_prompts, num_prompts = generate_prompts(template_car, class_name, c)
            else:
                class_prompts = [template_default.format(class_name)]
                num_prompts = 1
                print(f"Please check the class name {c}!")
    
            prompts.extend(class_prompts)
            num_prompts_per_class.append(num_prompts)
                    
        # print(f"Prompts: {prompts}")
        prompts = jt.concat([clip.tokenize(p) for p in prompts])
        return prompts, num_prompts_per_class

    def mixup_data(self, x, y, alpha=1.0):
        '''Compute the mixup data. Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = jt.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train(self):
        cfg = self.cfg

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            self.head.train()
            end = time.time()
            
            # for name, param in self.model.named_parameters():
            #     print(f"+{name}: {param.requires_grad}")

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]

                if cfg.mixup:
                    image, targets_a, targets_b, lam = self.mixup_data(image, label, alpha=cfg.alpha)

                output = self.model(image)
                if cfg.mixup:
                    loss = lam * self.criterion(output, targets_a) + (1 - lam) * self.criterion(output, targets_b)
                else:
                    loss = self.criterion(output, label)
                loss_micro = loss / self.accum_step
                if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                    self.optim.step(loss_micro)
                    self.optim.zero_grad()

                with jt.no_grad():
                    pred = output.argmax(dim=1)[0]
                    correct = pred.equal(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.lr
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                Animal_acc = np.mean(np.array(cls_accs)[self.animal_idxs])
                Caltech_acc = np.mean(np.array(cls_accs)[self.caltech_idxs])
                Food_acc = np.mean(np.array(cls_accs)[self.food_idxs])
                Thu_dog_acc = np.mean(np.array(cls_accs)[self.dog_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} Animal {Animal_acc:.4f} Caltech {Caltech_acc:.4f} Food {Food_acc:.4f} Thu_dog {Thu_dog_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                end = time.time()

            self.sched.step()
            jt.sync_all()
            jt.gc()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``validate_training_set True``.")
        jt.sync_all()
        jt.gc()

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

    @jt.no_grad()
    def inference(self, models, image):
        output = None
        if isinstance(models, list):
            for model in models:
                output = output + model(image) if output is not None else model(image)
        else:
            output = models(image)
        return output

    @jt.no_grad()
    def validate(self, mode="validate", models=None):
        if models:
            for model in models:
                model.eval()
        else:
            models = self.model.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_validate_loader
        elif mode == "validate":
            print(f"Evaluate on the validation set")
            data_loader = self.validate_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            output = self.inference(models, image)
            output = output.view(_bsz, _ncrops, -1).mean(dim=1)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return list(results.values())[0]

    @jt.no_grad()
    def test(self, models=None):
        if models:
            for model in models:
                model.eval()
        else:
            models = self.model.eval()
            
        data_loader = self.test_loader
        predictions = []
        image_names = []

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            image_path = batch[2]
            image_name = [path.split('/')[-1] for path in image_path]

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            output = self.inference(models, image)
            output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            
            predictions.extend(output.tolist())
            image_names.extend(image_name)
            
        animal_idxs = set(self.animal_idxs.tolist())
        caltech_idxs = set(self.caltech_idxs.tolist())
        food_idxs = set(self.food_idxs.tolist())
        dog_idxs = set(self.dog_idxs.tolist())
        animal_top1 = 0
        caltech_top1 = 0
        food_top1 = 0
        dog_top1 = 0
        car_top1 = 0
        def process_top1_prediction(top1_prediction):
            nonlocal animal_top1, caltech_top1, food_top1, dog_top1, car_top1
            if top1_prediction in animal_idxs:
                animal_top1 += 1
            elif top1_prediction in caltech_idxs:
                caltech_top1 += 1
            elif top1_prediction in food_idxs:
                food_top1 += 1
            elif top1_prediction in dog_idxs:
                dog_top1 += 1
            else:
                car_top1 +=1 
            
        with open(os.path.join(self.cfg.output_dir, 'result.txt'), 'w') as save_file:
            i = 0
            for prediction in predictions:
                prediction = np.asarray(prediction)
                top5_idx = prediction.argsort()[-1:-6:-1]
                process_top1_prediction(top5_idx[0])
                save_file.write(image_names[i] + ' ' +
                                ' '.join(str(idx) for idx in top5_idx) + '\n')
                i += 1
        
        print("Evaluate on the test set")
        print("=> result")
        print(f"* Count=> Animal: {animal_top1}  Caltech: {caltech_top1}  Food: {food_top1}  Thu_dog: {dog_top1}  Car: {car_top1}")


    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pkl")
        jt.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pkl")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = jt.load(load_path)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict)
        self.head.load_state_dict(head_dict)
