import jittor as jt
import os
import argparse
import copy

from utils.config import _C as cfg
from utils.logger import setup_logger
from trainer import Trainer
from result import process_files
from mapping_label import map_label

def preprocess():
    keywords_list = ['Thu-dog']
    map_label('datasets/Jittor4/classes.txt', 'datasets/Jittor4/classes_dog.txt', keywords_list)
    map_label('datasets/Jittor4/train_label.txt', 'datasets/Jittor4/train_label_dog.txt', keywords_list)
    map_label('datasets/Jittor4/val.txt', 'datasets/Jittor4/val_dog.txt', keywords_list)

def create_trainer(args):
    cfg_copy = copy.deepcopy(cfg)
    
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg_copy.defrost()
    cfg_copy.merge_from_file(cfg_data_file)
    cfg_copy.merge_from_file(cfg_model_file)
    cfg_copy.merge_from_list(args.opts)
    # cfg_copy.freeze()

    if cfg_copy.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg_copy.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg_copy.output_dir = os.path.join("./output", cfg_copy.output_dir)
    print("Output directory: {}".format(cfg_copy.output_dir))
    setup_logger(cfg_copy.output_dir)
    
    print("** Config **")
    print(cfg_copy)
    print("************")

    trainer = Trainer(cfg_copy)
    return trainer

def process(trainers):    
    if trainers[0].cfg.zero_shot:
        trainer = trainers[0]
        cfg = trainers[0].cfg
        if cfg.validate_training_set:
            trainer.validate("train")
        if cfg.validate:
            trainer.validate()
        if cfg.test:
            trainer.test()
        return
    
    else:
        models = []
        for trainer in trainers:
            cfg = trainer.cfg
            print("Model directory: {}".format(cfg.model_dir))
            trainer.load_model(cfg.model_dir)
            models.append(trainer.model)
        
        trainer = trainers[0]
        cfg = trainers[0].cfg
        if cfg.validate_training_set:
            trainer.validate("train", models=models)
        if cfg.validate:
            trainer.validate(models=models)
        if cfg.test:
            trainer.test(models=models)
        return


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", type=str, default="jittor4", help="data config file")
    
    parser.add_argument("--model1", type=str, default="zsclip_vit_b32_peft", help="First config file")
    parser.add_argument("--opts1", nargs=argparse.REMAINDER, 
                        default=["validate_training_set", "False",
                                 "test", "True",
                                 "output_dir", "jittor4_zsclip_vit_b32_peft"],
                        help="First options")
    
    parser.add_argument("--model2", type=str, default="clip_vit_b32_peft", help="Second config file")
    parser.add_argument("--opts2", nargs=argparse.REMAINDER, 
                        default=["train", "False", 
                                 "validate_training_set", "False",
                                 "test", "True",
                                 "model_dir", "output/jittor4_clip_vit_b32_peft_n1",
                                 "output_dir", "jittor4_clip_vit_b32_peft"], 
                        help="Second options")
    
    parser.add_argument("--model2_2", type=str, default="clip_vit_b32_peft", help="Second config file")
    parser.add_argument("--opts2_2", nargs=argparse.REMAINDER, 
                        default=["train", "False", 
                                 "validate_training_set", "False",
                                 "test", "True",
                                 "model_dir", "output/jittor4_clip_vit_b32_peft_n2",
                                 "output_dir", "jittor4_clip_vit_b32_peft"], 
                        help="Second options")
    
    parser.add_argument("--model2_3", type=str, default="clip_vit_b32_peft", help="Second config file")
    parser.add_argument("--opts2_3", nargs=argparse.REMAINDER, 
                        default=["train", "False", 
                                 "validate_training_set", "False",
                                 "test", "True",
                                 "model_dir", "output/jittor4_clip_vit_b32_peft_n3",
                                 "output_dir", "jittor4_clip_vit_b32_peft"], 
                        help="Second options")
    
    parser.add_argument("--model2_4", type=str, default="clip_vit_b32_peft", help="Second config file")
    parser.add_argument("--opts2_4", nargs=argparse.REMAINDER, 
                        default=["train", "False", 
                                 "validate_training_set", "False",
                                 "test", "True",
                                 "model_dir", "output/jittor4_clip_vit_b32_peft_n4",
                                 "output_dir", "jittor4_clip_vit_b32_peft"], 
                        help="Second options")
    
    parser.add_argument("--model3", type=str, default="in1k_lv_vit", help="Third config file")
    parser.add_argument("--opts3", nargs=argparse.REMAINDER, 
                        default=["train", "False", 
                                 "validate_training_set", "False",
                                 "test", "True",
                                 "model_dir", "output/jittor4_in1k_lv_vit_n1",
                                 "output_dir", "jittor4_in1k_lv_vit"], 
                        help="Third options")
    
    args = parser.parse_args()
    
    args1 = argparse.Namespace(data=args.data, model=args.model1, opts=args.opts1)
    trainer1 = create_trainer(args1)
    
    args2 = argparse.Namespace(data=args.data, model=args.model2, opts=args.opts2)
    trainer2 = create_trainer(args2)
    
    args2_2 = argparse.Namespace(data=args.data, model=args.model2_2, opts=args.opts2_2)
    trainer2_2 = create_trainer(args2_2)
    
    args2_3 = argparse.Namespace(data=args.data, model=args.model2_3, opts=args.opts2_3)
    trainer2_3 = create_trainer(args2_3)
    
    args2_4 = argparse.Namespace(data=args.data, model=args.model2_4, opts=args.opts2_4)
    trainer2_4 = create_trainer(args2_4)
    
    
    args3 = argparse.Namespace(data=args.data, model=args.model3, opts=args.opts3)
    trainer3 = create_trainer(args3)

    
    process([trainer1])
    process([trainer2, trainer2_2, trainer2_3, trainer2_4])
    process([trainer3])
    
    process_files("output/jittor4_zsclip_vit_b32_peft/result.txt",
                  "output/jittor4_clip_vit_b32_peft/result.txt",
                  "output/jittor4_in1k_lv_vit/result.txt",
                  "result.txt")

def params_total():
    jt.sync_all()
    jt.gc()
    
    params_paths = ['ViT-B-32.pkl',
                    'lvvit_l-150M-512-86.4.pkl']
    total_params = 0
    for path in params_paths:
        state_dict = jt.load(path)
        total_params += sum(param.size for param in state_dict.values())
        
    params_paths = ['output/jittor4_clip_vit_b32_peft_n1/checkpoint.pkl',
                    'output/jittor4_clip_vit_b32_peft_n2/checkpoint.pkl',
                    'output/jittor4_clip_vit_b32_peft_n3/checkpoint.pkl',
                    'output/jittor4_clip_vit_b32_peft_n4/checkpoint.pkl',
                    'output/jittor4_in1k_lv_vit_n1/checkpoint.pkl']
    for path in params_paths:
        state_dict = jt.load(path)
        for k in state_dict.keys():
            total_params += sum(param.size for param in state_dict[k].values())
    
    jt.sync_all()
    jt.gc()
    print("==================================================")
    print(f"Total number of parameters: {total_params}")
    return total_params

if __name__ == "__main__":
    preprocess()
    main()
    params_total()
