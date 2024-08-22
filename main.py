import os
import random
import argparse
import numpy as np

from utils.config import _C as cfg
from utils.logger import setup_logger

from trainer import Trainer


def main(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    setup_logger(cfg.output_dir)
    
    print("** Config **")
    print(cfg)
    print("************")
    

    trainer = Trainer(cfg)
    
    if cfg.zero_shot:
        if cfg.validate_training_set:
            trainer.validate("train")
        if cfg.validate:
            trainer.validate()
        if cfg.test:
            trainer.test()
        return
    
    if cfg.train:
        trainer.train()
    else:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir
            print("Model directory: {}".format(cfg.model_dir))
        trainer.load_model(cfg.model_dir)
        
    if cfg.validate_training_set:
        trainer.validate("train")
    if cfg.validate:
        trainer.validate()
    if cfg.test:
        trainer.test()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="jittor4", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="in1k_lv_vit", help="model config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
