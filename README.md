| The Fourth Jittor Artificial Intelligence Challenge

# Jittor Open Domain Few-Shot Visual Classification Task
## Requirements 

### Hardware

This project can run on a single A6000 (48G) GPU.

### Environment
- Ubuntu 22.04.4 LTS
- python == 3.9.19
- jittor == 1.3.8.5

### Dependencies
Run the following command to install the dependencies:
```
pip install -r requirements.txt
```

### Pre-trained Models

The **ViT-B/32** model can be downloaded from [uyzhang/JCLIP (github.com)](https://github.com/uyzhang/JCLIP). Place the model in the `Jittor-Few-Shot-Visual-Classification/` folder.

The **LV-ViT-L** model, pre-trained only on ImageNet-1K, can be downloaded from [zihangJiang/TokenLabeling: Pytorch implementation of "All Tokens Matter: Token Labeling for Training Better Vision Transformers" (github.com)](https://github.com/zihangJiang/TokenLabeling). After downloading the `lvvit_l-150M-512-86.4.pth.tar` model weights, place them in the `Jittor-Few-Shot-Visual-Classification/` folder and run `torch2jt.py` to convert them into the `lvvit_l-150M-512-86.4.pkl` weights file.

### Dataset

**Please modify** the `root` path in the `configs/data/jittor4.yaml` file to the **path of the downloaded dataset**.

The structure of the dataset should be:

```
│J4T1Dataset/
├──TestSetB/
│  ├── image_1.jpg
│  ├── ......
├──TrainSet/
│  ├── Animal
│  │   ├── Bear
│  │   │   ├──1.jpg
│  │   │   ├── ......
│  │   ├── ......
│  ├── Caltech-101
│  ├── Food-101
│  ├── Thu-dog
```

**The relative paths and labels for the 4 selected images per class** are stored in file `datasets/Jittor4/train_label.txt`

## Training

Only supports single GPU training. You can run the following command:

```sh
bash train.sh
```

## Inference

Only supports single GPU inference. You can run the following command:

```sh
python test.py
```

## Acknowledgment

This project references some code implementations from the paper *Long-tail Learning with Foundation Model: Heavy Fine-tuning Hurts* [shijxcs/LIFT: Source code for the paper "Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts" (ICML 2024) (github.com)](https://github.com/shijxcs/LIFT).

This project also references some code implementations from the paper *All Tokens Matter: Token Labeling for Training Better Vision Transformers* [zihangJiang/TokenLabeling: Pytorch implementation of "All Tokens Matter: Token Labeling for Training Better Vision Transformers" (github.com)](https://github.com/zihangJiang/TokenLabeling).
