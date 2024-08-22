import os
import random
from collections import defaultdict
from PIL import Image
from jittor.dataset import Dataset

class Jittor4(Dataset):    
    def __init__(self, root, train=True, transform=None, test_path=None, classes_path=None, train_path=None, validate_path=None):
        super().__init__()
        self.img_path = []
        self.labels = []
        self.train = train
        self.transform = transform
        self.classnames_txt = classes_path
        self.train_txt = train_path
        self.validate_txt = validate_path

        if not test_path:
            if train:
                self.txt = self.train_txt
            else:
                self.txt = self.validate_txt

            with open(self.txt) as f:
                for line in f:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.labels.append(int(line.split()[1]))
                
        elif test_path and not train:
            imgs = os.listdir(os.path.join(root, test_path))
            self.img_path = [os.path.join(root, test_path, img) for img in imgs]
            self.labels = [-1] * len(self.img_path)
            
        else:
            raise ValueError("Invalid condition: both 'test' and 'train' cannot be True simultaneously.")
    
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        self.classnames = self.read_classnames()
        self.total_len = len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        img_path = self.img_path[index]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label, img_path
    
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list
    
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                last_space_index = line.rfind(' ')
                classname = line[:last_space_index].strip()
                classnames.append(classname)
        return classnames
    
    # @classmethod
    # def split_dataset(self):
    #     with open(self.universe_txt, 'r') as f:
    #         lines = f.readlines()

    #     # Re-organize the data by class
    #     data = {}
    #     for line in lines:
    #         path, label = line.strip().split(' ')
    #         label = int(label)
    #         class_name = path.split('/')[2]
    #         if class_name not in data:
    #             data[class_name] = []
    #         data[class_name].append((path, label))

    #     train_set = []
    #     val_set = []
    #     for class_name, images in data.items():
    #         random.shuffle(images)
    #         train_set.extend(images[:4])
    #         val_set.extend(images[4:])

    #     with open(self.train_txt, 'w') as f:
    #         for path, label in train_set:
    #             f.write(f"{path} {label}\n")

    #     with open(self.validate_txt, 'w') as f:
    #         for path, label in val_set:
    #             f.write(f"{path} {label}\n")
