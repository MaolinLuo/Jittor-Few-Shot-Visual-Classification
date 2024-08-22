# Please install pytorch

import torch
import jittor as jt
from collections import OrderedDict
from models.lv_vit.lvvit import *

# https://github.com/zihangJiang/TokenLabeling
pytorch_params = torch.load('lvvit_l-150M-512-86.4.pth.tar', map_location='cpu', weights_only=True)
pytorch_params = pytorch_params['state_dict']

for k in pytorch_params.keys():
    pytorch_params[k] = pytorch_params[k].float().cpu().numpy()

jt.save(pytorch_params, 'lvvit_l-150M-512-86.4.pkl')

data = jt.load('lvvit_l-150M-512-86.4.pkl')

model = lvvit_l(img_size=512)
model.load_parameters(data)