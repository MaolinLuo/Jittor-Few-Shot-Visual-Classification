from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.root = ""  # Directory where datasets are stored
_C.test_path = ""   # The relative path of the test set with respect to root
_C.classes_path = ""
_C.train_path = ""
_C.validate_path = ""

_C.backbone = ""
_C.resolution = 224

_C.output_dir = None  # Directory to save the output files (like log.txt and model weights)
_C.print_freq = 10  # How often (batch) to print training information

_C.num_epochs = 10
_C.batch_size = 128
_C.micro_batch_size = 128  # for gradient accumulation, must be a divisor of batch size
_C.lr = 0.01
_C.weight_decay = 5e-4
_C.momentum = 0.9
_C.loss_type = "CE"

_C.classifier = "CosineClassifier"
_C.scale = 25  # for cosine classifier

_C.full_tuning = False  # full fine-tuning
_C.bias_tuning = False  # only fine-tuning the bias 
_C.ln_tuning = False  # only fine-tuning the layer norm
_C.bn_tuning = False  # only fine-tuning the batch norm (only for resnet)
_C.vpt_shallow = False
_C.vpt_deep = False
_C.adapter = False
_C.adaptformer = False
_C.lora = False
_C.ssf_attn = False
_C.ssf_mlp = False
_C.ssf_ln = False
_C.partial = None  # fine-tuning (or parameter-efficient fine-tuning) partial block layers
_C.vpt_len = None  # length of VPT sequence
_C.adapter_dim = None  # bottle dimension for adapter / adaptformer / lora.

_C.validate_ensemble = False  # validation-time ensemble
_C.ensemble_mode = "fivecrop" # "fivecrop" / "tencrop"
_C.expand = 24 # expand the width and height of images for validation-time ensemble

_C.zero_shot = False  # zero-shot CLIP
_C.train = False
_C.validate = False
_C.test = False
_C.validate_training_set = False  # load model and validate on the training set

_C.model_dir = None

_C.mixup = False
_C.alpha = 0.2