# 모듈 import
import wandb
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import argparse
import os 

def parse_arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default=os.path.expanduser('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'),
        help="config file for training",
    )
    return parser.parse_args()

args = parse_arguments()

# config file 들고오기
# cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
cfg = Config.fromfile(args.config_dir)
cfg_name = args.config_dir.split('/')[-1]

cfg.data.samples_per_gpu = 16

# train.py 수정
cfg.seed = 909
cfg.deterministic = True # 찾아보자
cfg.gpu_ids = [0]
cfg.work_dir = f"./work_dirs/{args.config_dir.split('/')[-1][:-3]}"

cfg.optimizer_config = dict(grad_clip = dict(max_norm=35, norm_type=2)) # Use gradient clip to stabilize training

# default_runtime.py 수정
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.log_config.interval = 50
config_name = args.config_dir.split('/')[-1][:-3]
cfg.log_config.hooks.append(
                            dict(type='WandbLoggerHook',
                                 interval=50,
                                init_kwargs=dict(
                                       project = 'waste_detection',
                                       name = f'dh){cfg_name}',
                                   )))
meta = dict()

# fp16 settings
# cfg.fp16 = dict(loss_scaler=[512.])

cfg.workflow = [('train', 1)]
cfg.device = get_device()

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

cfg.evaluation.classwise = True

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=True)
