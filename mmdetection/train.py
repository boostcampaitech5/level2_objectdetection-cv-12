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


classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") # 10개의 클래스

args = parse_arguments()

# config file 들고오기
# cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
cfg = Config.fromfile(args.config_dir)
# cfg_name = 'faster_rcnn_r50_fpn_1x_coco'
root='/opt/ml/dataset/'

# dataset.py 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'train_data_fold_3_411.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (512, 512) # Resize

# dataset.py 수정
cfg.data.val.classes = classes
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = root + 'val_data_fold_3_411.json' # valid json 정보
cfg.data.val.pipeline[1]['img_scale'] = (512, 512) # Resize

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (512, 512) # Resize

cfg.data.samples_per_gpu = 16

# train.py 수정
cfg.seed = 909
cfg.deterministic = True # 찾아보자
cfg.gpu_ids = [0]
cfg.work_dir = f"./work_dirs/{args.config_dir.split('/')[-1][:-3]}"

# retinanet_r50_fpn.py 수정
# cfg.model.bbox_head.num_classes = 10
cfg.model.roi_head.bbox_head.num_classes = 10

# schedule_1x.py 수정
# cfg.optimizer = dict(type='Adam', lr=2e-3, weight_decay=0.0001)
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
                                       name = f'hm){config_name}',
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

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=True)
