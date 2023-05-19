# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    './cascade_mask_rcnn_convnext_fpn.py',
    './dataset.py',
    './schedule_CosineAnnealing.py', './runtime.py'
]

model = dict(
    backbone=dict(
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[128, 256, 512, 1024], 
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    # neck=dict(in_channels=[128, 256, 512, 1024]),
neck=[
        dict(
            type='FPN',
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)],
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))


optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.8,
                                'decay_type': 'layer_wise',
                                'num_layers': 12})
#lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=20)

# do not use mmdet version fp16
fp16 = None
load_from='https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_base_22k_3x.pth'

# https://github.com/facebookresearch/ConvNeXt download from here


# lr_config = dict(
#     policy='CosineRestart', 
#     periods=[3900 * i for i in range(1,14)],
#     restart_weights=[1 for _ in range(1,14)],
#     by_epoch = False,
#     min_lr=1e-07)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
                init_kwargs=dict(
                    # 각각 자신에 맞춰서 Project이름 설정
                    project= 'waste_detection',
                    name = 'dh) _cascade_dyhead_convnext'
                ),
            ),
        # dict(type='MlflowLoggerHook')
    ])

work_dir = '/opt/ml/level2_objectdetection-cv-12/mmdetection/work_dirs/convnext-b_dyhead'
#$optimizer_config = dict(
#$    type="DistOptimizerHook",
#$    update_interval=1,
#$    grad_clip=None,
#$    coalesce=True,
#$    bucket_size_mb=-1,
#$    use_fp16=True,
#$)