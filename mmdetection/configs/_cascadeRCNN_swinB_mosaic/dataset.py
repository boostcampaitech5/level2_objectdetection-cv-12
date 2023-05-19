# dataset settings

dataset_type = 'CocoDataset'
data_root = '/opt/ml/dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

multi_scale = [(w,w) for w in range(512, 1024+1, 32)]

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type = 'HorizontalFlip', p = 1.0),
            dict(type = 'VerticalFlip', p = 1.0),
            dict(type='RandomRotate90',p=1.0)
        ],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.2, contrast_limit=0.2,
                p=1.0),
            dict(
                type='CLAHE',
                clip_limit=(1, 4),
                tile_grid_size=(8, 8),
                p=1.0),
        ],
        p=0.5),
    dict(type='HueSaturationValue',hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    dict(type='GaussNoise', p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', p=1.0),
            dict(type='GaussianBlur',blur_limit=5, p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='MotionBlur', blur_limit=5, p=1.0),
        ],
        p=0.1),
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='Mosaic', img_scale=(1024,1024), pad_val=114.0),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', 
        img_scale=multi_scale,
        multiscale_mode='value',
        keep_ratio=True,
        ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
    type='Albu',
    transforms=albu_train_transforms,
    bbox_params=dict(
        type='BboxParams',
        format='pascal_voc',
        label_fields=['gt_labels'],
        min_visibility=0.0,
        filter_lost_elements=True),
    keymap={
        'img': 'image',
        'gt_bboxes': 'bboxes'
    },
    update_pad_shape=False,
    skip_img_without_anno=True
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=multi_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_data_fold_3_seed_411.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes,
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_data_fold_3_seed_411.json',
        img_prefix=data_root,
        pipeline=val_pipeline,
        classes=classes,
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val_data_fold_3_seed_411.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
        ))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')