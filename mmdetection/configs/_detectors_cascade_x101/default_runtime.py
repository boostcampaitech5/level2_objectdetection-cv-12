work_dir = f"./work_dirs/_detectors_cascade_x101"
checkpoint_config = dict(max_keep_ckpts=3, interval=1)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
log_config['hooks'].append(
                            dict(type='WandbLoggerHook',
                                 interval=50,
                                 init_kwargs=dict(
                                 project = 'waste_detection',
                                 name = f'hm) _detectors_cascade_x101',#name = f'hm){config_name}',#
                                )))
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
