_base_ = [
    '../setr/setr_deit-base_pup_bs_8_512x512_80k_pascal_1over16_split_classic_semi_beta_1_th_0.95_MT_w_ours.py'
]

# BraTS dataset
dataset_type = 'BraTSDataset'
train_data_root = 'data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
val_data_root   = 'data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
num_classes = 4

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0, 1.0], to_rgb=False)
crop_size = (224, 224)
img_scale = (240, 240)

train_pipeline = [
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ExtraAttrs', tag='sup'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_semantic_seg'],
         meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg', 'tag')),
]

test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='SemiDataset',
        sup=dict(
            type=dataset_type,
            data_root=train_data_root,
            split='data/BraTS2023/splits/new_test.txt',
            pipeline=train_pipeline,
        ),
        unsup=dict(
            type=dataset_type,
            data_root=train_data_root,
            split='data/BraTS2023/splits/new_test.txt',
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=train_data_root,
        split='data/BraTS2023/splits/new_test.txt',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=train_data_root,
        split='data/BraTS2023/splits/new_test.txt',
        pipeline=test_pipeline,
    ),
    sampler=dict(
        train=dict(
            type='SemiBalanceSampler',
            sample_ratio=[1, 1],
            by_prob=False,
            max_iter_size=80000,
        )
    )
)

# Model — base config'i override et
model = dict(
    backbone=dict(in_channels=4),
    backbone_ema=dict(in_channels=4),
    decode_head_ema=dict(num_classes=4),
    auxiliary_head_ema=[
        dict(type='SETRUPHead', in_channels=768, channels=256, in_index=i,
             num_classes=4, dropout_ratio=0,
             norm_cfg=dict(type='BN', requires_grad=True),
             align_corners=False,
             loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
        for i in range(3)
    ],
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=[
        dict(type='SETRUPHead', in_channels=768, channels=256, in_index=i,
             num_classes=num_classes, dropout_ratio=0,
             norm_cfg=dict(type='BN', requires_grad=True),
             align_corners=False,
             loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
        for i in range(3)
    ],
    num_classes=num_classes,
    use_uapr=True,
    uapr_n_samples=10,
    uapr_dropout_p=0.3,
    use_adaptive_threshold=True,
    class_thresholds={0: 0.95, 1: 0.75, 2: 0.80, 3: 0.70},
)

test_cfg = dict(mode='whole')
