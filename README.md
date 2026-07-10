# S4Former for Brain Tumor Segmentation

This repository extends [S4Former](https://github.com/JoyHuYY1412/S4Former) — a Vision Transformer-based semi-supervised semantic segmentation framework — for multi-modal brain tumor segmentation on the **BraTS 2023 GLI** dataset.

Original paper: [Training Vision Transformers for Semi-Supervised Semantic Segmentation (CVPR 2024)](https://arxiv.org/abs/2405.02286)

---

## Extensions

Three domain-specific extensions are added on top of the original S4Former:

### 1. 4-Channel MRI Domain Adaptation
The original framework expects 3-channel RGB images. BraTS provides four MRI modalities — T1n, T1c, T2w, and T2 FLAIR — each revealing different tumor sub-regions. The patch embedding layer is modified to accept `in_channels=4`, and the EMA teacher backbone and decoder head are updated accordingly.

A new `BraTSDataset` class handles NIfTI file loading, per-modality z-score normalization, and 2D axial slice extraction.

### 2. Uncertainty-Aware Pseudo-Label Refinement (UAPR)
The original S4Former filters pseudo-labels with a fixed softmax confidence threshold (β=0.95). Softmax probabilities can be overconfident, especially at tumor boundaries. UAPR replaces this with Monte Carlo Dropout-based epistemic uncertainty estimation: the teacher model runs N=10 stochastic forward passes, and pixels with high prediction variance are down-weighted in the unsupervised loss.

### 3. Class-Adaptive Thresholding (CAT)
BraTS has severe class imbalance — the enhancing tumor (ET) can occupy less than 1% of voxels. A global threshold of 0.95 almost never selects ET as a pseudo-label. CAT replaces the single threshold with per-class thresholds tuned inversely to class frequency:

| Class | Label | Threshold |
|-------|-------|-----------|
| Background | 0 | 0.95 |
| ED (Edema) | 2 | 0.80 |
| NCR (Necrotic Core) | 1 | 0.75 |
| ET (Enhancing Tumor) | 3 | 0.70 |

---

## Results

All experiments use a clean 80/20 train/test split (seed=42, 1,001 train / 250 test patients).

| Split | Method | NCR IoU | ED IoU | ET IoU | mIoU |
|-------|--------|---------|--------|--------|------|
| 1/16 | S4Former-Base (BraTS) | 42.43 | 58.96 | 56.37 | 64.27 |
| 1/16 | **UAPR + CAT (Ours)** | **42.93** | 58.86 | **56.47** | **64.39** |
| 1/8  | S4Former-Base (BraTS) | 55.58 | **63.30** | 58.21 | 69.12 |
| 1/8  | **UAPR + CAT (Ours)** | **55.62** | 63.27 | **58.80** | **69.17** |

---

## Dataset

**BraTS 2023 GLI** — 1,251 pre-operative multi-parametric MRI scans with four modalities (T1n, T1c, T2w, T2 FLAIR). Labels: Background (0), NCR (1), ED (2), ET (3).

Download requires registration at [Synapse](https://www.synapse.org/):
```python
import synapseclient
syn = synapseclient.Synapse()
syn.login(authToken='YOUR_TOKEN')
syn.get('syn51514132', downloadLocation='data/BraTS2023')  # Training
syn.get('syn51514110', downloadLocation='data/BraTS2023')  # Validation
```

After downloading, create data splits:
```bash
python tools/create_brats_splits.py \
    --data_root data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
    --split_dir data/BraTS2023/splits \
    --seed 42
```

Expected structure:
```
data/BraTS2023/
├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
│   ├── BraTS-GLI-00000-000/
│   │   ├── BraTS-GLI-00000-000-t1n.nii.gz
│   │   ├── BraTS-GLI-00000-000-t1c.nii.gz
│   │   ├── BraTS-GLI-00000-000-t2w.nii.gz
│   │   ├── BraTS-GLI-00000-000-t2f.nii.gz
│   │   └── BraTS-GLI-00000-000-seg.nii.gz
│   └── ...
└── splits/
    ├── new_1over8_supervised.txt
    ├── new_1over8_unsupervised.txt
    ├── new_1over16_supervised.txt
    ├── new_1over16_unsupervised.txt
    └── new_test.txt
```

---

## Installation

### Option 1: Singularity Container (Recommended)

A Singularity definition file is provided. Build the container:

```bash
singularity build --fakeroot INSTALL/s4former.sif s4former.def
```

> **Note:** The container embeds the S4Former source at build time (`%files` section). Rebuild after code changes, or use bind mounts to override files at runtime.

Run with bind mounts to use local code changes without rebuilding:

```bash
singularity exec --nv \
    -B /path/to/S4Former/mmseg:/S4Former/mmseg \
    INSTALL/s4former.sif \
    bash tools/dist_train.sh configs/brats/s4former_brats_1over8_new_ours.py 1 --seed 1999
```

> **GPU Compatibility:** The container (PyTorch 1.11, CUDA 11.3) is **incompatible with H100 GPUs** due to NCCL version mismatches. Use **A100 GPUs**.

### Option 2: Manual Installation

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

pip install timm einops nibabel
pip install -e .
```


---

## Training

**BraTS 1/8 labeled — with UAPR + CAT:**
```bash
bash tools/dist_train.sh \
    configs/brats/s4former_brats_1over8_new_ours.py \
    1 --seed 1999
```

**BraTS 1/8 labeled — baseline (domain adaptation only):**
```bash
bash tools/dist_train.sh \
    configs/brats/s4former_brats_1over8_new_baseline.py \
    1 --seed 1999
```

**BraTS 1/16 labeled:**
```bash
bash tools/dist_train.sh \
    configs/brats/s4former_brats_1over16_new_ours.py \
    1 --seed 1999
```

> **Note:** The S4Former container is incompatible with H100 GPUs due to NCCL/CUDA version mismatches. Run on A100 GPUs.

---

## Evaluation

```bash
python tools/test.py \
    configs/brats/s4former_brats_1over8_new_ours.py \
    work_dirs/s4former_brats_1over8_new_ours_seed_1999/latest.pth \
    --eval mIoU
```

---

## Modified Files

| File | Change |
|------|--------|
| `mmseg/datasets/brats.py` | New — BraTS NIfTI dataset loader |
| `mmseg/datasets/__init__.py` | Register `BraTSDataset` |
| `mmseg/models/utils/uapr.py` | New — `UAPRModule` + `ClassAdaptiveThreshold` |
| `mmseg/models/segmentors/encoder_decoder.py` | Integrate UAPR + adaptive threshold |
| `mmseg/models/backbones/vit.py` | Fix attention weight extraction (forward hook) |
| `mmseg/utils/__init__.py` | Export `generate_unsup_patchmix_data` |
| `configs/brats/` | New BraTS config files |

---

## Citation

If you use this work, please cite the original S4Former paper:

```bibtex
@inproceedings{hu2024s4former,
  title={Training Vision Transformers for Semi-Supervised Semantic Segmentation},
  author={Hu, Xinting and Jiang, Li and Schiele, Bernt},
  booktitle={CVPR},
  year={2024}
}
```

---

## Acknowledgements

Built on top of [S4Former](https://github.com/JoyHuYY1412/S4Former) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
