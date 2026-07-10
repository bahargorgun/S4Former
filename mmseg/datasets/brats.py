import os
import os.path as osp
import numpy as np
from mmcv.utils import print_log
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class BraTSDataset(CustomDataset):
    CLASSES = ('background', 'NCR', 'ED', 'ET')
    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

    def __init__(self, data_root, split, pipeline,
                 slice_axis=2, skip_empty=True, **kwargs):
        self.data_root = data_root
        self.split = split
        self.slice_axis = slice_axis
        self.skip_empty = skip_empty
        self.ignore_index = 255
        self.reduce_zero_label = False
        self.ignore_index = 255
        self.reduce_zero_label = False
        from mmseg.datasets.pipelines import Compose
        self.pipeline = Compose(pipeline)
        self.img_infos = self.load_annotations()

    def load_annotations(self):
        import sys, subprocess
        subprocess.run(['pip', 'install', 'nibabel', '--target=/tmp/pypackages', '-q'])
        sys.path.insert(0, '/tmp/pypackages')
        import nibabel as nib
        with open(self.split, 'r') as f:
            case_ids = [l.strip() for l in f if l.strip()]
        img_infos = []
        for case_id in case_ids:
            case_dir = osp.join(self.data_root, case_id)
            seg_path = osp.join(case_dir, f'{case_id}-seg.nii.gz')
            if not osp.exists(seg_path):
                continue
            seg = nib.load(seg_path).get_fdata().astype(np.uint8)
            for s in range(seg.shape[2]):
                if self.skip_empty and np.sum(seg[:, :, s] > 0) == 0:
                    continue
                img_infos.append({'case_id': case_id, 'case_dir': case_dir, 'slice_idx': s,
                                  'filename': f'{case_id}_s{s:03d}'})
        print_log(f'BraTS: {len(case_ids)} volume, {len(img_infos)} slices', logger='root')
        return img_infos

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        import sys
        sys.path.insert(0, '/tmp/pypackages')
        import nibabel as nib
        info = self.img_infos[idx]
        case_dir, case_id, s = info['case_dir'], info['case_id'], info['slice_idx']
        channels = []
        for mod in ['t1n', 't1c', 't2w', 't2f']:
            vol = nib.load(osp.join(case_dir, f'{case_id}-{mod}.nii.gz')).get_fdata().astype(np.float32)
            slc = vol[:, :, s]
            mask = slc > 0
            if mask.sum() > 0:
                slc[mask] = (slc[mask] - slc[mask].mean()) / (slc[mask].std() + 1e-8)
            channels.append(slc)
        img = np.stack(channels, axis=-1).astype(np.float32)
        seg_vol = nib.load(osp.join(case_dir, f'{case_id}-seg.nii.gz')).get_fdata().astype(np.uint8)
        gt_seg = seg_vol[:, :, s]
        results = {
            'img': img, 'gt_semantic_seg': gt_seg[:, :, np.newaxis],
            'img_shape': img.shape, 'ori_shape': img.shape, 'pad_shape': img.shape,
            'filename': info['filename'], 'ori_filename': info['filename'],
            'scale_factor': 1.0, 'flip': False, 'flip_direction': None,
            'img_norm_cfg': {'mean': [0]*4, 'std': [1]*4, 'to_rgb': False},
            'seg_fields': ['gt_semantic_seg'],
        }
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        return self.__getitem__(idx)

    def get_gt_seg_maps(self, efficient_test=False):
        import sys; sys.path.insert(0, '/tmp/pypackages')
        import nibabel as nib
        for info in self.img_infos:
            seg_vol = nib.load(osp.join(info['case_dir'], f'{info["case_id"]}-seg.nii.gz')).get_fdata().astype(np.uint8)
            yield seg_vol[:, :, info['slice_idx']]
