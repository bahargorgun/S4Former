import torch
import torch.nn as nn
import torch.nn.functional as F


class UAPRModule(nn.Module):
    def __init__(self, n_samples=10, dropout_p=0.3):
        super().__init__()
        self.n_samples = n_samples
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, logits):
        samples = []
        for _ in range(self.n_samples):
            probs = F.softmax(self.dropout(logits), dim=1)
            samples.append(probs)
        samples = torch.stack(samples, dim=0)
        mean_probs = samples.mean(dim=0)
        variance = samples.var(dim=0).mean(dim=1)
        uncertainty_weight = torch.exp(-variance)
        return uncertainty_weight, mean_probs


class ClassAdaptiveThreshold:
    def __init__(self, num_classes=4, class_thresholds=None):
        self.num_classes = num_classes
        self.class_thresholds = class_thresholds or {
            0: 0.95, 1: 0.75, 2: 0.80, 3: 0.70
        }

    def get_mask(self, probs):
        max_probs, hard_label = probs.max(dim=1)
        conf_mask = torch.zeros_like(max_probs, dtype=torch.bool)
        for c in range(self.num_classes):
            thr = self.class_thresholds.get(c, 0.95)
            conf_mask |= (hard_label == c) & (max_probs > thr)
        pseudo_label = hard_label.clone()
        pseudo_label[~conf_mask] = 255
        return conf_mask, pseudo_label
