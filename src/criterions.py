import torch 
import torch.nn as nn
import torch.nn.functional as F

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            # Ensure it's a tensor and register as a buffer for automatic device moving
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.register_buffer('pos_weight', pos_weight_tensor)
        else:
            self.pos_weight = None

    def forward(self, inputs, targets, mask):
        # Compute raw BCE loss
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            reduction='none',
            pos_weight=self.pos_weight
        )
        
        # Apply the mask (mask is 1 for valid residues, 0 for padding)
        loss = loss * mask 
        
        # Mean only over valid positions
        return loss.sum() / (mask.sum() + 1e-8)


class MaskedMCC(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        # Explicitly set dtype to float32
        self.register_buffer('TP', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('TN', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('FP', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('FN', torch.tensor(0.0, dtype=torch.float32))

    def update(self, preds, targets, mask):
        # Ensure preds are thresholded and types match
        y_pred = (preds > self.threshold).float()
        targets = targets.float()
        mask = mask.float()

        self.TP += (y_pred * targets * mask).sum()
        self.TN += ((1 - y_pred) * (1 - targets) * mask).sum()
        self.FP += (y_pred * (1 - targets) * mask).sum()
        self.FN += ((1 - y_pred) * targets * mask).sum()

    def compute(self):
        numerator = (self.TP * self.TN) - (self.FP * self.FN)
        denominator = torch.sqrt(
            (self.TP + self.FP) *
            (self.TP + self.FN) *
            (self.TN + self.FP) *
            (self.TN + self.FN)
        )
        return numerator / (denominator + 1e-8)

    def reset(self):
        self.TP.zero_()
        self.TN.zero_()
        self.FP.zero_()
        self.FN.zero_()

# Repeat similar register_buffer logic for MaskedRecall and MaskedPrecision
class MaskedRecall(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.register_buffer('true_positives', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('actual_positives', torch.tensor(0.0, dtype=torch.float32))

    def update(self, preds, targets, mask):
        y_pred = (preds > self.threshold).float()
        self.true_positives += (y_pred * targets * mask).sum()
        self.actual_positives += (targets * mask).sum()

    def compute(self):
        return self.true_positives / (self.actual_positives + 1e-8)

    def reset(self):
        self.true_positives.zero_()
        self.actual_positives.zero_()

class MaskedPrecision(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.register_buffer('true_positives', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('predicted_positives', torch.tensor(0.0, dtype=torch.float32))

    def update(self, preds, targets, mask):
        y_pred = (preds > self.threshold).float()
        self.true_positives += (y_pred * targets * mask).sum()
        self.predicted_positives += (y_pred * mask).sum()

    def compute(self):
        return self.true_positives / (self.predicted_positives + 1e-8)

    def reset(self):
        self.true_positives.zero_()
        self.predicted_positives.zero_()