"""Image-predicted building affinity; instance IDs are used only by the loss."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


OFFSETS = tuple((y, x) for y in range(-2, 3) for x in range(-2, 3) if (y, x) != (0, 0))
DILATIONS = (1, 4, 8)


def shifted(value, dy, dx):
    h, w = value.shape[-2:]
    py, px = abs(dy), abs(dx)
    padded = F.pad(value, (px, px, py, py))
    return padded[..., py + dy : py + dy + h, px + dx : px + dx + w]


class BuildingAffinityContext(nn.Module):
    def __init__(self, channels, embedding=16):
        super().__init__()
        self.embedding = nn.Conv2d(channels, embedding, 1)
        self.log_scale = nn.Parameter(torch.full((3,), math.log(math.expm1(5.0))))
        self.bias = nn.Parameter(torch.full((3,), -2.5))

    def forward(self, feature, footprint_weight):
        embedding = F.normalize(self.embedding(feature).float(), dim=1, eps=1e-6)
        feature = feature.float()
        contexts, all_logits = [], []
        for scale, dilation in enumerate(DILATIONS):
            # Bind the scale: checkpoint recomputes this function during backward.
            def aggregate(f, e, w, s=scale, d=dilation):
                return self.aggregate(f, e, w, s, d)

            if self.training and torch.is_grad_enabled():
                context, logits = checkpoint(
                    aggregate, feature, embedding, footprint_weight, use_reentrant=False
                )
            else:
                context, logits = aggregate(feature, embedding, footprint_weight)
            contexts.append(context)
            all_logits.append(logits)
        return contexts, all_logits

    def aggregate(self, feature, embedding, footprint_weight, scale, dilation):
        support = torch.ones_like(footprint_weight)
        numerator = feature * footprint_weight
        denominator = footprint_weight
        logits = []
        for oy, ox in OFFSETS:
            dy, dx = oy * dilation, ox * dilation
            neighbour = shifted(embedding, dy, dx)
            logit = (embedding * neighbour).sum(1, keepdim=True)
            logit = logit * F.softplus(self.log_scale[scale]) + self.bias[scale]
            weight = (
                logit.sigmoid()
                * shifted(footprint_weight, dy, dx)
                * shifted(support, dy, dx)
            )
            numerator = numerator + weight * shifted(feature, dy, dx)
            denominator = denominator + weight
            logits.append(logit)
        return numerator / denominator.clamp_min(1e-6), torch.cat(logits, dim=1)


def affinity_supervision(logits, instance, valid):
    """Equal weight for observed same/different-building classes per scale.

    Mixed feature-grid cells, invalid heights and background are excluded.
    Tile-local IDs are compared only inside each image, never across batches.
    """
    shape = logits[0].shape[-2:]
    if instance.shape != valid.shape:
        raise ValueError("Instance IDs and validity must have identical shapes")
    ids = instance.float()
    maximum = F.adaptive_max_pool2d(ids, shape)
    minimum = -F.adaptive_max_pool2d(-ids, shape)
    reliable = -F.adaptive_max_pool2d(-valid.float(), shape) > 0.5
    reliable = reliable & (maximum == minimum) & (minimum > 0)
    total = logits[0].sum() * 0.0
    scale_losses, positive_count, negative_count = [], 0, 0
    positive_hits = total.detach()
    negative_hits = total.detach()
    for values, dilation in zip(logits, DILATIONS):
        positive_sum, negative_sum = values.sum() * 0.0, values.sum() * 0.0
        npairs, nn_pairs = 0, 0
        for index, (oy, ox) in enumerate(OFFSETS):
            dy, dx = oy * dilation, ox * dilation
            eligible = reliable & shifted(reliable.float(), dy, dx).bool()
            same = minimum == shifted(minimum, dy, dx)
            pos, neg = eligible & same, eligible & ~same
            score = values[:, index : index + 1].float()
            positive_sum = positive_sum + F.softplus(-score)[pos].sum()
            negative_sum = negative_sum + F.softplus(score)[neg].sum()
            npos, nneg = int(pos.sum()), int(neg.sum())
            npairs += npos
            nn_pairs += nneg
            positive_hits = positive_hits + ((score.detach() >= 0) & pos).sum()
            negative_hits = negative_hits + ((score.detach() < 0) & neg).sum()
        terms = []
        if npairs:
            terms.append(positive_sum / npairs)
        if nn_pairs:
            terms.append(negative_sum / nn_pairs)
        if terms:
            scale_losses.append(torch.stack(terms).mean())
        positive_count += npairs
        negative_count += nn_pairs
    loss = torch.stack(scale_losses).mean() if scale_losses else total
    return loss, positive_count, negative_count, positive_hits, negative_hits
