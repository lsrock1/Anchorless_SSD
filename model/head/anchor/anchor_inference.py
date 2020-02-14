import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import nms


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms_topk(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = nms(boxes, scores, overlap)
    ids = torch.nonzero(keep).squeeze()
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    keep[: min(keep.shape[0], top_k)]
    return keep


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    xy_min = boxes[:, :2] - boxes[:, 2:]/2
    xy_max = xy_min + boxes[:, 2:]
    return torch.cat([xy_min, xy_max], dim=1)


class Postprocessor(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = 0
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg.VARIANCE

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        conf_data = F.softmax(conf_data, 2)
        num_priors = prior_data.size(0)

        conf_data = conf_data.view(-1, num_priors, self.num_classes).transpose(2, 1)
        prior_data = prior_data.view(-1, num_priors, 4).expand(conf_data.size(0), num_priors, 4)
        prior_data = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4), prior_data, self.variance)
        decoded_boxes = decoded_boxes.view(conf_data.size(0), -1, 4)
        
        output = []
        for localization, confidence in zip(decoded_boxes, conf_data):
            for class_idx in range(1, self.num_classes):
                confidence_per_class = confidence[class_idx]
                mask = torch.nonzero(confidence_per_class > self.conf_thresh).squeeze()
                scores = confidence_per_class[mask]
                if scores.size(0) == 0:
                    continue

                boxes = localization[mask].reshape(-1, 4)
                ids = nms_topk(boxes, scores, self.nms_thresh, self.top_k)
                output.append(torch.cat([scores[ids].unsqueeze(1), box_[ids]], 1))
        
        return output


def make_anchor_postprocessor(config):
    box_selector = PostProcessor(
        config.NUM_CLASSES,
        config.TEST.DETECTIONS_PER_IMG,
        config.ANCHOR.INFERENCE_TH,
        config.NMS_TH)

    return box_selector