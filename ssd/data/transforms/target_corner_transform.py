import numpy as np
import torch

from ssd.utils import cornerbox_utils


class SSDTargetCornerBoxTransform:
    def __init__(self, corner_form_priors, coordinate_variance, iou_threshold):
        #self.center_form_priors = center_form_priors
        self.corner_form_priors = corner_form_priors
        self.coordinate_variance = coordinate_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        
        boxes, labels = cornerbox_utils.assign_priors_corner_boxes(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        #boxes = box_utils.corner_form_to_center_form(boxes)
        locations = cornerbox_utils.convert_corner_boxes_to_locations(boxes, self.corner_form_priors, self.coordinate_variance)
       
        return locations, labels
