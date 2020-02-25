import torch
import math


def convert_locations_to_corner_boxes(locations, priors, coordinate_variance):
    """Convert regressional location results of SSD into boxes in the form of (topleft_x, topleft_y, bottomright_x, bottomright_y).

    The conversion:
        REF: $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        REF: $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        REF: center_variance: a float used to change the scale of center.
        REF: size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[tl_x, tl_y, br_x, br_y]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)

    prior_sizes = torch.abs( priors[..., 2:] - priors[..., :2] ) 

    return torch.cat([
        ( locations[..., :2] * coordinate_variance * prior_sizes ) + priors[..., :2],
        ( locations[..., 2:] * coordinate_variance * prior_sizes ) + priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_corner_boxes_to_locations(corner_form_boxes, corner_form_priors, coordinate_variance):
    # priors can have one dimension less
    if corner_form_priors.dim() + 1 == corner_form_boxes.dim():
        corner_form_priors = corner_form_priors.unsqueeze(0)

    prior_sizes = torch.abs( corner_form_priors[..., 2:] - corner_form_priors[..., :2] ) 

    return torch.cat([
        ( corner_form_boxes[..., :2] - corner_form_priors[..., :2]) / prior_sizes / coordinate_variance,
        ( corner_form_boxes[..., 2:] - corner_form_priors[..., 2:]) / prior_sizes / coordinate_variance
    ], dim=corner_form_boxes.dim() - 1 )


def area_of_corner_boxes( left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of_corner_boxes(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])
    
    overlap_area = area_of_corner_boxes(overlap_left_top, overlap_right_bottom)
    area0 = area_of_corner_boxes(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of_corner_boxes(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors_corner_boxes(gt_boxes, gt_labels, corner_form_priors,
                iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    ious = iou_of_corner_boxes(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels


def hard_negative_mining_corner_boxes(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
        cut the number of negative predictions to make sure the ratio
        between the negative examples and positive examples is no more
        the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask

"""
def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                        locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


import numpy as np

if __name__ == '__main__':
    locations = np.array( [ [ [ 200, 200, 750, 750 ], [ 250, 250, 700, 700 ] ] ] ) / np.array( [ 1000, 800, 1000, 800 ] )
    locations = torch.from_numpy( locations )
    
    priors = np.array( [ [ 250, 250, 750, 750 ], [ 100, 100, 900, 900 ] ] ) / np.array( [ 1000, 800, 1000, 800 ] )
    priors = torch.from_numpy( priors )

    corner_boxes = convert_locations_to_corner_boxes( locations, priors, 0.5  ) 
    location_boxes = convert_corner_boxes_to_locations( corner_boxes, priors, 0.5 )
    print( 'BOXES Center2Corners conversions', torch.sum( locations - location_boxes ) )
"""
