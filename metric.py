import cv2
import numpy as np
import random
import pickle
import os
from sklearn import metrics
# from icecream import ic
from IPython import embed
# from imgcat import imgcat

def select(array, indices):
    selection = []
    for idx in indices:
        selection.append(array[idx])
    return selection

def jafroc(heatmaps, masks, version=2, relative_kernel_size=0.1, get_extra=False,threshold=0.15):

    # check arguments
    assert version in [1, 2]
    # heatmaps, masks = _check_heatmaps_and_masks(heatmaps, masks)

    # define pos/neg scores
    neg_image_scores = []    # neg index to score
    pos_contour_scores = [] # pos index to contour index to score
    for heatmap, mask in zip(heatmaps, masks):

        # heatmap, mask = _check_heat_map_and_mask(heatmap, mask)
        mask = mask.astype(np.uint8)
        if mask.sum() > 0: # this is postive,
            contours, _ = cv2.findContours(mask,
                                              cv2.RETR_EXTERNAL, # each external boundary matters only
                                              cv2.CHAIN_APPROX_SIMPLE)
            contour_scores = []
            for contour in contours: # and get max score for each contour
                image = np.zeros_like(mask)
                image = cv2.drawContours(image, [contour], 0, color=255, thickness=-1)
                contour_scores.append(heatmap[image > 0].max())
            pos_contour_scores.append(contour_scores)

            if version == 1:
                # extract the eroded negative region and add it to negative score list
                mask_neg = 1 - mask
                kernel_size = int(round(min(list(mask.shape)) * relative_kernel_size))
                assert kernel_size > 1, 'kernel size too small'
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_neg = cv2.erode(mask_neg, kernel)
                if mask_neg.any():
                    neg_image_scores.append(heatmap[mask_neg > 0].max())
        else:
            neg_image_scores.append(heatmap.max())

    if len(neg_image_scores) == 0 or len(pos_contour_scores) == 0:
        print('warning) all positives or all negatives, return 0')
        print(len(neg_image_scores), len(pos_contour_scores))
        if get_extra:
            extra = {"sensitivity": 0., "specificity": 0., "threshold": threshold}
            return 0., extra
        else:
            return 0.

    # compute jafroc
    jafroc = 0.
    for ns in neg_image_scores:
        for contour_scores in pos_contour_scores:
            for cs in contour_scores:
                jafroc += 1. / len(contour_scores) * ((ns < cs and 1.) or (ns == cs and .5) or 0.)
    jafroc_auc = jafroc / (len(neg_image_scores) * len(pos_contour_scores))

    if get_extra:
        # compute sensitivity and specificity
        sensitivity_list = []
        for contour_scores in pos_contour_scores:
            for cs in contour_scores:
                sensitivity_list.append(float(cs > threshold))
        sen = np.mean(sensitivity_list)

        specificity_list = []
        for ns in neg_image_scores:
            specificity_list.append(float(ns < threshold))
        spe = np.mean(specificity_list)

        extra = {"sensitivity":sen, "specificity":spe, "threshold":threshold}
        return jafroc_auc, extra
    else:
        return jafroc_auc

def bootstrap_jafroc_ci(heatmaps, masks, n_bootstraps, alpha, rng_seed):

    # ref: https://stackoverflow.com/a/19132400 bootstrap auc score
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(masks) - 1, len(masks))

        selected_masks = select(masks, indices)
        selected_heatmaps = select(heatmaps, indices)

        if len(np.unique([mask.max() for mask in selected_masks])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score, jafroc_extra = jafroc(selected_heatmaps, selected_masks, get_extra=True)
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int((alpha/2) * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1.0-(alpha/2)) * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))

    return (round(confidence_lower,3), round(confidence_upper,3))
