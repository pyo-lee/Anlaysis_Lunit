#-*- coding:utf-8 -*-

import os
import pandas as pd
import json
import dicom_handler
import numpy as np
from PIL import Image
import cv2
import metric.metric as metric
import pprint
import argparse
import pickle


UINT8_MAX = np.iinfo(np.uint8).max
UINT16_MAX = np.iinfo(np.uint16).max

#from visual import make_overlay_image, make_overlay_image2
#import visual as visual_lib

def convert_dict_to_array(contour):
    arr = [[row["x"], row["y"]] for row in contour]
    arr = np.array(arr)
    return arr


def get_masks(contours, width, height, dtype=np.float32, mask_size=1000):
    assert isinstance(contours, list) or isinstance(contours, tuple)

    images = []
    for contour in contours:
        for conts in contour:
            images.append(get_mask(conts, width, height, dtype, mask_size=mask_size))

    return images


def get_mask(contour, width, height, dtype=np.float32, mask_size=1000):
    if width < height:
        mask_height = mask_size
        mask_width = int(mask_size * width / height)
    else:
        mask_width = mask_size
        mask_height = int(mask_size * height / width)

    image = np.zeros((mask_height, mask_width), np.uint8)

    assert len(contour) > 0

    for subcontour in contour:
        assert len(subcontour) > 0
        pts = np.array(
            [[[(x + width / 2) / width * mask_width, (y + height / 2) / height * mask_height]] for x, y in subcontour],
            np.int32)
        cv2.fillPoly(image, [pts], (255, 255, 255))
    return convert_image_type(image, dtype)


def convert_image_type(image, dtype=np.float32):
    if image.dtype == np.uint8:

        if dtype == np.float32:
            image = image.astype(np.float32)
            image /= UINT8_MAX
            return image
        elif dtype == np.uint8:
            return image
        else:
            raise TypeError('numpy.float32 or numpy.uint8 supported as a target dtype')

    elif image.dtype == np.uint16:

        if dtype == np.float32:
            image = image.astype(np.float32)
            image /= UINT16_MAX
            return image
        elif dtype == np.uint8:
            image = image.astype(np.float32)
            image *= UINT8_MAX / UINT16_MAX
            image = image.astype(np.uint8)
            return image
        elif dtype == np.uint16:
            return image
        else:
            raise TypeError('numpy.float32 or numpy.uint8 or numpy.uint16 supported as a target dtype')

    else:
        raise TypeError('numpy.uint8 or numpy.uint16 supported as an input dtype')

def get_human_output(pixel_array, human_data):
    _dict = eval(human_data['contour_list'])
    _ratings = eval(human_data['rating_list'])

    height, width = pixel_array.shape

    if _ratings:
        masks = []
        for _key in _dict.keys():
            _contour = _dict[_key]

            for _rating in _ratings:
                if _rating['contourId']  == _key:
                    lesion_rating = _rating['rating']

            arr = convert_dict_to_array(_contour)

            arr[..., :, 0] = arr[:, 0] * width - width / 2
            arr[..., :, 1] = arr[:, 1] * height - height / 2

            arr = arr.astype(np.int64)

            arr = np.expand_dims(arr, 0)
            arr = np.expand_dims(arr, 0)

            mask = sum(get_masks([arr], width=width, height=height, mask_size=max(width, height)))
            mask = mask * (float(lesion_rating)/5)
            masks.append(mask)

        final_mask = np.stack(masks, axis=0).max(0)

    else:
        final_mask = np.zeros((height, width), dtype=np.float32)

    return final_mask

def get_gt_final_mask(data):
    if 'abnormal_finding' in data.keys() and data['abnormal_finding']:
        masks = []
        for _dict in data['abnormal_finding']:
#             if _dict['label_text'] in mca_list:
            if _dict['label_text']:
                for contour_key in _dict['contour_list'].keys():
                    contour = _dict['contour_list'][contour_key]
                    arr = convert_dict_to_array(contour)
                    # print(width, height)
                    # print(arr)

                    # print(width, height)
                    # print(arr)

                    arr[..., :, 0] = arr[:, 0]
                    arr[..., :, 1] = arr[:, 1]
                    arr = arr.astype(np.int64)

                    arr = np.expand_dims(arr, 0)
                    arr = np.expand_dims(arr, 0)

                    mask = sum(get_masks([arr], width=width, height=height, mask_size=max(width, height)))
                    masks.append(mask)
            else:
                mask = np.zeros((height, width), dtype=np.float32)
                masks.append(mask)

        final_mask = np.stack(masks, axis=0).max(0)
    else:
        final_mask = np.zeros((height, width), dtype=np.float32)
        
    return final_mask


mapping_df = pd.read_csv('brmh_1_mapping_table.csv')
mapping_cases = mapping_df['case_no'].tolist()

interest_list = []
for i in range(24):
    interest_list.append('u{}/t2'.format(str(i+2)))

#interest_list = ['u2/t2']


with open('jafroc_respiratory(mca).txt', 'w') as csvfile:
    for interest_dir in interest_list:
        gt_masks = []
        human_masks = []
        for index, file_name in enumerate(mapping_cases):
            if file_name.split('-')[0] == 'B':
                hospital_name = 'brmh'
            elif file_name.split('-')[0] == 'K':
                hospital_name = 'kyuh'
            elif file_name.split('-')[0] == 'G':
                hospital_name = 'gugh'
            else:
                raise ValueError('invalid hospital name')

            json_root_path = '/storage2/ctr/original/cxr/external_validation/BRMH-GIL-KONYANG/BRMH-GIL-KONYANG/review_result/{}-A1/{}/respiratory'.format(hospital_name.upper(), hospital_name)
            json_file = os.path.join(json_root_path, (file_name+'.dcm.json'))

            heatmap_root_path = '/storage2/ctr/original/cxr/external_validation/BRMH-GIL-KONYANG/{}/respiratory'.format(hospital_name)

            with open(json_file, "r") as f:
                data = json.load(f)

            handler = dicom_handler.get_handler(os.path.join(heatmap_root_path, (file_name+'.dcm')), modality='CXR')
            pixel_array = handler.pixels
            pixel_array = (pixel_array * 255).astype(np.uint8)
            height, width = pixel_array.shape

            mca_list = ['강화', '결절 / 질량', 'Nodule / Mass', 'Consolidation']
            gt_masks.append(get_gt_final_mask(data))

            human_root_path = '/lunit/home/hyunsuky/opt_pilot/brmh_opt2/annotation_1'
            human_json_name = str(mapping_df['seq'].tolist()[index]) + '.json'
            human_json_full = os.path.join(human_root_path, interest_dir, human_json_name)
            with open(human_json_full, "r") as f:
                human_data = json.load(f)

            human_masks.append(get_human_output(pixel_array, human_data))

        new_shape = (512, 512)
        human_outputs = [cv2.resize(np.asarray(human_mask), new_shape, interpolation=cv2.INTER_NEAREST) for human_mask in human_masks]

        gt_masks = [gt_mask.astype(bool) for gt_mask in gt_masks]


        resized_human_outputs = []
        for index, human_output in enumerate(human_outputs):
            resized_human_outputs.append(np.resize(human_output, gt_masks[index].shape))

        jafroc_value = metric.jafroc(resized_human_outputs, gt_masks)
        print("jafroc:\t{:.2f}".format(jafroc_value), file=csvfile)

        jaf_ci = metric.bootstrap_jafroc_ci(resized_human_outputs, gt_masks, n_bootstraps=100, alpha=0.05, rng_seed=123)
        print(jaf_ci, file=csvfile)

csvfile.close()
