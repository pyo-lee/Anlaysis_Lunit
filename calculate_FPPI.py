import os
import numpy as np
import cv2


gt_base = '/lunit/home/hyunsuky/new-insight-engine-cxr/insight-engine-cxr/test/large_test/results/2.4.11.0_healthcheck_gugh_fppi/gt_mask'
base = '/lunit/home/hyunsuky/new-insight-engine-cxr/insight-engine-cxr/test/large_test/results/2.4.11.0_healthcheck_gugh_fppi/ai_mask'
ai_mask_list = os.listdir(base)
sum_image = len(ai_mask_list)

fppi_cnt = 0
for index, ai_mask in enumerate(sorted(ai_mask_list)):
    # if index==10:
    #     break

    gt_file = '{}_gt.png'.format(ai_mask.split('_ai_mask')[0])
    gt_im = cv2.imread(os.path.join(gt_base, gt_file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)

    print("started reading gt file: {}".format(gt_file))

    im = cv2.imread(os.path.join(base, ai_mask))
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    ai_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if gt_im.sum() == 0:
        fppi_cnt += len(ai_contours)
    else:
        for ai_contour in ai_contours:
            _image = np.zeros_like(gt_im)
            _image = cv2.drawContours(_image, [ai_contour], 0, color=255, thickness=-1)
            if _image[gt_im>0].sum() == 0:
                fppi_cnt += 1
    print("intermediate sum ffpi cnt: {}".format(fppi_cnt))


print('Calculated FPPI: {}'.format(fppi_cnt/sum_image))



    # for ai_contour in ai_contours:
