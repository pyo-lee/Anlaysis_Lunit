import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,roc_curve,auc
import ast

def convert_dict_to_array(contour):
    arr = [[row["x"], row["y"]] for row in contour]
    arr = np.array(arr)
    return arr

def CreateFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def json_open(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def AUC_score(label,prob):
    fp,tp,_=roc_curve(label,prob)
    auc_score = auc(fp,tp)
    return auc_score

def json_gt_open(path):
    contours = []
    labels = []
    with open(path, "r") as f:
        data = json.load(f)
        keys = data.keys()
        
        if 'abnormal_finding' in keys:
            abnormal_finding = data['abnormal_finding']
            for i in range(len(abnormal_finding)):
                abnormal_keys = abnormal_finding[i].keys()

                if 'contour_list' in abnormal_keys:
                    contour = abnormal_finding[i]['contour_list']
                    label = abnormal_finding[i]['label_text']
                    contours.append(contour)
                    labels.append(label)
            
        w = data['width']
        h = data['height']

    return contours, labels, w, h

def json_opt_open(path):
    contours = []
    labels = []
    contourIds = []
    with open(path, "r") as f:
        data = json.load(f)
        rating_list = ast.literal_eval(data['rating_list'])
        rates = []
        rate = 0
        if rating_list!=[]:
            for rate in rating_list:
                rating_tmp = rate['rating'] 
                rates.append(rating_tmp)
            rate = max(rates)
        
        if data['contour_list']:
            contour_list = ast.literal_eval(data['contour_list'])
            
            for i in range(len(rating_list)):
                contourId = rating_list[i]['contourId']
                contour = contour_list[contourId]
                contourIds.append(contourId)
                contours.append(contour)
                        
    return contours, contourIds, rating_list, int(rate)

def json_ai_open(path):
    with open(path, "r") as f:
        data = json.load(f)
        prob = data['pos_prob']
    
    return prob

def json_ai_map_open(path):
    with open(path, "r") as f:
        data = json.load(f)
        prob_map = data['pos_map']
        prob = data['pos_prob']
    
    return prob_map, prob

def cutoff(data_np_cutoff, cutoff):
    
    data_np_cutoff[data_np_cutoff>cutoff]=255
    data_np_cutoff[data_np_cutoff<cutoff]=0
    
    return data_np_cutoff
