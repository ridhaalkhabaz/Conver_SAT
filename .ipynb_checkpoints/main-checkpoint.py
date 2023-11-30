# import 
import time 
import os
import sys
import cv2
import glob
import torch 
import scipy 
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import geopandas as gpd
import geohash_hilbert as ghh
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm 
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from ultralyticsplus import YOLO, render_result
## our scripts 
sys.path.insert(0, './utils/')
from indexing import *
from filtering_mods import *
from detect import *
torch.manual_seed(0)
np.random.seed(0)

def indexing(mod_type, scope_filtering=False, model_pre_train_path=None):
    ## indexing 
    res = {}
    keys = []
    begin_time = time.time()
    kdtree = KdTree('./buildings/samples_bld.geojson')
    indexing_time = time.time()
    indexing_period = indexing_time-begin_time
    res['indexing took'] = indexing_period
    meta_data = filter_data()
    reading_meta_time = time.time()
    meta_reading_period = reading_meta_time-indexing_time
    res['meta reading took'] = meta_reading_period
    # filter models training
    # log filter model 
    if mod_type == 'log':
        log_mod_class = binary_search_models('log',kdtree, meta_data)
        log_mod = log_mod_class.model
        log_fit_time = time.time()
        log_fit_period = log_fit_time-reading_meta_time
        res['log fitting took'] = log_fit_period
        keys = find_interestings_subtrees(kdtree, log_mod, 'log', meta_data, 0.1, 0.1)
        # print('=============================================')
        log_filter_set_size = len(keys)
        log_tp = meta_data.df_meta.loc[keys].label.sum()
        res['log filtered set size (our)'] = log_filter_set_size
        res['log filtering TP (our)'] = log_tp
        # print(log_filter_set_size)
        # print(log_tp)
        log_filtering_time = time.time()
        log_filtering_period = log_filtering_time-log_fit_time
        # print('=============================================')
        # print('log filtering took {}'.format(log_filtering_period))
        # print('=============================================')
        res['log filtering took (our)'] = log_filtering_period
        if scope_filtering:
            log_keys_scope = find_interestings_subtrees(kdtree, log_mod, 'log', meta_data, 0.1, 0.1, True)
            # print('=============================================')
            log_filter_set_size_scope = len(log_keys_scope)
            log_tp_scope = meta_data.df_meta.loc[log_keys_scope].label.sum()
            res['log filtered set size (Scope)'] = log_filter_set_size_scope
            res['log filtering TP (Scope)'] = log_tp_scope
            # # print(log_filter_set_size_scope)
            # # print(log_tp_scope)
            log_filtering_time_scope = time.time()
            log_filtering_period_scope = log_filtering_time_scope-log_filtering_time
            # print('=============================================')
            # print('log scope filtering took {}'.format(log_filtering_period_scope))
            # print('=============================================')
            res['log filtering took (Scope)'] = log_filtering_period_scope
    else: 
        # cnn log model 
        cnn_mod_class = binary_search_models('cnn',kdtree, meta_data)
        cnn_mod = cnn_mod_class.model
        cnn_fit_time = time.time()
        cnn_fit_period = cnn_fit_time-log_filtering_time_scope
        # print('=============================================')
        # print('cnn fitting took {}'.format(cnn_fit_period))
        # print('=============================================')
        res['cnn fitting took'] = cnn_fit_period
        keys = find_interestings_subtrees(kdtree, cnn_mod, 'cnn', meta_data, 0.2, 0.2)
        # print('=============================================')
        cnn_filter_set_size = len(keys)
        cnn_tp = meta_data.df_meta.loc[keys].label.sum()
        res['cnn filtered set size (Our)'] = cnn_filter_set_size
        res['cnn filtering TP (Our)'] = cnn_tp
        # print(cnn_filter_set_size)
        # print(cnn_tp)
        cnn_filtering_time = time.time()
        cnn_filtering_period = cnn_filtering_time-cnn_fit_time
        # print('cnn filtering took {}'.format(cnn_filtering_period))
        # print('=============================================')
        res['cnn filtering took (our)'] = cnn_filtering_period
        # print('=============================================')
        if scope_filtering:
            cnn_keys_scope = find_interestings_subtrees(kdtree, cnn_mod, 'cnn', meta_data, 0.155, 0.155, True)
            # print('=============================================')
            cnn_filter_set_size_scope = len(cnn_keys_scope)
            cnn_tp_scope = meta_data.df_meta.loc[cnn_keys_scope].label.sum()
            res['cnn filtered set size (Scope)'] = cnn_filter_set_size_scope
            res['cnn filtering TP (Scope)'] = cnn_tp_scope
            # print(cnn_filter_set_size_scope)
            # print(cnn_tp_scope)
            cnn_filtering_time_scope = time.time()
            cnn_filtering_period_scope = cnn_filtering_time_scope-cnn_filtering_time
            # print('cnn filtering took {}'.format(cnn_filtering_period_scope))
            # print('=============================================')
            res['cnn filtering took (Scope)'] = cnn_filtering_period_scope
    indexingfiltering = time.time()
    indexingfiltering_period = indexingfiltering-begin_time
    # print('indexing and filtering took {}'.format(indexingfiltering_period))
    # print('=============================================')
    # del log_mod, cnn_mod
    return res, keys

def detection(mod_type, indices, dataObj, pretrain_mod ='./results/detection/resnet34_best_sec.pt', img_path = '/Users/ridhaalkhabaz/Documents/mlds/images/' ):
    begin_time = time.time()
    out = 0
    if mod_type == 'tiny':
        transform = v2.Compose([
        # v2.Resize(256),
        # v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) # used the same thing for training 
        detec_dataset = detect_dataset(indices, transform=transform)
        detect_loader = DataLoader(detec_dataset, batch_size=32)
        detector = models.resnet34(pretrained=True)
        detector.fc = nn.Linear(detector.fc.in_features, 57)
        detector.load_state_dict(torch.load(pretrain_mod,map_location=torch.device('cpu')))
        detector.eval()
        count_exp = 0
        count_truth = 0
        # iter = 0
        for interst in tqdm(detect_loader):
            output = detector(interst[0])
            result = torch.argmax(output, dim=1)
            count_exp+=result.sum().detach().numpy()
            # iter +=1 
            # if iter %20:
            #     print(count_exp, count_truth)
        out += count_exp
    else:
        print('YOLO')
        count = 0
        model = YOLO(pretrain_mod)
        for ndx in tqdm(indices):   
            img = img_path+'imag_'+str(ndx)+'.png'
            results = model.predict(img)
            count += len(results[0].boxes)
        
        out += count
    truth = dataObj.df_meta.loc[indices].FID.sum()
    end_tim = time.time()
    period = end_tim-begin_time
    print('detection using {}'.format(mod_type))
    print('it took this amount of seconds {}'.format(period))
    return out, truth
        

res, idices = indexing('log')
print(res, len(idices))
dataObj = filter_data()
# count, truth = detection('yolov8', idices, dataObj, pretrain_mod ='keremberke/yolov8n-building-segmentation', img_path = '/Users/ridhaalkhabaz/Documents/mlds/images/' )
# print(count, truth)
c, tru = detection('tiny', idices, dataObj)
print(c, tru)
    # print(res)
    # filename = "iteration_results_num"+str(i) +".txt"
    # with open(filename, 'w') as f:
        
    #     for key, value in res.items():
            
    #         f.write('%s:%s\n' % (key, value))
    
    