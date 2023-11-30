import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
import geohash_hilbert as ghh
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)
#======
# make sure you have the right folder to the images and labels 
# for reference labels here are just laser made masks
class KdTree:
    def __init__(self, bath_to_directory, parent_precision=4, child_precision=6, train_split=0.1, path_images='/Users/ridhaalkhabaz/Documents/mlds/images/', path_labels='/Users/ridhaalkhabaz/Documents/mlds/labels/'):
        ## please note that our
        self.direc = gpd.read_file(bath_to_directory).set_crs(3443, allow_override=True).to_crs(4326) #please not 3443 projection key is only becasue samples.geojson is corrupted 
        self.direc['center'] = self.direc['geometry'].centroid
        self.parent_precision = parent_precision
        self.child_precision = child_precision
        self.tree = {}
        self.tree = self._init_tree()
        self.path_images = path_images
        self.path_labels = path_labels
    def _init_tree(self):
        part = {}
        n = len(self.direc)
        for i in range(n):
            ## find the record we hash 
            rec = self.direc.iloc[i]
            center = rec['center']
            ## getting the lon, lat for the sample 
            lng, lat = center.x, center.y
            ## geohash our sample 
            hash_child = ghh.encode(lng, lat, precision=self.child_precision)
            hash_parent = ghh.encode(lng, lat, precision=self.parent_precision)
            ## index the parent node if it does not exist 
            if part.get(hash_parent) is None:
                part[hash_parent] = {}
            ## index the parent node if it does not exist 
            if part[hash_parent].get(hash_child) is None:
                part[hash_parent][hash_child] = i 
        return part
    def _find_train_test_indxs(self, split):
        train_split = split 
        trainig_input = []
        for key in self.tree:
            sub_tree = self.tree.get(key)
            subtree_n  = len(sub_tree)
            num_recs = int(train_split*subtree_n)
            keys_list = list(sub_tree.keys())
            samples_kys = np.random.choice(keys_list, num_recs)
            samples_indx = [sub_tree.get(ky) for ky in samples_kys]
            trainig_input.extend(samples_indx)
        return trainig_input
    def _get_item_idx(self, key):
        if len(key) > self.parent_precision:
            parent_key = key[:self.parent_precision]
            desired = self.tree[parent_key].get(key)
            return desired 
        return 'check input'
    def _show_example(self, key, show_label=True):
        indx = self._get_item_idx(key)
        path_to_label = self.path_labels+'label_'+str(indx)+'.png'
        path_to_image = self.path_images+'imag_'+str(indx)+'.png'
        if show_label:
            label = np.array(Image.open(path_to_label))
            label[label > 0] = 255
            plt.imshow(label)
            plt.show()
        else:
            plt.imshow(Image.open(path_to_image))
            plt.show()
    def _get_subtree_indxs(self, key, breadth_search=False):
        desired_kys = [key]
        desired = []
        if breadth_search:
            desired_kys.extend(list(ghh.neighbours(key).values()))
        for key in desired_kys:
            desired.extend(self.tree[key].values())
        return desired 
    def _get_key(self, idx):
        if not isinstance(idx, int):
            return None 
        for key in self.tree.keys():
            if idx in list(self.tree[key].values()):
                for ky, indx in self.tree[key].items():
                    if idx == indx:
                        return ky
        return None 
    def _get_subtree_sample_indx(self, key, ratio):
        indices = list(self.tree[key].values())
        n = len(indices)
        n_sams = int(ratio*n)
        return list(np.random.choice(indices, n_sams))
    

#-----
# make sure you have the right folder to the images and labels 
class filter_data:
    def __init__(self, path_to_metadata='./buildings/samples_bld.geojson', path_to_texture='./samples_texture.geojson', path_to_images= '/Users/ridhaalkhabaz/Documents/mlds/images/'):
        self.df_meta = gpd.read_file(path_to_metadata)
        self.labels = [self._binary_map(indx) for indx in range(len(self.df_meta))]
        self.df_meta['label'] = self.labels
        self.df_text = gpd.read_file(path_to_texture)
        self.df_text['label'] = self.labels 
        self.imgs_path = path_to_images 
        self.num_recs = len(self.df_meta)
    def _binary_map(self, idx):
        if self.df_meta.iloc[idx]['FID']>0:
            return 1
        return 0
    def _get_label(self, idx):
        return self.df_meta.iloc[idx]['label']
    def _get_log_data(self, training, indices, indices_test):
        cols = list(self.df_text.columns)
        cols.remove('label')
        cols.remove('geometry')
        if training:
            df_train = self.df_text.loc[indices]
            df_test = self.df_text.loc[indices_test]
            x_train = np.array(df_train[cols], dtype=np.float32)
            x_test = np.array(df_test[cols], dtype=np.float32)
            y_train = np.array(df_train['label'], dtype=np.float32)
            y_test = np.array(df_test['label'], dtype=np.float32)
            return torch.tensor(x_train), torch.tensor(y_train), torch.tensor(x_test), torch.tensor(y_test)
        df_res = self.df_text.loc[indices]
        res = np.array(df_res[cols], dtype=np.float32)
        return torch.tensor(res)
    def _img_to_tensor(self, idx):
        img_path = self.imgs_path+'imag_'+str(idx)+'.png'
        img = np.array(Image.open(img_path))
        img_tensor = torch.tensor(img).to(torch.float32)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        return torch.unsqueeze(img_tensor, 0)
    