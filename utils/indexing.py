import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
import geohash_hilbert as ghh
import matplotlib.pyplot as plt
class BBTree:
    def __init__(self, bath_to_directory, parent_precision=4, child_precision=6, train_split=0.1, path_images='/Users/ridhaalkhabaz/Documents/mlds/images/', path_labels='/Users/ridhaalkhabaz/Documents/mlds/labels/'):
        ## please note that our
        self.direc = gpd.read_file(bath_to_directory).set_crs(3443, allow_override=True).to_crs(4326) #please not 3443 projection key is only becasue samples.geojson is corrupted 
        self.direc['center'] = self.direc['geometry'].centroid
        self.parent_precision = parent_precision
        self.child_precision = child_precision
        self.tree = {}
        self.tree = self._init_tree()
        self.train_split = train_split
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
    def _find_train_indxs(self):
        train_split = self.train_split 
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
        else:
            plt.imshow(Image.open(path_to_image))
        