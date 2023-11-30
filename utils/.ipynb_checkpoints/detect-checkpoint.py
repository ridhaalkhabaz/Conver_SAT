### imports 
import os
import sys 
import cv2
import torch 
import geopandas as gpd
from PIL import Image

from torch.utils.data import Dataset, DataLoader


#- class for tiny models images -> make sure to change the folders to appropiate fashion. 
class detect_dataset(Dataset):
    def __init__(self, indices, transform=None, metadata_path='./buildings/samples_bld.geojson', path_images= '/Users/ridhaalkhabaz/Documents/mlds/images/', path_labels='/Users/ridhaalkhabaz/Documents/mlds/labels/'):
        self.indices = indices
        self.data = []
        self.imgs_path = path_images
        self.labs_path = path_labels
        self.df_meta = gpd.read_file(metadata_path)
        self.transform = transform
        for indx in indices:
            # path_to_label = self.labs_path+'label_'+str(indx)+'.png'
            path_to_image = self.imgs_path+'imag_'+str(indx)+'.png'
            label_ind = self.df_meta.iloc[indx]['FID']
            label = 1 if label_ind > 0 else 0
            self.data.append([path_to_image, label])
        self.img_dim = (224, 224)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        # img = np.expand_dims(img, axis=0)
        class_id = class_name
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1)
        if self.transform:
            self.transform(img_tensor)          
        class_id = torch.tensor([class_id])
        return img_tensor, class_id