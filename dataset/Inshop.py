# https://github.com/htdt/hyp_metric/blob/master/proxy_anchor/dataset/Inshop.py
from .base import *

import numpy as np, os, sys, pandas as pd, csv, copy
import torch
import torchvision
import PIL.Image


class Inshop_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None, occlusion_type=None):
        self.root = root + '/Inshop_Clothes'
        self.mode = mode
        self.transform = transform
        self.train_ys, self.train_im_paths = [], []
        self.query_ys, self.query_im_paths = [], []
        self.gallery_ys, self.gallery_im_paths = [], []
        
        data_info = np.array(pd.read_table(self.root +'/Eval/list_eval_partition.txt', header=1, delim_whitespace=True))[:,:]
        #Separate into training dataset and query/gallery dataset for testing.
        train, query, gallery = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]

        #Generate conversions
        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
        train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])

        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
        query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
        gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

        #Generate Image-Dicts for training, query and gallery of shape {class_idx:[list of paths to images belong to this class] ...}
        for img_path, key in train:
            self.train_im_paths.append(os.path.join(self.root, 'Img', img_path))
            self.train_ys += [int(key)]

        for img_path, key in query:
            if occlusion_type is None:
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path)) # when inferencing with occluded data, only query image is occluded!
            elif occlusion_type == 'black_mask_small':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "img_black")) # when inferencing with occluded data, only query image is occluded!
            elif occlusion_type == 'black_mask_big_upl':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "img_upperleft"))
            elif occlusion_type == 'black_mask_big_upr':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "img_upperright"))
            elif occlusion_type == 'black_mask_big_lowl':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "img_lowerleft"))
            elif occlusion_type == 'black_mask_big_lowr':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "img_lowerright"))
            elif occlusion_type == 'white_mask':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "img_white")) # when inferencing with occluded data, only query image is occluded!
            elif occlusion_type == 'black_box_50':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "occluded_img")) # when inferencing with occluded data, only query image is occluded!
            elif occlusion_type == 'black_box_30':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "mid_30_30_blackbox_inshop_img"))
            elif occlusion_type == 'random_black_box':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "80_100_140_160_rand_occluded_img")) # when inferencing with occluded data, only query image is occluded!
            elif occlusion_type == 'object_big':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "object_occluded_img"))
            elif occlusion_type == 'object_small':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "small_object_occluded_img"))
            elif occlusion_type == 'hand_big':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "hand_occluded_img"))
            elif occlusion_type == 'hand_small':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "small_hand_occluded_img"))
            elif occlusion_type == 'random':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "rand_merge_occluded_img"))

            # Final Version    
            elif occlusion_type == '10_bottom':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio10_bottom_img"))
            elif occlusion_type == '10_center':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio10_center_img"))
            elif occlusion_type == '10_top':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio10_top_img"))
            elif occlusion_type == '10_random':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio10_random_object_img"))
                
            elif occlusion_type == '20_center':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio20_center_object_img"))
            elif occlusion_type == '20_random':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio20_random_object_img"))
            elif occlusion_type == '20_top':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio20_top_object_img"))
            elif occlusion_type == '20_bottom':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio20_bottom_object_img"))


            elif occlusion_type == '5_bottom':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio5_bottom_object_img"))
            elif occlusion_type == '5_center':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio5_center_img"))
            elif occlusion_type == '5_random':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio5_random_img"))
            elif occlusion_type == '5_top':
                self.query_im_paths.append(os.path.join(self.root, 'Img', img_path).replace("img", "inshop_ratio5_top_img"))
            else:
                raise ValueError("Unavailable occlusion type for Inshop Datset!!")


            self.query_ys += [int(key)]

        for img_path, key in gallery:
            self.gallery_im_paths.append(os.path.join(self.root, 'Img', img_path))
            self.gallery_ys += [int(key)]
            
        if self.mode == 'train':
            self.im_paths = self.train_im_paths
            self.ys = self.train_ys
        elif self.mode == 'query':
            self.im_paths = self.query_im_paths
            self.ys = self.query_ys
        elif self.mode == 'gallery':
            self.im_paths = self.gallery_im_paths
            self.ys = self.gallery_ys

    def nb_classes(self):
        return len(set(self.ys))
            
    def __len__(self):
        return len(self.ys)
            
    def __getitem__(self, index):
        
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im
        
        im = img_load(index)
        target = self.ys[index]

        return im, target