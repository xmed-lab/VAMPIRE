from prompt_points import label_to_point_prompt_local, label_to_point_prompt_global
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import random
import numpy as np
from collections import *

import albumentations as alb
from tqdm import tqdm
import csv

from octa_preprocess import get_points


class octainhouse_2d_dataset(Dataset):
    def __init__(self, root, isTraining=True):
        super(octainhouse_2d_dataset,self).__init__()
        self.root = root
        self.isTraining = isTraining
        self.allItems = self.getAllPath(root, isTraining)
        
        prob = 0.3
        self.transform = alb.Compose([
            alb.RandomBrightnessContrast(p=prob),
            alb.CLAHE(p=prob), 
            alb.VerticalFlip(p=prob),
            alb.HorizontalFlip(p=prob),
            alb.AdvancedBlur(p=prob),
        ])

    def __len__(self):
        return len(self.allItems)

    def __getitem__(self, index):
        imageid = self.allItems[index]
        image = []
        image_paths = [os.path.join(self.root, 'Superficial', imageid), os.path.join(self.root, 'Deep', imageid), os.path.join(self.root, 'Choriocapillaris', imageid)]
        for image_path in image_paths:
            image.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

        pos_points, neg_points = get_points(image[0])
        
        image = np.array(image)
        
        if self.isTraining:
            image = self.transform(**{"image": image.transpose((1,2,0))})["image"].transpose((2,0,1))
            
        prompt_type = np.array([1] * len(pos_points) + [0] * len(neg_points))
        prompt_points = np.array(pos_points + neg_points)
        
        return image, prompt_points, prompt_type, imageid

    def getAllPath(self, root, isTraining):
        items = []
        filePath = '/home/lwangdk/Enface_FSSY-220825-221230/v1_6binary/folds/train.csv'

        with open(filePath,'r') as csvFile:
            reader = csv.reader(csvFile)
            header = next(reader)
            for item in reader:
                left_right = item[0].split('_')[0].split('-')[-1]
                front_back = item[0][:-4].split('_')[-1]

                if left_right == '0' and front_back == 'R':         
                    items.append(item[0])
                elif left_right == '1' and front_back == 'L':
                    items.append(item[0])
        
        filePath = '/home/lwangdk/Enface_FSSY-220825-221230/v1_6binary/folds/test.csv'
        with open(filePath,'r') as csvFile:
            reader = csv.reader(csvFile)
            header = next(reader)
            for item in reader:
                left_right = item[0].split('_')[0].split('-')[-1]
                front_back = item[0][:-4].split('_')[-1]
                if left_right == '0' and front_back == 'R':         
                    items.append(item[0])
                elif left_right == '1' and front_back == 'L':
                    items.append(item[0])
        
        return items
    