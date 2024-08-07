from torch.utils.data import Dataset
import os
from PIL import Image
import imageio
import numpy as np
import torch
import json

class ImageDataset(Dataset):

    def __init__(
        self,
        data_path,
        transform, 
        caption_path = "",
        train_set = True
    ):
        super(ImageDataset).__init__()
        self.data_path = data_path
        self.images = [os.path.join(data_path,x) for x in os.listdir(data_path) if x.endswith(".jpg")]
        self.transform = transform
        self.train_set = train_set
        if self.train_set:
            with open(caption_path, newline='') as jsonfile:
                self.caption_set = json.load(jsonfile)
            self.caption_list = []
            for i in self.images:
                name = i.split("/")[-1]
                for c in self.caption_set[name]:
                    self.caption_list.append((i, c))
  
    def __len__(self):
        #return len(self.images)
        if self.train_set:
            return len(self.caption_list)
        else:
            return len(self.images)
  
    def __getitem__(self,index):
        if self.train_set:
            image_name = self.caption_list[index][0]
            image = Image.open(image_name)
            if self.transform is not None:
                image = self.transform(image)
            image = image.expand(3, 224, 224)
            return image, self.caption_list[index][1]
        
        else:
            image_name = self.images[index]
            image = Image.open(image_name)
            if self.transform is not None:
                image = self.transform(image)
            image = image.expand(3, 224, 224)
            image_name = image_name.split('/')[-1].split('.')[-2]
            return image, image_name