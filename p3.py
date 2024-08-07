from tokenizers import Tokenizer
import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import timm
from dataset import ImageDataset
from types import SimpleNamespace
from timm.optim.optim_factory import create_optimizer
import json
from model import p2_model

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" 
#print(device)
torch.cuda.empty_cache()

# set random seed
seed = 5203
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

test_transform = transforms.Compose([
    transforms.Resize((224, 224), max_size=None, antialias=None), 
    transforms.ToTensor(), 
])

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def visulize_attention_ratio(mask_name, img_path, attention_mask, ratio=0.5, cmap="jet"):
    #print("load image from: ", img_path)
    # load the image
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')
    
    # normalize the attention mask
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    plt.savefig("p3/" + str(mask_name) + ".png")

ckpt_path = 'hw3_model/p2.pth'
encoder_name = 'vit_large_patch16_224'
encoder_dim = 1024
model = p2_model(encoder_name, encoder_dim, device).to(device)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt)
model = model.to(device)

# prepare data
batch_size = 1
dataset_path_test = "hw3_data/p3_data/images"
test_set = ImageDataset(dataset_path_test, transform=test_transform, train_set=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

tokenizer = Tokenizer.from_file("hw3_data/caption_tokenizer.json")

max_len = 15
bos = torch.full((1, 1), 2)
result = []
result_dict = {}

with torch.no_grad():
    for i, (image, image_name) in enumerate(tqdm(test_loader)):
        ids = bos
        ch = 0
        pred_caption = []
        image, ids = image.to(device), ids.to(device)
        mask_name = 0
        while ch != 3 and len(pred_caption) < max_len:
            output = model(image, ids)
            attention_mask = torch.reshape(model.decoder.layers[-1].attention_weight[0,-1,1:], (14, 14))
            attention_mask = attention_mask.cpu().detach().numpy()
            visulize_attention_ratio(image_name[0] + str(mask_name), "hw3_data/p3_data/images/" + image_name[0] + ".jpg", attention_mask)
            mask_name += 1
            last_pred = output[0][-1]
            ch = last_pred.max(-1)[1]
            pred_caption.append(ch.item())
            ids = torch.cat([ids, ch.reshape(1,1)], dim=1)
        print(tokenizer.decode(pred_caption))
        result_dict[image_name[0]] = tokenizer.decode(pred_caption)
    