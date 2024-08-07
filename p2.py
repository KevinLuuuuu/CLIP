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

train_transform = transforms.Compose([
    transforms.Resize((224, 224), max_size=None, antialias=None),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(), 
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224), max_size=None, antialias=None),
    transforms.ToTensor(),
])

lr = 6e-5
#step_size = 180
batch_size = 100
encoder_name = 'vit_large_patch16_224'
model_name = "p2_large_bs100_flip_6e-5_6layers_notpretrained.pth"
epochs = 20

if 'large' in encoder_name:
    encoder_dim = 1024 # 768, 1024
else:
    encoder_dim = 768

tokenizer = Tokenizer.from_file("hw3_data/caption_tokenizer.json")

model = p2_model(encoder_name, encoder_dim, device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.9)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'))

# prepare data
dataset_path_train = "hw3_data/p2_data/images/train"
dataset_path_val = "hw3_data/p2_data/images/val"

train_set = ImageDataset(dataset_path_train, transform=train_transform, caption_path="caption_train.json")
valid_set = ImageDataset(dataset_path_val, transform=test_transform, caption_path="caption_val.json")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

best_acc = 0

#print(model)
# freeze encoder 

for param in model.encoder.parameters():
    param.requires_grad = False

best_loss = 1000

for epoch in range(epochs):

    model.train()
    train_loss = 0
    train_loss_record = []
    train_correct = 0
    
    for i, (image, caption) in enumerate(tqdm(train_loader)):
        image = image.to(device)
        optimizer.zero_grad()
        tokenized_caption = tokenizer.encode_batch(caption)
        ids = torch.LongTensor([tc.ids for tc in tokenized_caption]).to(device)
        output = model(image, ids)

        output = output[:,:-1,:]
        #print(output.shape)
        output = output.contiguous().view(-1, 18022)
        ids = ids[:,1:]
        ids =  ids.contiguous().view(-1)
        #print(ids.shape)
        #output = torch.swapaxes(output, 1, 2)
        train_loss = criterion(output, ids)
        train_loss_record.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
        #scheduler.step()
    
    model.eval()
    eval_loss = 0
    eval_loss_record = []
    eval_correct = 0

    with torch.no_grad():
        for i, (image, caption) in enumerate(tqdm(valid_loader)):
            image = image.to(device)
            tokenized_caption = tokenizer.encode_batch(caption)
            ids = torch.LongTensor([tc.ids for tc in tokenized_caption]).to(device)
            output = model(image, ids)
            output = output[:,:-1,:]
            output = output.contiguous().view(-1, 18022)
            ids = ids[:,1:]
            ids =  ids.contiguous().view(-1)
            eval_loss = criterion(output, ids)
            eval_loss_record.append(eval_loss.item())
    
    mean_train_loss = sum(train_loss_record)/len(train_loss_record)
    mean_eval_loss = sum(eval_loss_record)/len(eval_loss_record)
    print("Epoch:", epoch)
    print('Train loss: {:.4f}'.format( mean_train_loss))
    print('Evaluate loss: {:.4f}'.format(mean_eval_loss))     
         
    if best_loss > mean_eval_loss:
        best_loss = mean_eval_loss
        print('This epoch has best loss is {:.4f} and save model'.format(best_loss))
        #model_scripted.save('p1_b.pth')
        torch.save(model.state_dict(), model_name)

    print('The best loss is {:.4f} '.format(best_loss))
    print("Save model, " + model_name)

'''
caption = "I love my jobs"
tokenized_caption = tokenizer.encode(caption)

print(tokenized_caption.ids)
print(tokenized_caption.tokens)
'''
