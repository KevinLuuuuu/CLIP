import os
import clip
import torch
from dataset import ImageDataset
import torchvision.transforms as transforms
import json
from PIL import Image
from tqdm.auto import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
batch_size = 1
eval_transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset_path_val = "hw3_data/p1_data/val"
images = [os.path.join(dataset_path_val,x) for x in os.listdir(dataset_path_val) if x.endswith(".png")]
#valid_set = ImageDataset(dataset_path_val, transform=eval_transform)
#valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

#cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
#print(cifar100[3637])
#image, class_id = cifar100[3637]
#image_input = preprocess(image).unsqueeze(0).to(device)

with open('hw3_data/p1_data/id2label.json', newline='') as jsonfile:
    data = json.load(jsonfile)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in data.values()]).to(device)


eval_correct = 0
# Calculate features
label = 0
with torch.no_grad():
    for image_name in tqdm(images):
        image = Image.open(image_name)
        label = int(image_name.split("/")[-1].split("_")[0])
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        #print(values)
        if label == indices.item():
            #print("true")
            eval_correct = eval_correct + 1
    
print('Accuracy: {}/{} ({:.3f}%)'.format(eval_correct, len(images), 100 * eval_correct / len(images))) 


'''

for i, (image, label) in enumerate(tqdm(valid_loader)):
    image, label = image.to(device), label.to(device)
    #_, output = model(image) # momdel A
    output = model(image) # model B
    eval_loss = criterion(output, label)
    eval_loss_record.append(eval_loss.item())
    pred_label = torch.max(output.data, 1)[1]
    eval_correct = eval_correct + pred_label.eq(label.view_as(pred_label)).sum().item()




# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
'''