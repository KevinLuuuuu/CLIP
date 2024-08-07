import os
import clip
import torch
from dataset import ImageDataset
import torchvision.transforms as transforms
import json
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
batch_size = 1
eval_transform = transforms.Compose([
    transforms.ToTensor(),
])

with open('hw3_data/p1_data/id2label.json', newline='') as jsonfile:
    data = json.load(jsonfile)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in data.values()]).to(device)

eval_correct = 0
# Calculate features
label = 0
image_name = "hw3_data/p1_data/val/2_455.png"

with torch.no_grad():
    image = Image.open(image_name)
    label = int(image_name.split("/")[-1].split("_")[0])
    image_input = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

item = []
for id in indices:
    print(id.item())
    print(data[str(id.item())])
print(values)

fig = plt.figure()
for i in range(4, -1, -1):
    item.append(data[str(indices[i].item())])
prob = [i * 100 for i in values.tolist()]
prob.reverse()
plt.barh(item, prob, color=['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'limegreen'])
#fig, ax = plt.subplots(2,2)
#ax[0][0].plot(x)
#ax[1][1].plot(y)

plt.title("correct probability: " + str(round(prob[4], 2)) + "%")
plt.xlim(0,100)
plt.show()
plt.savefig("p1_3_p.png")

plt.clf()

plt.title("correct label: " + item[4])
plt.imshow(image)
plt.savefig("p1_3_i.png")


#*100