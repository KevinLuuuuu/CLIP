import os
import clip
import torch
import torchvision.transforms as transforms
import json
from PIL import Image
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm.auto import tqdm

def main(args):

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    dataset_path_val = args.input_dir
    images = [os.path.join(dataset_path_val,x) for x in os.listdir(dataset_path_val) if x.endswith(".png")]

    with open(args.id2label_dir, newline='') as jsonfile:
        data = json.load(jsonfile)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in data.values()]).to(device)

    name_list = []
    predict_list = []
    # Calculate features
    with torch.no_grad():
        for image_name in tqdm(images):
            image = Image.open(image_name)
            name = image_name.split("/")[-1]
            #print(name)
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            name_list.append(name)
            predict_list.append(indices)

    ################# check ####################
    with open(args.output_dir, 'w', newline="") as fp:        
        file_writer = csv.writer(fp)
        file_writer.writerow(['filename', 'label'])
        for i in range(len(predict_list)):
            file_writer.writerow([name_list[i], predict_list[i].item()])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Path to the input file.",
        required=True
    )
    parser.add_argument(
        "--id2label_dir",
        type=Path,
        help="Path to the input file.",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the input file.",
        required=True
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda"
    else:
        device = "cpu"

    args = parse_args()
    main(args)