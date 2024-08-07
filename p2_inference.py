from tokenizers import Tokenizer
import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import ImageDataset
import json
from model import p2_model
from argparse import ArgumentParser, Namespace
from pathlib import Path

def main(args):

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

    ckpt_path = 'hw3_model/p2.pth'
    encoder_name = 'vit_large_patch16_224'

    if 'large' in encoder_name:
        encoder_dim = 1024
    else:
        encoder_dim = 768

    model = p2_model(encoder_name, encoder_dim, device).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    # prepare data
    batch_size = 1
    dataset_path_test = args.input_dir
    test_set = ImageDataset(dataset_path_test, transform=test_transform, train_set=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    tokenizer = Tokenizer.from_file("hw3_model/caption_tokenizer.json")

    max_len = 15
    bos = torch.full((1, 1), 2)
    result_dict = {}

    with torch.no_grad():
        for i, (image, image_name) in enumerate(tqdm(test_loader)):
            ids = bos
            ch = 0
            pred_caption = []
            image, ids = image.to(device), ids.to(device)
            while ch != 3 and len(pred_caption) < max_len:
                output = model(image, ids)
                last_pred = output[0][-1]
                ch = last_pred.max(-1)[1]
                pred_caption.append(ch.item())
                ids = torch.cat([ids, ch.reshape(1,1)], dim=1)
            result_dict[image_name[0]] = tokenizer.decode(pred_caption)
            #print(tokenizer.decode(pred_caption))

        json_result = json.dumps(result_dict, indent=2)
        with open(args.output_dir, 'w') as outFile:
            outFile.write(json_result)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
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


