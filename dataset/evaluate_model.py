import torch
import argparse

from spurious_score import eval_spurious_score, get_model, get_loaders
from torchvision import models
from utils.model_normalization import MVPWrapper, NormalizationWrapper


def get_devices(gpus):
    if len(gpus) == 0:
        device_ids = None
        device = torch.device('cpu')
        print('Warning! Computing on CPU')
    elif len(gpus) == 1:
        device_ids = None
        device = torch.device('cuda:' + str(gpus[0]))
    else:
        device_ids = [int(i) for i in gpus]
        device = torch.device('cuda:' + str(min(device_ids)))
    return device, device_ids


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments', prefix_chars='-')
    parser.add_argument('--gpu', '--list', nargs='+', default=[0],
                        help='GPU indices, if more than 1 parallel modules will be called')
    parser.add_argument('--bs', default=16, type=int,
                    help='batch size.')
    parser.add_argument('--model', type=str, default='robust_resnet')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    device, device_ids = get_devices(args.gpu)
    
    bs = args.bs

    # Model
    """
    evaluate pre-trained models from timm :
        just set the model_name accordingly
    
    evaluate your own models:
        - replace the get_model function
        - model should include a normalization wrapper (see utils.model_normalization.py)
        - img_size format (3, <size>, <size>)
    """
    model_name = args.model

    if "cleaned" in model_name:
        mean=torch.tensor([0.4500, 0.4448, 0.4061])
        std=torch.tensor([0.2646, 0.2575, 0.2818])
        print("Cleaned")
    elif "1.0" in model_name:
        mean=torch.tensor([0.4499, 0.4447, 0.4062])
        std=torch.tensor([0.2619, 0.2541, 0.2716])
        print("Neg")
    else:
        mean=torch.tensor([0.4504, 0.4448, 0.4054])
        std=torch.tensor([0.2608, 0.2529, 0.2706])
        print("Baseline")
    
    model = models.resnet18(num_classes=9)
    model.load_state_dict(torch.load(model_name))
    model = MVPWrapper(model)
    model.eval()
    model.to(device)
    img_size = (3, 224, 224)
    model_name = f"{model_name.split('/')[-2]}_norm"

    # load datasets
    spurious_loader, in_subset_loader = get_loaders(img_size, bs)

    eval_spurious_score(model, model_name, device, spurious_loader, in_subset_loader)