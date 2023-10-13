# %%
import cv2
import numpy as np
import torch
from dataclasses import dataclass 

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--use-cuda', action='store_true', default=False,
#                         help='Use NVIDIA GPU acceleration')
#     parser.add_argument(
#         '--image-path',
#         type=str,
#         default='./examples/both.png',
#         help='Input image path')
#     parser.add_argument('--aug_smooth', action='store_true',
#                         help='Apply test time augmentation to smooth the CAM')
#     parser.add_argument(
#         '--eigen_smooth',
#         action='store_true',
#         help='Reduce noise by taking the first principle componenet'
#         'of cam_weights*activations')

#     parser.add_argument(
#         '--method',
#         type=str,
#         default='gradcam',
#         help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

#     args = parser.parse_args()
#     args.use_cuda = args.use_cuda and torch.cuda.is_available()
#     if args.use_cuda:
#         print('Using GPU for acceleration')
#     else:
#         print('Using CPU for computation')

#     return args

# %%
######################
# Script Arguments
######################

@dataclass
class InitArguments:
    """Class for initialization arguments"""
    use_cuda: bool = False
    image_path: str = './examples/both.png'
    aug_smooth: bool = False
    eigen_smooth: bool = False
    method: str = 'scorecam'

    def __post_init__(self):
        methods = ["gradcam", "scorecam", "gradcam++", ...]
        assert self.method in methods, f"method should be one of {methods}"


args_v1 = InitArguments(
    image_path="./data/WBC_100/val/data/Basophil/20190526_162951_0.jpg"
)


# %%
########################################
# Load pre-trained model
# v1 - loading from jsons
########################################

import torch
import matplotlib.pyplot as plt
from transformers import ViTConfig, Trainer, TrainingArguments
from WORKING_vit_fromscratch import compute_metrics, ViTForImageClassificationFromScratch # Import the compute_metrics function from your previous script


SAVE_PATH = './models/WBC_1_testv2' 
# Path where the model was saved
TEST_PATH = './data/WBC_100/val/data/'
SAVED_MODEL_PATH = './models/WBC_10_saveModel'

import json

# Load the model config from a JSON file
with open(SAVE_PATH + '/model_config.json', 'r') as f:
    loaded_config_dict = json.load(f)

loaded_config = ViTConfig.from_dict(loaded_config_dict)

model = ViTForImageClassificationFromScratch(loaded_config)  # Initialize the model
model.load_state_dict(torch.load(SAVE_PATH + '/model_state_dict.pth', map_location=torch.device('cpu')))

# Make sure to call this if you plan to use the model for inference
model.eval()


# %%
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = args_v1
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True)
    # model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
# %%
