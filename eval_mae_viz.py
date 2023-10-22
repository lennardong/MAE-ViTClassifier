
########################
# Initialize
########################

# Load Model
# -------------------
from transformers import ViTMAEForPreTraining
from transformers import ViTFeatureExtractor
from PIL import Image

MODEL_PATH = './models/MAE_full100_3'
model = ViTMAEForPreTraining.from_pretrained(MODEL_PATH)

# Generate input tensor
# -------------------
# The pixel_values output of the ViTFeatureExtractor is the sequence of embedded patches. 
# It is a PyTorch tensor of shape [batch_size(1), Channels (3), Height(224), Width (224)]
# IMG_PATH = 'data/pRCC_nolabel/3610_6089_2000.jpg'
# IMG_PATH = 'data/CAM16_100cls_10mask/train/data/normal/normal_002-500-2-384ver.jpg' # CAM16
IMG_PATH = 'data/WBC_1/train/data/Basophil/20190527_111443_0.jpg' # WBC

feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH)
image = Image.open(IMG_PATH)
pixel_values = feature_extractor(image, return_tensors='pt').pixel_values

########################
# Visualization
########################
import torch
import numpy as np
import matplotlib.pyplot as plt

# Fix the error: OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.manual_seed(2)

# Helper Functions 
# -------------------
imagenet_mean = np.array(feature_extractor.image_mean)
imagenet_std = np.array(feature_extractor.image_std)


def show_image(image_, title=''):
    # image is [H, W, 3]
    assert image_.shape[2] == 3
    plt.imshow(torch.clip((image_ * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    # plt.imshow(torch.clip(image_ * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def visualize(pixel_values_, model_):
    # forward pass
    outputs = model_(pixel_values_)
    y = model_.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    # visualize the mask
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model_.config.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model_.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', pixel_values_)

    # masked image
    im_masked = x * (1 - mask)

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 3, 1)
    show_image(x[0], "original")

    plt.subplot(1, 3, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 3, 3)
    show_image(y[0], "reconstruction")

    plt.show()

# Visualize
# -------------------


if __name__ == "__main__":
    visualize(pixel_values, model)
