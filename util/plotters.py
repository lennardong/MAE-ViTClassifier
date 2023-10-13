# Math
import torch
import numpy as np

# Models
from transformers import ViTFeatureExtractor, ViTMAEForPreTraining

# Viz
import matplotlib.pyplot as plt
from PIL import Image


#############################################
# MAE 
#############################################

def _show_image(image_, imagenet_std_, imagenet_mean_, title_=''):
    # image is [H, W, 3]
    assert image_.shape[2] == 3
    plt.imshow(torch.clip((image_ * imagenet_std_ + imagenet_mean_) * 255, 0, 255).int())
    # TODO why use std and mean? check against transforms done prior
    plt.title(title_, fontsize=16)
    plt.axis('off')
    return

def visualize_mae_reconstruction(image_: Image, model_: ViTMAEForPreTraining, feature_extractor_: ViTFeatureExtractor):
    """
    Visualizes the masked image, reconstruction, and reconstruction with visible patches.

    Args:
    - image: an image using PIL. example: image = Image.open(requests.get(url, stream=True).raw)
    - model (torch.transformers): a pretrained ViT MAE model

    Returns:
    - None
    """

    # Init
    # --------------------
    imagenet_mean = np.array(feature_extractor_.image_mean)
    imagenet_std = np.array(feature_extractor_.image_std)
    # TODO replace this to use pretrained model_
    feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base") 
    pixel_values = feature_extractor(image_, return_tensors="pt").pixel_values

    
    # Forward pass
    # --------------------
    outputs = model_(pixel_values)
    y = model_.unpatchify(outputs.logits)
    # TODO explain this 
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    # Visualize the mask
    # --------------------
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model_.config.patch_size**2 *3)  # (N, H*W, p*p*3)
    mask = model_.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', pixel_values)
    im_masked = x * (1 - mask)

    
    # Plot
    # --------------------
    plt.rcParams['figure.figsize'] = [24, 24] #upsize

    plt.subplot(1, 3, 1)
    _show_image(x[0], imagenet_std, imagenet_mean, "original")

    plt.subplot(1, 3, 2)
    _show_image(im_masked[0], imagenet_std, imagenet_mean, "masked")

    plt.subplot(1, 3, 3)
    _show_image(y[0], imagenet_std, imagenet_mean, "reconstruction")

    plt.show()