########################################################
# Load Model
########################################################
import torch
import matplotlib.pyplot as plt
from transformers import ViTConfig, Trainer, TrainingArguments
from WORKING_vit_fromscratch import compute_metrics, ViTForImageClassificationFromScratch # Import the compute_metrics function from your previous script

SAVE_PATH = './models/WBC_1_testv3' # Path where the model was saved
# TEST_PATH = './data/WBC_100/val/data/'
# SAVED_MODEL_PATH = './models/WBC_10_saveModel'

import json

# Load the model config from a JSON file
with open(SAVE_PATH + '/model_config.json', 'r') as f:
    loaded_config_dict = json.load(f)

loaded_config = ViTConfig.from_dict(loaded_config_dict)

model = ViTForImageClassificationFromScratch(loaded_config)  # Initialize the model
model.load_state_dict(torch.load(SAVE_PATH + '/model_state_dict.pth', map_location=torch.device('cpu')))
model.eval() # freeze model for inference


########################################################
# Generate prediction
########################################################
from PIL import Image
from transformers import ViTConfig, ViTImageProcessor

IMG_PATH = 'data/WBC_100/val/data/Basophil/20190526_162951_0.jpg'

processor = ViTImageProcessor(loaded_config)
id2label = {
    0: 'Basophil', 
    1: 'Eosinophil', 
    2: 'Lymphocyte', 
    3: 'Monocyte', 
    4: 'Neutrophil'
}

def predict(path: str):
    
    # Encode
    im=Image.open(path)
    
    encoding = processor(images=im, return_tensors="pt")
    encoding.keys()
    pixel_values = encoding['pixel_values']

    # Softmax results 
    outputs = model(pixel_values)
    results = outputs.logits.softmax(1)

    # Process
    result_idx = results.argmax(1).tolist()[0]
    result_label = id2label[result_idx]
    result_p = results.tolist()[0][result_idx]

    return result_label, result_p

l, p = predict(IMG_PATH)

print(f'{l}: {p}')

