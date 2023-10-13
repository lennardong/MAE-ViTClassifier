# # System Utilities
import logging
from transformers import ViTConfig

from typing import Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# INPUTS
TRAINING_PATH = './data/WBC_1/train/data/' #will be automatically split to train/val
TEST_PATH = './data/WBC_100/val/data/'
MODEL_CONFIG = ViTConfig(
    image_size=224,
    patch_size=16,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
OUTPUT_PATH = './models/WBC_1_testv1'
DATASET_PATHS = {
    'pRCC': './data/pRCC_nolabel',
    'CAM16_test': './data/CAM16_100cls_10mask/test/data/normal',
    'CAM16_train': './data/CAM16_100cls_10mask/train/data/normal',
    'CAM16_val': './data/CAM16_100cls_10mask/val/data/normal',
}

#########################################
# LOAD DATA
#########################################
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, Dataset
import random
from PIL import ImageDraw, Image


from datasets import concatenate_datasets, load_dataset
import logging

logger = logging.getLogger(__name__)

def load_and_split_datasets(paths: Dict[str, str], val_split: float = 0.15):
    '''
    Returns a Dataset tuple for train and validation

    Dataset Format
    ------------------- 
    Dataset({
        features: ['image'],
        num_rows: 294
    })
    
    Item Schema
    ------------------
    For each item, schema is: 
    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=575x575>,
    'label': 4}
    '''
    datasets = [load_dataset("imagefolder", data_dir=path)['train'] for path in paths.values()]
    merged_dataset = concatenate_datasets(datasets)

    if val_split > 0.0:
        split = merged_dataset.train_test_split(test_size=val_split)
        return split["train"], split["test"]
    else:
        return merged_dataset


# Test


########################################
# Prepare Dataset
########################################
from transformers import ViTConfig, ViTImageProcessor
from PIL import ImageOps
from random import random

processor = ViTImageProcessor(MODEL_CONFIG)

def transform(example_batch):
    # logger.info(f'/n#####################/nTRANSFORM: {example_batch}')
    # Take a list of PIL images and turn them to pixel values
    inputs = processor(images = [x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    # inputs['label'] = example_batch['label']
    # logger.info(f'/nRETURN: /n{inputs}')
    return inputs

def transform_with_augmentation(example_batch):
    # logger.info(f'/n#####################/nTRANSFORM w augmentation: {example_batch}')
    augmented_images = []
    for image in example_batch['image']:
        # Apply horizontal flip with 50% probability
        if random() > 0.5:
            image = ImageOps.mirror(image)
        # Apply vertical flip with 50% probability
        if random() > 0.5:
            image = ImageOps.flip(image)
        augmented_images.append(image)
    
    # Continue with your existing transformation logic
    inputs = processor(images=augmented_images, return_tensors='pt')
    # inputs['label'] = example_batch['label'] # removed for MAE
    # logger.info(f'/nRETURN: /n{inputs}')
    return inputs


################################
# Setup the Trainer
################################

# Data Collator
# ---------------------
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values}

# Eval Metric
# ---------------------

# Load Model
# ---------------------
from transformers import ViTMAEConfig, ViTMAEForPreTraining

model_args = ViTMAEConfig(
    hidden_size= 768,
    num_hidden_layers =12,
    num_attention_heads= 12,
    intermediate_size= 3072, # x4 hidden size
    hidden_act= "gelu",
    hidden_dropout_prob= 0.1, # dropout probability for all FC in embeddings, encoder, pooler
    attention_probs_dropout_prob = 0.1,
    image_size= 224,
    patch_size= 16, # resolution of each patch
    decoder_num_attention_heads= 16, # modified from default 12
    decoder_hidden_size= 512,
    decoder_num_hidden_layers= 8, 
    decoder_intermediate_size= 2048,
    mask_ratio= 0.75,
    norm_pix_loss= False
)  
model = ViTMAEForPreTraining(model_args)


# Load Trainer
# ---------------------
from transformers import TrainingArguments

use_fp16 = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()

training_args = TrainingArguments(
  output_dir = OUTPUT_PATH,
  per_device_train_batch_size= 64 // num_gpus if num_gpus > 0 else 16,  # Adjust batch size
  evaluation_strategy="epoch", #instead of 'steps'
  save_strategy= "epoch",
  num_train_epochs=800,
  fp16=use_fp16,
#   save_steps=100, # set very high to avoid saving
#   eval_steps=100, # ignored because eval set to epoch
  logging_steps=100,
  learning_rate=0.0001,
  save_total_limit=2,
  remove_unused_columns=False,
  report_to='none', # alt is tensorboard
  load_best_model_at_end=True,
  lr_scheduler_type="cosine",  # Set scheduler to cosine
  warmup_steps=100  # Number of warmup steps
)


################################
# COMPUTE
################################

if __name__ == '__main__':
    # Load & Prepare Data
    train_ds, test_ds = load_and_split_datasets(DATASET_PATHS, 0.15)
    prepared_train_ds = train_ds.with_transform(transform_with_augmentation)
    prepared_test_ds = test_ds.with_transform(transform)

    # Load Model
    model = ViTMAEForPreTraining(model_args)

    # Init Trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_train_ds,
        # eval_dataset=ds["validation"] if training_arguments.do_eval else None,
        # tokenizer=processor,
        # NOTE to check if tokenizer is necessary
        data_collator=collate_fn,
    )

    train_results = trainer.train()

    # Save
    # HACK manual saves
    import json
    trainer.save_model(OUTPUT_PATH) # <- doesn't work!
    
    # Save the model state dictionary
    torch.save(model.state_dict(), OUTPUT_PATH + '/model_state_dict.pth')

    # Save log history (for plotting)
    log_history = trainer.state.log_history
    with open(OUTPUT_PATH + '/log_history.json', 'w') as f:
        json.dump(log_history, f)

    # Save the model config as a JSON file
    with open(OUTPUT_PATH + '/model_config.json', 'w') as f:
        json.dump(MODEL_CONFIG.to_dict(), f)

    # Assuming `training_args` is your TrainingArguments object
    training_args_dict = training_args.to_dict()

    # Save to JSON file
    with open(OUTPUT_PATH + '/training_args.json', 'w') as f:
        json.dump(training_args_dict, f)

