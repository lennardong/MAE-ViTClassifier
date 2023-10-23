from typing import List, Dict, Tuple, Union, Optional
"""
Utility functions for ViT
"""

# Load Data
# ------------------
from datasets import load_dataset
import random
from PIL import ImageDraw, Image



def load_and_split_dataset(path: str, val_split: float = 0.15):
    """
    Returns a DatasetDict.

    DatasetDict Format
    -------------------
    DatasetDict({
        train: Dataset({
            features: ['image', 'label'],
            num_rows: 6757
        })
        validation: Dataset({
            features: ['image', 'label'],
            num_rows: 1690
        })
    })

    Item Schema
    ------------------
    For each item, schema is:
    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=575x575>,
    'label': 4}
    """

    dataset = load_dataset("imagefolder", data_dir=path)

    if "validation" not in dataset.keys() and val_split > 0.0:
        split = dataset['train'].train_test_split(test_size=val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

        return dataset["train"], dataset["validation"]

    else:
        return dataset['train']


def show_examples(ds, seed: int = 12, examples_per_class: int = 3, size=(350, 350)):
    """
    Generate an image grid of samples with labels
    """
    w, h = size
    labels = ds.features['label'].names
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)
    # font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 24)

    for label_id, label in enumerate(labels):

        # Filter the dataset by a single label, shuffle it, and grab a few samples
        ds_slice = ds.filter(lambda ex: ex['label'] == label_id).shuffle(seed).select(range(examples_per_class))

        # Plot this label's examples along a row
        for i, example in enumerate(ds_slice):
            image = example['image']
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            draw.text(box, label, (255, 255, 255))

    return grid


# Image Augmentation
# ------------------
from transformers import ViTImageProcessor
from PIL import ImageOps
from random import random


def transform(processor: ViTImageProcessor, example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor(images=[x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['label'] = example_batch['label']
    # logger.info(f'/nRETURN: /n{inputs}')
    return inputs


def transform_with_augmentation(processor: ViTImageProcessor, example_batch):
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
    inputs['label'] = example_batch['label']
    # logger.info(f'/nRETURN: /n{inputs}')
    return inputs


# Data Collator
# -----------------
import torch


def collate_fn_classifier(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


def collate_fn_mae(batch):
    """Masked Autoencoder"""
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


# Metrics
# -----------------
import json
import numpy as np
from datasets import load_metric


def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


# Evals
# -----------------
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from transformers import Trainer, TrainingArguments, ViTConfig
from datetime import datetime

# Settings
sns.set_style('darkgrid')


def save_session(save_path: str,
                 trainer: Trainer,
                 model: nn.Module = None,
                 model_config: ViTConfig = None,
                 training_args: TrainingArguments = None,
                 processor_args: dict = None):
    """
    Save the model, model config, training args, and log history to a directory.
    Args:
        model:
        training_args:
        model_config:
        trainer:
        save_path:

    Returns:

    """
    print("\nSAVE SESSION...")
    # Save Model
    # try:
    #     trainer.save_model(save_path)
    # except Exception as e:
    #     print(f"...trainer.save_model() failed due to {e}")

    # State dictionary
    print('...saving model state_dict')
    if model is not None:
        torch.save(model.state_dict(), save_path + '/model_state_dict.pth')
        # torch.save(model, save_path + '/entire_model.pth')

    # Log history (for plotting)
    print('...saving training history')
    if trainer is not None:
        log_history = trainer.state.log_history
        with open(save_path + '/log_history.json', 'w') as f:
            json.dump(log_history, f)

    # Config
    print('...saving model config')
    if model_config is not None:
        with open(save_path + '/model_config.json', 'w') as f:
            json.dump(model_config.to_dict(), f)

    # Training Args
    print('...saving training args')
    if training_args is not None:
        training_args_dict = training_args.to_dict()
        with open(save_path + '/training_args.json', 'w') as f:
            json.dump(training_args_dict, f)

    # save processor args to json
    print('...saving processor args')
    if processor_args is not None:
        with open(save_path + '/processor_args.json', 'w') as f:
            json.dump(processor_args, f)

    print("...session saved")


########################################################################################
# EVALUATION HELPERS
########################################################################################

# # Load Models for Eval
# # -----------------
# import os
# from main_vitclassifier_fromscratch import ViTForImageClassificationFromScratch
# from main_vitclassifier_transferlearning import ViTForImageClassificationFromMAE
# from datasets import Dataset
#
#
# def load_model_fromscratch(directory_path: str, filename_model_args: str, filename_model_state: str) -> ViTForImageClassificationFromScratch:
#     """Returns the model"""
#
#     file_model_args = os.path.join(directory_path, filename_model_args)
#     file_state_dict = os.path.join(directory_path, filename_model_state)
#
#     with open(file_model_args, 'r') as f:
#         loaded_args = json.load(f)
#     config = ViTConfig.from_dict(loaded_args)
#     model = ViTForImageClassificationFromScratch(config)  # Initialize the model
#     model.load_state_dict(torch.load(file_state_dict, map_location=torch.device('cpu')))
#
#     # Make sure to call this if you plan to use the model for inference
#     model.eval()
#
#     return model
#
#
# def load_trainer(directory_path: str, filename: str, model) -> Trainer:
#     """Returns the trainer to run inference"""
#     file = os.path.join(directory_path, filename)
#     with open(file, 'r') as f:
#         loaded_args = json.load(f)
#
#     config = TrainingArguments(**loaded_args)
#     trainer = Trainer(
#         model=model,
#         args=config,
#         data_collator=collate_fn_classifier,
#         compute_metrics=compute_metrics,
#     )
#
#     return trainer
#
#
# def load_training_logs(directory_path: str, filename: str) -> dict:
#     """Returns the training logs for eval.
#
#     Schema: [dict_eval_loss, dict_loss, dict_eval_loss, dict_loss, ...]
#
#     eval loss keys:
#     ['eval_loss', 'eval_accuracy', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch', 'step']
#
#     loss keys:
#     ['loss', 'learning_rate', 'epoch', 'step']
#     """
#     file = os.path.join(directory_path, filename)
#     with open(file, 'r') as f:
#         loaded_args = json.load(f)
#
#     return loaded_args
#
#
# def load_test_ds(directory_path: str, filename: str, test_path: str) -> (ViTImageProcessor, Dataset):
#     """Returns the test set df"""
#     file = os.path.join(directory_path, filename)
#     with open(file, 'r') as f:
#         loaded_args = json.load(f)
#
#     processor_ = ViTImageProcessor(**loaded_args)
#     file_test = os.path.join(directory_path, filename)
#     ds_raw = load_and_split_dataset(test_path, 0.0)
#     ds_transformed = ds_raw.with_transform(lambda example_batch: transform(processor_, example_batch))
#
#     return processor_, ds_transformed


# Viz
# -----------------
import seaborn as sns
import pandas as pd


def plot_losses_train_eval(trainer: Trainer, title: str = 'Loss Landscape, Train vs. Eval',
                           save_path: Optional[str] = None):
    """
    Plot the training and evaluation losses from the log history.
    Args:
        trainer: HF trainer instance
        title: string, title of plot e.g. "My Loss Plot"
        save_path: string, save path in format e.g. './models/WBC_1'

    Returns:
        na. saved file with timestamp prefix if save_path is provided

    """
    # Assuming `trainer.state.log_history` contains training loss and epoch
    train_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    train_epochs = [log['epoch'] for log in trainer.state.log_history if 'loss' in log]

    # Assuming `trainer.state.log_history` contains evaluation loss and epoch
    eval_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    eval_epochs = [log['epoch'] for log in trainer.state.log_history if 'eval_loss' in log]

    plt.plot(train_epochs, train_loss, color='lightblue', label='Training Loss')
    plt.plot(eval_epochs, eval_loss, color='darkblue', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    if save_path:
        # save to folder path
        # get datetime in format YYMMDD-HHMM
        prefix = datetime.now().strftime("%y%m%d-%H%M")
        filename = prefix + '-losses.png'
        plt.savefig(save_path + '/' + filename)
        print(f'Loss plot saved to {save_path}/{filename}')


def plot_losses_from_logs(logs: List[Dict[str, float]], plot_title: str, save_path: str):
    logs = pd.DataFrame(logs)

    pass