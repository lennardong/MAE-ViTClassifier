# # System Utilities
import logging
from transformers import ViTConfig
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# INPUTS
TRAINING_PATH = './data/WBC_100/train/data/'  # will be automatically split to train/val
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
OUTPUT_PATH = './models/WBC_100_v1'

#########################################
# LOAD DATA
#########################################
from datasets import load_dataset
import random
from PIL import ImageDraw, Image
from util.utils import load_and_split_dataset, show_examples


def load_and_split_dataset(path: str, val_split: float = 0.15):
    '''
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
    '''
    logger.info(f'/n######################/nLOAD AND SPLIT DATASET: {path}')

    dataset = load_dataset("imagefolder", data_dir=path)

    if "validation" not in dataset.keys() and val_split > 0.0:
        split = dataset['train'].train_test_split(test_size=val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

        return dataset["train"], dataset["validation"]

    else:
        return dataset['train']


def show_examples(ds, seed: int = 12, examples_per_class: int = 3, size=(350, 350)):
    '''
    Generate an image grid of samples with labels
    '''
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


# Load data
# train_ds, val_ds = load_and_split_dataset(TRAINING_PATH, 0.15)
# test_ds = load_and_split_dataset(TEST_PATH, 0.0)
# show_examples(val_ds, seed=random.randint(0, 1337), examples_per_class=3)

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
    inputs = processor(images=[x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['label'] = example_batch['label']
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
    inputs['label'] = example_batch['label']
    # logger.info(f'/nRETURN: /n{inputs}')
    return inputs


# Test
# prepared_train_ds = train_ds.with_transform(transform_with_augmentation)
# prepared_val_ds = val_ds.with_transform(transform)  # No extra augmentation for validation
# prepared_test_ds = test_ds.with_transform(transform)  # No extra augmentation for test

################################
# Setup the Trainer
################################

# Data Collator
# -----------------
import torch


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


# Evaluation Metric
# -----------------
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


# Load Model
# -----------------
from transformers import ViTModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


class ViTForImageClassificationFromScratch(nn.Module):

    def __init__(self, config, labels: int = 5):
        super().__init__()

        # Initialize the base ViT model
        self.vit = ViTModel(config)

        # Add a classifier layer
        self.classifier = nn.Linear(config.hidden_size, labels)

        # Pass in list of labels
        self.num_labels = labels
        # self.id2label={str(i): c for i, c in enumerate(label_titles)},
        # self.label2id={c: str(i) for i, c in enumerate(label_titles)}

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Training arguments
# -----------------
from transformers import TrainingArguments

use_fp16 = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()

training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    per_device_train_batch_size=64 // num_gpus if num_gpus > 0 else 16,  # Adjust batch size
    evaluation_strategy="epoch",  # instead of 'steps'
    save_strategy="epoch",
    num_train_epochs=5,
    fp16=use_fp16,
    #   save_steps=100, # set very high to avoid saving
    #   eval_steps=100, # ignored because eval set to epoch
    logging_steps=5,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to='none',  # alt is tensorboard
    load_best_model_at_end=True,
)

################################################################################################
# COMPUTE
################################################################################################

if __name__ == '__main__':
    # Load Data    
    train_ds, val_ds = load_and_split_dataset(TRAINING_PATH, 0.15)
    test_ds = load_and_split_dataset(TEST_PATH, 0.0)

    # Prepare Data
    prepared_train_ds = train_ds.with_transform(transform_with_augmentation)
    prepared_val_ds = val_ds.with_transform(transform)  # No extra augmentation for validation
    prepared_test_ds = test_ds.with_transform(transform)  # No extra augmentation for test

    # Init Model
    label_titles = prepared_train_ds.features['label'].names
    model = ViTForImageClassificationFromScratch(MODEL_CONFIG)

    # Init Trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_train_ds,
        eval_dataset=prepared_val_ds,
        # tokenizer=processor,
    )

    # Train
    train_results = trainer.train()

    # Save
    # HACK manual saves
    import json

    trainer.save_model(OUTPUT_PATH)  # <- doesn't work!

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

    ################################
    # EVALUATION
    ################################

    # Plot
    # -----------------
    import matplotlib.pyplot as plt

    # Assuming `trainer.state.log_history` contains training loss and epoch
    train_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    train_epochs = [log['epoch'] for log in trainer.state.log_history if 'loss' in log]

    # Assuming `trainer.state.log_history` contains evaluation loss and epoch
    eval_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    eval_epochs = [log['epoch'] for log in trainer.state.log_history if 'eval_loss' in log]

    plt.plot(train_epochs, train_loss, label='Train Loss')
    plt.plot(eval_epochs, eval_loss, color='red', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Accuracy
    # -----------------
    test_results = trainer.predict(prepared_test_ds)
    metrics = compute_metrics(test_results)
    print(f"Test Accuracy: {metrics['accuracy']}")
    # WBC_1: Test Accuracy: 0.6128472222222222
    # WBC_10: Test Accuracy: 0.6128472222222222
    # WBC_50: Test Accuracy: 0.7262731481481481
    # WBC_100:
