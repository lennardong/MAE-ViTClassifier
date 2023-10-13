# Pretraining a Masked Autoencoder

# Data
from transformers import DataCollatorForMaskedImageModeling
from datasets import load_dataset, concatenate_datasets

# Models
from transformers import ViTFeatureExtractor, ViTForMaskedImageModeling, ViTConfig

# Training
from transformers import TrainingArguments, Trainer

# Maths
import torch

# Misc
from typing import Dict
import util.plotters as plotters

####################################
# Helper Functions
####################################

'''
Make sure that the final preprocessing output for all datasets has the same shape and features.
This is so that they can be used to train a single model. 
If the preprocessing steps produce outputs of different shapes or types, 
you may need to add an additional step to standardize them.
'''
def load_and_preprocess_datasets(feature_extractor, **dataset_paths):
    """
    Load and preprocess datasets from multiple folder paths and merge them into a single dataset.
    
    Parameters:
        feature_extractor (ViTFeatureExtractor): The feature extractor for preprocessing.
        dataset_paths (dict): Keyword arguments representing dataset names and their corresponding folder paths.
        
    Returns:
        DatasetDict: A merged dataset containing examples from all provided datasets.
    """
    datasets = []
    
    # Function to preprocess images
    def preprocess_images(batch):
        batch["pixel_values"] = feature_extractor(batch["image_path"])["pixel_values"]
        return batch
    
    for dataset_name, folder_path in dataset_paths.items():
        # Load the dataset
        dataset = load_dataset(dataset_name, data_dir=folder_path)
        
        # Preprocess the dataset
        dataset = dataset.map(preprocess_images, batched=True)
        
        # Append to the list
        datasets.append(dataset)
        
    # Merge all datasets
    merged_dataset = concatenate_datasets(datasets)
    
    return merged_dataset



####################################
# Main
####################################

def main(config_datasetPaths_: Dict[str, str], config_model_: ViTConfig, config_trainingArgs_: TrainingArguments):
    # Load Data
    feature_extractor = ViTFeatureExtractor(size=(224, 224))
    merged_dataset = load_and_preprocess_datasets(feature_extractor, config_datasetPaths_)

    # LOAD MODEL
    model = ViTForMaskedImageModeling(config_model_)
    data_collator = DataCollatorForMaskedImageModeling(feature_extractor, masking_probability=0.75) 

    trainer = Trainer(
    model=model,
    args=config_trainingArgs_,
    data_collator=data_collator,
    train_dataset=merged_dataset["train"],
    )

    trainer.train()
    model.save_pretrained("path/to/save/mae_model")


########################
# Execution
########################


# CONFIGS - Default
# ------------------

config_datasetPaths = {
    'pRCC': './data/pRCC_nolabel',
    'CAM16_test': './data/CAM16_100cls_10mask/test/data/normal',
    'CAM16_train': './data/CAM16_100cls_10mask/train/data/normal',
    'CAM16_val': './data/CAM16_100cls_10mask/val/data/normal',
}

config_model = ViTConfig(
    image_size=224, 
    patch_size=16, 
    hidden_size=768, 
    num_hidden_layers=12, 
    num_attention_heads=12)

config_trainingArgs = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,)

config_misc = {
    "save_path":"...",}

# RUN
# ------------------

# if __name__ == '__main__':
    # main(config_datasetPaths, config_model, config_trainingArgs)

'''
TODO 1. How to implement logging? e.g. logging loss per epoch, final accuracy
x TODO 2. How to implement a raw ViT (i.e. without pre-trained weights. It will be trained from scratch using the data)
x TODO 3. How to implement GPU training? 
TODO 4. How to implement a plot of original image, masked image, generated image
TODO 5. What other metrics would a researcher use to track learning?
'''
