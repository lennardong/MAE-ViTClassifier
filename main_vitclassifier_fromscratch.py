"""
From-scratch training of a VIT Classifier for White-Blood Cell Classification
- ViT Architecture based on BASE ViT Model Architecture
- Pre-trained model is a ViTMAE model, trained on the related medical dataset
"""

# Internal Helpers
import util.utils as utils
import json

# System Utilities
import logging
from dataclasses import dataclass

# Tensor Libraries
import torch
import torch.nn as nn

# Deep Learning Libraries
from transformers import (
    ViTModel, ViTConfig, ViTImageProcessor,  # Model and Config
    TrainingArguments, Trainer,  # Training
)
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------
# Models
# ------------------


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


def save_session(save_path: str,
                 trainer: Trainer,
                 model: nn.Module = None,
                 model_config: ViTConfig = None,
                 training_args: TrainingArguments = None):
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
    print('saving model state_dict')
    if model is not None:
        torch.save(model.state_dict(), save_path + '/model_state_dict.pth')
        # torch.save(model, save_path + '/entire_model.pth')

    # Log history (for plotting)
    print('saving training history')
    if trainer is not None:
        log_history = trainer.state.log_history
        with open(save_path + '/log_history.json', 'w') as f:
            json.dump(log_history, f)

    # Config
    print('saving model config')
    if model_config is not None:
        with open(save_path + '/model_config.json', 'w') as f:
            json.dump(model_config.to_dict(), f)

    # Training Args
    print('saving training args')
    if training_args is not None:
        training_args_dict = training_args.to_dict()
        with open(save_path + '/training_args.json', 'w') as f:
            json.dump(training_args_dict, f)

    print("...session saved")

# ------------------
# INPUTS
# ------------------


def run_model(TRAINING_FOLDER_, TEST_FOLDER_, OUTPUT_FOLDER_, WBC_LABEL: str = ''):
    num_gpus = torch.cuda.device_count()

    # ARGUMENTS
    MODEL_CONFIG = ViTConfig(
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    TRAINING_ARGS = TrainingArguments(
        output_dir=OUTPUT_FOLDER_,
        per_device_train_batch_size=64 // num_gpus if num_gpus > 0 else 16,  # Adjust batch size
        evaluation_strategy="epoch",  # instead of 'steps'
        save_strategy="epoch",
        num_train_epochs=20,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    PROCESSOR_ARGS = {
        "do_resize": True,  # resize to default (224, 224)
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],  # Normalize to [0.5, 0.5, 0.5]
        "image_std": [0.5, 0.5, 0.5]  # Normalize with standard deviation [0.5, 0.5, 0.5]
        # rest will take default values
    }

    # DATA
    print(f"LOAD DATA: \nsource: {TRAINING_FOLDER_}, output: {OUTPUT_FOLDER_}")
    train_ds, val_ds = utils.load_and_split_dataset(TRAINING_FOLDER_, 0.15)
    test_ds = utils.load_and_split_dataset(TEST_FOLDER_, 0.0)

    # Use lambda to pass processor to the transform functions
    processor = ViTImageProcessor(**PROCESSOR_ARGS)
    prepared_train_ds = train_ds.with_transform(
        lambda example_batch: utils.transform_with_augmentation(processor, example_batch))
    prepared_val_ds = val_ds.with_transform(lambda example_batch: utils.transform(processor, example_batch))
    prepared_test_ds = test_ds.with_transform(lambda example_batch: utils.transform(processor, example_batch))

    # MODEL
    print("INITIALIZE MODEL")
    label_titles = prepared_train_ds.features['label'].names
    model = ViTForImageClassificationFromScratch(MODEL_CONFIG)

    # TRAINING
    print("START TRAINING")
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        data_collator=utils.collate_fn_classifier,
        compute_metrics=utils.compute_metrics,
        train_dataset=prepared_train_ds,
        eval_dataset=prepared_val_ds,
    )
    training_results = trainer.train()

    print("SAVE SESSION")
    utils.save_session(OUTPUT_FOLDER_, trainer, model, MODEL_CONFIG, TRAINING_ARGS)

    # EVALUATE
    print("EVALUATE RESULTS")
    test_results = trainer.predict(prepared_test_ds)
    print(utils.compute_metrics(test_results))
    utils.plot_losses_train_eval(trainer, f"{WBC_LABEL}\nLoss Landscape, from scratch", OUTPUT_FOLDER_)


# ------------------
# Run
# ------------------


@dataclass
class Folder:
    dataset_label: str
    training_folder: str
    output_folder: str
    test_folder: str = './data/WBC_100/val/data/'
    pretrained_model_folder: str = './models/MAE_test2'


if __name__ == '__main__':
    WBC1 = Folder(dataset_label="WBC1", training_folder='./data/WBC_1/train/data/', output_folder='./models/fromScratch_WBC_1')
    WBC10 = Folder(dataset_label="WBC10", training_folder='./data/WBC_10/train/data/', output_folder='./models/fromScratch_WBC_10')
    WBC50 = Folder(dataset_label="WBC50", training_folder='./data/WBC_50/train/data/', output_folder='./models/fromScratch_WBC_50')
    WBC100 = Folder(dataset_label="WBC100", training_folder='./data/WBC_100/train/data/', output_folder='./models/fromScratch_WBC_100')

    # for item in [WBC1, WBC10, WBC50, WBC100]:
    for item in [WBC1]: # this is for debugging
        run_model(item.training_folder, item.test_folder, item.output_folder, item.dataset_label)
