"""
Transfer Learning and end-to-end retraining of a VIT Classifier for White-Blood Cell Classification
- ViT Architecture based on BASE ViT Model Architecture
- Pre-trained model is a ViTMAE model, trained on the related medical dataset
"""

# Internal Helpers
import util.utils as utils

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


class ViTForImageClassificationFromMAE(nn.Module):

    def __init__(self, pretrained_model_path: str, num_labels: int = 5):
        super().__init__()

        # Initialize the ViT model from the encoder part of a pre-trained ViTMAE model (for memory considerations)
        # to load the whole model, use:
        # vit_mae_model = ViTMAEForPreTraining.from_pretrained("path/to/your/pretrained/MAE_model")
        self.vit = ViTModel.from_pretrained(pretrained_model_path)

        # Get the hidden size from the config
        hidden_size = self.vit.config.hidden_size

        # Add a classifier layer
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

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


# ------------------
# INPUTS
# ------------------


def run_model(PRETRAINED_MODEL_FOLDER_, TRAINING_FOLDER_, TEST_FOLDER_, OUTPUT_FOLDER_, WBC_LABEL: str = ''):
    use_fp16 = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()

    # ARGUMENTS
    MODEL_CONFIG = ViTConfig.from_pretrained(PRETRAINED_MODEL_FOLDER_)
    TRAINING_ARGS = TrainingArguments(
        output_dir=OUTPUT_FOLDER_,
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
        # report_to='none',  # alt is tensorboard
        load_best_model_at_end=True,
    )

    # DATA
    print(f"LOAD DATA: \nsource: {TRAINING_FOLDER_}, output: {OUTPUT_FOLDER_}")
    train_ds, val_ds = utils.load_and_split_dataset(TRAINING_FOLDER_, 0.15)
    test_ds = utils.load_and_split_dataset(TEST_FOLDER_, 0.0)

    # Use lambda to pass processor to the transform functions
    processor = ViTImageProcessor(MODEL_CONFIG)
    prepared_train_ds = train_ds.with_transform(
        lambda example_batch: utils.transform_with_augmentation(processor, example_batch))
    prepared_val_ds = val_ds.with_transform(lambda example_batch: utils.transform(processor, example_batch))
    prepared_test_ds = test_ds.with_transform(lambda example_batch: utils.transform(processor, example_batch))

    # MODEL
    print("INITIALIZE MODEL")
    label_titles = prepared_train_ds.features['label'].names
    model = ViTForImageClassificationFromMAE(PRETRAINED_MODEL_FOLDER_)

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
    train_results = trainer.train()
    utils.save_session(OUTPUT_FOLDER_, trainer, model, MODEL_CONFIG, TRAINING_ARGS)

    # EVALUATE
    print("EVALUATE RESULTS")
    test_results = trainer.predict(prepared_test_ds)
    print(utils.compute_metrics(test_results))
    utils.plot_losses_train_eval(trainer, f"{WBC_LABEL}\nLoss Landscape, MAE transfer learning", OUTPUT_FOLDER_)


# ------------------
# Run
# ------------------


@dataclass
class Folder:
    dataset_label: str
    training_folder: str
    output_folder: str
    pretrained_model_folder: str = './models/MAE_full100_3'
    test_folder: str = './data/WBC_100/val/data/'


if __name__ == '__main__':
    WBC1 = Folder(dataset_label="WBC1", training_folder='./data/WBC_1/train/data/', output_folder='./models/transfer_WBC_1')
    WBC10 = Folder(dataset_label="WBC10", training_folder='./data/WBC_10/train/data/', output_folder='./models/transfer_WBC_10')
    WBC50 = Folder(dataset_label="WBC50", training_folder='./data/WBC_50/train/data/', output_folder='./models/transfer_WBC_50')
    WBC100 = Folder(dataset_label="WBC100", training_folder='./data/WBC_100/train/data/', output_folder='./models/transfer_WBC_1')

    # for item in [WBC1, WBC10, WBC50, WBC100]:
    for item in [WBC1]:
        run_model(item.pretrained_model_folder, item.training_folder, item.test_folder, item.output_folder, item.dataset_label)
