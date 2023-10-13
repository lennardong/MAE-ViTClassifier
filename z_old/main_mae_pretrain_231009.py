""" Pre-training a 🤗 ViT model as an MAE (masked autoencoder), as proposed in https://arxiv.org/abs/2111.06377.

-----------------------------------
# LOG
-----------------------------------

## 20231006

Progress
- confirmed that dataloader is loading the right items 
- located other resources for MAE

Problems 
- Issue with trainer persists. Suspect its somethign to do with the batching.

Plans
- move items out of main() so its easier to debug in iPython
- rip out and simplify the training process. if reinstatement needed, refer to `main_mae_pretrain_231006.py`

"""

# System Utilities
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# Data Handling and Manipulation
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch

# Data Augmentation and Preprocessing
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode

# Model and Training Utilities
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
    ViTMAEConfig,
    ViTMAEForPreTraining,
    IntervalStrategy,
    SchedulerType,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)
# check_min_version("4.35.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")

################################
# Config Dataclasses
################################

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command line.

    Attributes:
        dataset_name: Name of a dataset from the datasets package. Default is "cifar10".
        dataset_config_name: The configuration name of the dataset to use (via the datasets library). Default is None.
        image_column_name: The column name of the images in the files. Default is None.
        train_dir: A folder containing the training data. Default is None.
        validation_dir: A folder containing the validation data. Default is None.
        train_val_split: Percent to split off of train for validation. Default is 0.15.
        max_train_samples: For debugging purposes or quicker training, truncate the number of training examples to this value if set. Default is None.
        max_eval_samples: For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set. Default is None.
    """
    dataset_name: Optional[str] = None
    dataset_paths: Optional[Dict[str, str]] =field(default_factory=dict)
    dataset_config_name: Optional[str] = None
    image_column_name: Optional[str] = None
    train_dir: Optional[str] = None
    validation_dir: Optional[str] = None
    train_val_split: Optional[float] = 0.15
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    def __post_init__(self):
        """
        Initialize a dictionary to store the paths for the training and validation directories.

        The dictionary is stored in the 'data_files' attribute of the instance. 
        The keys are 'train' and 'val', and the values are the respective directory paths.
        If neither 'train_dir' nor 'validation_dir' are provided, 'data_files' is set to None.
        """
        data_files = {}
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None

@dataclass
class ModelArguments:
    """
    Arguments for model/config/image processor to pre-train

    Attributes:
        model_name_or_path: The model checkpoint for weights initialization. Don't set if you want to train a model from scratch. Default is None.
        
        config_name: Pretrained config name or path if not the same as model_name_or_path. Default is None.
        
        config_overrides: Override some existing default config settings when a model is trained from scratch. Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index. Default is None.
        
        cache_dir: Where do you want to store the pretrained models downloaded from s3. Default is None.
        
        model_revision: The specific model version to use (can be a branch name, tag name or commit id). Default is "main".
        
        image_processor_name: Name or path of preprocessor config. Default is None.
        
        token: The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`). Default is None.
        
        mask_ratio: The ratio of the number of masked tokens in the input sequence. Default is 0.75.
        
        norm_pix_loss: Whether or not to train with normalized pixel values as target. Default is True.
    """
    model_name_or_path: Optional[str] = None
    config_name: Optional[str] = None
    config_overrides: Optional[str] = None
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    image_processor_name: Optional[str] = None
    image_size: int = 224
    token: Optional[str] = None
    # use_auth_token: Optional[bool] = None
    mask_ratio: float = 0.75
    norm_pix_loss: bool = True

@dataclass
class CustomTrainingConfig(TrainingArguments):
    """
    Custom training arguments that extend the default TrainingArguments from Hugging Face's Transformers library.

    Attributes:
        output_dir: The output directory where the model predictions and checkpoints will be written.
        do_train: Whether to run training.
        do_eval: Whether to run evaluation on the dev set.
        evaluation_strategy: Evaluation and Save strategy.
        base_learning_rate: Base learning rate used to calculate the absolute learning rate.
        lr_scheduler_type: The scheduler type to use.
        weight_decay: Weight decay for optimizer.
        num_train_epochs: Number of training epochs.
        warmup_ratio: Ratio for the warmup learning rate.
        per_device_train_batch_size: Batch size per device during training.
        per_device_eval_batch_size: Batch size for evaluation.
        logging_strategy: Logging strategy to use.
        logging_steps: Log every X updates steps.
        save_strategy: Save strategy to use.
        load_best_model_at_end: Whether or not to load the best model found during training at the end of training.
        save_total_limit: Limit the total amount of checkpoints, delete the older checkpoints.
        seed: Random seed for initialization.
    """
    
    output_dir: str = "./vit-mae-demo"
    do_train: bool = True
    do_eval: bool = True
    evaluation_strategy: IntervalStrategy = IntervalStrategy.EPOCH
    base_learning_rate: float = 1.5e-4
    lr_scheduler_type: SchedulerType = SchedulerType.COSINE
    weight_decay: float = 0.05
    num_train_epochs: int = 800
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    logging_strategy: IntervalStrategy = IntervalStrategy.STEPS
    logging_steps: int = 10
    save_strategy: IntervalStrategy = IntervalStrategy.EPOCH
    load_best_model_at_end: bool = True
    save_total_limit: int = 3
    seed: int = 1337
    logging_dir: str = "./logs"
    remove_unused_columns = False


############################
# Helper Functions
############################

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values}

# ---------------------

def setup_logging(training_config):
    """
    Sets up logging based on the training configuration.

    Parameters:
        training_config: The configuration object containing training parameters and settings.
    """
    # Basic logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set verbosity level if logging is enabled
    if training_config.should_log:
        transformers.utils.logging.set_verbosity_info()

    # Set the log level based on the training configuration
    log_level = training_config.get_process_log_level()
    logger = logging.getLogger()
    logger.setLevel(log_level) # logging.basicConfig(level=logging.WARNING)  # Will capture WARNING and above messages


    # Enable default and explicit logging formats
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log a summary of the training process
    logger.warning(
        f"Process rank: {training_config.local_rank}, device: {training_config.device}, n_gpu: {training_config.n_gpu}"
        + f"distributed training: {bool(training_config.parallel_mode.value == 'distributed')}, 16-bits training: {training_config.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_config}")

# ---------------------

def load_dataset_from_folders(paths: Dict[str, str], test_split_: float = 0.15) -> DatasetDict:
    """
    Flattens images in folders into one dataset with train/validation split.
    If test_split_ > 0, returns with 'train' and 'validation' keys. Else, just a train key.

    Arguments:
        paths: dictionary of dataset names and their paths.
        test_split: float value for test split ratio.
    """
    
    logger.warning('\n#################################')
    logger.warning('# LOADING DATASETS FROM FOLDERS')
    logger.warning('#################################')
    
    datasets = list()
    
    # Load datasets
    for dataset_name, folder_path in paths.items():
        logger.warning(f'loading: {dataset_name}')
        dataset = load_dataset(
            path= "imagefolder",
            data_dir = folder_path,
            split='train'
        )

        # Preprocess
        # TODO: Check if preprocessing is necessary

        # Add to list
        datasets.append(dataset)

    # Merge
    datasets_merged = concatenate_datasets(datasets)
    datasets_merged = DatasetDict({"train": datasets_merged})
    
    logger.warning(f"DATASET MERGED\n----------------\n{datasets_merged}")

    # Split
    if "validation" not in datasets_merged.keys() and test_split_ > 0.0:
        split = datasets_merged['train'].train_test_split(test_size=test_split_)
        datasets_merged["train"] = split["train"]
        datasets_merged["validation"] = split["test"]
    
    logger.warning(f'DATASET DICT\n-----------------\n{datasets_merged}')

    return datasets_merged


# TESTING
# config_datasetPaths = {
#     'pRCC': './data/pRCC_nolabel',
#     'CAM16_test': './data/CAM16_100cls_10mask/test/data/normal',
#     'CAM16_train': './data/CAM16_100cls_10mask/train/data/normal',
#     'CAM16_val': './data/CAM16_100cls_10mask/val/data/normal',
# }

# dataconfig_test = DataTrainingArguments(
#     dataset_paths= config_datasetPaths
# )

# test_ds = load_dataset_from_folders(dataconfig_test.dataset_paths, dataconfig_test.train_val_split)



# ---------------------

def load_model(mc: ModelArguments) -> Tuple[ViTMAEConfig, ViTImageProcessor,ViTMAEForPreTraining]:
    """
    Load ViTMAE model for pre-training. 

    If no folderpath given in `mc` (model config), a new one will be created using MAE Base_Patch16 Architecture from MAE repo
    If folderpath given in `mc` (model config), model will be loaded accordingly, but config updated with any customizations (e.g. mask ration)

    Args:
        mc (ModelArguments): dict-like DataObject containing model configuration.

    Returns:
        ViTMAEConfig : configuration
        ViTImageProcessor : ImageProcessor
        ViTMAEForPreTraining : Model
    """
    
    logger.warning('\n#################################')
    logger.warning('# LOADING MODEL FROM MODEL CONFIG')
    logger.warning('#################################')
    
    # Inits
    mc_kwargs = {
        "cache_dir": mc.cache_dir,
        "revision": mc.model_revision,
        "token": mc.token
    }

    # Model Architecture (model)
    # ref: https://huggingface.co/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEConfig
    # ----------------------
    # Check for pre-trained (BLOCK A)
    # Check for pre-trained
    if mc.model_name_or_path:
        config = ViTMAEConfig.from_pretrained(mc.model_name_or_path, **mc_kwargs)
        model = ViTMAEForPreTraining.from_pretrained(
            mc.model_name_or_path,
            # from_tf=bool(".ckpt" in mc.model_name_or_path),
            config=config,
            cache_dir= mc.cache_dir,
            revision= mc.model_revision,
            token=mc.token,
        )
        logger.warning(f"ViTMAE loaded from {mc.model_name_or_path}")
        
    # Else build from scratch, using MAE Base Architecture
    else: 
        config = ViTMAEConfig(
            hidden_size= 768,
            num_hidden_layers =12,
            num_attention_heads= 12,
            intermediate_size= 3072,
            hidden_act= "gelu",
            hidden_dropout_prob= 0.0, # dropout probability for all FC in embeddings, encoder, pooler
            attention_probs_dropout_prob = 0.0,
            image_size= mc.image_size,
            patch_size= 16, # resolution of each patch
            decoder_num_attention_heads= 16, # modified from default 12
            decoder_hidden_size= 512,
            decoder_num_hidden_layers= 8, 
            decoder_intermediate_size= 2048,
            mask_ratio= 0.75,
            norm_pix_loss= False
        )  
        model = ViTMAEForPreTraining(config)
        logger.warning("New ViTMAE instance from scratch")

    # Update configs with customized settings
    config.update({
        "mask_ratio": mc.mask_ratio,
        "norm_pix_loss": mc.norm_pix_loss,
        "image_size": mc.image_size,
    })

    # Image Processor (image_processor)
    # ref: https://huggingface.co/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor
    # ----------------------
    # Check for pre-trained to inherit
    if mc.model_name_or_path:
        image_processor = ViTImageProcessor.from_pretrained(mc.model_name_or_path, **mc_kwargs)
    else:
        image_processor = ViTImageProcessor(
            do_resize= True,
            size = {"height": mc.image_size, "width": mc.image_size}, # can be overriden by preprocess
            do_rescale= True,
            rescale_factor= 1/255, # <- rescale pixels from 0, 255 -> 0, 1
            do_normalize= True,
            image_mean= [0.485, 0.456, 0.406],
            image_std= [0.229, 0.224, 0.225],
            # ref: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        )

    logger.warning(f"CONFIG\n-------------\n{config}")
    logger.warning(f"IMAGE PROCESSOR\n-------------\n{image_processor}")
    logger.warning(f"MODEL\n-------------\n{model}")
    
    return config, image_processor, model

# TESTING 
# model_config_test = ModelArguments()
# config_test, image_processor_test, model_test = load_model(model_config_test)

# ----------------------

def get_col_names(ds: DatasetDict, dc: DataTrainingArguments, tc: TrainingArguments) -> str:
    """
    Returns column name of images. 
    """
    # Set for training or validation
    column_names = ds['train'].column_names if tc.do_train else ds['validation'].column_names
    
    # Get column name
    if dc.image_column_name is not None:
        image_column_name = dc.image_column_name
    elif "image" in column_names:
        image_column_name = "image"
    elif "img" in column_names:
        image_column_name = "img"
    else:
        image_column_name = column_names[0]

    logger.warning(f"IMAGE COLUMN NAME: {image_column_name}")
    
    return image_column_name

def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms."""
        print(f'PREPROCESS: {examples}')

        examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
        return examples
# DEBUG
# training_config_test = CustomTrainingConfig(
#     output_dir="./test/"
#     )
# col_name_test = get_col_names(test_ds, dataconfig_test, training_config_test)

# ----------------------

# Simplified Training Setup
def initialize_trainer(ds, data_config, training_config, preprocess_images, model):
    # Check for required datasets
    if "train" not in ds:
        raise ValueError("--do_train requires a train dataset")
    if "validation" not in ds:
        raise ValueError("--do_eval requires a validation dataset")

    # Optional: Shuffle and limit dataset size
    if data_config.max_train_samples:
        ds["train"] = ds["train"].shuffle(seed=training_config.seed).select(range(data_config.max_train_samples))
    if data_config.max_eval_samples:
        ds["validation"] = ds["validation"].shuffle(seed=training_config.seed).select(range(data_config.max_eval_samples))

    # Apply transformations
    ds["train"].set_transform(preprocess_images)
    ds["validation"].set_transform(preprocess_images)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collate_fn,
    )
    return trainer

# Simplified Training Execution
def execute_training(trainer, training_config, last_checkpoint):
    checkpoint = training_config.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_session()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

# Simplified Evaluation
def execute_evaluation(trainer):
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

############################
# CONFIGS
############################

config_datasetPaths = {
'pRCC_cropped': './data/pRCC_cropped',
'CAM16_test': './data/CAM16_100cls_10mask/test/data/normal',
'CAM16_train': './data/CAM16_100cls_10mask/train/data/normal',
'CAM16_val': './data/CAM16_100cls_10mask/val/data/normal',
}

data_arguments = DataTrainingArguments(
    dataset_paths= config_datasetPaths
)

# Model
model_arguments = ModelArguments(
    norm_pix_loss= False
)

# Training
training_arguments = CustomTrainingConfig(
    output_dir="./test/",
)


############################
# Main
############################

if __name__ == "__main__":
    setup_logging(training_arguments)

    # load data
    ds = load_dataset_from_folders(data_arguments.dataset_paths, data_arguments.train_val_split)
    config, image_processor, model = load_model(model_arguments)

    ### DATA AUGMENTATIONS ###

    size = (image_processor.size["height"], image_processor.size["width"])
    # NOTE hardcoding
    image_column_name = get_col_names(ds, data_arguments, training_arguments)
    # image_column_name = 'image'
    transforms = Compose(
        [
            Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )
    ### TRAINING SETUP ###
    if training_arguments.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_arguments.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_arguments.seed).select(range(data_arguments.max_train_samples))
        # Set the training transforms
        ds["train"].set_transform(preprocess_images)

    if training_arguments.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_arguments.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"].shuffle(seed=training_arguments.seed).select(range(data_arguments.max_eval_samples))
            )
        # Set the validation transforms
        ds["validation"].set_transform(preprocess_images)

    # Compute absolute learning rate
    total_train_batch_size = (
        training_arguments.train_batch_size * training_arguments.gradient_accumulation_steps * training_arguments.world_size
    )
    if training_arguments.base_learning_rate is not None:
        training_arguments.learning_rate = training_arguments.base_learning_rate * total_train_batch_size / 256

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=ds["train"] if training_arguments.do_train else None,
        eval_dataset=ds["validation"] if training_arguments.do_eval else None,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    ### TRAINING ###
    if training_arguments.do_train:
        checkpoint = None
        if training_arguments.resume_from_checkpoint is not None:
            checkpoint = training_arguments.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        # what plots would a researcher use to evaluate?
        # e.g. How to plot loss curve?

    # Evaluation
    if training_arguments.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    









