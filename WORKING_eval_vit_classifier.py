import torch
import matplotlib.pyplot as plt
from transformers import ViTConfig, Trainer, TrainingArguments
import util.utils as utils
import json
from main_vitclassifier_fromscratch import ViTForImageClassificationFromScratch
from transformers import ViTImageProcessor
from main_vitclassifier_transferlearning import ViTForImageClassificationFromMAE
import os
# Path where the model was saved
SAVE_PATH = './models/fromScratch_WBC_10'
TEST_PATH = './data/WBC_100/val/data/'

########################################
# Load pre-trained model
########################################
filenames = {
    'model_config': 'model_config.json',
    'model_state_dict': 'model_state_dict.pth',
    'log_history': 'log_history.json',
    'training_args': 'training_args.json',
    'processor_args': 'processor_args.json'
}

# %%
def load_model(directory_path: str, filename: str) -> ViTForImageClassificationFromScratch:
    """Returns the model"""

    file = os.path.join(directory_path, filename)
    with open(file, 'r') as f:
        loaded_config_dict = json.load(f)

    MODEL_ARGS = ViTConfig.from_dict(loaded_config_dict)
    model = ViTForImageClassificationFromScratch(MODEL_ARGS)  # Initialize the model
    model.load_state_dict(torch.load(SAVE_PATH + '/model_state_dict.pth', map_location=torch.device('cpu')))

    # Make sure to call this if you plan to use the model for inference
    model.eval()

    return model


load_model(SAVE_PATH, filenames['model_config'])
#%%

def load_trainer(directory_path: str, filename: str) -> Trainer:
    """Returns the trainer to run inference"""
    file = os.path.join(directory_path, filename)
    with open(file, 'r') as f:
        loaded_config_dict = json.load(f)
    pass

# %%

def load_test_ds(path: str) -> Dataset:
    """Returns the test set df"""
    pass


def load_training_logs(path: str) -> dict:
    """Returns the training logs for eval"""
    pass


# Load the model config from a JSON file

########################################
# Plot the training metrics
########################################


train_loss = [log['loss'] for log in log_history if 'loss' in log]
train_epochs = [log['epoch'] for log in log_history if 'loss' in log]
eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
eval_epochs = [log['epoch'] for log in log_history if 'eval_loss' in log]

plt.plot(train_epochs, train_loss, label='Train Loss')
plt.scatter(eval_epochs, eval_loss, color='red', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

########################################
# Run Predictions
########################################


PROCESSOR_ARGS = {
        "do_resize": True,  # resize to default (224, 224)
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],  # Normalize to [0.5, 0.5, 0.5]
        "image_std": [0.5, 0.5, 0.5]  # Normalize with standard deviation [0.5, 0.5, 0.5]
        # rest will take default values
    }

processor = ViTImageProcessor(**PROCESSOR_ARGS)
test_ds = utils.load_and_split_dataset(TEST_PATH, 0.0)
prepared_test_ds = test_ds.with_transform(lambda example_batch: utils.transform(processor, example_batch))

# Load Training ARgs
with open(SAVE_PATH + '/training_args.json', 'r') as f:
    loaded_training_args_dict = json.load(f)

# Recreate the TrainingArguments object
use_fp16 = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
training_args = TrainingArguments(**loaded_training_args_dict)

trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=utils.collate_fn_classifier,
        compute_metrics=utils.compute_metrics,
    )

# Run inference
test_results = trainer.predict(prepared_test_ds)
metrics = utils.compute_metrics(test_results)
print(f"Test Accuracy: {metrics['accuracy']}")
