########################################################
# Load Model
########################################################
import torch
from transformers import ViTConfig
from main_vitclassifier_fromscratch import ViTForImageClassificationFromScratch
import json
from transformers import ViTImageProcessor
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix, multilabel_confusion_matrix
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


###########################
# Helper Functions
###########################
def load_model(artefact_location_):
    # Load the model config from a JSON file
    with open(artefact_location_ + '/model_config.json', 'r') as f:
        loaded_config_dict = json.load(f)

    loaded_config_ = ViTConfig.from_dict(loaded_config_dict)

    model_ = ViTForImageClassificationFromScratch(loaded_config_)  # Initialize the model
    model_.load_state_dict(torch.load(artefact_location_ + '/model_state_dict.pth', map_location=torch.device('cpu')))
    model_.eval()  # freeze model for inference
    processor_ = ViTImageProcessor(loaded_config_)
    return model_, processor_


def build_df_pred(img_folder_: str, processor_, model_, model_tag_: str, id2label_: dict, lim: int = 0):
    """
    Builds a DataFrame with prediction and actual label information for multiple images.

    Parameters:
    - img_folder: str, path to the root folder containing subfolders with images
    - processor: ViTImageProcessor, image processor object
    - model: PyTorch model for prediction
    - model_tag: str, tag for the model
    - id2label_: dict, a dictionary mapping class IDs to their labels
    - lim: int, limit the number of images to be processed (0 for no limit)

    Returns:
    - df_pred: DataFrame, contains prediction and label information
    """
    rows_list = []

    for subdir, _, files in os.walk(img_folder_):
        true_label = os.path.basename(subdir)  # Use subfolder name as the true label

        print(f'\nProcessing {true_label}...')
        count = 0
        for img_name in tqdm(files):
            if lim != 0 and count >= lim:
                print(f'\nLimit of {lim} images reached.')
                break

            img_path = os.path.join(subdir, img_name)

            # Skip if not an image file
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Encode the image
            im = Image.open(img_path)
            encoding = processor_(images=im, return_tensors="pt")
            pixel_values = encoding['pixel_values']

            with torch.no_grad():  # No gradient is needed for evaluation
                # Generate Softmax results
                outputs = model_(pixel_values)
                results = outputs.logits.softmax(dim=1)

                # Process the results
                result_idx = results.argmax(1).tolist()[0]
                predicted_label = id2label_[result_idx]
                predicted_val = results.tolist()[0][result_idx]

                # Check if the prediction is correct
                is_correct = (predicted_label == true_label)

                # Append to DataFrame
                new_row = {
                    'model': model_tag_,
                    'input': img_name,
                    'predicted_label': predicted_label,
                    'predicted_val': predicted_val,
                    'actual_label': true_label,
                    'is_correct': is_correct
                }
                rows_list.append(new_row)
                count += 1

    df_pred_ = pd.DataFrame(rows_list)

    return df_pred_


def generate_statistics(df_pred_):
    # List to store metrics for each model
    model_metrics = []

    # Extract unique models
    models = df_pred_['model'].unique()

    # For each unique model, calculate the metrics
    for model_ in models:
        df_model = df_pred_[df_pred_['model'] == model_]

        accuracy = accuracy_score(df_model['actual_label'], df_model['predicted_label'])
        report = classification_report(df_model['actual_label'], df_model['predicted_label'], output_dict=True,
                                       zero_division=0)

        # Extracting other metrics from the classification report
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']

        model_metrics.append([model_, accuracy, precision, recall, f1])

    # Convert list to DataFrame
    df_eval = pd.DataFrame(model_metrics, columns=['model', 'accuracy', 'precision', 'recall', 'f1'])

    return df_eval

###########################
# MAIN
###########################

if __name__ == '__main__':
    ARTEFACT_LOCATION = './models/fromScratch_WBC_1'  # Path where the model was saved
    TEST_DIRECTORY = './data/WBC_100/val/data/'
    ID2LABEL = {
        0: 'Basophil',
        1: 'Eosinophil',
        2: 'Lymphocyte',
        3: 'Monocyte',
        4: 'Neutrophil'
    }

    EVALS = {
        'fromScratch_WBC1': './models/fromScratch_WBC_1',
        'fromScratch_WBC10': './models/fromScratch_WBC_10',
        'fromScratch_WBC50': './models/fromScratch_WBC_50',
        'fromScratch_WBC100': './models/fromScratch_WBC_100',
        'transferLearning_WBC1': './models/transfer_WBC_1',
        'transferLearning_WBC10': './models/transfer_WBC_10',
        'transferLearning_WBC50': './models/transfer_WBC_50',
        'transferLearning_WBC100': './models/transfer_WBC_100',
    }

    for model_tag, artefact_location in EVALS.items():
        print(f'######################\n#EVALUATING "{model_tag}"...')
        # Load model
        model, processor = load_model(artefact_location)

        # Build DFs
        df_pred = build_df_pred(TEST_DIRECTORY, processor, model, model_tag, ID2LABEL)
        df_eval = generate_statistics(df_pred)

        # Save DFs
        df_pred.to_csv(os.path.join(artefact_location, 'df_pred.csv'), index=False)
        df_eval.to_csv(os.path.join(artefact_location, 'df_eval.csv'), index=False)

        # Get model metrics
        print(df_eval)

#################
# OLD
'''
# LOAD MODEL

# Load the model config from a JSON file
with open(ARTEFACT_LOCATION + '/model_config.json', 'r') as f:
    loaded_config_dict = json.load(f)

loaded_config = ViTConfig.from_dict(loaded_config_dict)

model = ViTForImageClassificationFromScratch(loaded_config)  # Initialize the model
model.load_state_dict(torch.load(ARTEFACT_LOCATION + '/model_state_dict.pth', map_location=torch.device('cpu')))
model.eval()  # freeze model for inference
processor = ViTImageProcessor(loaded_config)
'''
# def predict(path: str):
#     # Encode
#     im = Image.open(path)
#
#     encoding = processor(images=im, return_tensors="pt")
#     encoding.keys()
#     pixel_values = encoding['pixel_values']
#
#     # Softmax results
#     outputs = model(pixel_values)
#     results = outputs.logits.softmax(1)
#
#     # Process
#     result_idx = results.argmax(1).tolist()[0]
#     result_label = ID2LABEL[result_idx]
#     result_p = results.tolist()[0][result_idx]
#
#     return result_label, result_p
#
#
# l, p = predict(IMG_PATH)
#
# print(f'{l}: {p}')
#
# def build_df_pred_v1(img_folder: str, processor, model, model_tag: str, actual_label_dict: dict):
#     """
#     Builds a DataFrame with prediction and actual label information for multiple images.
#
#     Parameters:
#     - img_folder: str, path to the folder containing images
#     - processor: ViTImageProcessor, image processor object
#     - model: PyTorch model for prediction
#     - model_tag: str, tag for the model
#     - actual_label_dict: dict, a dictionary mapping image filenames to their actual labels
#
#     Returns:
#     - df_pred: DataFrame, contains prediction and label information
#     """
#     # Initialize an empty DataFrame
#     df_pred_ = pd.DataFrame(columns=['model', 'input', 'predicted_label', 'predicted_val', 'actual_label', 'is_correct'])
#
#     # Iterate through each image in the folder
#     for img_name in os.listdir(img_folder):
#         img_path = os.path.join(img_folder, img_name)
#
#         # Skip if not an image file
#         if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             continue
#
#         # Encode the image
#         im = Image.open(img_path)
#         encoding = processor(images=im, return_tensors="pt")
#         encoding.keys()
#         pixel_values = encoding['pixel_values']
#
#         # Generate Softmax results
#         outputs = model(pixel_values)
#         results = outputs.logits.softmax(dim=1)
#
#         # Process the results
#         result_idx = results.argmax(1).tolist()[0]
#         predicted_label = actual_label_dict[result_idx]
#         predicted_val = results.tolist()[0][result_idx]
#
#         # Retrieve the actual label from the dictionary
#         actual_label = actual_label_dict.get(img_name, "Unknown")
#
#         # Check if the prediction is correct
#         is_correct = (predicted_label == actual_label)
#
#         # Append to DataFrame
#         new_row = {
#             'model': model_tag,
#             'input': img_name,
#             'predicted_label': predicted_label,
#             'predicted_val': predicted_val,
#             'actual_label': actual_label,
#             'is_correct': is_correct
#         }
#         df_pred_ = df_pred_.append(new_row, ignore_index=True)
#
#     return df_pred_
