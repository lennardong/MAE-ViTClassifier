# Utilities
from dataclasses import dataclass

# Viz + Analytics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Helper functions
import util.utils as utils


########################################
# Inputs
########################################

@dataclass
class EvaluationSet:
    label: str
    model_path: str
    test_path: str = "./data/WBC_100/val/data/"
    model_config: str = "model_config.json"
    model_state_dict: str = "model_state_dict.pth"
    log_history: str = "log_history.json"
    training_args: str = "training_args.json"
    processor_args: str = "processor_args.json"


id2label = {
    0: 'Basophil',
    1: 'Eosinophil',
    2: 'Lymphocyte',
    3: 'Monocyte',
    4: 'Neutrophil'
}

########################################
# Eval individual & Build Aggregate
########################################

WBC1_scratch = EvaluationSet(label="WBC1_scratch", model_path="./models/fromScratch_WBC_1")
WBC10_scratch = EvaluationSet(label="WBC10_scratch", model_path="./models/fromScratch_WBC_10")
WBC50_scratch = EvaluationSet(label="WBC50_scratch", model_path="./models/fromScratch_WBC_50")
WBC100_scratch = EvaluationSet(label="WBC100_scratch", model_path="./models/fromScratch_WBC_100")

preds_df = pd.DataFrame()
metrics_df = pd.DataFrame()

for item in [WBC10_scratch]:
    label = item.label
    model = utils.load_model_fromscratch(item.model_path, item.model_config, item.model_state_dict)
    trainer = utils.load_trainer(item.model_path, item.training_args, model)
    training_logs = utils.load_training_logs(item.model_path, item.log_history)
    processor, test_ds = utils.load_test_ds(item.model_path, item.processor_args, item.test_path)

    print(model)

    # -------------------
    # Evaluate training
    # -------------------

    # Plot: loss curves vs accuracy

    # -------------------
    # Evaluate predictions
    # -------------------

    # Construct prediction DF: predicted, actual, prediction_score
    pred_df = pd.DataFrame()

    # Construct metrics DF: Accuracy Precision Recall F1
    metric_df = pd.DataFrame()

    # Plot: Confusion Matrix

    # Plot: ROC Curve


########################################
# Eval Aggregate (spread of scores)
########################################

# box and whisker plots, x: score, y: model, hue: cell type. Data: correct predictions

# box and whisker plots, x: score, y: model, hue: cell type. Data: wrong predictions
