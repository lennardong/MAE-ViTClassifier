"""
This script ingests a series of prebuilt prediction and evalution CSVs from the model folders
It consolidates the files and plots the spread of P values
"""

########################################################
# Load Model
########################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


###########################
# MAIN
###########################


if __name__ == '__main__':

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

    # Define filenames for DFs
    data_eval = "df_eval.csv"
    data_preds = "df_pred.csv"

    # Consolidate the dataframes
    eval_dfs = []
    preds_dfs = []

    # Build a consolidated DataFrame for evaluation
    for model_name, folder_location in EVALS.items():
        # Paths to the CSVs
        eval_csv_path = f"{folder_location}/{data_eval}"
        preds_csv_path = f"{folder_location}/{data_preds}"

        # Read the CSVs
        df_eval_temp = pd.read_csv(eval_csv_path)
        df_preds_temp = pd.read_csv(preds_csv_path)

        # Optionally, add a column to specify which model the data came from:
        df_eval_temp['model'] = model_name
        df_preds_temp['model'] = model_name

        # Store the temporary dataframes in the lists
        eval_dfs.append(df_eval_temp)
        preds_dfs.append(df_preds_temp)

    df_eval_all = pd.concat(eval_dfs, ignore_index=True)
    df_preds_all = pd.concat(preds_dfs, ignore_index=True)

    # Filter the dataframe to include only correct predictions
    correct_predictions = df_preds_all[df_preds_all['is_correct']]
    wrong_predictions = df_preds_all[~df_preds_all['is_correct']]

    # Plot Distributions
    for df in [correct_predictions, wrong_predictions]:
        flierprops = dict(marker='o', markersize=0.5, markerfacecolor='gray', alpha=0.5)
        g = sns.catplot(
            data=df,
            x="model",
            y="predicted_val",
            col="actual_label",
            kind="box",
            height=6,
            aspect=0.5,
            color='lightblue',  # Uniform color for all bars
            flierprops=flierprops  # Adjusting the properties for outlier dots
        )

        # Additional plot adjustments
        g.set_axis_labels("Model", "Predicted Value")
        g.set_titles("{col_name}")
        g.set_xticklabels(rotation=90)
        g.despine(left=True)

        plt.tight_layout()
        plt.show()
