# Importing required libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Given data
data = {
    "WBC_Datasize": [0.01, 0.1, 0.5, 1, 0.01, 0.1, 0.5, 1],
    "Accuracy": [0.61227, 0.73148, 0.94965, 0.97164, 0.61285, 0.90104, 0.95486, 0.97049],
    "Precision": [0.12252, 0.36244, 0.89466, 0.95332, 0.12257, 0.78875, 0.90775, 0.93961],
    "Recall": [0.19981, 0.37126, 0.89335, 0.93709, 0.2, 0.78074, 0.91559, 0.93979],
    "Type": ['From Scratch', 'From Scratch', 'From Scratch', 'From Scratch',
             'Transfer Learning', 'Transfer Learning', 'Transfer Learning', 'Transfer Learning']
}

# Converting data to DataFrame
df = pd.DataFrame(data)

# Setting the aesthetics to 'darkgrid'
sns.set(style="darkgrid")

# Initializing the figure
plt.figure(figsize=(12, 6))

# Defining a color mapping for 'Type'
color_mapping = {'From Scratch': 'pink', 'Transfer Learning': 'blue'}

# Creating the line plot with the specified changes
sns.lineplot(x="WBC_Datasize", y="value", hue="Type", style="variable", markers=True,
            markersize=10,
             palette=color_mapping,
             data=pd.melt(df, ['WBC_Datasize', 'Type'], ['Accuracy', 'Precision', 'Recall']))

# Update so x axes only shows 0.01, 0.1, 0.5, 1
plt.xticks([0.01, 0.1, 0.5, 1])


# Adding labels and title
plt.xlabel("WBC Dataset Size")
plt.ylabel("Metric Scores")
plt.title(f"Comparison of Metrics\nTransfer Learning vs From Scratch")



# Adjusting the legend
legend = plt.legend(title="Metrics and Type", bbox_to_anchor=(1.05, 1), loc='upper left')
legend.set_title("Legend")

# Displaying the plot
plt.tight_layout()
plt.show()
