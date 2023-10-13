# Load the JSON data from the uploaded file
import json
from matplotlib import pyplot as plt

plt.style.use('seaborn-darkgrid')
# Path to the uploaded JSON file
file_path = './models/MAE_full100_3/trainer_state.json'

# Read the file and load the JSON data
with open(file_path, 'r') as f:
    log_history_data = json.load(f)

# Extract 'step', 'loss', and 'eval_loss' from the 'log_history' field
steps = []
losses = []
eval_losses = []
epochs = []

for entry in log_history_data['log_history']:
    if 'step' in entry and 'loss' in entry:
        steps.append(entry['step'])
        epochs.append(entry['epoch'])
        losses.append(entry['loss'])
    if 'step' in entry and 'eval_loss' in entry:
        eval_losses.append(entry['eval_loss'])

# Ensure the lengths match for plotting
if len(steps) != len(eval_losses):
    min_len = min(len(steps), len(eval_losses))
    steps = steps[:min_len]
    losses = losses[:min_len]
    eval_losses = eval_losses[:min_len]

# Plotting the loss and eval_loss by steps
plt.figure(figsize=(12, 6))

plt.axes().set_ylim([0.5, 1.0])
# plt.axes().set_yticks([i for i in range(0, 1, 0.1)])
plt.plot(steps, losses,linestyle='-', color='r', label='Training Loss')
plt.plot(steps, eval_losses, linestyle='--', color='b', label='Evaluation Loss')
# plt.plot(steps, losses, marker='o', linestyle='-', color='r', label='Training Loss')
# plt.plot(steps, eval_losses, marker='x', linestyle='--', color='b', label='Evaluation Loss')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Masked Autoencoder \nTraining and Evaluation Loss')

plt.legend()

# plt.axes().set_xticks(epochs)


# set y-axis range to start from 0

plt.grid(True)
plt.tight_layout()

plt.show()
