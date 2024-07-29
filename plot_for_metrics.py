import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

"""
Plot metrics from a CSV file generated after YOLOv8 training

"""

def log(msg: str) -> str:
    print(msg)    

# Parse the arguments
parser = argparse.ArgumentParser(description='Plot the metrics from a CSV file')
parser.add_argument('--path', type=str, required=True, help='Path to the CSV file')
parser.add_argument('--metric', type=str, required=True, help='Name of the metric to plot')
parser.add_argument('--epochs', type=int, default=-1, help='Number of epochs to plot')
args = parser.parse_args()

metrics_to_csv = {
    'precision': 'metrics/precision(B)',
    'recall': 'metrics/recall(B)',
    'map': 'metrics/mAP50(B)',
}

metrics_to_plot_title = {
    'precision': 'Precision',
    'recall': 'Recall',
    'map': 'mAP50',
}
metric_value = args.metric.lower()


# Load the data from the CSV file
log(f'Loading {metrics_to_plot_title[metric_value]} data from {args.path}...')
file_path = args.path  # Replace with the path to your CSV file
data = pd.read_csv(file_path)
data.columns = [col.replace(' ', '') for col in data.columns]
epochs_quantity = args.epochs if args.epochs > 0 else len(data['epoch'])


log(f'Setting to plot the last {epochs_quantity} epochs...')
# Extract relevant columns
epochs = data['epoch'][-epochs_quantity:] # The last n epochs
metrics = data[metrics_to_csv[metric_value]][-epochs_quantity:]

# Set the style
sns.set_theme(style="whitegrid")

# Create a plot
plt.figure(figsize=(20, 8))
plt.plot(epochs, metrics, marker='', linestyle='-', 
         color='b', label=metrics_to_plot_title[metric_value], markersize=8, 
         linewidth=2)

# Add titles and labels
plt.title(metrics_to_plot_title[metric_value] + ' Metric Over Epochs', fontsize=20, fontweight='bold')
plt.xlabel('Epochs', fontsize=16)
plt.ylabel(metrics_to_plot_title[metric_value], fontsize=16)
plt.ylim(metrics.min() - 0.05, metrics.max() + 0.05)

step = 20

plt.xticks(ticks=range(0, len(epochs)+1, step), labels=range(0, len(epochs)+1, step), fontsize=12) # Show every 10th epoch
plt.yticks(fontsize=12)

# Highlighting the max metric value
max_metric = metrics.max()
max_epoch = epochs[metrics.idxmax()]
last_metric = metrics.iloc[-1]
last_epoch =  epochs.iloc[-1]
log(f'Plotting the max value and final value of the {metrics_to_plot_title[metric_value]} metric...')
plt.plot(max_epoch, max_metric, marker='o', markersize=6, color='r')
plt.text(max_epoch, max_metric, f'Max: {max_metric:.2f}', fontsize=12, ha='right')

plt.plot(last_epoch, last_metric, marker='o', markersize=6, color='r')
plt.text(last_epoch, last_metric + 0.01, f'Final: {last_metric:.2f}', fontsize=12, ha='right', va='bottom')

# Add a legend
plt.legend(fontsize=14)

log(f'Displaying the plot!\nWaiting for the user to close the plot . . . ')
# Show the plot
plt.show()

log(f'Program ended !')
