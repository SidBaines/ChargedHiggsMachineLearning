# Local Metrics Storage and Plotting System

This system provides local storage and visualization of training metrics, independent of wandb. It stores metrics in lightweight JSON format and provides flexible plotting capabilities.

## Overview

The system consists of three main components:

1. **`utils/metrics_storage.py`** - Utilities for storing and loading metrics locally
2. **`plot_training_metrics.py`** - Command-line script for plotting metrics  
3. **Modified `TrainHighLevelClassifier.py`** - Training script now saves metrics locally

## Features

- **Lightweight storage**: Metrics stored in human-readable JSON format
- **Easy comparison**: Compare metrics across multiple experiments
- **Train/Val comparison**: Plot training and validation metrics on the same axis for easy comparison
- **Flexible plotting**: Choose specific metrics to plot or create comprehensive dashboards
- **Programmatic access**: Easy to access metric values for custom analysis
- **Independence**: Works without wandb, purely local storage

## How It Works

### During Training

The modified training script automatically:
1. Creates a `MetricsStorage` instance in the experiment output directory
2. Saves the training configuration to `{experiment_name}_config.json`
3. Stores metrics at the end of each epoch in `{experiment_name}_metrics.json`

The metrics JSON file has this structure:
```json
{
  "train": {
    "0": {"accuracy_qqbb": 0.85, "auc_qqbb_mHlow0_mHhigh1000": 0.92, ...},
    "1": {"accuracy_qqbb": 0.87, "auc_qqbb_mHlow0_mHhigh1000": 0.93, ...}
  },
  "val": {
    "0": {"accuracy_qqbb": 0.82, "auc_qqbb_mHlow0_mHhigh1000": 0.90, ...},
    "1": {"accuracy_qqbb": 0.84, "auc_qqbb_mHlow0_mHhigh1000": 0.91, ...}
  },
  "train_MC": {...},
  "val_MC": {...}
}
```

### Metrics Stored

The system stores all metrics computed by the `HEPMetrics` class:

- **Accuracy metrics**: Overall and per-channel accuracy scores
- **AUC metrics**: Area under curve for different mass ranges and channels  
- **Signal selection metrics**: Expected signal counts for different background levels
- **Loss values**: Training and validation losses

## Train/Val Comparison Feature

The `combine_train_val` parameter allows you to plot both training and validation metrics on the same axis, making it easy to spot overfitting and compare learning curves.

### How it works:
- When `combine_train_val=True`, you specify the base metric name (e.g., `"accuracy_qqbb"`)
- The system automatically plots both `train/accuracy_qqbb` and `val/accuracy_qqbb`
- Training curves are shown as solid lines (-) with circles (o)
- Validation curves are shown as dashed lines (--) with squares (s)
- Different experiments get different colors

### Example usage:
```python
# Single metric comparison
plotter.plot_single_metric("accuracy_qqbb", combine_train_val=True)

# Multiple metrics with train/val comparison
plotter.plot_multiple_metrics(["accuracy_qqbb", "auc_qqbb_mHlow95_mHhigh140"], 
                             combine_train_val=True)

# Dashboard with train/val comparisons
plotter.create_summary_dashboard(combine_train_val=True)
```

This is particularly useful for:
- Detecting overfitting (when training continues to improve but validation plateaus)
- Comparing model architectures across both training and validation performance
- Understanding convergence behavior

## Usage

### Command Line Interface

#### Basic Usage
```bash
# Plot metrics from a single experiment
python plot_training_metrics.py --experiment_dirs output/20241201-123456_TrainingOutput/

# Compare multiple experiments with custom names
python plot_training_metrics.py \
    --experiment_dirs exp1/ exp2/ exp3/ \
    --experiment_names "Baseline" "High LR" "With Dropout"

# Save plots instead of displaying them
python plot_training_metrics.py \
    --experiment_dirs output/20241201-123456_TrainingOutput/ \
    --save_plots --output_dir my_plots/
```

#### Advanced Options
```bash
# List all available metrics
python plot_training_metrics.py \
    --experiment_dirs output/20241201-123456_TrainingOutput/ \
    --list_metrics

# Plot specific metrics
python plot_training_metrics.py \
    --experiment_dirs exp1/ exp2/ \
    --metrics "train/accuracy_qqbb" "val/accuracy_qqbb" "val/auc_qqbb_mHlow95_mHhigh140"

# Plot training and validation on same axis
python plot_training_metrics.py \
    --experiment_dirs exp1/ exp2/ \
    --metrics "accuracy_qqbb" "auc_qqbb_mHlow95_mHhigh140" \
    --combine_train_val

# Create comprehensive dashboard
python plot_training_metrics.py \
    --experiment_dirs exp1/ exp2/ \
    --dashboard --save_plots

# Create dashboard with train/val comparisons
python plot_training_metrics.py \
    --experiment_dirs exp1/ exp2/ \
    --dashboard --combine_train_val --save_plots
```

### Programmatic Usage

#### Load and Plot Single Experiment
```python
from utils.metrics_storage import load_multiple_experiments
from plot_training_metrics import MetricsPlotter

# Load experiment
experiments = load_multiple_experiments(["output/20241201-123456_TrainingOutput/"])
plotter = MetricsPlotter(experiments)

# Plot specific metric
plotter.plot_single_metric("train/accuracy_qqbb")

# Create dashboard
plotter.create_summary_dashboard(save_dir="plots/")
```

#### Compare Multiple Experiments
```python
# Load multiple experiments
experiment_dirs = ["exp1/", "exp2/", "exp3/"]
experiment_names = ["Baseline", "High LR", "With Dropout"]
experiments = load_multiple_experiments(experiment_dirs, experiment_names)

plotter = MetricsPlotter(experiments)

# Compare loss curves
plotter.plot_loss_curves()

# Compare specific metrics
metrics = ["train/accuracy_qqbb", "val/accuracy_qqbb"]
plotter.plot_multiple_metrics(metrics, title="Accuracy Comparison")

# Plot training and validation on same axis
plotter.plot_single_metric("accuracy_qqbb", combine_train_val=True, 
                          title="Training vs Validation Accuracy")

# Compare multiple metrics with train/val on same plots
base_metrics = ["accuracy_qqbb", "auc_qqbb_mHlow95_mHhigh140"] 
plotter.plot_multiple_metrics(base_metrics, combine_train_val=True,
                             title="Train vs Val Comparison")

# Create dashboard with train/val comparisons
plotter.create_summary_dashboard(save_dir="plots/", combine_train_val=True)
```

#### Direct Data Access
```python
from utils.metrics_storage import MetricsStorage

# Load metrics directly
storage = MetricsStorage("output/20241201-123456_TrainingOutput/", "HighLevelClassifier")
metrics = storage.load_metrics()
config = storage.load_config()

# Access specific values
epochs = storage.get_epochs()
final_epoch = max(epochs)
final_train_acc = metrics["train"][str(final_epoch)]["accuracy_qqbb"]
print(f"Final training accuracy: {final_train_acc:.3f}")
```

### Custom Analysis

```python
# Extract data for custom plotting
plotter = MetricsPlotter(experiments)
exp_name = list(experiments.keys())[0]

# Get raw data
epochs, values = plotter.get_metric_values(exp_name, "train/accuracy_qqbb")

# Use with any plotting library
import matplotlib.pyplot as plt
plt.plot(epochs, values)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
```

## Examples

See `example_plot_metrics.py` for comprehensive examples showing:

- Single experiment visualization
- Multi-experiment comparison  
- Programmatic data access
- Custom plotting with matplotlib

## File Structure

After training, your experiment directory will contain:
```
output/20241201-123456_TrainingOutput/
├── HighLevelClassifier_metrics.json    # Metrics data
├── HighLevelClassifier_config.json     # Training configuration  
├── TrainHighLevelClassifier.py         # Copy of training script
└── models/                             # Model checkpoints
    └── qqbb_Nplits2_ValIdx0/
        ├── chkpt12345.pth
        ├── mean.npy
        └── std.npy
```

## Choosing Metrics to Store

The current implementation stores all computed metrics. To customize which metrics are stored, you can modify the `store_epoch_metrics` call in the training script or filter metrics in the `MetricsStorage.store_epoch_metrics` method.

## Performance

- **Storage**: JSON files are typically 10-100KB per experiment
- **Loading**: Fast loading even for long training runs
- **Memory**: Metrics loaded on-demand, minimal memory footprint

## Troubleshooting

### Common Issues

1. **No metrics found**: Ensure the experiment directory contains `*_metrics.json` files
2. **Import errors**: Make sure you're running from the project root directory
3. **Empty plots**: Check that the metric names are correct using `--list_metrics`

### Debugging

```python
# Check what metrics are available
from utils.metrics_storage import MetricsStorage
storage = MetricsStorage("your/experiment/dir", "HighLevelClassifier")
print(storage.get_metric_keys("train"))
print(storage.get_epochs())
```

## Integration with Existing Workflow

This system works alongside your existing wandb logging - it doesn't replace it. Both systems can run simultaneously, giving you:

- **Wandb**: Real-time monitoring, cloud storage, team collaboration
- **Local storage**: Offline access, custom analysis, long-term archival

The local metrics are automatically stored every epoch without any impact on training performance. 