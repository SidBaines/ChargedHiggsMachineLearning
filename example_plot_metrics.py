#!/usr/bin/env python3
"""
Example script demonstrating how to use the metrics plotting functionality.

This script shows different ways to load and visualize training metrics
from locally stored JSON files.
"""

import os
from utils.metrics_storage import load_multiple_experiments, MetricsStorage
from plot_training_metrics import MetricsPlotter

def example_single_experiment():
    """Example: Load and plot metrics from a single experiment."""
    # Path to your experiment directory
    experiment_dir = "output/20250708-110653_TrainingOutput/"
    
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory {experiment_dir} not found!")
        return
    
    # Load the experiment
    experiments = load_multiple_experiments([experiment_dir])
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Create plotter
    plotter = MetricsPlotter(experiments)
    
    # List available metrics
    print("Available training metrics:")
    for metric in plotter.get_available_metrics("train"):
        print(f"  {metric}")
    
    print("\nAvailable validation metrics:")
    for metric in plotter.get_available_metrics("val"):
        print(f"  {metric}")
    
    # Plot specific metrics - traditional way (single split)
    plotter.plot_single_metric("train/accuracy_qqbb", title="Training Accuracy (qqbb channel)")
    
    # NEW: Plot both training and validation on same axis
    plotter.plot_single_metric("accuracy_qqbb", title="Training vs Validation Accuracy", combine_train_val=True)
    
    # Create a dashboard with all key metrics
    plotter.create_summary_dashboard()
    
    # NEW: Create dashboard with train/val comparisons
    plotter.create_summary_dashboard(save_dir="single_exp_comparison", combine_train_val=True)

def example_compare_experiments():
    """Example: Compare metrics from multiple experiments."""
    if 0:
        # Paths to your experiment directories
        experiment_dirs = [
            "output/20250708-110653_TrainingOutput/",
            "output/20250708-110955_TrainingOutput/",
            "output/20250708-111001_TrainingOutput/",
        ]
        
        # Custom experiment names for cleaner plots
        experiment_names = ["Baseline", "With Dropout", "Different LR"]
        
        # Filter to existing directories
        existing_dirs = [d for d in experiment_dirs if os.path.exists(d)]
        existing_names = experiment_names[:len(existing_dirs)]
        
        if not existing_dirs:
            print("No experiment directories found!")
            return
        
        # Load experiments
        experiments = load_multiple_experiments(existing_dirs, existing_names)
    else:
        experiments = {
            # "output/20250708-110653_TrainingOutput/" : "[256]",
            # "output/20250708-110955_TrainingOutput/" : "[256, 256]", # Copied below!
            # "output/20250708-111001_TrainingOutput/" : "[512]",
            # "output/20250708-140222_TrainingOutput/" : "[256, 256, 256]", # Copied below!
            # "output/20250708-140205_TrainingOutput/" : "[128, 128, 128, 128, 128]", # Copied below!

            "output/20250708-110955_TrainingOutput/" : "[256, 256]" + r"LR $5\times 10^{-6} \rightarrow 2\times 10^{-8}$",
            "output/20250708-140205_TrainingOutput/" : "[128, 128, 128, 128, 128]" + r"LR $5\times 10^{-6} \rightarrow 2\times 10^{-8}$",
            "output/20250708-140222_TrainingOutput/" : "[256, 256, 256]" + r"LR $5\times 10^{-6} \rightarrow 2\times 10^{-8}$",
            "output/20250708-140300_TrainingOutput/" : "[256, 256]" + r"LR $5\times 10^{-4} \rightarrow 2\times 10^{-7}$",
            "output/20250708-150441_TrainingOutput/" : "[512, 512, 512]" + r"LR $5\times 10^{-4} \rightarrow 2\times 10^{-7}$",
        }
        existing_experiments = {d:experiments[d] for d in experiments.keys() if os.path.exists(d)}
        
        # Load experiments
        experiments = load_multiple_experiments(list(existing_experiments.keys()), list(existing_experiments.values()))
    
    # Create plotter
    plotter = MetricsPlotter(experiments)
    
    # Compare loss curves
    # plotter.plot_loss_curves()
    
    # Compare accuracy metrics
    plotter.plot_accuracy_comparison(combine_train_val=True)
    
    # Compare specific metrics
    metrics_to_compare = [
        "train/accuracy_qqbb",
        "val/accuracy_qqbb", 
        "train/auc_qqbb_mHlow95_mHhigh140",
        "val/auc_qqbb_mHlow95_mHhigh140"
    ]
    plotter.plot_multiple_metrics(metrics_to_compare, save_path="comparison_plots/key_performance_metrics.png", title="Key Performance Metrics")
    plotter.plot_multiple_metrics(metrics_to_compare, save_path="comparison_plots/key_performance_metrics.pdf", title="Key Performance Metrics")
    
    # NEW: Compare metrics with train/val on same axis
    base_metrics_to_compare = [
        # "accuracy_qqbb",
        # "auc_qqbb_mHlow95_mHhigh140"
        {'name':"loss/loss_total", 'LegendEntry':r'Cross-entropy loss', 'ylim':(0.61, 0.68)},
        {'name':"auc_qqbb_mHlow95_mHhigh140", 'LegendEntry':r'ROC AUC (central $m_{h}$)', 'ylim':(0.675, 0.765)},
        # "auc_qqbb_mHlow0_mHhigh10000000":'AUC (all mH)',
        {'name':"_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_0.8_mHlow0_mHhigh10000000000.0", 'LegendEntry':r'$S_{B_{200}}$ (0.8TeV $H^{+}$)', 'ylim':(300, 425)},
        {'name':"_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_3.0_mHlow0_mHhigh10000000000.0", 'LegendEntry':r'$S_{B_{200}}$ (3.0TeV $H^{+}$)', 'ylim':(1300, 1800)},
    ]
    plotter.plot_multiple_metrics([base_metrics_to_compare[i]['name'] for i in range(len(base_metrics_to_compare))], 
                                    save_path="comparison_plots/train_val_comparison.png", 
                                    title="Train vs Val Comparison", 
                                    combine_train_val=True, 
                                    metric_names_to_legend=[base_metrics_to_compare[i]['LegendEntry'] for i in range(len(base_metrics_to_compare))],
                                    ylims=[base_metrics_to_compare[i]['ylim'] for i in range(len(base_metrics_to_compare))]
                                    )
    plotter.plot_multiple_metrics([base_metrics_to_compare[i]['name'] for i in range(len(base_metrics_to_compare))], 
                                  save_path="comparison_plots/train_val_comparison.pdf", 
                                  title="Train vs Val Comparison", 
                                  combine_train_val=True, 
                                  metric_names_to_legend=[base_metrics_to_compare[i]['LegendEntry'] for i in range(len(base_metrics_to_compare))],
                                  ylims=[base_metrics_to_compare[i]['ylim'] for i in range(len(base_metrics_to_compare))]
                                  )
    
    # Save all plots to a directory
    # plotter.create_summary_dashboard(save_dir="comparison_plots")

def example_programmatic_access():
    """Example: Programmatically access metric values for custom analysis."""
    experiment_dir = "output/20250708-111001_TrainingOutput/"
    
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory {experiment_dir} not found!")
        return
    
    # Load experiment directly using MetricsStorage
    storage = MetricsStorage(experiment_dir, "HighLevelClassifier")
    metrics = storage.load_metrics()
    config = storage.load_config()
    
    print(f"Experiment configuration:")
    print(f"  Learning rate: {config.get('learning_rate_high', 'N/A')}")
    print(f"  Batch size: {config.get('batch_size', 'N/A')}")
    print(f"  Architecture: {config.get('architecture', 'N/A')}")
    
    print(f"\nTraining epochs: {storage.get_epochs()}")
    
    # Access specific metric values
    if "train" in metrics and metrics["train"]:
        last_epoch = max(int(k) for k in metrics["train"].keys())
        last_train_metrics = metrics["train"][str(last_epoch)]
        
        print(f"\nFinal training metrics (epoch {last_epoch}):")
        for metric, value in last_train_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    if "val" in metrics and metrics["val"]:
        last_epoch = max(int(k) for k in metrics["val"].keys())
        last_val_metrics = metrics["val"][str(last_epoch)]
        
        print(f"\nFinal validation metrics (epoch {last_epoch}):")
        for metric, value in last_val_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")

def example_custom_plotting():
    """Example: Create custom plots with matplotlib."""
    import matplotlib.pyplot as plt
    
    experiment_dir = "output/20250708-111001_TrainingOutput/"
    
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory {experiment_dir} not found!")
        return
    
    # Load experiments
    experiments = load_multiple_experiments([experiment_dir])
    plotter = MetricsPlotter(experiments)
    
    # Get data for custom plotting
    exp_name = list(experiments.keys())[0]
    
    # Get training and validation accuracy
    train_epochs, train_acc = plotter.get_metric_values(exp_name, "train/accuracy_qqbb")
    val_epochs, val_acc = plotter.get_metric_values(exp_name, "val/accuracy_qqbb")
    
    # Create custom plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_acc, 'b-', label='Training', linewidth=2)
    plt.plot(val_epochs, val_acc, 'r--', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy (qqbb channel)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # print("=== Single Experiment Example ===")
    # example_single_experiment()
    
    print("\n=== Compare Multiple Experiments Example ===")
    example_compare_experiments()
    
    # print("\n=== Programmatic Access Example ===")
    # example_programmatic_access()
    
    # print("\n=== Custom Plotting Example ===")
    # example_custom_plotting() 