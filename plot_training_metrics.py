#!/usr/bin/env python3
"""
Plot Training Metrics Script

This script loads and plots training metrics from locally stored JSON files.
Supports single experiment visualization and multi-experiment comparisons.

Usage:
    python plot_training_metrics.py --experiment_dirs output/20241201-123456_TrainingOutput/
    python plot_training_metrics.py --experiment_dirs dir1/ dir2/ dir3/ --experiment_names exp1 exp2 exp3
    python plot_training_metrics.py --experiment_dirs dir1/ --metrics "train_MC/accuracy" "val_MC/accuracy" --save_plots
    python plot_training_metrics.py --experiment_dirs dir1/ --metrics "accuracy_qqbb" --combine_train_val --save_plots
    python plot_training_metrics.py --experiment_dirs dir1/ dir2/ --dashboard --combine_train_val
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from utils.metrics_storage import load_multiple_experiments
import seaborn as sns

# Set plotting style
plt.style.use('default')
plt.rcParams['text.usetex'] = True
sns.set_palette("husl")

class MetricsPlotter:
    """Class for plotting training metrics from JSON files."""
    
    def __init__(self, experiments: Dict[str, Dict], figsize=(12, 8)):
        self.experiments = experiments
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, len(experiments)))
        
    def get_available_metrics(self, split: str = "train_MC") -> List[str]:
        """Get all available metrics across all experiments."""
        all_metrics = set()
        for exp_name, exp_data in self.experiments.items():
            if split in exp_data and exp_data[split]:
                first_epoch = list(exp_data[split].keys())[0]
                metrics = exp_data[split][first_epoch].keys()
                all_metrics.update([f"{split}/{metric}" for metric in metrics])
        return sorted(list(all_metrics))
    
    def get_metric_values(self, exp_name: str, metric_name: str, split: Optional[str] = None) -> tuple:
        """Extract epochs and values for a specific metric from an experiment."""
        # Parse metric name (e.g., "train_MC/accuracy_qqbb" -> split="train_MC", metric="accuracy_qqbb")
        if split is None:
            if "/" in metric_name:
                split, metric = metric_name.split("/", 1)
            else:
                # Default to train if no split specified
                split, metric = "train_MC", metric_name
        else:
            metric = metric_name
            
        exp_data = self.experiments[exp_name]
        if split not in exp_data:
            return [], []
            
        epochs = []
        values = []
        
        for epoch_str, metrics in exp_data[split].items():
            try:
                epoch = int(epoch_str)
                if metric_name in metrics:
                    epochs.append(epoch)
                    values.append(metrics[metric_name])
            except (ValueError, KeyError):
                continue
                
        return epochs, values
    
    def plot_single_metric(self, metric_name: str, save_path: Optional[str] = None, 
                          show_legend: bool = True, title: Optional[str] = None,
                          combine_train_val: bool = False):
        """Plot a single metric across all experiments."""
        plt.figure(figsize=self.figsize)
        
        if combine_train_val:
            # Plot both train and val versions of the metric
            base_metric = metric_name.split('/')[-1] if '/' in metric_name else metric_name
            
            for i, (exp_name, _) in enumerate(self.experiments.items()):
                # Plot training
                train_epochs, train_values = self.get_metric_values(exp_name, f"train_MC/{base_metric}")
                if train_epochs and train_values:
                    plt.plot(train_epochs, train_values, label=f'{exp_name} (Train)', 
                            color=self.colors[i], linestyle='-', linewidth=2, marker='o', markersize=4)
                
                # Plot validation
                val_epochs, val_values = self.get_metric_values(exp_name, f"val_MC/{base_metric}")
                if val_epochs and val_values:
                    plt.plot(val_epochs, val_values, label=f'{exp_name} (Val)', 
                            color=self.colors[i], linestyle='--', linewidth=2, marker='s', markersize=4)
        else:
            # Original behavior - plot only the specified metric
            for i, (exp_name, _) in enumerate(self.experiments.items()):
                epochs, values = self.get_metric_values(exp_name, metric_name)
                if epochs and values:
                    plt.plot(epochs, values, label=exp_name, color=self.colors[i], 
                            linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('Epoch')
        base_metric_name = metric_name.split('/')[-1] if '/' in metric_name else metric_name
        plt.ylabel(base_metric_name.replace('_', ' ').title())
        
        if title:
            plt.title(title)
        elif combine_train_val:
            plt.title(f'{base_metric_name.replace("_", " ").title()} - Training vs Validation')
        else:
            plt.title(f'{metric_name.replace("/", " - ").replace("_", " ").title()} vs Epoch')
        
        plt.grid(True, alpha=0.3)
        
        if show_legend and (len(self.experiments) > 1 or combine_train_val):
            plt.legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def plot_multiple_metrics(self, metric_names: List[str], save_path: Optional[str] = None,
                            ncols: int = 2, title: Optional[str] = None,
                            combine_train_val: bool = False,
                            metric_names_to_legend: Optional[List[str]] = None,
                            ylims: Optional[List[Tuple[float, float]]] = None):
        """Plot multiple metrics in subplots."""
        n_metrics = len(metric_names)
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(self.figsize[0]*ncols//2, self.figsize[1]*nrows//2))
        # if n_metrics == 1:
        #     # axes = [axes]
        #     pass
        # elif nrows == 1:
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, metric_name in enumerate(metric_names):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col] #if nrows > 1 else axes[col]
            
            if combine_train_val:
                # Plot both train and val versions of the metric
                # base_metric = metric_name.split('/')[-1] if '/' in metric_name else metric_name
                base_metric = metric_name
                base_label = metric_names_to_legend[idx] if metric_names_to_legend is not None else None
                
                for i, (exp_name, _) in enumerate(self.experiments.items()):
                    # Plot training
                    if 'loss' in base_metric:
                        train_epochs, train_values = self.get_metric_values(exp_name, f"{base_metric}", split="train")
                    elif 'ByMassAcceptance' in base_metric:
                        train_epochs, train_values = self.get_metric_values(exp_name, f"train_MC{base_metric}", split="train_MC")
                    else:
                        train_epochs, train_values = self.get_metric_values(exp_name, f"train_MC/{base_metric}")
                    if train_epochs and train_values:
                        ax.plot(train_epochs, train_values, label=f'{exp_name} (Train)', 
                               color=self.colors[i], linestyle='-', linewidth=2)#, marker='o', markersize=3)
                    if ylims is not None:
                        ax.set_ylim(ylims[idx])
                for i, (exp_name, _) in enumerate(self.experiments.items()):
                    # Plot validation
                    if 'loss' in base_metric:
                        val_epochs, val_values = self.get_metric_values(exp_name, f"loss/loss_ce", split="val")
                    elif 'ByMassAcceptance' in base_metric:
                        val_epochs, val_values = self.get_metric_values(exp_name, f"val_MC{base_metric}", split="val_MC")
                    else:
                        val_epochs, val_values = self.get_metric_values(exp_name, f"val_MC/{base_metric}")
                    if val_epochs and val_values:
                        ax.plot(val_epochs, val_values, label=f'{exp_name} (Val)', 
                               color=self.colors[i], linestyle='--', linewidth=2)#, marker='s', markersize=3)
                    if ylims is not None:
                        ax.set_ylim(ylims[idx])
                ax.set_title(f'{base_label} - Train vs Val')
            else:
                base_label = metric_names_to_legend[idx] if metric_names_to_legend is not None else metric_name.split('/')[-1] if '/' in metric_name else metric_name
                # Original behavior
                for i, (exp_name, _) in enumerate(self.experiments.items()):
                    epochs, values = self.get_metric_values(exp_name, metric_name)
                    if epochs and values:
                        ax.plot(epochs, values, label=exp_name, color=self.colors[i], 
                               linewidth=2, marker='o', markersize=3)
                    if ylims is not None:
                        ax.set_ylim(ylims[idx])
                ax.set_title(base_label.replace("/", " - ").replace("_", " ").title())
            
            ax.set_xlabel('Epoch')
            base_metric_name = metric_name.split('/')[-1] if '/' in metric_name else metric_name
            # ax.set_ylabel(base_metric_name.replace('_', ' ').title())
            ax.set_ylabel(base_label)
            ax.grid(True, alpha=0.3)
            
            if idx == 0 and (len(self.experiments) > 1 or combine_train_val):  # Only show legend on first subplot
                ax.legend(ncol=2)
        
        # Hide empty subplots
        for idx in range(n_metrics, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col] #if nrows > 1 else axes[col]
            ax.set_visible(False)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            # plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def plot_loss_curves(self, save_path: Optional[str] = None):
        """Plot training and validation loss curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot total loss
        for i, (exp_name, _) in enumerate(self.experiments.items()):
            train_epochs, train_loss = self.get_metric_values(exp_name, "train_MC/loss/total")
            val_epochs, val_loss = self.get_metric_values(exp_name, "val_MC/loss/total")
            
            if train_epochs and train_loss:
                ax1.plot(train_epochs, train_loss, label=f'{exp_name} (Train)', 
                        color=self.colors[i], linestyle='-', linewidth=2)
            if val_epochs and val_loss:
                ax1.plot(val_epochs, val_loss, label=f'{exp_name} (Val)', 
                        color=self.colors[i], linestyle='--', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot cross-entropy loss if available
        for i, (exp_name, _) in enumerate(self.experiments.items()):
            train_epochs, train_ce = self.get_metric_values(exp_name, "train_MC/loss_total")
            val_epochs, val_ce = self.get_metric_values(exp_name, "val_MC/loss_ce")
            
            if train_epochs and train_ce:
                ax2.plot(train_epochs, train_ce, label=f'{exp_name} (Train)', 
                        color=self.colors[i], linestyle='-', linewidth=2)
            if val_epochs and val_ce:
                ax2.plot(val_epochs, val_ce, label=f'{exp_name} (Val)', 
                        color=self.colors[i], linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Cross-Entropy Loss')
        ax2.set_title('Cross-Entropy Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def plot_accuracy_comparison(self, save_path: Optional[str] = None, combine_train_val: bool = False):
        """Plot accuracy metrics comparison."""
        # Find all accuracy metrics
        if combine_train_val:
            # Get base accuracy metrics (without train/val prefix)
            train_accuracy_metrics = [m.split('/')[-1] for m in self.get_available_metrics("train_MC") 
                                    if "accuracy" in m.lower() and "qqbb" in m.lower()]
            val_accuracy_metrics = [m.split('/')[-1] for m in self.get_available_metrics("val_MC") 
                                  if "accuracy" in m.lower() and "qqbb" in m.lower()]
            # Get intersection to ensure we have both train and val for each metric
            base_metrics = list(set(train_accuracy_metrics) & set(val_accuracy_metrics))
            accuracy_metrics = base_metrics
        else:
            accuracy_metrics = [m for m in self.get_available_metrics("train_MC") + self.get_available_metrics("val_MC") 
                               if "accuracy" in m.lower() and "qqbb" in m.lower()]
        
        if not accuracy_metrics:
            print("No accuracy metrics found!")
            return
        
        self.plot_multiple_metrics(accuracy_metrics, save_path=save_path, 
                                 title="Accuracy Metrics Comparison", combine_train_val=combine_train_val)
    
    def plot_auc_comparison(self, save_path: Optional[str] = None, combine_train_val: bool = False):
        """Plot AUC metrics comparison."""
        # Find all AUC metrics
        if combine_train_val:
            # Get base AUC metrics (without train/val prefix)
            train_auc_metrics = [m.split('/')[-1] for m in self.get_available_metrics("train_MC") 
                               if "auc" in m.lower()]
            val_auc_metrics = [m.split('/')[-1] for m in self.get_available_metrics("val_MC") 
                             if "auc" in m.lower()]
            # Get intersection to ensure we have both train and val for each metric
            base_metrics = list(set(train_auc_metrics) & set(val_auc_metrics))
            auc_metrics = base_metrics
        else:
            auc_metrics = [m for m in self.get_available_metrics("train_MC") + self.get_available_metrics("val_MC") 
                          if "auc" in m.lower()]
        
        if not auc_metrics:
            print("No AUC metrics found!")
            return
        
        self.plot_multiple_metrics(auc_metrics, save_path=save_path, 
                                 title="AUC Metrics Comparison", combine_train_val=combine_train_val)
    
    def create_summary_dashboard(self, save_dir: Optional[str] = None, combine_train_val: bool = False):
        """Create a comprehensive dashboard with all key metrics."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Plot loss curves (always shows both train and val)
        loss_path = os.path.join(save_dir, "loss_curves.png") if save_dir else None
        self.plot_loss_curves(save_path=loss_path)
        
        # Plot accuracy comparison
        acc_path = os.path.join(save_dir, "accuracy_comparison.png") if save_dir else None
        self.plot_accuracy_comparison(save_path=acc_path, combine_train_val=combine_train_val)
        
        # Plot AUC comparison  
        auc_path = os.path.join(save_dir, "auc_comparison.png") if save_dir else None
        self.plot_auc_comparison(save_path=auc_path, combine_train_val=combine_train_val)
        
        suffix = " with train/val comparison" if combine_train_val else ""
        print(f"Dashboard created{' in ' + save_dir if save_dir else ''}{suffix}!")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from JSON files")
    parser.add_argument("--experiment_dirs", nargs="+", required=True,
                       help="Directories containing experiment results")
    parser.add_argument("--experiment_names", nargs="+", 
                       help="Names for experiments (optional)")
    parser.add_argument("--metrics", nargs="+",
                       help="Specific metrics to plot (e.g., 'train/accuracy' 'val/loss')")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save plots instead of displaying them")
    parser.add_argument("--output_dir", default="plots",
                       help="Directory to save plots (default: plots)")
    parser.add_argument("--dashboard", action="store_true",
                       help="Create a comprehensive dashboard")
    parser.add_argument("--list_metrics", action="store_true",
                       help="List all available metrics and exit")
    parser.add_argument("--combine_train_val", action="store_true",
                       help="Plot both training and validation on the same axis")
    
    args = parser.parse_args()
    
    # Load experiments
    print(f"Loading experiments from: {args.experiment_dirs}")
    experiments = load_multiple_experiments(args.experiment_dirs, args.experiment_names)
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"Loaded {len(experiments)} experiments: {list(experiments.keys())}")
    
    # Create plotter
    plotter = MetricsPlotter(experiments)
    
    # List metrics and exit if requested
    if args.list_metrics:
        print("\nAvailable metrics:")
        for split in ["train", "val", "train_MC", "val_MC"]:
            metrics = plotter.get_available_metrics(split)
            if metrics:
                print(f"\n{split.upper()}:")
                for metric in metrics:
                    print(f"  {metric}")
        return
    
    # Set up output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dashboard if requested
    if args.dashboard:
        save_dir = args.output_dir if args.save_plots else None
        plotter.create_summary_dashboard(save_dir=save_dir, combine_train_val=args.combine_train_val)
        return
    
    # Plot specific metrics if provided
    if args.metrics:
        if len(args.metrics) == 1:
            save_path = os.path.join(args.output_dir, f"{args.metrics[0].replace('/', '_')}.png") if args.save_plots else None
            plotter.plot_single_metric(args.metrics[0], save_path=save_path, combine_train_val=args.combine_train_val)
        else:
            save_path = os.path.join(args.output_dir, "multiple_metrics.png") if args.save_plots else None
            plotter.plot_multiple_metrics(args.metrics, save_path=save_path, combine_train_val=args.combine_train_val)
    else:
        # Default: create dashboard
        save_dir = args.output_dir if args.save_plots else None
        plotter.create_summary_dashboard(save_dir=save_dir, combine_train_val=args.combine_train_val)


if __name__ == "__main__":
    main() 