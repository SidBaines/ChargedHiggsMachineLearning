import json
import os
import numpy as np
from typing import Dict, List, Any, Optional

class MetricsStorage:
    """
    Utility class for storing and loading training metrics locally.
    Stores metrics in JSON format for easy reading and plotting.
    """
    
    def __init__(self, save_dir: str, experiment_name: str = "experiment"):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.metrics_file = os.path.join(save_dir, f"{experiment_name}_metrics.json")
        self.config_file = os.path.join(save_dir, f"{experiment_name}_config.json")
        
        # Initialize metrics storage
        self.metrics = {
            "train": {},
            "val": {},
            "train_MC": {},
            "val_MC": {}
        }
        
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
    
    def store_epoch_metrics(self, epoch: int, train_metrics: Dict[str, Any], 
                           val_metrics: Dict[str, Any], 
                           train_mc_metrics: Optional[Dict[str, Any]] = None,
                           val_mc_metrics: Optional[Dict[str, Any]] = None,
                           loss_metrics: Optional[Dict[str, float]] = None):
        """
        Store metrics for a single epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary  
            train_mc_metrics: Training MC weighted metrics (optional)
            val_mc_metrics: Validation MC weighted metrics (optional)
            loss_metrics: Loss values (optional, e.g., {"train_loss": 0.5, "val_loss": 0.3})
        """
        # Convert numpy values to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            else:
                return obj
        
        # Store training metrics
        clean_train_metrics = convert_for_json(train_metrics)
        clean_val_metrics = convert_for_json(val_metrics)
        
        self.metrics["train"][str(epoch)] = clean_train_metrics
        self.metrics["val"][str(epoch)] = clean_val_metrics
        
        if train_mc_metrics:
            self.metrics["train_MC"][str(epoch)] = convert_for_json(train_mc_metrics)
        if val_mc_metrics:
            self.metrics["val_MC"][str(epoch)] = convert_for_json(val_mc_metrics)
            
        # Add loss metrics if provided
        if loss_metrics:
            for split in ["train", "val"]:
                if str(epoch) not in self.metrics[split]:
                    self.metrics[split][str(epoch)] = {}
                for loss_name, loss_value in loss_metrics.items():
                    if loss_name.startswith(split):
                        clean_key = loss_name.replace(f"{split}_", "").replace(f"{split}/", "")
                        self.metrics[split][str(epoch)][f"loss/{clean_key}"] = float(loss_value)
        
        # Save to file
        self.save_metrics()
    
    def save_config(self, config: Dict[str, Any]):
        """Save training configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_metrics(self) -> Dict[str, Dict]:
        """Load metrics from JSON file."""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {"train": {}, "val": {}, "train_MC": {}, "val_MC": {}}
    
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_metric_keys(self, split: str = "train") -> List[str]:
        """Get all available metric keys for a given split."""
        metrics = self.load_metrics()
        if split in metrics and metrics[split]:
            # Get keys from the first epoch
            first_epoch = list(metrics[split].keys())[0]
            return list(metrics[split][first_epoch].keys())
        return []
    
    def get_epochs(self) -> List[int]:
        """Get list of available epochs."""
        metrics = self.load_metrics()
        if metrics["train"]:
            return sorted([int(epoch) for epoch in metrics["train"].keys()])
        return []


def load_multiple_experiments(base_dirs: List[str], experiment_names: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Load metrics from multiple experiments for comparison.
    
    Args:
        base_dirs: List of base directories containing experiments
        experiment_names: Optional list of experiment names. If None, uses directory names.
    
    Returns:
        Dictionary mapping experiment names to their metrics
    """
    all_metrics = {}
    
    for i, base_dir in enumerate(base_dirs):
        if experiment_names and i < len(experiment_names):
            exp_name = experiment_names[i]
        else:
            exp_name = os.path.basename(base_dir.rstrip('/'))
        
        # Look for metrics files in the directory
        metrics_files = [f for f in os.listdir(base_dir) if f.endswith('_metrics.json')]
        
        if metrics_files:
            # Take the first metrics file found
            metrics_file = os.path.join(base_dir, metrics_files[0])
            with open(metrics_file, 'r') as f:
                all_metrics[exp_name] = json.load(f)
        else:
            print(f"Warning: No metrics file found in {base_dir}")
    
    return all_metrics 