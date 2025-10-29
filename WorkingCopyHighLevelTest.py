# %% # Load required modules
import time
ts = []
ts.append(time.time())

import numpy as np
import os
import torch
from datetime import datetime
from jaxtyping import Float
import einops
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg') # If you want to run in batch mode, and not see the plots made
from utils import decode_y_eval_to_info
from mynewdataloaderHighLevel import ProportionalMemoryMappedDatasetHighLevel
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor, nn
import einops
import wandb
import torch.nn.functional as F
# from torchmetrics import Accuracy, AUC, ConfusionMatrix
# from torchmetrics import ConfusionMatrix
# from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import shutil
from MyMetricsHighLevel import HEPMetrics, HEPLoss, init_wandb

# %%
timeStr = datetime.now().strftime("%Y%m%d-%H%M%S")
saveDir = "output/" + timeStr  + "_TrainingOutput/"
os.makedirs(saveDir)
print(saveDir)
if 1:
    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
# Save the current script to the saveDir so we know what the training script was
def save_current_script(destination_directory):
    current_script_path = os.path.abspath(__file__)
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    destination_path = os.path.join(destination_directory, os.path.basename(current_script_path))
    shutil.copy(current_script_path, destination_path)
    print(f"This script copied to: {destination_path}")
# save_current_script('%s'%(saveDir))

# Some choices about the training process
# Assumes that the data has already been binarised
MET_CUT_ON = True
MH_SEL = False
N_TARGETS = 2 # Number of target classes (needed for one-hot encoding)
BIN_WRITE_TYPE=np.float32
N_Real_Vars = 7 # x, y, z, energy, d0val, dzval.  BE CAREFUL because this might change and if it does you ahve to rebinarise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Set up stuff to read in data from bin file

batch_size = 64*32
DATA_PATH='/data/atlas/baines/tmp3_highLevel' + '_MetCut'*MET_CUT_ON + '_mHSel'*MH_SEL + '/'
memmap_paths = {}
channel = 'lvbb'
means = np.load(f'{DATA_PATH}{channel}_mean.npy')
stds = np.load(f'{DATA_PATH}{channel}_std.npy')
for file_name in os.listdir(DATA_PATH):
    # if not '510124' in file_name:
    #     continue
    if 'shape' in file_name:
        continue
    if (channel in file_name) and ('dsid' in file_name):
        dsid = file_name[5:11]
        # if dsid < 510120:
        memmap_paths[int(dsid)] = DATA_PATH+file_name
    else:
        pass
train_split = 1.0
train_dataloader = ProportionalMemoryMappedDatasetHighLevel(
                 memmap_paths = memmap_paths,  # DSID to memmap path
                 N_Real_Vars=N_Real_Vars, # Will auto add 4 (for the y, w, dsid, mWh) inside the funciton
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 n_targets=N_TARGETS,
                 train_split=train_split,
                 means=means,
                 stds=stds,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)

# batch = next(train_dataloader)

# %%
batch = next(train_dataloader)
# plt.hist(batch['mWh'].detach().cpu())

# %%
# Create a new model
models = {}
fit_histories = {}
model_n = 0

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfigurableNN(nn.Module):
    def __init__(self, N_inputs, N_targets, hidden_layers, dropout_prob=0.0, use_batchnorm=True):
        """
        Initializes a configurable neural network.
        Args:
            N_inputs (int): Number of input features.
            N_targets (int): Number of output neurons (for classification).
            hidden_layers (list of int): List of integers specifying the number of neurons in each hidden layer.
            dropout_prob (float): Probability of dropout. 0.0 means no dropout.
            use_batchnorm (bool): Whether to use batch normalization in each layer.
        """
        super(ConfigurableNN, self).__init__()
        layers = []
        self.N_inputs = N_inputs
        self.N_targets = N_targets
        input_dim = self.N_inputs
        for hidden_dim in hidden_layers:
            # Input to first hidden layer and subsequent hidden layers
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())  # Activation function
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(p=dropout_prob))
            input_dim = hidden_dim
        # Output layer
        layers.append(nn.Linear(input_dim, N_targets))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def summary(self):
        """
        Prints the summary of the network architecture.
        """
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Network Summary:\n{'-'*20}")
        print(f"Total Parameters: {num_params}")
        print(f"Input size: {self.N_inputs}")
        print(f"Output size: {self.N_targets}")
        print(f"Layers:")
        for i, layer in enumerate(self.network):
            print(f"Layer {i+1}: {layer}")
        print(f"{'-'*20}")


# models[model_n] = {'model' : Net(model_cfg).to(device), 'inputs' : inputs}
# models[model_n] = {'model' : MyHookedTransformer(model_cfg).to(device)}

models[model_n] = {'model':ConfigurableNN(N_inputs=N_Real_Vars, N_targets=N_TARGETS, hidden_layers=[128, 128, 128], dropout_prob=0.00, use_batchnorm=False).to(device)}
models[model_n]['model'].summary()


# %% Cell to load in old model if we want to BE CAREFUL if you don't want to overwrite
if 1: # Just an example of how to load one back in
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250205-002440_TrainingOutput/models/0/chkpt199_202600.pth" # Hidden layers [128, 128, 128], batchnorm=False
    loaded_state_dict = torch.load(modelfile, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    models[model_n]['model'].load_state_dict(loaded_state_dict)



model, train_loader = models[model_n]['model'], train_dataloader


# %%
if 1:
    train_metrics_MCWts = HEPMetrics(channel=channel, total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[1000, 5000], num_classes=N_TARGETS) # TODO should 'total_weights_per_dsid' here be abs or not-abs
    train_loader._reset_indices()
    sweights=0
    sweights2=0
    for i in range(len(train_loader)+1):
    # for _ in range(100):
        batch = next(train_loader)
        dsid = 510121
        x, y, mWh, dsids, w, MCWts, mH = batch.values()
        outputs = model(x)
        train_metrics_MCWts.update(outputs, y.argmax(dim=-1), MCWts, mWh, dsids, mH)
        sweights += w[dsids==dsid].sum().item()
        sweights2 += MCWts[dsids==dsid].sum().item()
        if ((i%100)==0):
            print(f'{i}: {sweights:10.5f}\t{sweights2:10.5f}')
    print(f'{i}: {sweights:10.5f}\t{sweights2:10.5f}')


# %%
train_metrics_MCWts.compute_and_log(0, prefix="train_MC", step=0, log_level=3, save=False)
# %%
