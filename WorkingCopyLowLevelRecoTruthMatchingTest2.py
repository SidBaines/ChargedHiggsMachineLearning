# %% # Load required modules
import time
ts = []
ts.append(time.time())

import numpy as np
import os
import torch
from datetime import datetime
from jaxtyping import Float
from typing import List
import einops
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Agg')
# from utils import save_current_script#, decode_y_eval_to_info
from utils import decode_y_eval_to_info
from mynewdataloader import ProportionalMemoryMappedDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor, nn
import einops
import wandb
import torch.nn.functional as F
import dsidmappings
# from torchmetrics import Accuracy, AUC, ConfusionMatrix
# from torchmetrics import ConfusionMatrix
# from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import shutil
# from MyMetricsLowLevel import HEPMetrics, HEPLoss, init_wandb
from MyMetricsLowLevelRecoTruthMatching import HEPMetrics, HEPLoss, init_wandb
import functools

# %%
timeStr = datetime.now().strftime("%Y%m%d-%H%M%S")
# saveDir = "output/" + timeStr  + "_TrainingOutput/"
# os.makedirs(saveDir)
# print(saveDir)
# # Save the current script to the saveDir so we know what the training script was
# def save_current_script(destination_directory):
#     current_script_path = os.path.abspath(__file__)
#     if not os.path.exists(destination_directory):
#         os.makedirs(destination_directory)
#     destination_path = os.path.join(destination_directory, os.path.basename(current_script_path))
#     shutil.copy(current_script_path, destination_path)
#     print(f"This script copied to: {destination_path}")
# # save_current_script('%s'%(saveDir))

# Some choices about the training process
# Assumes that the data has already been binarised
REMOVE_WHERE_TRUTH_WOULD_BE_CUT = False # Because we're predicting, want to include ALL events
INCLUDE_NEGATIVE_SELECTIONS = True
ONLY_READ_SIGNAL = True
PHI_ROTATED = True
MODEL_ARCH="DEEPSETS_SELFATTENTION_RESIDUAL_X2"
# if not (MODEL_ARCH=="HYBRID_SELFATTENTION_GATED"):
if True:
    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: # For testing new model architecture classes, probably run on cpu, since CUDA will just give difficult errors if there is some problem with the arch
    device = "cpu"
USE_LORENTZ_INVARIANT_FEATURES = True
TOSS_UNCERTAIN_TRUTH = True
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
USE_OLD_TRUTH_SETTING = False
# if USE_OLD_TRUTH_SETTING:
#     raise NotImplementedError # Need to check if we should require truth_agreement variable here (well, really in the prep data script) or not
SHUFFLE_OBJECTS = False
NORMALISE_DATA = False
SCALE_DATA = True
assert(not (NORMALISE_DATA and SCALE_DATA))
CONVERT_TO_PT_PHI_ETA_M = False
IS_XBB_TAGGED = False
REQUIRE_XBB = False # If we only select categories 0, 3, 8, 9, 10 (ie, if INCLUDE_ALL_SELECTIONS is False) then I think this is satisfied anyway
assert (~((REQUIRE_XBB and (~IS_XBB_TAGGED))))
INCLUDE_TAG_INFO = True
INCLUDE_ALL_SELECTIONS = True
MET_CUT_ON = True
MH_SEL = False
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
if IS_XBB_TAGGED:
    N_CTX = 7 # the SIX types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}
else:
    N_CTX = 6 # the five types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet'}
USE_DROPOUT = True
BIN_WRITE_TYPE=np.float32
max_n_objs_in_file = 14 # BE CAREFUL because this might change and if it does you ahve to rebinarise
max_n_objs_to_read = 14
if INCLUDE_TAG_INFO:
    N_Real_Vars=7 # px, py, pz, energy, tagInfo.  BE CAREFUL because this might change and if it does you ahve to rebinarise
else:
    N_Real_Vars=6 # px, py, pz, energy.  BE CAREFUL because this might change and if it does you ahve to rebinarise


# %%
# Set up stuff to read in data from bin file


batch_size = 256*4
DATA_PATH=f'/data/atlas/baines/tmp2_SingleXbbSelected_XbbTagged_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired' + '_mHSel'*MH_SEL + '/'
DATA_PATH=f'/data/atlas/baines/tmp_SingleXbbSelected_XbbTagged_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired' + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '/'
DATA_PATH=f'/data/atlas/baines/tmp_SingleXbbSelected' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '/'
DATA_PATH=f'/data/atlas/baines/tmp2_LowLevelRecoTruthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '/'
DATA_PATH=f'/data/atlas/baines/tmp3_LowLevelRecoTruthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '/'
# DATA_PATH=f'/data/atlas/baines/tmp7_LowLevelRecoTrsuthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT + '/'
DATA_PATH = '/data/atlas/baines/20250305v9' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
if NORMALISE_DATA:
    means = np.load(f'{DATA_PATH}mean.npy')[1:]
    stds = np.load(f'{DATA_PATH}std.npy')[1:]
else:
    means = None
    if SCALE_DATA:
        stds = np.ones(N_Real_Vars)
        stds[:4] = 1e5
    else:
        stds = None
memmap_paths_train = {}
memmap_paths_val = {}
for file_name in os.listdir(DATA_PATH):
    if ('shape' in file_name) or ('npy' in file_name):
        continue
    dsid = file_name[5:11]
    # if (int(dsid)!=510115) and (int(dsid)!=510116) and (int(dsid)!=510117):
    if ((int(dsid) > 500000) and (int(dsid) < 600000)) or (not ONLY_READ_SIGNAL):
        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
    if ((int(dsid) > 500000) and (int(dsid) < 600000)) or (not ONLY_READ_SIGNAL):
        print("For now, don't include the background even in the val set, but later we might want to add some backgorund performance logging to the Metric tracking code")
        memmap_paths_val[int(dsid)] = DATA_PATH+file_name
n_splits=2
validation_split_idx=0
train_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths_train,  # DSID to memmap path
                 max_objs_in_memmap=max_n_objs_in_file,
                 N_Real_Vars_In_File=N_Real_Vars,
                 N_Real_Vars_To_Return=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                 shuffle_batch=False,
                 means=means,
                 stds=stds,
                 objs_to_output=max_n_objs_to_read,
                 signal_only=ONLY_READ_SIGNAL,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
val_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths_val,  # DSID to memmap path
                 max_objs_in_memmap=max_n_objs_in_file,
                 N_Real_Vars_In_File=N_Real_Vars,
                 N_Real_Vars_To_Return=N_Real_Vars,
                 class_proportions = None,
                #  batch_size=64*8*64*8,
                 batch_size=batch_size,
                 device=device,
                 is_train=False,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                 shuffle_batch=False,
                 means=means,
                 stds=stds,
                 objs_to_output=max_n_objs_to_read,
                 signal_only=ONLY_READ_SIGNAL,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
print(train_dataloader.get_total_samples())
# assert(False) # Need to check if the weighting is correct - it seemed likely that the sum of training weights for signal was not the same as for background?
# %%
batch = next(train_dataloader)

# %%
if 0:
    plt.figure(figsize=(4,4))
    nbins=20
    bins=np.arange(nbins+1)/nbins*200e3+50e3
    plt.hist(batch['mH'][((batch['dsids']<500000)|(batch['dsids']>600000))].cpu(),histtype='step',bins=bins, label='Bkg')
    plt.hist(batch['mH'][~((batch['dsids']<500000)|(batch['dsids']>600000))].cpu(),histtype='step',bins=bins, label='Sig')
    plt.show()
    plt.figure(figsize=(4,4))
    nbins=20
    bins=np.arange(nbins+1)/nbins*200e3+50e3
    plt.hist(batch['mH'][batch['y'].argmax(dim=-1)==0].cpu(),histtype='step',bins=bins, label='Bkg')
    plt.hist(batch['mH'][batch['y'].argmax(dim=-1)!=0].cpu(),histtype='step',bins=bins, label='Sig')
    plt.show()
# plt.hist(batch['mH'][batch['dsids']==510124],histtype=='step')

# %%


class LorentzInvariantFeatures(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, four_momenta):
        # four_momenta shape: [..., 4] where last dim is (px, py, pz, E)
        # Calculate Lorentz invariant quantities
        mass_squared = four_momenta[..., 3]**2 - (four_momenta[..., 0]**2 + 
                                                 four_momenta[..., 1]**2 + 
                                                 four_momenta[..., 2]**2)
        pt = torch.sqrt(four_momenta[..., 0]**2 + four_momenta[..., 1]**2)
        eta = torch.asinh(four_momenta[..., 2] / pt)
        eta = torch.nan_to_num(eta, nan=0.0)
        phi = torch.atan2(four_momenta[..., 1], four_momenta[..., 0])
        
        return torch.stack([mass_squared, pt, eta, phi], dim=-1)

if MODEL_ARCH=="HYBRID_SELFATTENTION_GATED":
    assert(False) # Need to turn this into a classifier per object
    class HybridAttentionDeepSets(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, embedding_size=32, dropout_p=0.0):
            super().__init__()
            if USE_LORENTZ_INVARIANT_FEATURES:
                self.invariant_features = LorentzInvariantFeatures()
            
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
            
            # Initial per-object processing
            self.object_net = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Self-attention layer for object interactions (with fewer heads)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            
            # Gated attention mechanism
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Sigmoid()
            )
            
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh()
            )
            
            # Attention integration layer
            self.attention_integration = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # Combine self-attention and gated attention
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            
            # Final classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
        def forward(self, object_features, object_types):
            # batch_size, num_objects, feature_dim = x.shape
            # Assuming last feature is object type
            if USE_LORENTZ_INVARIANT_FEATURES:
                object_features[...,:4] = self.invariant_features(object_features[...,:4])
            
            # Get type embeddings and combine with features
            type_emb = self.type_embedding(object_types)
            combined = torch.cat([object_features, type_emb], dim=-1)
            
            # Process each object
            object_features = self.object_net(combined)
            
            # Apply self-attention to model interactions between objects
            self_attention_output, _ = self.self_attention(
                object_features, object_features, object_features
            )
            
            # Apply gated attention mechanism
            gates = self.gate_net(object_features)
            attention_values = self.attention_net(object_features)
            gated_features = gates * attention_values
            
            # Create two pooled representations
            # 1. Self-attention based pooling
            pooled_self_attention = torch.sum(self_attention_output, dim=1)
            
            # 2. Gated attention based pooling
            pooled_gated_attention = torch.sum(gated_features, dim=1)
            
            # Combine both attention mechanisms
            combined_features = torch.cat([pooled_self_attention, pooled_gated_attention], dim=1)
            integrated_features = self.attention_integration(combined_features)
            
            return self.classifier(integrated_features)
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X2":
    # assert(False) # Need to turn this into a classifier per object
    class DeepSetsWithResidualSelfAttentionDouble(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32):
            super().__init__()
            if USE_LORENTZ_INVARIANT_FEATURES:
                self.invariant_features = LorentzInvariantFeatures()
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
            # Initial per-object processing
            self.object_net = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # Self-attention layer for object interactions
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout_p,
                # dropout=0.0,
                batch_first=True,
            )
            self.self_attention2 = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout_p,
                # dropout=0.0,
                batch_first=True,
            )
            # Processing after attention with normalization
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.post_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.post_attention2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            # Final classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
        def forward(self, object_features, object_types):
            batch_size, num_objects, feature_dim = x.shape
            # Get type embeddings and combine with features
            type_emb = self.type_embedding(object_types)
            if USE_LORENTZ_INVARIANT_FEATURES:
                object_features[...,:4] = self.invariant_features(object_features[...,:4])
            combined = torch.cat([object_features, type_emb], dim=-1)
            # Process each object
            object_features = self.object_net(combined)
            # Store original features for residual connection
            identity = object_features
            # Apply self-attention to model interactions between objects
            # This creates a mechanism for objects to attend to each other
            attention_output, _ = self.self_attention(
                object_features, object_features, object_features,
                key_padding_mask=(object_types==(N_CTX-1))
            )
            # Add residual connection and normalize
            attention_output = identity + attention_output
            attention_output = self.layer_norm(attention_output)
            # Post-attention processing
            attention_output = self.post_attention(attention_output)
            # Store original features for residual connection
            identity2 = attention_output
            # Apply self-attention to model interactions between objects
            # This creates a mechanism for objects to attend to each other
            attention_output2, _ = self.self_attention2(
                attention_output, attention_output, attention_output,
                key_padding_mask=(object_types==(N_CTX-1))
            )
            # Add residual connection and normalize
            attention_output2 = identity2 + attention_output2
            attention_output2 = self.layer_norm2(attention_output2)
            # Post-attention processing
            attention_output2 = self.post_attention2(attention_output2)
            return self.classifier(attention_output2)
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL":
    # assert(False) # Need to turn this into a classifier per object
    class DeepSetsWithResidualSelfAttention(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32):
            super().__init__()
            if USE_LORENTZ_INVARIANT_FEATURES:
                self.invariant_features = LorentzInvariantFeatures()
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
            # Initial per-object processing
            self.object_net = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # Self-attention layer for object interactions
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                # dropout=dropout_p/2,
                dropout=0.0,
                batch_first=True,
            )
            # Processing after attention with normalization
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.post_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            # Final classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
        def forward(self, object_features, object_types):
            batch_size, num_objects, feature_dim = x.shape
            # Get type embeddings and combine with features
            type_emb = self.type_embedding(object_types)
            if USE_LORENTZ_INVARIANT_FEATURES:
                object_features[...,:4] = self.invariant_features(object_features[...,:4])
            combined = torch.cat([object_features, type_emb], dim=-1)
            # Process each object
            object_features = self.object_net(combined)
            # Store original features for residual connection
            identity = object_features
            # Apply self-attention to model interactions between objects
            # This creates a mechanism for objects to attend to each other
            attention_output, _ = self.self_attention(
                object_features, object_features, object_features,
                key_padding_mask=(object_types==(N_CTX-1))
            )
            # Add residual connection and normalize
            attention_output = identity + attention_output
            attention_output = self.layer_norm(attention_output)
            # Post-attention processing
            object_features = self.post_attention(attention_output)
            return self.classifier(attention_output)
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION":
    class DeepSetsWithSelfAttention(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32):
            super().__init__()
            if USE_LORENTZ_INVARIANT_FEATURES:
                self.invariant_features = LorentzInvariantFeatures()
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
            # Initial per-object processing
            self.object_net = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # Self-attention layer for object interactions
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                # dropout=dropout_p/2,
                dropout=0.0,
                batch_first=True,
            )
            # Processing after attention
            self.post_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            # Final classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
        def forward(self, object_features, object_types):
            batch_size, num_objects, feature_dim = x.shape
            # Get type embeddings and combine with features
            type_emb = self.type_embedding(object_types)
            if USE_LORENTZ_INVARIANT_FEATURES:
                object_features[...,:4] = self.invariant_features(object_features[...,:4])
            combined = torch.cat([object_features, type_emb], dim=-1)
            # Process each object
            object_features = self.object_net(combined)
            # Apply self-attention to model interactions between objects
            # This creates a mechanism for objects to attend to each other
            attention_output, _ = self.self_attention(
                object_features, object_features, object_features,
                key_padding_mask=(object_types==(N_CTX-1))
            )
            # Post-attention processing
            object_features = self.post_attention(attention_output)
            return self.classifier(object_features)
elif MODEL_ARCH=="DEEPSETS":
    assert(False) # Need to turn this into a classifier per object
    class DeepSetsWithGatedAttention(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, dropout_p=0.0, embedding_size=32):
            super().__init__()
            if USE_LORENTZ_INVARIANT_FEATURES:
                self.invariant_features = LorentzInvariantFeatures()
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
            # Initial per-object processing
            self.object_net = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # +32 for type embedding
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # Gated attention mechanism
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            # Final classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                # nn.Dropout(dropout_p),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        
        def forward(self, object_features, types):
            # Get type embeddings and combine with features
            type_emb = self.type_embedding(types)
            if USE_LORENTZ_INVARIANT_FEATURES:
                object_features[...,:4] = self.invariant_features(object_features[...,:4])
            combined = torch.cat([object_features, type_emb], dim=-1)
            # Process each object
            object_features = self.object_net(combined)
            # Apply gated attention
            gates = self.gate_net(object_features)
            attention = self.attention_net(object_features)
            gated_features = gates * attention
            # Permutation-invariant pooling
            pooled = torch.sum(gated_features, dim=1)
            return self.classifier(pooled)
elif MODEL_ARCH=="PARTICLE_FLOW":
    assert(False) # Need to turn this into a classifier per object
    class LorentzInvariantParticleFlow(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, dropout_p=0.0, embedding_size=32):
            super().__init__()
            
            self.invariant_features = LorentzInvariantFeatures()
            
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)

            # Initial node feature network
            self.node_init = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # 4 invariant features + tagInfo + type embedding
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Edge network for computing pairwise features
            self.edge_network = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),  # 4 invariant features + tagInfo per object
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Node update network
            self.node_network = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),  # edge features + invariant features
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim//2, num_classes)
            )
        
        def forward(self, object_features, types):
            batch_size, n_particles, _ = object_features.shape
            
            # Split input into components
            four_momenta = object_features[..., :4]
            tag_info = object_features[..., 4:5]
            obj_type = types.long()
            
            # Calculate Lorentz invariant features
            invariant_features = self.invariant_features(four_momenta)
            type_emb = self.type_embedding(obj_type)
            
            # Combine invariant features with tag info and type embeddings
            node_features = torch.cat([invariant_features, tag_info, type_emb], dim=-1)
            
            # Initialize node representations
            node_features = self.node_init(node_features)
            
            # Compute pairwise interactions
            node_features_i = node_features.unsqueeze(2).expand(-1, -1, n_particles, -1)
            node_features_j = node_features.unsqueeze(1).expand(-1, n_particles, -1, -1)
            edge_input = torch.cat([node_features_i, node_features_j], dim=-1)
            
            # Process edges
            edge_features = self.edge_network(edge_input)
            edge_features = torch.mean(edge_features, dim=2)  # Average over neighbors
            
            # Update node features
            node_input = torch.cat([edge_features, node_features], dim=-1)
            final_features = self.node_network(node_input)
            
            # Global pooling
            pooled = torch.mean(final_features, dim=1)
            
            return self.classifier(pooled)
elif MODEL_ARCH=="TRANSFORMER":
    class MyHookedTransformer(HookedTransformer):
        def __init__(self, cfg, mass_input_layer=2, mass_hidden_dim=256, **kwargs):
            super(MyHookedTransformer, self).__init__(cfg, **kwargs)
            if USE_LORENTZ_INVARIANT_FEATURES:
                self.invariant_features = LorentzInvariantFeatures()
            self.hook_dict['hook_mytokens'] = HookPoint()
            self.hook_dict['hook_mytokens'].name = 'hook_mytokens'
            self.mod_dict['hook_mytokens'] = self.hook_dict['hook_mytokens']
            self.W_Embed = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_vocab, cfg.d_model)))
            nn.init.normal_(self.W_Embed, std=0.02)
            
        def forward(self, tokens: Float[Tensor, "batch object d_input"], token_types: Float[Tensor, "batch object"], **kwargs) -> Float[Tensor, "batch d_model"]:
            self.hook_dict['hook_mytokens'](tokens)
            if USE_LORENTZ_INVARIANT_FEATURES:
                tokens[...,:4] = self.invariant_features(tokens[...,:4])
            expanded_W_E = self.W_Embed.unsqueeze(0).expand(token_types.shape[0], -1, -1, -1)
            expanded_types = token_types.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.W_Embed.shape[-2], self.W_Embed.shape[-1])
            W_E_selected = torch.gather(expanded_W_E, dim=1, index=expanded_types)
            output = einops.einsum(tokens, W_E_selected, "batch object d_input, batch object d_input d_model -> batch object d_model")
            if 'start_at_layer' in kwargs:
                raise NotImplementedError
            else:
                class_outs = super(MyHookedTransformer, self).forward(output, start_at_layer=0, **kwargs)
            return class_outs
else:
    assert(False)


# %%
# Create a new model
models = {}
fit_histories = {}
model_n = 0

if MODEL_ARCH=="TRANSFORMER":
    # Create the model with the desired properties
    model_cfg = HookedTransformerConfig(
        # normalization_type='LN',
        normalization_type='LN',
        d_model=128,
        d_head=8,
        n_layers=4,
        n_heads=4,
        n_ctx=N_CTX, # Max number of types of object per event + 1 because we want a dummy row in the embedding matrix for non-existing particles
        d_vocab=N_Real_Vars-2, # Number of inputs per object
        d_vocab_out=1,  # 1 because we're doing binary classification
        d_mlp=256,
        attention_dir="bidirectional",  # defaults to "causal"
        act_fn="relu",
        use_attn_result=True,
        device=str(device),
        use_hook_tokens=True,
    )
elif MODEL_ARCH=="DEEPSETS" or MODEL_ARCH=="PARTICLE_FLOW":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":32}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":32, "num_heads":1}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":32, "num_heads":1}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X2":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":10, "num_heads":4}
elif MODEL_ARCH=="HYBRID_SELFATTENTION_GATED":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":32, "num_heads":1}
else:
    assert(False)

if MODEL_ARCH=="HYBRID_SELFATTENTION_GATED":
    models[model_n] = {'model' : HybridAttentionDeepSets(input_dim=N_Real_Vars-2, hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads']).to(device)}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL":
    models[model_n] = {'model' : DeepSetsWithResidualSelfAttention(input_dim=N_Real_Vars-2, hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads']).to(device)}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X2":
    models[model_n] = {'model' : DeepSetsWithResidualSelfAttentionDouble(input_dim=N_Real_Vars-2, hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads']).to(device)}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION":
    models[model_n] = {'model' : DeepSetsWithSelfAttention(input_dim=N_Real_Vars-2, hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads']).to(device)}
elif MODEL_ARCH=="DEEPSETS":
    models[model_n] = {'model' : DeepSetsWithGatedAttention(input_dim=N_Real_Vars-2, hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p']).to(device)}
elif MODEL_ARCH=="PARTICLE_FLOW":
    models[model_n] = {'model' : LorentzInvariantParticleFlow(input_dim=N_Real_Vars-2, hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p']).to(device)}
elif MODEL_ARCH=="TRANSFORMER":
    models[model_n] = {'model' : MyHookedTransformer(model_cfg).to(device)}
else:
    assert(False)
print(sum(p.numel() for p in models[model_n]['model'].parameters()))

# %%
if MODEL_ARCH=="TRANSFORMER":
    def add_perma_hooks_to_mask_pad_tokens(
        model: HookedTransformer
    ) -> HookedTransformer:
        # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
        def cache_padding_tokens_mask(tokens: Float[Tensor, "batch object d_input"], hook: HookPoint) -> None:
            # print("Caching padding tokens!")
            hook.ctx["padding_tokens_mask"] = einops.rearrange(torch.all(tokens==0, dim=-1), "b sK -> b 1 1 sK")

        # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
        def apply_padding_tokens_mask(
            attn_scores: Float[Tensor, "batch head seq_Q seq_K"],
            hook: HookPoint,
        ) -> None:
            attn_scores.masked_fill_(model.hook_dict["hook_mytokens"].ctx["padding_tokens_mask"], -1e5)
            if hook.layer() == model.cfg.n_layers - 1:
                del model.hook_dict["hook_mytokens"].ctx["padding_tokens_mask"]

        # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
        for name, hook in model.hook_dict.items():
            if name == "hook_mytokens":
                hook.add_perma_hook(cache_padding_tokens_mask)  # type: ignore
            elif name.endswith("attn_scores"):
                hook.add_perma_hook(apply_padding_tokens_mask)  # type: ignore

        return model

    def dropout_hook(
        resid: Float[Tensor, "batch object d_model"],
        hook: HookPoint,
        p=0.1,
        v=-1e5,
    ) -> None:
        resid.masked_fill_((torch.rand(resid.shape) < p).to(resid.device), v)

    models[model_n]['model'].reset_hooks(including_permanent=True)
    models[model_n]['model'] = add_perma_hooks_to_mask_pad_tokens(models[model_n]['model'])


class_weights = [1 for _ in range(N_TARGETS)]
labels = ['Bkg', 'Lep', 'Had'] # truth==0 is bkg, truth==1 is leptonic decay, truth==2 is hadronic decay
class_weights_expanded = einops.repeat(torch.Tensor(class_weights), 't -> batch t', batch=batch_size).to(device)



# SHOULD CHANGE WEIGHT DECAY BACK (IT WAS 1e-5 before)
# criterion = nn.CrossEntropyLoss(reduction='none')  # 'none' to handle sample weights
# for epoch in range(num_epochs):
#     dataloader._reset_indices()
    
#     # Training
#     models[model_n]['model'].train()
#     num_steps = len_train_dataloader-1
#     for n_step in range(num_steps): #TODO currently we don't handle the last batch, fix this
#         batch = next(dataloader)
#         x, y, w, types, y_eval, mWH_qqbb, mWH_lvbb = batch['x'], batch['y'], batch['w'], batch['types'], batch['y_eval'], batch['mWH_qqbb'], batch['mWH_lvbb']
#         class_weights_expanded = einops.repeat(torch.Tensor(class_weights), 't -> l t', l=y.shape[0]).to(device)
#         outputs = models[model_n]['model'](x, types)
#         class_loss = (criterion(outputs, y) * w).mean()
#         optimizer.zero_grad()
#         class_loss.backward()
#         optimizer.step()
#         if (n_step % 10) == 0:
#             print('[%d/%d][%d/%d]\tLoss_C: %.4e' %(epoch, num_epochs, n_step, num_steps,class_loss.item()))

# %%
from utils import basic_lr_scheduler
if 0:
    n_epochs = 100
    plt.plot([basic_lr_scheduler(i, 1, 1e-3, n_epochs=n_epochs, log=False) for i in range(n_epochs)])
    plt.plot([basic_lr_scheduler(i, 1, 1e-3, n_epochs=n_epochs, log=True) for i in range(n_epochs)])
    plt.yscale('log')
    plt.show()
# [basic_lr_scheduler(i, 1, 1e-3, n_epochs=n_epochs) for i in range(n_epochs)]


# %%

# %%
model, train_loader, val_loader = models[model_n]['model'], train_dataloader, val_dataloader
num_epochs = 1
# log_interval = 100
log_interval = int(50e3/batch_size)
# longer_log_interval = log_interval*10
longer_log_interval = 100000000000
SAVE_MODEL_EVERY = 25
name_mapping = {"DEEPSETS":"DS", 
                "HYBRID_SELFATTENTION_GATED":"DSSAGA", 
                "DEEPSETS_SELFATTENTION":"DSSA", 
                "DEEPSETS_SELFATTENTION_RESIDUAL":"DSSAR", 
                "DEEPSETS_SELFATTENTION_RESIDUAL_X2":"DSSAR2", 
                "PARTICLE_FLOW":"PF",
                "TRANSFORMER":"TF",
                }
config = {
        "learning_rate": 2e-4,
        "learning_rate_low": 1e-8,
        "learning_rate_log_decay":True,
        "architecture": "PhysicsTransformer",
        "dataset": "ATLAS_ChargedHiggs",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "wandb":False,
        "name":"_"+timeStr+"_LowLevel_"+name_mapping[MODEL_ARCH],
        "weight_decay":1e-10,
    }
optimizer = torch.optim.Adam(models[model_n]['model'].parameters(), lr=1e-4, weight_decay=config['weight_decay'])
if config['wandb']:
    init_wandb(config)
    # wandb.watch(model, log_freq=100)

# %%
if 1: # 
    print("WARNING: You are starting from a semi-pre-trained model state")
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250302-224613_TrainingOutput/models/0/chkpt74_414975.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250304-145036_TrainingOutput/models/0/chkpt163050.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250305-195515_TrainingOutput/models/0/chkpt163050.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250305-232917_TrainingOutput/models/0/chkpt162930.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    loaded_state_dict = torch.load(modelfile, map_location=torch.device(device))
    models[model_n]['model'].load_state_dict(loaded_state_dict)
# %%
dsid_group_names = set([i for i in dsidmappings.DSID_TYPE_MAPPING_0.values() if 'H+' not in i])
dsid_groups = {grpname:torch.tensor([i for i in dsidmappings.DSID_TYPE_MAPPING_0 if dsidmappings.DSID_TYPE_MAPPING_0[i]==grpname]) for grpname in dsid_group_names}
train_metrics = HEPMetrics(N_CTX-1, max_n_objs_to_read, max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000], dsid_groups=dsid_groups) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics = HEPMetrics(N_CTX-1, max_n_objs_to_read, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000], dsid_groups=dsid_groups)
train_metrics_MCWts = HEPMetrics(N_CTX-1, max_n_objs_to_read, max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000], dsid_groups=dsid_groups) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics_MCWts = HEPMetrics(N_CTX-1, max_n_objs_to_read, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000], dsid_groups=dsid_groups)
global_step = 0
total_train_samples_processed = 0
train_loader._reset_indices()
# orig_len_train_dataloader=len(train_loader)
# num_lr_steps = num_epochs*orig_len_train_dataloader
for epoch in range(1):
    model.eval()
    # Update learning rate based on the cosine scheduler
    train_loader._reset_indices()
    val_loader._reset_indices()
    train_metrics.reset()
    train_metrics_MCWts.reset()
    val_metrics.reset()
    val_metrics_MCWts.reset()
    # model.train()
    # orig_len_train_dataloader=len(train_loader)
    # for batch_idx in range(orig_len_train_dataloader):
    for batch_idx in range(100):
        # print(f"Batch {batch_idx + 1}/{orig_len_train_dataloader}")
        print(f"Batch {batch_idx + 1}")
        batch = next(train_loader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        outputs = model(x[...,:-2], types).squeeze()
        total_train_samples_processed += len(y)
        train_metrics.update(outputs, x[...,-1], w, dsids, types)
        train_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types)
    # orig_len_val_dataloader=len(val_loader)
    for batch_idx in range(100):
    # for batch_idx in range(orig_len_val_dataloader):
        # print(f"Batch {batch_idx + 1}/{orig_len_val_dataloader}")
        print(f"Batch {batch_idx + 1}")
        batch = next(val_loader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        outputs = model(x[...,:-2], types).squeeze()
        val_metrics.update(outputs, x[...,-1], w, dsids, types)
        val_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types)
# wandb.finish()

# %%
val_metrics_MCWts.reset_starts()
r=val_metrics_MCWts.compute_and_log(1, log_level=10, save=False)

print(f"mass  PctPerfectReco  PctValidReco PctPerfectOfValid")
for mass in val_metrics_MCWts.DSID_MASS_MAPPING.values():
    print(f"{mass}: {r[f'val/PerfectRecoPct_{mass}']*100:10.2f}/{r[f'val/validPct_{mass}']*100:10.2f} = {r[f'val/PerfectRecoPct_{mass}']/r[f'val/validPct_{mass}']*100:10.2f}")

# %%
if 0:
    batch = next(val_loader)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
# %%        
from utils import Get_PtEtaPhiM_fromXYZT
# for dsid in val_metrics_MCWts.DSID_MASS_MAPPING.keys():
# for dsid in list(val_metrics_MCWts.DSID_MASS_MAPPING.keys())[:1]:
nbins=100
bins = np.arange(nbins+1)/nbins*3e3
bins = nbins
for dsid in [510115, 510118, 510123]:
    dsid_mask = (dsids == dsid).cpu().numpy()
    pred_inclusion = outputs>0
    true_inclusion = x[...,-1]
    # print(x[...,:4].shape)
    # print(pred_inclusion.shape)
    if 0:
        fourmom=(x[...,:4]).cpu().numpy()*1e5
    elif 1:
        fourmom=(x[...,:4] * (true_inclusion!=0).unsqueeze(-1)).cpu().numpy()*1e5
    else:
        fourmom=(x[...,:4] * pred_inclusion.unsqueeze(-1)).cpu().numpy()*1e5
    # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,0].sum(dim=-1).cpu().numpy(),fourmom[:,1].sum(dim=-1).cpu().numpy(),fourmom[:,2].sum(dim=-1).cpu().numpy(),fourmom[:,3].sum(dim=-1).cpu().numpy())
    # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[...,0].sum(dim=-1).cpu().numpy(),fourmom[...,1].sum(dim=-1).cpu().numpy(),fourmom[...,2].sum(dim=-1).cpu().numpy(),fourmom[...,3].sum(dim=-1).cpu().numpy())
    _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,2,0],fourmom[:,2,1],fourmom[:,2,2],fourmom[:,2,3])
    _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,:2,0].sum(axis=-1),fourmom[:,:2,1].sum(axis=-1),fourmom[:,:2,2].sum(axis=-1),fourmom[:,:2,3].sum(axis=-1))
    # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,:,0].sum(axis=-1),fourmom[:,:,1].sum(axis=-1),fourmom[:,:,2].sum(axis=-1),fourmom[:,:,3].sum(axis=-1))
    _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[dsid_mask,:,0].sum(axis=-1),fourmom[dsid_mask,:,1].sum(axis=-1),fourmom[dsid_mask,:,2].sum(axis=-1),fourmom[dsid_mask,:,3].sum(axis=-1))
    plt.hist(m*1e-3, weights=MCWts.cpu()[dsid_mask], histtype='step', label=val_metrics_MCWts.DSID_MASS_MAPPING[dsid],bins=bins)
plt.legend()
# plt.savefig('tmp.pdf')
plt.show()
plt.close('all')


# %%
from utils import Get_PtEtaPhiM_fromXYZT
# for dsid in val_metrics_MCWts.DSID_MASS_MAPPING.keys():
# for dsid in list(val_metrics_MCWts.DSID_MASS_MAPPING.keys())[:1]:
nbins=100
bins = np.arange(nbins+1)/nbins*3e3
bins = nbins
for dsid in [510115, 510118, 510123]:
    dsid_mask = (dsids == dsid).cpu()
    pred_inclusion = (outputs>0).cpu()
    true_inclusion = x[...,-1]
    # print(x[...,:4].shape)
    # print(pred_inclusion.shape)
    if 0:
        fourmom=(x[...,:4]).cpu().numpy()*1e5
    elif 1:
        fourmom=(x[...,:4] * (true_inclusion!=0).unsqueeze(-1)).cpu().numpy()*1e5
    else:
        fourmom=(x[...,:4] * pred_inclusion.unsqueeze(-1)).cpu().numpy()*1e5
    # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,0].sum(dim=-1).cpu().numpy(),fourmom[:,1].sum(dim=-1).cpu().numpy(),fourmom[:,2].sum(dim=-1).cpu().numpy(),fourmom[:,3].sum(dim=-1).cpu().numpy())
    # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[...,0].sum(dim=-1).cpu().numpy(),fourmom[...,1].sum(dim=-1).cpu().numpy(),fourmom[...,2].sum(dim=-1).cpu().numpy(),fourmom[...,3].sum(dim=-1).cpu().numpy())
    _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,2,0],fourmom[:,2,1],fourmom[:,2,2],fourmom[:,2,3])
    _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,:2,0].sum(axis=-1),fourmom[:,:2,1].sum(axis=-1),fourmom[:,:2,2].sum(axis=-1),fourmom[:,:2,3].sum(axis=-1))
    # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,:,0].sum(axis=-1),fourmom[:,:,1].sum(axis=-1),fourmom[:,:,2].sum(axis=-1),fourmom[:,:,3].sum(axis=-1))
    _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[dsid_mask,:,0].sum(axis=-1),fourmom[dsid_mask,:,1].sum(axis=-1),fourmom[dsid_mask,:,2].sum(axis=-1),fourmom[dsid_mask,:,3].sum(axis=-1))
    plt.hist(m*1e-3, weights=MCWts.cpu()[dsid_mask], histtype='step', label=val_metrics_MCWts.DSID_MASS_MAPPING[dsid],bins=bins)
plt.legend()
# plt.savefig('tmp.pdf')
plt.show()
plt.close('all')


# %%
print(torch.where(y.argmax(dim=-1)[:50]==1))
print(torch.where(y.argmax(dim=-1)[:50]==2))
print(x[torch.where(y.argmax(dim=-1)[:50]==1)[0],:,-1])
print(x[torch.where(y.argmax(dim=-1)[:50]==2)[0],:,-1])
print(x[:,:,-1].shape)
simplified_true_inclusion = ((x[:,:,-1]==1)*1 + (x[:,:,-1]>1)*2)
one_hot_true_inclusion = torch.nn.functional.one_hot(simplified_true_inclusion.to(torch.long))
print(one_hot_true_inclusion.shape)
ex=torch.randn(one_hot_true_inclusion.shape)
ce=torch.nn.CrossEntropyLoss()
# ce(ex[0],)
ex[0].shape

tmp_logits=(one_hot_true_inclusion==0)*-10.0 + (one_hot_true_inclusion!=0)*1.0
tmp_logits=(one_hot_true_inclusion!=0)*1.

ce(tmp_logits.flatten(), one_hot_true_inclusion.float().flatten())
# num_nonempty_objs = (types!=padding_token).sum(dim=-1)
ce_loss = ce(einops.rearrange(tmp_logits[:10], 'batch object cls -> (batch object) cls'),
              einops.rearrange(one_hot_true_inclusion[:10].float(), 'batch object cls -> (batch object) cls')
              )
print(ce_loss)


# %%

# %%
# %%
import torch
import torch.nn as nn

# Define batch size, number of objects, and number of classes
batch_size = 10000
num_objects = 14
num_classes = 3

# CrossEntropyLoss expects raw logits (not softmax probabilities)
criterion = nn.CrossEntropyLoss()

# Random example
logits_random = torch.randn(batch_size, num_objects, num_classes)  # Random logits
labels_random = torch.randint(0, num_classes, (batch_size, num_objects))  # Random labels
loss_random = criterion(logits_random.view(-1, num_classes), labels_random.view(-1))
print(f"Random case loss: {loss_random.item()}")

# Correct classification example
logits_correct = torch.full((batch_size, num_objects, num_classes), -100.0)  # Start with very low values
labels_correct = torch.randint(0, num_classes, (batch_size, num_objects))  # Random correct labels
logits_correct[torch.arange(batch_size).unsqueeze(1), torch.arange(num_objects).unsqueeze(0), labels_correct] = 100.0  # Set correct class logits high
loss_correct = criterion(logits_correct.view(-1, num_classes), labels_correct.view(-1))
print(f"Correct classification loss: {loss_correct.item()}")  # Should be close to zero

# Incorrect classification example
logits_incorrect = torch.full((batch_size, num_objects, num_classes), 100.0)  # Set everything high
labels_incorrect = torch.randint(0, num_classes, (batch_size, num_objects))  # Random labels
wrong_classes = (labels_incorrect + 1) % num_classes  # Shift all labels to incorrect classes
logits_incorrect[torch.arange(batch_size).unsqueeze(1), torch.arange(num_objects).unsqueeze(0), wrong_classes] = -100.0  # Force incorrect predictions
loss_incorrect = criterion(logits_incorrect.view(-1, num_classes), labels_incorrect.view(-1))
print(f"Incorrect classification loss: {loss_incorrect.item()}")  # Should be high
