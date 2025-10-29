# %%
# File to compare the performance of pre-trained high and low level classifier models, which we'll read in from the .pth files

# %% IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import torchvision.transforms.functional as FT
import torchvision.transforms.functional as FT
import numpy as np
import os

from metrics import highlevelmetrics, lowlevelmetrics
from dataloaders import lowleveldataloader, highleveldataloader
from models.models import TestNetwork, ConfigurableNN
from utils.utils import DSID_MASS_MAPPING
sorted_masses = sorted(list(DSID_MASS_MAPPING.values()))
# from interp.mechinterputils import run_with_cache_and_bottleneck













############  LOW LEVEL DATA PREP CONFIG  ################
# These are all variables which I need to specifcy exactly what the data prep was, 
#    so the model knows what form it's getting it in
ONLY_CORRECT_TRUTH_FOR_TRAINING = True
ONLY_CORRECT_RECO_FOR_TRAINING = True
REMOVE_WHERE_TRUTH_WOULD_BE_CUT=False # Should be false for classification train/test, and false for reco test
INCLUDE_INCLUSION_TAGS = True # This is only for newer files which contain these tags
# Want to re-do the prepdata and calculate the mH on the fly (using the truth reco) so we can put the correlation loss back in/trust the mH calculations
INCLUDE_ALL_SELECTIONS = False
INCLUDE_NEGATIVE_SELECTIONS = False
USE_OLD_TRUTH_SETTING = True
PHI_ROTATED = False
USE_LORENTZ_INVARIANT_FEATURES = True
TAG_INFO_INPUT = True
TOSS_UNCERTAIN_TRUTH = True
SHUFFLE_OBJECTS = True
NORMALISE_DATA = False
SCALE_DATA = True
CONVERT_TO_PT_PHI_ETA_M = False
IS_XBB_TAGGED = False
REQUIRE_XBB = False # If we only select categories 0, 3, 8, 9, 10 (ie, if INCLUDE_ALL_SELECTIONS is False) then I think this is satisfied anyway
INCLUDE_TAG_INFO = True
MET_CUT_ON = True
MH_SEL = False
USE_DROPOUT = True
if True:
    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: # For testing new model architecture classes, probably run on cpu, since CUDA will just give difficult errors if there is some problem with the arch
    device = "cpu"
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
assert(not (NORMALISE_DATA and SCALE_DATA))
assert (~((REQUIRE_XBB and (~IS_XBB_TAGGED))))
if (not INCLUDE_ALL_SELECTIONS) and INCLUDE_NEGATIVE_SELECTIONS:
    assert(False)
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
if IS_XBB_TAGGED:
    N_CTX = 7 # the SIX types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}
else:
    N_CTX = 6 # the five types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet'}
BIN_WRITE_TYPE=np.float32
max_n_objs_in_file = 30 # BE CAREFUL because this might change and if it does you ahve to rebinarise
max_n_objs_to_read = 14

if INCLUDE_TAG_INFO:
    N_Real_Vars_In_File = 5
    N_Real_Vars=5 # px, py, pz, energy, tagInfo.  BE CAREFUL because this might change and if it does you ahve to rebinarise
else:
    N_Real_Vars_In_File = 4
    N_Real_Vars=4 # px, py, pz, energy.  BE CAREFUL because this might change and if it does you ahve to rebinarise
if INCLUDE_INCLUSION_TAGS:
    N_Real_Vars_In_File += 2
    N_Real_Vars += 0

############   LOW LEVEL DATA LOADING CONFIG  ################
batch_size = 256*8
target_channel = 'qqbb'
validation_split_idx=0
n_splits=2
KEEP_DSID= None # a dsid if we only want to keep that DSID in training, or None
MIN_DSID = None # a dsid if we only want to keep this DSID or above (inclusive) or None
MAX_DSID = None # a dsid if we only want to keep this DSID or below (inclusive) or None


############  LOW LEVEL MODEL TRAINING CONFIG  ################
MODEL_ARCH="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP"
ATTENTION_OUTPUT_BOTTLENECK_SIZE = None
USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION = False
num_blocks_variable=3
model_cfg = {'d_attn':None, 'include_mlp':True, 'd_model': 152, 'd_mlp': 400, 'num_blocks':num_blocks_variable, 'dropout_p': 0.0, "embedding_size":N_CTX, "num_heads":4}

num_epochs = 30
log_interval = int(50e3/batch_size)
longer_log_interval = 100000000000
SAVE_MODEL_EVERY = 5
name_mapping = {"DEEPSETS":"DS", 
                "HYBRID_SELFATTENTION_GATED":"DSSAGA", 
                "DEEPSETS_SELFATTENTION":"DSSA", 
                "DEEPSETS_SELFATTENTION_RESIDUAL":"DSSAR", 
                "DEEPSETS_SELFATTENTION_RESIDUAL_X2":"DSSAR2", 
                "DEEPSETS_SELFATTENTION_RESIDUAL_X3":"DSSAR3", 
                "DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP":f"DSSARVTS{num_blocks_variable}",
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
        # "name":"_"+timeStr+"_LowLevel_"+name_mapping[MODEL_ARCH]+"_"+target_channel,
        "weight_decay":1e-10,
    }






# %%
##############################################
#########    LOW LEVEL DATA LOADING     #############
##############################################
# DATA_PATH = '/data/atlas/baines/20250321v2_AppliedRecoNNSplit_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_RemovedWrongRecoForTraining'*ONLY_CORRECT_RECO_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
DATA_PATH = '/data/atlas/baines/20250619v1_Split_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
DATA_PATH = '/data/atlas/baines/20250619v2_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
# assert(validation_split_idx<n_splits)
target_channel_num = {'lvbb':1, 'qqbb':2}[target_channel]
if 'AppliedRecoNN' in DATA_PATH: # This will have an extra variable, for the reco networks selection
    N_Real_Vars_In_File += 1
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

# %%
####### Get list of relevant input files ########
memmap_paths_train = {}
memmap_paths_val = {}
for file_name in os.listdir(DATA_PATH):
    if ('shape' in file_name) or ('npy' in file_name):
        continue
    if not (target_channel in file_name):
        continue
    dsid = file_name[5:11]
    if (int(dsid) > 500000) and (int(dsid) < 600000):
        if KEEP_DSID is not None:
            if (int(dsid)==KEEP_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif MIN_DSID is not None: # Train with only certain DSID or above
            if (int(dsid)>=MIN_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif MAX_DSID is not None: # Train with only certain DSID or below
            if (int(dsid)<=MAX_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        else: # train with all
            memmap_paths_train[int(dsid)] = DATA_PATH+file_name
    else: # Keep all the background
        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
    memmap_paths_val[int(dsid)] = DATA_PATH+file_name

####### Create the dataloaders ########
train_dataloader_lowlevel = lowleveldataloader.ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths_train,  # DSID to memmap path
                 max_objs_in_memmap=max_n_objs_in_file,
                 N_Real_Vars_In_File=N_Real_Vars_In_File,
                 N_Real_Vars_To_Return=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                #  shuffle_batch=False,
                 means=means,
                 stds=stds,
                 objs_to_output=max_n_objs_to_read,
                 has_eventNumbers=True,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
val_dataloader_lowlevel = lowleveldataloader.ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths_val,  # DSID to memmap path
                 max_objs_in_memmap=max_n_objs_in_file,
                 N_Real_Vars_In_File=N_Real_Vars_In_File,
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
                #  shuffle_batch=False,
                 means=means,
                 stds=stds,
                 objs_to_output=max_n_objs_to_read,
                 has_eventNumbers=True,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
print(train_dataloader_lowlevel.get_total_samples())
print(val_dataloader_lowlevel.get_total_samples())
print(val_dataloader_lowlevel.abs_weight_sums)
print(val_dataloader_lowlevel.weight_sums)





# %%
##############################################
#########    HIGH LEVEL DATA LOADING     #############
##############################################
# Some choices about the training process
# Assumes that the data has already been binarised
PARAMETRISED_NN = True
TOSS_UNCERTAIN_TRUTH = True
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
USE_OLD_TRUTH_SETTING = True
# if USE_OLD_TRUTH_SETTING:
#     raise NotImplementedError # Need to check if we should require truth_agreement variable here (well, really in the prep data script) or not
MET_CUT_ON = True
MH_SEL = False
N_TARGETS = 2 # Number of target classes (needed for one-hot encoding)
BIN_WRITE_TYPE=np.float32
N_Real_Vars = 7 # x, y, z, energy, d0val, dzval.  BE CAREFUL because this might change and if it does you ahve to rebinarise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Set up stuff to read in data from bin file

# batch_size = 64*2
DATA_PATH = '/data/atlas/baines/20250322v4_highLevel' + '_MetCut'*MET_CUT_ON + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '/'
# DATA_PATH = '/data/atlas/baines/20250429v1_HighLevelAppliedRecoNNSplit/' # For data which was recostructed using low-level network then the relevant high-level data was calculated
memmap_paths_train = {}
memmap_paths_val = {}
channel = 'qqbb'
n_splits=2
validation_split_idx=0
KEEP_DSID= None # a dsid if we only want to keep that DSID in training, or None
MIN_DSID = None # a dsid if we only want to keep this DSID or above (inclusive) or None
MAX_DSID = None # a dsid if we only want to keep this DSID or below (inclusive) or None
EXCL_DSID = None # a dsid if we only want to exclude that DSID in training, or None
assert(sum([i is not None for i in [KEEP_DSID, MIN_DSID, MAX_DSID, EXCL_DSID]])<=1)
assert(not (PARAMETRISED_NN and (KEEP_DSID is not None)))
means = np.load(f'{DATA_PATH}{channel}_mean.npy')
stds = np.load(f'{DATA_PATH}{channel}_std.npy')
for file_name in os.listdir(DATA_PATH):
    # if not '510124' in file_name:
    #     continue
    if 'shape' in file_name:
        continue
    if (channel in file_name) and ('dsid' in file_name):
        dsid = file_name[5:11]
        memmap_paths_val[int(dsid)] = DATA_PATH+file_name
        # if (int(dsid) > 510116) or (int(dsid) < 500000) or (int(dsid) > 600000):
        if KEEP_DSID is not None:
            if (int(dsid)==KEEP_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif MIN_DSID is not None: # Train with only certain DSID or above
            if (int(dsid)>=MIN_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif MAX_DSID is not None: # Train with only certain DSID or below
            if (int(dsid)<=MAX_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif EXCL_DSID is not None:
            if (int(dsid)!=EXCL_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        else: # train with all
            memmap_paths_train[int(dsid)] = DATA_PATH+file_name
    else:
        pass
train_dataloader_highlevel = highleveldataloader.ProportionalMemoryMappedDatasetHighLevel(
                 memmap_paths = memmap_paths_train,  # DSID to memmap path
                 N_Real_Vars=N_Real_Vars, # Will auto add 6 (for the y, w, dsid, mWh, mH, eventNumbers) inside the funciton
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 means=means,
                 stds=stds,
                 has_eventNumbers=True,
                 return_pole_mass=PARAMETRISED_NN,
                #  signal_reweights=np.array([3,3,3,1,1,1,1,1,1,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
val_dataloader_highlevel = highleveldataloader.ProportionalMemoryMappedDatasetHighLevel(
                 memmap_paths = memmap_paths_val,  # DSID to memmap path
                 N_Real_Vars=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 means=means,
                 stds=stds,
                 has_eventNumbers=True,
                 return_pole_mass=PARAMETRISED_NN,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
# batch = next(train_dataloader)

print(train_dataloader_highlevel.get_total_samples())
print(val_dataloader_highlevel.get_total_samples())
print(val_dataloader_highlevel.abs_weight_sums)
print(val_dataloader_highlevel.weight_sums)




# %%

## Read in the models
if 1:
    high_level_modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250613-082618_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt135200.pth"
    high_level_model = ConfigurableNN(
        N_inputs=N_Real_Vars+int(PARAMETRISED_NN), 
        N_targets=N_TARGETS, 
        hidden_layers=[512, 512, 512],
        dropout_prob=0.0,
        use_batchnorm=False
        ).to(device)
else:
    assert(False)
if 1:
    low_level_modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250617-171842_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt29_275130.pth"
    low_level_model = TestNetwork(
        is_reconstruction_model=False, 
        hidden_dim_attn=model_cfg['d_attn'], 
        use_lorentz_invariant_features=USE_LORENTZ_INVARIANT_FEATURES, 
        bottleneck_attention=ATTENTION_OUTPUT_BOTTLENECK_SIZE, 
        feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, 
        num_particle_types=N_CTX, 
        hidden_dim_mlp=model_cfg['d_mlp'], include_mlp=model_cfg['include_mlp'], num_attention_blocks=model_cfg['num_blocks'], hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size'])
loaded_state_dict = torch.load(high_level_modelfile, map_location=torch.device(device))
high_level_model.load_state_dict(loaded_state_dict)





# %%
# Actually run the models and record in the metric trackers
# Low level first
low_level_val_metrics = lowlevelmetrics.HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader_lowlevel.get_total_samples()), total_weights_per_dsid=val_dataloader_lowlevel.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
low_level_val_metrics.reset()
low_level_model.eval()
low_level_model.to(device)
val_dataloader_lowlevel._reset_indices()
orig_len_train_dataloader = len(val_dataloader_lowlevel)
for batch_idx in range(orig_len_train_dataloader):
    if ((batch_idx%10)==0):
        print(f"Low level batch {batch_idx} of {orig_len_train_dataloader}")
    batch = next(val_dataloader_lowlevel)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
    with torch.no_grad():
        if (ATTENTION_OUTPUT_BOTTLENECK_SIZE is None):# and (not criterion.calc_entropy_loss):
            outputs = low_level_model(x, types)
        else:
            outputs, cache = run_with_cache_and_bottleneck(low_level_model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
        outputs[:, 3-target_channel_num] = -100 # Have to zero out the other channel, since we are only training on one channel at a time
    low_level_val_metrics.update(outputs, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)


# %%
# Now High level
high_level_val_metrics = highlevelmetrics.HEPMetrics(
parametrised_nn=PARAMETRISED_NN, channel=channel, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader_highlevel.get_total_samples()), total_weights_per_dsid=val_dataloader_highlevel.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
high_level_val_metrics.reset()
high_level_model.eval()
high_level_model.to(device)
val_dataloader_highlevel._reset_indices()
orig_len_train_dataloader = len(val_dataloader_highlevel)
for batch_idx in range(orig_len_train_dataloader):
    if ((batch_idx%10)==0):
        print(f"High level batch {batch_idx} of {orig_len_train_dataloader}")
    batch = next(val_dataloader_highlevel)
    if PARAMETRISED_NN:
        x, y, mWh, dsid, w, MC_w, mH, pole_mass = batch.values()
    else:
        x, y, dsid, w, MC_w, mH = batch.values()
    with torch.no_grad():

        if PARAMETRISED_NN:
            # Need to re-do the outputs with each of the different masses as input
            outputs = torch.zeros(x.shape[0], 10, 2).to(device)
            for i in range(10):
                outputs[:, i, :] = high_level_model(torch.cat([x, torch.ones_like(pole_mass).unsqueeze(-1)*sorted_masses[i]], dim=-1))
        else:
            outputs = high_level_model(x)
    high_level_val_metrics.update(outputs, y.argmax(dim=-1), MC_w, mWh, dsid, mH)


# %%
# Compute the metrics
low_level_val_metrics.reset_starts()
high_level_val_metrics.reset_starts()
llm = low_level_val_metrics.compute_and_log(epoch=0, prefix="val", step=None, log_level=3, save=False, commit=None, verbose=False, calc_all=True)
hlm = high_level_val_metrics.compute_and_log(epoch=0, prefix="val", step=None, log_level=3, save=False, commit=None)
print(llm)
print(hlm)


# %%
for mass in sorted_masses:
    print(f"Mass: {mass}", end=" ")
    print(f"Low level: {llm[f'val/auc_{channel}_{mass}_mHlow95_mHhigh140']}", end=" ")
    print(f"High level: {hlm[f'val/auc_{channel}_{mass}_mHlow95_mHhigh140']}")
# %%
