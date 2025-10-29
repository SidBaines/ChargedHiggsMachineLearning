# %% # Load required modules
import time
ts = []
ts.append(time.time())

import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
import numpy as np
import torch
from datetime import datetime
from jaxtyping import Float
import einops
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Agg') # If you want to run in batch mode, and not see the plots made
# from utils import decode_y_eval_to_info
from dataloaders.highleveldataloader import ProportionalMemoryMappedDatasetHighLevel
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset as ProportionalMemoryMappedDatasetLowLevel
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor, nn
import einops
import wandb
import torch.nn.functional as F
from models.models import ConfigurableNN, TestNetwork
# from torchmetrics import Accuracy, AUC, ConfusionMatrix
# from torchmetrics import ConfusionMatrix
# from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import shutil
from metrics.highlevelmetrics import HEPMetrics, init_wandb
import metrics.lowlevelmetrics as lowlevelmetrics
from typing import List
from utils.utils import DSID_MASS_MAPPING
sorted_masses = sorted(list(DSID_MASS_MAPPING.values()))

# %%

model_params = {
    # 'Low-level NN (transformer-reco inputs)' : {
    #     'type': 'Low-level NN (transformer-reco inputs)',
    #     # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
    #     # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250510-105524_TrainingOutput/models/lvbb_Nplits2_ValIdx0/chkpt29_396270.pth',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250702-164642_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt_latest.pth',
    #     'd_attn':None,
    #     'include_mlp':True,
    #     'd_model': 200,
    #     'd_mlp': 400,
    #     'num_blocks':3,
    #     'dropout_p': 0.0,
    #     'num_heads':2,
    #     'device':'cpu',
    # },
    # 'Low-level pre-split NN' : {
    #     'type': 'Low-level pre-split NN',
    #     # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250619-113114_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt_latest.pth',
    #     'd_attn':None,
    #     'include_mlp':True,
    #     'd_model': 200,
    #     'd_mlp': 400,
    #     'num_blocks':3,
    #     'dropout_p': 0.0,
    #     'num_heads':2,
    #     'device':'cpu',
    # },
    # 'Low-level combined NN' : {
    #     'type': 'Low-level combined NN',
    #     # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250619-113045_TrainingOutput/models/combined_Nplits2_ValIdx0/chkpt_latest.pth',
    #     'd_attn':None,
    #     'include_mlp':True,
    #     'd_model': 200,
    #     'd_mlp': 400,
    #     'num_blocks':3,
    #     'dropout_p': 0.0,
    #     'num_heads':2,
    #     'device':'cpu',
    # },
    'High-level Joint NN' : {
        'type': 'hl',
        # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-114052_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt86528.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    # 'High-level Parametrised NN' : {
    #     'type': 'hl',
    #     # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-114146_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt86528.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':True,
    #     'device':'cpu',
    # },
    # 'High-level Joint NN (excl 0.9TeV)' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250630-183253_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83616.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level Parametrised NN (excl 2.0TeV)' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250702-074126_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt82272.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':True,
    #     'device':'cpu',
    # },
    # 'High-level Joint NN (excl 2.0TeV)' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250702-074155_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt82272.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level Parametrised NN (excl 0.9TeV)' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250630-183150_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83616.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':True,
    #     'device':'cpu',
    # },
    # 'High-level 0.8-trained' : {
    #     'type': 'hl',
    #     # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt80650.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level 0.9-trained' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-114122_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt81500.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level 1.0-trained' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-175944_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt82100.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level 1.2-trained' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-180006_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt82750.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level 1.4-trained' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-180023_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83150.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level 1.6-trained' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-180158_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83400.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level 1.8-trained' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-195616_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83400.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level 2.0-trained' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-195637_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83600.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level 2.5-trained' : {
    #     'type': 'hl',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-195658_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83600.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
    # 'High-level 3.0-trained' : {
    #     'type': 'hl',
    #     # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
    #     'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090928_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83400.pth',
    #     'layers':[512,256,128],
    #     'PARAMETRISED_NN':False,
    #     'device':'cpu',
    # },
}

vms = {}
vms_MCWts = {}
sig_rems = {}
sig_rems_100 = {}
sig_rems_central = {}
sig_rems_100_central = {}
roc_aucs = {}
roc_aucs_central = {}

model_n = 0
for model_name in model_params:
    model_n+=1
    print(f"Starting model: {model_name} ({model_n}/{len(model_params)})")
    if model_params[model_name]['type']=='Low-level NN (transformer-reco inputs)':
        if 0:
            device = torch.device(model_params[model_name]['device'])
            USE_EVENT_NUMBERS = False
            ONLY_CORRECT_TRUTH_FOR_TRAINING = True
            ONLY_CORRECT_RECO_FOR_TRAINING = True
            REMOVE_WHERE_TRUTH_WOULD_BE_CUT=False # Should be false for classification train/test, and false for reco test
            INCLUDE_INCLUSION_TAGS = True # This is only for newer files which contain these tags
            # Want to re-do the prepdata and calculate the mH on the fly (using the truth reco) so we can put the correlation loss back in/trust the mH calculations
            INCLUDE_ALL_SELECTIONS = True
            INCLUDE_NEGATIVE_SELECTIONS = True
            USE_OLD_TRUTH_SETTING = False
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


            ############   DATA LOADING CONFIG  ################
            batch_size = 256*16
            validation_split_idx=0
            n_splits=2
            KEEP_DSID= None # a dsid if we only want to keep that DSID in training, or None
            MIN_DSID = None # a dsid if we only want to keep this DSID or above (inclusive) or None
            MAX_DSID = None # a dsid if we only want to keep this DSID or below (inclusive) or None
            EXCL_DSID = None # a dsid if we only want to exclude that DSID in training, or None
            assert(sum([i is not None for i in [KEEP_DSID, MIN_DSID, MAX_DSID, EXCL_DSID]])<=1)


            ############   MODEL TRAINING CONFIG  ################
            ATTENTION_OUTPUT_BOTTLENECK_SIZE = None
            USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION = False
            if USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION or (ATTENTION_OUTPUT_BOTTLENECK_SIZE is not None):
                from interp.mechinterputils import run_with_cache_and_bottleneck
            else:
                run_with_cache_and_bottleneck = None







            
            ##############################################
            #########       DATA LOADING     #############
            ##############################################
            DATA_PATH = '/data/atlas/baines/20250321v2_AppliedRecoNNSplit_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_RemovedWrongRecoForTraining'*ONLY_CORRECT_RECO_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
            # DATA_PATH = '/data/atlas/baines/20250618v2_Split_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
            # DATA_PATH = '/data/atlas/baines/20250619v2_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
            # assert(validation_split_idx<n_splits)
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

            
            ####### Get list of relevant input files ########
            memmap_paths_train = {}
            memmap_paths_val = {}
            for file_name in os.listdir(DATA_PATH):
                if ('shape' in file_name) or ('npy' in file_name):
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
                    elif EXCL_DSID is not None:
                        if (int(dsid)!=EXCL_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                            memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                    else: # train with all
                        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                else: # Keep all the background
                    memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                memmap_paths_val[int(dsid)] = DATA_PATH+file_name

            ####### Create the dataloaders ########
            # train_dataloader = ProportionalMemoryMappedDatasetLowLevel(
            #                 memmap_paths = memmap_paths_train,  # DSID to memmap path
            #                 max_objs_in_memmap=max_n_objs_in_file,
            #                 N_Real_Vars_In_File=N_Real_Vars_In_File,
            #                 N_Real_Vars_To_Return=N_Real_Vars,
            #                 class_proportions = None,
            #                 batch_size=batch_size,
            #                 device=device, 
            #                 is_train=True,
            #                 validation_split_idx=validation_split_idx,
            #                 n_splits=n_splits,
            #                 n_targets=N_TARGETS,
            #                 shuffle=SHUFFLE_OBJECTS,
            #                 #  shuffle_batch=False,
            #                 means=means,
            #                 stds=stds,
            #                 objs_to_output=max_n_objs_to_read,
            #                 has_eventNumbers=USE_EVENT_NUMBERS,
            #                 #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
            #                 #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
            # )
            val_dataloader = ProportionalMemoryMappedDatasetLowLevel(
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
                            has_eventNumbers=USE_EVENT_NUMBERS,
                            #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                            #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
            )

            
            ##############################################
            ########       CREATE MODEL      #############
            ##############################################
            model = TestNetwork(is_reconstruction_model=False, hidden_dim_attn=model_params[model_name]['d_attn'], use_lorentz_invariant_features=USE_LORENTZ_INVARIANT_FEATURES, bottleneck_attention=ATTENTION_OUTPUT_BOTTLENECK_SIZE, feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, num_particle_types=N_CTX, hidden_dim_mlp=model_params[model_name]['d_mlp'], include_mlp=model_params[model_name]['include_mlp'], num_attention_blocks=model_params[model_name]['num_blocks'], hidden_dim=model_params[model_name]['d_model'],  dropout_p=model_params[model_name]['dropout_p'],  num_heads=model_params[model_name]['num_heads'], embedding_size=N_CTX).to(device)
            model.load_state_dict(torch.load(model_params[model_name]['filepath'], map_location=device))




            
            ##############################################
            #######   SET UP METRIC TRACKERS      ########
            ##############################################
            have_saved_a_model = False
            if not USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION:
                criterion = lowlevelmetrics.HEPLoss(N_CTX, apply_correlation_penalty=False, alpha=1.0, calc_entropy_loss=False)
            else:
                criterion = lowlevelmetrics.HEPLoss(N_CTX, apply_correlation_penalty=False, alpha=1.0, use_entropy_loss=True, entropy_weight=1e-3)

            # train_metrics = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
            # val_metrics = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
            # train_metrics_MCWts = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
            val_metrics_MCWts = lowlevelmetrics.HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
            # Validation phase
            model.eval()
            with torch.no_grad():
                orig_len_val_dataloader=len(val_dataloader)
                for batch_idx in range(orig_len_val_dataloader):
                    if (batch_idx % 10 == 0):
                        print(f"Batch {batch_idx} of {orig_len_val_dataloader}")
                    batch = next(val_dataloader)
                    # if (batch_idx >= orig_len_val_dataloader-5):
                    #     continue

                    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
                    # x, y, w, types, mqq, mlv, MCWts = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device), MCWts.to(device)
                    if (ATTENTION_OUTPUT_BOTTLENECK_SIZE is None) and (not criterion.calc_entropy_loss):
                        outputs = model(x, types)
                    else:
                        outputs, cache = run_with_cache_and_bottleneck(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
                    # if 0:
                    #     loss += criterion(outputs[:,[0, target_channel_num]], y[:,[0, target_channel_num]], w, config['wandb'], mqq, mlv, mHs).sum() * w.sum()
                    #     wt_sum += w.sum()
                    #     # outputs[:, 3-target_channel_num] = -100
                    # else:
                    #     # outputs[:, 3-target_channel_num] = -100
                    #     if (not criterion.calc_entropy_loss):
                    #         loss += criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs).sum() * w.sum()
                    #     else:
                    #         loss += criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs, cache=cache).sum() * w.sum()
                        wt_sum += w.sum()
                    # val_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv, dsids, mHs)
                    val_metrics_MCWts.update(outputs, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
            log_level = 3
            val_metrics_MCWts.reset_starts()
            # vms[model_name] = val_metrics.compute_and_log(0, prefix="val", step=0, log_level=log_level, save=False, commit=False, verbose=False)
            vms_MCWts[model_name] = vms_MCWts[model_name] = val_metrics_MCWts.compute_and_log(0, prefix="val_MC", step=0, log_level=log_level, save=False, commit=False, verbose=False)
            sig_rems[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
            # sig_rems_100[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
            roc_aucs[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow0_mHhigh10000000'] for mass in sorted_masses]
            roc_aucs_central[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow95_mHhigh140'] for mass in sorted_masses]
            sig_rems_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
            sig_rems_100_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
        else:
            ############   DATA PREP CONFIG  ################
            # These are all variables which I need to specifcy exactly what the data prep was, 
            #    so the model knows what form it's getting it in
            USE_EVENT_NUMBERS = False
            ONLY_CORRECT_TRUTH_FOR_TRAINING = True
            ONLY_CORRECT_RECO_FOR_TRAINING = True
            REMOVE_WHERE_TRUTH_WOULD_BE_CUT=False # Should be false for classification train/test, and false for reco test
            INCLUDE_INCLUSION_TAGS = True # This is only for newer files which contain these tags
            # Want to re-do the prepdata and calculate the mH on the fly (using the truth reco) so we can put the correlation loss back in/trust the mH calculations
            INCLUDE_ALL_SELECTIONS = True
            INCLUDE_NEGATIVE_SELECTIONS = True
            USE_OLD_TRUTH_SETTING = False
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



            ############   DATA LOADING CONFIG  ################
            batch_size = 256*64
            target_channel = 'qqbb'
            validation_split_idx=0
            n_splits=2
            KEEP_DSID= None # a dsid if we only want to keep that DSID in training, or None
            MIN_DSID = None # a dsid if we only want to keep this DSID or above (inclusive) or None
            MAX_DSID = None # a dsid if we only want to keep this DSID or below (inclusive) or None
            EXCL_DSID = None # a dsid if we only want to exclude that DSID in training, or None
            assert(sum([i is not None for i in [KEEP_DSID, MIN_DSID, MAX_DSID, EXCL_DSID]])<=1)


            ############   MODEL TRAINING CONFIG  ################
            MODEL_ARCH="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP"
            ATTENTION_OUTPUT_BOTTLENECK_SIZE = None
            USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION = False
            if USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION or (ATTENTION_OUTPUT_BOTTLENECK_SIZE is not None):
                from interp.mechinterputils import run_with_cache_and_bottleneck
            else:
                run_with_cache_and_bottleneck = None
            num_blocks_variable=3
            model_cfg = {'d_attn':None, 'include_mlp':True, 'd_model': 200, 'd_mlp': 400, 'num_blocks':num_blocks_variable, 'dropout_p': 0.0, "embedding_size":N_CTX, "num_heads":2}



            
            ##############################################
            #########       DATA LOADING     #############
            ##############################################
            DATA_PATH = '/data/atlas/baines/20250321v2_AppliedRecoNNSplit_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_RemovedWrongRecoForTraining'*ONLY_CORRECT_RECO_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
            # DATA_PATH = '/data/atlas/baines/20250619v1_Split_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
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
                    elif EXCL_DSID is not None:
                        if (int(dsid)!=EXCL_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                            memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                    else: # train with all
                        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                else: # Keep all the background
                    memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                memmap_paths_val[int(dsid)] = DATA_PATH+file_name

            ####### Create the dataloaders ########
            # train_dataloader = ProportionalMemoryMappedDatasetLowLevel(
            #                 memmap_paths = memmap_paths_train,  # DSID to memmap path
            #                 max_objs_in_memmap=max_n_objs_in_file,
            #                 N_Real_Vars_In_File=N_Real_Vars_In_File,
            #                 N_Real_Vars_To_Return=N_Real_Vars,
            #                 class_proportions = None,
            #                 batch_size=batch_size,
            #                 device=device, 
            #                 is_train=True,
            #                 validation_split_idx=validation_split_idx,
            #                 n_splits=n_splits,
            #                 n_targets=N_TARGETS,
            #                 shuffle=SHUFFLE_OBJECTS,
            #                 #  shuffle_batch=False,
            #                 means=means,
            #                 stds=stds,
            #                 objs_to_output=max_n_objs_to_read,
            #                 has_eventNumbers=USE_EVENT_NUMBERS,
            #                 #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
            #                 #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
            # )
            val_dataloader = ProportionalMemoryMappedDatasetLowLevel(
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
                            has_eventNumbers=USE_EVENT_NUMBERS,
                            #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                            #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
            )
            # print(train_dataloader.get_total_samples())
            print(val_dataloader.get_total_samples())
            # assert(False) # Need to check if the weighting is correct - it seemed likely that the sum of training weights for signal was not the same as for background?


            model = TestNetwork(is_reconstruction_model=False, hidden_dim_attn=model_params[model_name]['d_attn'], use_lorentz_invariant_features=USE_LORENTZ_INVARIANT_FEATURES, bottleneck_attention=ATTENTION_OUTPUT_BOTTLENECK_SIZE, feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, num_particle_types=N_CTX, hidden_dim_mlp=model_params[model_name]['d_mlp'], include_mlp=model_params[model_name]['include_mlp'], num_attention_blocks=model_params[model_name]['num_blocks'], hidden_dim=model_params[model_name]['d_model'],  dropout_p=model_params[model_name]['dropout_p'],  num_heads=model_params[model_name]['num_heads'], embedding_size=N_CTX).to(device)
            model.load_state_dict(torch.load(model_params[model_name]['filepath'], map_location=device))


            
            ##############################################
            #######   SET UP METRIC TRACKERS      ########
            ##############################################
            
            val_metrics_MCWts = lowlevelmetrics.HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
    
            val_dataloader._reset_indices()
            val_metrics_MCWts.reset()
        
            model.eval()
            with torch.no_grad():
                orig_len_val_dataloader=len(val_dataloader)
                loss = 0
                wt_sum = 0
                for batch_idx in range(orig_len_val_dataloader):
                    if (batch_idx % 10 == 0):
                        print(f"Batch {batch_idx} of {orig_len_val_dataloader}")
                    batch = next(val_dataloader)
                    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
                    if (ATTENTION_OUTPUT_BOTTLENECK_SIZE is None) and (not USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION):
                        outputs = model(x, types)
                    else:
                        outputs, cache = run_with_cache_and_bottleneck(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
                    outputs[:, 3-target_channel_num] = -100
                    val_metrics_MCWts.update(outputs, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
                    # print('[%d/%d][%d/%d] Val' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader))
                    
            log_level = 3
            val_metrics_MCWts.reset_starts()
            # vms[model_name] = val_metrics.compute_and_log(0, prefix="val", step=0, log_level=log_level, save=False, commit=False, verbose=False)
            vms_MCWts[model_name] = vms_MCWts[model_name] = val_metrics_MCWts.compute_and_log(0, prefix="val_MC", step=0, log_level=log_level, save=False, commit=False, verbose=False)
            sig_rems[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
            # sig_rems_100[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
            roc_aucs[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow0_mHhigh10000000'] for mass in sorted_masses]
            roc_aucs_central[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow95_mHhigh140'] for mass in sorted_masses]
            sig_rems_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
            sig_rems_100_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
    elif model_params[model_name]['type']=='hl':
        TOSS_UNCERTAIN_TRUTH = True
        if not TOSS_UNCERTAIN_TRUTH:
            raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
        USE_OLD_TRUTH_SETTING = True
        # if USE_OLD_TRUTH_SETTING:
        #     raise NotImplementedError # Need to check if we should require truth_agreement variable here (well, really in the prep data script) or not
        MET_CUT_ON = True
        MH_SEL = False
        N_TARGETS = 2 # Number of target classes (needed for one-hot encoding)
        BIN_WRITE_TYPE=np.float32
        N_Real_Vars = 7 # x, y, z, energy, d0val, dzval. BE CAREFUL because this might change and if it does you ahve to rebinarise
        device = torch.device(model_params[model_name]['device'])

        # Set up stuff to read in data from bin file

        batch_size = 64*2
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
        assert(not (model_params[model_name]['PARAMETRISED_NN'] and (KEEP_DSID is not None)))
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
                elif MIN_DSID is not None: # Train with only certain DSID or above
                    if (int(dsid)>=MIN_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                elif MAX_DSID is not None: # Train with only certain DSID or below
                    if (int(dsid)<=MAX_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                elif EXCL_DSID is not None:
                    if (int(dsid)!=EXCL_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                else: # train with all
                    memmap_paths_train[int(dsid)] = DATA_PATH+file_name
            else:
                pass
        # train_dataloader = ProportionalMemoryMappedDatasetHighLevel(
        #                 memmap_paths = memmap_paths_train,  # DSID to memmap path
        #                 N_Real_Vars=N_Real_Vars, # Will auto add 6 (for the y, w, dsid, mWh, mH, eventNumbers) inside the funciton
        #                 class_proportions = None,
        #                 batch_size=batch_size,
        #                 device=device, 
        #                 is_train=True,
        #                 validation_split_idx=validation_split_idx,
        #                 n_splits=n_splits,
        #                 n_targets=N_TARGETS,
        #                 means=means,
        #                 stds=stds,
        #                 has_eventNumbers=True,
        #                 return_pole_mass=model_params[model_name]['PARAMETRISED_NN'],
        #                 shuffle_batch=False,
        #                 #  signal_reweights=np.array([3,3,3,1,1,1,1,1,1,1]),
        #                 #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
        # )
        val_dataloader = ProportionalMemoryMappedDatasetHighLevel(
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
                        return_pole_mass=model_params[model_name]['PARAMETRISED_NN'],
                        shuffle_batch=False,
                        #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                        #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
        )

        model = ConfigurableNN(
                N_inputs=N_Real_Vars+int(model_params[model_name]['PARAMETRISED_NN']), 
                N_targets=N_TARGETS, 
                # hidden_layers=[400,800,800,400,400],
                # hidden_layers=[128, 128, 128],
                hidden_layers=[512, 256, 128],
                dropout_prob=0.0,
                use_batchnorm=False
                ).to(device)
        model.load_state_dict(torch.load(model_params[model_name]['filepath'], map_location=device))

        # SHOULD CHANGE WEIGHT DECAY BACK (IT WAS 1e-5 before)
        num_epochs = 50

        val_loader = val_dataloader

        val_metrics_MCWts = HEPMetrics(parametrised_nn=model_params[model_name]['PARAMETRISED_NN'], max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), channel=channel, total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[1000])
        tot_wts_per_dsid = val_metrics_MCWts.total_weights_per_dsid
        
        val_loader._reset_indices()
        # val_metrics.reset()
        val_metrics_MCWts.reset()
        model.eval()
        with torch.no_grad():
            orig_len_val_dataloader=len(val_loader)
            for batch_idx in range(orig_len_val_dataloader):
                # print(f"batch {batch_idx} of {orig_len_val_dataloader}")
                # if (batch_idx >= orig_len_val_dataloader-5):
                #     continue
                batch = next(val_loader)
                if model_params[model_name]['PARAMETRISED_NN']:
                    x, y, mWh, dsid, w, MC_w, mH, pole_mass = batch.values()
                else:
                    x, y, mWh, dsid, w, MC_w, mH = batch.values()
                # x, y, w, mWh, dsid = x.to(device), y.to(device), w.to(device), mWh.to(device), dsid.to(device)
            
                if model_params[model_name]['PARAMETRISED_NN']:
                    outputs = model(torch.cat([x, pole_mass.unsqueeze(-1)], dim=-1))
                else:
                    outputs = model(x)
                # print(x[:3])
                # print(outputs[:3])

                # Now calculate outputs again for each mass point for metrics
                if model_params[model_name]['PARAMETRISED_NN']:
                    outputs = torch.zeros(outputs.shape[0], 10, outputs.shape[1]).to(device)
                    for i in range(10):
                        outputs[:, i, :] = model(torch.cat([x, torch.ones_like(pole_mass).unsqueeze(-1)*sorted_masses[i]], dim=-1))
                else:
                    pass
                # val_metrics.update(outputs, y.argmax(dim=-1), w, mWh, dsid, mH)
                val_metrics_MCWts.update(outputs, y.argmax(dim=-1), MC_w, mWh, dsid, mH)
                # print('[%d/%d][%d/%d] Val' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader))
                # break
        log_level = 3
        val_metrics_MCWts.reset_starts()
        # vms[model_name] = val_metrics.compute_and_log(0, prefix="val", step=0, log_level=log_level, save=False, commit=False, verbose=False)
        vms_MCWts[model_name] = vms_MCWts[model_name] = val_metrics_MCWts.compute_and_log(0, prefix="val_MC", step=0, log_level=log_level, save=False, commit=False, verbose=False)
        sig_rems[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
        # sig_rems_100[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
        roc_aucs[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow0_mHhigh10000000'] for mass in sorted_masses]
        roc_aucs_central[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow95_mHhigh140'] for mass in sorted_masses]
        sig_rems_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
        sig_rems_100_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
    elif model_params[model_name]['type']=='Low-level pre-split NN':
        device = torch.device(model_params[model_name]['device'])
        USE_EVENT_NUMBERS = False
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


        ############   DATA LOADING CONFIG  ################
        batch_size = 256*16
        target_channel = 'qqbb'
        validation_split_idx=0
        n_splits=2
        KEEP_DSID= None # a dsid if we only want to keep that DSID in training, or None
        MIN_DSID = None # a dsid if we only want to keep this DSID or above (inclusive) or None
        MAX_DSID = None # a dsid if we only want to keep this DSID or below (inclusive) or None
        EXCL_DSID = None # a dsid if we only want to exclude that DSID in training, or None
        assert(sum([i is not None for i in [KEEP_DSID, MIN_DSID, MAX_DSID, EXCL_DSID]])<=1)


        ############   MODEL TRAINING CONFIG  ################
        ATTENTION_OUTPUT_BOTTLENECK_SIZE = None
        USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION = False
        if USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION or (ATTENTION_OUTPUT_BOTTLENECK_SIZE is not None):
            from interp.mechinterputils import run_with_cache_and_bottleneck
        else:
            run_with_cache_and_bottleneck = None

        
        ##############################################
        #########       DATA LOADING     #############
        ##############################################
        # DATA_PATH = '/data/atlas/baines/20250321v2_AppliedRecoNNSplit_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_RemovedWrongRecoForTraining'*ONLY_CORRECT_RECO_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
        DATA_PATH = '/data/atlas/baines/20250619v1_Split_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
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
                elif EXCL_DSID is not None:
                    if (int(dsid)!=EXCL_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                else: # train with all
                    memmap_paths_train[int(dsid)] = DATA_PATH+file_name
            else: # Keep all the background
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
            memmap_paths_val[int(dsid)] = DATA_PATH+file_name

        ####### Create the dataloaders ########
        # train_dataloader = ProportionalMemoryMappedDatasetLowLevel(
        #                 memmap_paths = memmap_paths_train,  # DSID to memmap path
        #                 max_objs_in_memmap=max_n_objs_in_file,
        #                 N_Real_Vars_In_File=N_Real_Vars_In_File,
        #                 N_Real_Vars_To_Return=N_Real_Vars,
        #                 class_proportions = None,
        #                 batch_size=batch_size,
        #                 device=device, 
        #                 is_train=True,
        #                 validation_split_idx=validation_split_idx,
        #                 n_splits=n_splits,
        #                 n_targets=N_TARGETS,
        #                 shuffle=SHUFFLE_OBJECTS,
        #                 #  shuffle_batch=False,
        #                 means=means,
        #                 stds=stds,
        #                 objs_to_output=max_n_objs_to_read,
        #                 has_eventNumbers=USE_EVENT_NUMBERS,
        #                 #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
        #                 #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
        # )
        val_dataloader = ProportionalMemoryMappedDatasetLowLevel(
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
                        has_eventNumbers=USE_EVENT_NUMBERS,
                        #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                        #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
        )
        # print(train_dataloader.get_total_samples())
        print(val_dataloader.get_total_samples())
        # assert(False) # Need to check if the weighting is correct - it seemed likely that the sum of training weights for signal was not the same as for background?


        
        ##############################################
        ########       CREATE MODEL      #############
        ##############################################
        model = TestNetwork(is_reconstruction_model=False, hidden_dim_attn=model_params[model_name]['d_attn'], use_lorentz_invariant_features=USE_LORENTZ_INVARIANT_FEATURES, bottleneck_attention=ATTENTION_OUTPUT_BOTTLENECK_SIZE, feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, num_particle_types=N_CTX, hidden_dim_mlp=model_params[model_name]['d_mlp'], include_mlp=model_params[model_name]['include_mlp'], num_attention_blocks=model_params[model_name]['num_blocks'], hidden_dim=model_params[model_name]['d_model'],  dropout_p=model_params[model_name]['dropout_p'],  num_heads=model_params[model_name]['num_heads'], embedding_size=N_CTX).to(device)
        model.load_state_dict(torch.load(model_params[model_name]['filepath'], map_location=device))

        ##############################################
        #######   SET UP METRIC TRACKERS      ########
        ##############################################
        have_saved_a_model = False
        criterion = lowlevelmetrics.HEPLoss(N_CTX, apply_correlation_penalty=False, alpha=1.0, calc_entropy_loss=False)

        # train_metrics = lowlevelmetrics.HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
        # val_metrics = lowlevelmetrics.HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
        # train_metrics_MCWts = lowlevelmetrics.HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
        val_metrics_MCWts = lowlevelmetrics.HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
        val_dataloader._reset_indices()
        val_metrics_MCWts.reset()
        model.eval()
        with torch.no_grad():
            orig_len_val_dataloader=len(val_dataloader)
            loss = 0
            wt_sum = 0
            for batch_idx in range(orig_len_val_dataloader):
                batch = next(val_dataloader)
                # if (batch_idx >= orig_len_val_dataloader-5):
                #     continue

                x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
                # x, y, w, types, mqq, mlv, MCWts = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device), MCWts.to(device)
                if (ATTENTION_OUTPUT_BOTTLENECK_SIZE is None) and (not criterion.calc_entropy_loss):
                    outputs = model(x, types)
                else:
                    outputs, cache = run_with_cache_and_bottleneck(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
                outputs[:, 3-target_channel_num] = -100
                # if 0:
                #     loss += criterion(outputs[:,[0, target_channel_num]], y[:,[0, target_channel_num]], w, config['wandb'], mqq, mlv, mHs).sum() * w.sum()
                #     wt_sum += w.sum()
                #     outputs[:, 3-target_channel_num] = -100
                # else:
                #     outputs[:, 3-target_channel_num] = -100
                #     if (not criterion.calc_entropy_loss):
                #         loss += criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs).sum() * w.sum()
                #     else:
                #         loss += criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs, cache=cache).sum() * w.sum()
                #     wt_sum += w.sum()
                # val_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv, dsids, mHs)
                val_metrics_MCWts.update(outputs, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
                # val_metrics.compute_and_log(epoch, prefix="val", log_level=log_level, save=config['wandb'], commit=False)
        log_level = 3
        val_metrics_MCWts.reset_starts()
        # vms[model_name] = val_metrics.compute_and_log(0, prefix="val", step=0, log_level=log_level, save=False, commit=False, verbose=False)
        vms_MCWts[model_name] = val_metrics_MCWts.compute_and_log(0, prefix="val_MC", step=0, log_level=log_level, save=False, commit=False, verbose=False)
        sig_rems[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
        # sig_rems_100[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
        roc_aucs[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow0_mHhigh10000000'] for mass in sorted_masses]
        roc_aucs_central[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow95_mHhigh140'] for mass in sorted_masses]
        sig_rems_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
        sig_rems_100_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
    elif model_params[model_name]['type']=='Low-level combined NN':
        device = torch.device(model_params[model_name]['device'])
        USE_EVENT_NUMBERS = False
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


        ############   DATA LOADING CONFIG  ################
        batch_size = 256*16
        validation_split_idx=0
        n_splits=2
        KEEP_DSID= None # a dsid if we only want to keep that DSID in training, or None
        MIN_DSID = None # a dsid if we only want to keep this DSID or above (inclusive) or None
        MAX_DSID = None # a dsid if we only want to keep this DSID or below (inclusive) or None
        EXCL_DSID = None # a dsid if we only want to exclude that DSID in training, or None
        assert(sum([i is not None for i in [KEEP_DSID, MIN_DSID, MAX_DSID, EXCL_DSID]])<=1)


        ############   MODEL TRAINING CONFIG  ################
        ATTENTION_OUTPUT_BOTTLENECK_SIZE = None
        USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION = False
        if USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION or (ATTENTION_OUTPUT_BOTTLENECK_SIZE is not None):
            from interp.mechinterputils import run_with_cache_and_bottleneck
        else:
            run_with_cache_and_bottleneck = None







        
        ##############################################
        #########       DATA LOADING     #############
        ##############################################
        # DATA_PATH = '/data/atlas/baines/20250321v2_AppliedRecoNNSplit_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_RemovedWrongRecoForTraining'*ONLY_CORRECT_RECO_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
        # DATA_PATH = '/data/atlas/baines/20250618v2_Split_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
        DATA_PATH = '/data/atlas/baines/20250619v2_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
        # assert(validation_split_idx<n_splits)
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

        
        ####### Get list of relevant input files ########
        memmap_paths_train = {}
        memmap_paths_val = {}
        for file_name in os.listdir(DATA_PATH):
            if ('shape' in file_name) or ('npy' in file_name):
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
                elif EXCL_DSID is not None:
                    if (int(dsid)!=EXCL_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
                else: # train with all
                    memmap_paths_train[int(dsid)] = DATA_PATH+file_name
            else: # Keep all the background
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
            memmap_paths_val[int(dsid)] = DATA_PATH+file_name

        ####### Create the dataloaders ########
        # train_dataloader = ProportionalMemoryMappedDatasetLowLevel(
        #                 memmap_paths = memmap_paths_train,  # DSID to memmap path
        #                 max_objs_in_memmap=max_n_objs_in_file,
        #                 N_Real_Vars_In_File=N_Real_Vars_In_File,
        #                 N_Real_Vars_To_Return=N_Real_Vars,
        #                 class_proportions = None,
        #                 batch_size=batch_size,
        #                 device=device, 
        #                 is_train=True,
        #                 validation_split_idx=validation_split_idx,
        #                 n_splits=n_splits,
        #                 n_targets=N_TARGETS,
        #                 shuffle=SHUFFLE_OBJECTS,
        #                 #  shuffle_batch=False,
        #                 means=means,
        #                 stds=stds,
        #                 objs_to_output=max_n_objs_to_read,
        #                 has_eventNumbers=USE_EVENT_NUMBERS,
        #                 #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
        #                 #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
        # )
        val_dataloader = ProportionalMemoryMappedDatasetLowLevel(
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
                        has_eventNumbers=USE_EVENT_NUMBERS,
                        #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                        #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
        )

        
        ##############################################
        ########       CREATE MODEL      #############
        ##############################################
        model = TestNetwork(is_reconstruction_model=False, hidden_dim_attn=model_params[model_name]['d_attn'], use_lorentz_invariant_features=USE_LORENTZ_INVARIANT_FEATURES, bottleneck_attention=ATTENTION_OUTPUT_BOTTLENECK_SIZE, feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, num_particle_types=N_CTX, hidden_dim_mlp=model_params[model_name]['d_mlp'], include_mlp=model_params[model_name]['include_mlp'], num_attention_blocks=model_params[model_name]['num_blocks'], hidden_dim=model_params[model_name]['d_model'],  dropout_p=model_params[model_name]['dropout_p'],  num_heads=model_params[model_name]['num_heads'], embedding_size=N_CTX).to(device)
        model.load_state_dict(torch.load(model_params[model_name]['filepath'], map_location=device))




        
        ##############################################
        #######   SET UP METRIC TRACKERS      ########
        ##############################################
        have_saved_a_model = False
        if not USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION:
            criterion = lowlevelmetrics.HEPLoss(N_CTX, apply_correlation_penalty=False, alpha=1.0, calc_entropy_loss=False)
        else:
            criterion = lowlevelmetrics.HEPLoss(N_CTX, apply_correlation_penalty=False, alpha=1.0, use_entropy_loss=True, entropy_weight=1e-3)

        # train_metrics = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
        # val_metrics = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
        # train_metrics_MCWts = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
        val_metrics_MCWts = lowlevelmetrics.HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
        # Validation phase
        model.eval()
        with torch.no_grad():
            orig_len_val_dataloader=len(val_dataloader)
            for batch_idx in range(orig_len_val_dataloader):
                batch = next(val_dataloader)
                # if (batch_idx >= orig_len_val_dataloader-5):
                #     continue

                x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
                # x, y, w, types, mqq, mlv, MCWts = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device), MCWts.to(device)
                if (ATTENTION_OUTPUT_BOTTLENECK_SIZE is None) and (not criterion.calc_entropy_loss):
                    outputs = model(x, types)
                else:
                    outputs, cache = run_with_cache_and_bottleneck(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
                # if 0:
                #     loss += criterion(outputs[:,[0, target_channel_num]], y[:,[0, target_channel_num]], w, config['wandb'], mqq, mlv, mHs).sum() * w.sum()
                #     wt_sum += w.sum()
                #     # outputs[:, 3-target_channel_num] = -100
                # else:
                #     # outputs[:, 3-target_channel_num] = -100
                #     if (not criterion.calc_entropy_loss):
                #         loss += criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs).sum() * w.sum()
                #     else:
                #         loss += criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs, cache=cache).sum() * w.sum()
                    wt_sum += w.sum()
                # val_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv, dsids, mHs)
                val_metrics_MCWts.update(outputs, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
        log_level = 3
        val_metrics_MCWts.reset_starts()
        # vms[model_name] = val_metrics.compute_and_log(0, prefix="val", step=0, log_level=log_level, save=False, commit=False, verbose=False)
        vms_MCWts[model_name] = vms_MCWts[model_name] = val_metrics_MCWts.compute_and_log(0, prefix="val_MC", step=0, log_level=log_level, save=False, commit=False, verbose=False)
        sig_rems[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
        # sig_rems_100[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
        roc_aucs[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow0_mHhigh10000000'] for mass in sorted_masses]
        roc_aucs_central[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow95_mHhigh140'] for mass in sorted_masses]
        sig_rems_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
        sig_rems_100_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
    else:
        raise NotImplementedError
    for k in range(len(sig_rems[model_name])):
        print(f"{sorted_masses[k]}: {sig_rems[model_name][k]:10.1f}", end=', ')
    print()
    for k in range(len(roc_aucs[model_name])):
        print(f"{sorted_masses[k]}: {roc_aucs[model_name][k]:10.1f}", end=', ')


# %%
for model_name in model_params:
    sig_rems[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
    sig_rems_100[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
    roc_aucs[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow0_mHhigh10000000'] for mass in sorted_masses]
    roc_aucs_central[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow95_mHhigh140'] for mass in sorted_masses]
    sig_rems_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
    sig_rems_100_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]



# %%
if 0:
    # Save the calculated data
    import pickle
    with open('sig_rems_qqbb.pkl', 'wb') as f:
        pickle.dump(sig_rems, f)
    with open('roc_aucs_qqbb.pkl', 'wb') as f:
        pickle.dump(roc_aucs, f)
    with open('sig_rems_central_qqbb.pkl', 'wb') as f:
        pickle.dump(sig_rems_central, f)
    with open('sig_rems_100_qqbb.pkl', 'wb') as f:
        pickle.dump(sig_rems_100, f)
    with open('sig_rems_100_central_qqbb.pkl', 'wb') as f:
        pickle.dump(sig_rems_100_central, f)
    with open('roc_aucs_central_qqbb.pkl', 'wb') as f:
        pickle.dump(roc_aucs_central, f)
    with open('tot_wts_per_dsid_qqbb.pkl', 'wb') as f:
        pickle.dump(tot_wts_per_dsid, f)

# %%
if 1:
    # Load the calculated data
    import pickle
    with open('sig_rems_qqbb.pkl', 'rb') as f:
        sig_rems = pickle.load(f)
    with open('roc_aucs_qqbb.pkl', 'rb') as f:
        roc_aucs = pickle.load(f)
    with open('sig_rems_central_qqbb.pkl', 'rb') as f:
        sig_rems_central = pickle.load(f)
    with open('sig_rems_100_qqbb.pkl', 'rb') as f:
        sig_rems_100 = pickle.load(f)
    with open('sig_rems_100_central_qqbb.pkl', 'rb') as f:
        sig_rems_100_central = pickle.load(f)
    with open('roc_aucs_central_qqbb.pkl', 'rb') as f:
        roc_aucs_central = pickle.load(f)
    with open('tot_wts_per_dsid_qqbb.pkl', 'rb') as f:
        tot_wts_per_dsid = pickle.load(f)

# %%
for model_name in model_params:
    print(f"{model_name}:", end=' ')
    for k in range(len(sig_rems[model_name])):
        print(f"{sorted_masses[k]}: {sig_rems[model_name][k]:8.2f}", end=', ')
    print()
for model_name in model_params:
    print(f"{model_name}:", end=' ')
    for k in range(len(roc_aucs[model_name])):
        print(f"{sorted_masses[k]}: {roc_aucs[model_name][k]:5.3f}", end=', ')
    print()


# %%
if 0:
    # Plots optionally including mass points separated
    mycolors = {
        'Low-level NN (transformer-reco inputs)':'tab:olive',
        'Low-level pre-split NN': 'tab:orange',
        'Low-level combined NN': 'tab:pink',
        'High-level Joint NN': 'tab:green', 
        'High-level Parametrised NN': 'tab:purple', 
        'High-level Individual NNs': 'tab:blue',
        'High-level 0.8-trained': 'tab:red', 
        'High-level 1.2-trained': 'tab:blue', 
        'High-level 3.0-trained': 'tab:brown'
        }
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(5, 3))
    # for k, model_name in enumerate(model_params):
    for k, model_name in enumerate(['Low-level pre-split NN', 'Low-level combined NN', 'High-level Joint NN', 'High-level Parametrised NN', 'High-level 0.8-trained', 'High-level 1.2-trained', 'High-level 3.0-trained']):
        ls = '--' if 'trained' in model_name else '-'
        line_width = 1.0 if 'trained' in model_name else 1.5
        plt.plot(sorted_masses, sig_rems[model_name], label=model_name, linestyle=ls, linewidth=line_width, color=mycolors[model_name])
    # Make a legend outside of the main plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Mass (GeV)')
    # plt.xlim([0.8,3.8])
    plt.ylabel('Expected Signal $S_{b_{200}}$')
    plt.title('Expected Signal vs Mass')
    plt.show()

    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(5, 3))
    for k, model_name in enumerate(model_params):
        ls = '--' if 'trained' in model_name else '-'
        line_width = 1.0 if 'trained' in model_name else 1.5
        plt.plot(sorted_masses, sig_rems_central[model_name], label=model_name, linestyle=ls, linewidth=line_width)
    # Make a legend outside of the main plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Mass (GeV)')
    # plt.xlim([0.8,3.8])
    plt.ylabel('Expected Signal $S_{b_{200}}$ (Central Mass Region)')
    plt.title('Expected Signal vs Mass (Central Mass Region)')
    plt.show()


    plt.figure(figsize=(5, 3))
    for k, model_name in enumerate(model_params):
        ls = '--' if 'trained' in model_name else '-'
        plt.plot(sorted_masses, roc_aucs[model_name], label=model_name, linestyle=ls)
    # Make a legend outside of the main plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Mass (GeV)')
    # plt.xlim([0.5,3.1])
    plt.ylabel('$S_{ROC}$')
    plt.title('ROC AUC vs Mass')
    plt.show()
    plt.close('all')

    plt.figure(figsize=(5, 3))
    for k, model_name in enumerate(model_params):
        ls = '--' if 'trained' in model_name else '-'
        plt.plot(sorted_masses, roc_aucs_central[model_name], label=model_name, linestyle=ls)
    # Make a legend outside of the main plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Mass (GeV)')
    # plt.xlim([0.5,3.1])
    plt.ylabel('$S_{ROC}$')
    plt.title('ROC AUC vs Mass (Central Mass Region)')
    plt.show()
    plt.close('all')


# %%
# Plots including individual mass point NNs combined

models_to_plot = [
    # 'Low-level NN (transformer-reco inputs)',
    'Low-level pre-split NN', 
    'Low-level combined NN', 
    # 'High-level Individual NNs',
    'High-level Joint NN', 
    'High-level Parametrised NN', 
    # 'High-level Joint NN (excl 0.9TeV)', 
    # 'High-level Parametrised NN (excl 0.9TeV)', 
    # 'High-level Joint NN (excl 2.0TeV)', 
    # 'High-level Parametrised NN (excl 2.0TeV)'
]
# plotSaveDir = 'NNPerformancePlotsForThesisQqbb/jNNpNNiNN'
plotSaveDir = 'NNPerformancePlotsForThesisQqbb/jNNpNNlowLevel'
# plotSaveDir = 'NNPerformancePlotsForThesisQqbb/jNNpNNlowLevelTransformerReco'




mycolors = {
    'Low-level NN (transformer-reco inputs)':'tab:olive',
    'Low-level pre-split NN': 'tab:orange',
    'Low-level combined NN': 'tab:pink',
    'High-level Joint NN': 'tab:green', 
    'High-level Individual NNs': 'tab:blue',
    'High-level Parametrised NN': 'tab:purple', 
    'High-level Joint NN (excl 0.9TeV)':'tab:green', 
    'High-level Parametrised NN (excl 0.9TeV)':'tab:purple',
    'High-level Joint NN (excl 2.0TeV)':'tab:green', 
    'High-level Parametrised NN (excl 2.0TeV)':'tab:purple',
    # 'High-level 0.8-trained': 'tab:red', 
    # 'High-level 1.2-trained': 'tab:blue', 
    # 'High-level 3.0-trained': 'tab:brown'
    }
mylinestyles = {
    'Low-level NN (transformer-reco inputs)':'-',
    'Low-level pre-split NN': '--',
    'Low-level combined NN': ':',
    'High-level Joint NN': '-', 
    'High-level Individual NNs': ':',
    'High-level Parametrised NN': '-.',
    'High-level Joint NN (excl 0.9TeV)':'--', 
    'High-level Parametrised NN (excl 0.9TeV)':'--',
    'High-level Joint NN (excl 2.0TeV)':':', 
    'High-level Parametrised NN (excl 2.0TeV)':':',
    # 'High-level 0.8-trained': 'tab:red', 
    # 'High-level 1.2-trained': 'tab:blue', 
    # 'High-level 3.0-trained': 'tab:brown'
    }
plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
PLOT_AS_PERCENTAGE = True
if PLOT_AS_PERCENTAGE:
    reweight_factor = [100/tot_wts_per_dsid[dsid] for dsid in sorted(DSID_MASS_MAPPING.keys())]
else:
    reweight_factor = [1.0 for _ in sorted(DSID_MASS_MAPPING.keys())]

# for k, model_name in enumerate(['High-level Joint NN', 'High-level Parametrised NN', 'High-level Individual NNs', 'High-level Joint NN (excl 0.9TeV)', 'High-level Parametrised NN (excl 0.9TeV)']):
# for k, model_name in enumerate(['Low-level pre-split NN', 'Low-level combined NN', 'High-level Joint NN', 'High-level Parametrised NN', 'High-level Individual NNs', 'High-level Joint NN (excl 0.9TeV)', 'High-level Parametrised NN (excl 0.9TeV)', 'High-level Joint NN (excl 2.0TeV)', 'High-level Parametrised NN (excl 2.0TeV)']):
plt.figure(figsize=(5, 3))
os.makedirs(plotSaveDir, exist_ok=True)
for k, model_name in enumerate(models_to_plot):
    if model_name == 'High-level Individual NNs':
        sig_rems_individual = [sig_rems[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] * reweight_factor[mass_n] for mass_n in range(len(sorted_masses))]
        plt.plot(sorted_masses, sig_rems_individual, label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
    elif '(excl 0.9TeV)' in model_name:
        plt.plot(sorted_masses, [sig_rems[model_name][mass_n] * reweight_factor[mass_n] for mass_n in range(len(sorted_masses))], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
    elif '(excl 2.0TeV)' in model_name:
        plt.plot(sorted_masses, [sig_rems[model_name][mass_n] * reweight_factor[mass_n] for mass_n in range(len(sorted_masses))], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
    else:
        plt.plot(sorted_masses, [sig_rems[model_name][mass_n] * reweight_factor[mass_n] for mass_n in range(len(sorted_masses))], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
# Make a legend outside of the main plot
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.legend(loc='lower right', prop={'size': 6})
plt.xlabel('Mass (GeV)')
# plt.xlim([0.8,3.8])
if PLOT_AS_PERCENTAGE:
    # plt.ylabel('$\mathbf{S_{b_{200}}}$ (as percentage)')
    plt.ylabel('$S_{b_{200}}$ (as percentage)')
else:
    plt.ylabel('$S_{b_{200}}$')
# plt.title('Expected Signal vs Mass')
plt.title('qqbb')
# plt.yscale('log')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(os.path.join(plotSaveDir, 'ExpectedSignalAt200BkgAccepted_vs_Mass.pdf'))
plt.show()

if 0:
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(5, 3))
    for k, model_name in enumerate(models_to_plot):
        if model_name == 'High-level Individual NNs':
            sig_rems_individual = [sig_rems_central[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] for mass_n in range(len(sorted_masses))]
            plt.plot(sorted_masses, sig_rems_individual, label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
        elif '(excl 0.9TeV)' in model_name:
            plt.plot(sorted_masses, sig_rems_central[model_name], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
        else:
            plt.plot(sorted_masses, sig_rems_central[model_name], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
    # Make a legend outside of the main plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Mass (GeV)')
    # plt.xlim([0.8,3.8])
    if PLOT_AS_PERCENTAGE:
        assert False
        plt.ylabel('$S_{b_{200}}$ (Central Mass Region, as percentage)')
    else:
        plt.ylabel('$S_{b_{200}}$ (Central Mass Region)')
    plt.title('Expected Signal vs Mass (Central Mass Region)')
    # plt.yscale('log')
    plt.show()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(plotSaveDir, 'ExpectedSignal_vs_Mass_centralMassRegion.pdf'))

# Same for roc aucs
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(5, 3))
for k, model_name in enumerate(models_to_plot):
    if model_name == 'High-level Individual NNs':
        roc_aucs_individual = [roc_aucs[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] for mass_n in range(len(sorted_masses))]
        plt.plot(sorted_masses, roc_aucs_individual, label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
    elif '(excl 0.9TeV)' in model_name:
        plt.plot(sorted_masses, roc_aucs[model_name], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
    else:
        plt.plot(sorted_masses, roc_aucs[model_name], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
# Make a legend outside of the main plot
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.legend(loc='lower right', prop={'size': 6})
plt.xlabel('Mass (GeV)')
# plt.xlim([0.8,3.8])
plt.ylabel('$S_{ROC}$')
# plt.title('ROC AUC vs Mass')
plt.title('qqbb')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(os.path.join(plotSaveDir, 'ROC_AUC_vs_Mass.pdf'))
plt.show()

if 0:
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(5, 3))
    for k, model_name in enumerate(models_to_plot):
        if model_name == 'High-level Individual NNs':
            roc_aucs_individual = [roc_aucs_central[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] for mass_n in range(len(sorted_masses))]
            plt.plot(sorted_masses, roc_aucs_individual, label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
        elif '(excl 0.9TeV)' in model_name:
            plt.plot(sorted_masses, roc_aucs_central[model_name], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
        else:
            plt.plot(sorted_masses, roc_aucs_central[model_name], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
    # Make a legend outside of the main plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Mass (GeV)')
    # plt.xlim([0.8,3.8])
    plt.ylabel('$S_{ROC}$')
    plt.title('ROC AUC vs Mass (Central Mass Region)')
    plt.show()

# Same for the 100 background accepted 

plt.figure(figsize=(5, 3))
# for k, model_name in enumerate(['High-level Joint NN', 'High-level Parametrised NN', 'High-level Individual NNs', 'High-level Joint NN (excl 0.9TeV)', 'High-level Parametrised NN (excl 0.9TeV)']):
for k, model_name in enumerate(models_to_plot):
    if model_name == 'High-level Individual NNs':
        sig_rems_individual = [sig_rems_100[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] * reweight_factor[mass_n] for mass_n in range(len(sorted_masses))]
        plt.plot(sorted_masses, sig_rems_individual, label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
    else:
        plt.plot(sorted_masses, [sig_rems_100[model_name][mass_n] * reweight_factor[mass_n] for mass_n in range(len(sorted_masses))], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
# Make a legend outside of the main plot
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.legend(loc='lower right', prop={'size': 6})
plt.xlabel('Mass (GeV)')
# plt.xlim([0.8,3.8])
if PLOT_AS_PERCENTAGE:
    plt.ylabel('$S_{b_{100}}$ (as percentage)')
else:
    plt.ylabel('$S_{b_{100}}$')
# plt.title('Expected Signal vs Mass')
plt.title('qqbb')
# plt.yscale('log')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(os.path.join(plotSaveDir, 'ExpectedSignalAt100BkgAccepted_vs_Mass.pdf'))
plt.show()

if 0:
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(5, 3))
    for k, model_name in enumerate(models_to_plot):
        if model_name == 'High-level Individual NNs':
            sig_rems_individual = [sig_rems_100_central[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] for mass_n in range(len(sorted_masses))]
            plt.plot(sorted_masses, sig_rems_individual, label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
        elif '(excl 0.9TeV)' in model_name:
            plt.plot(sorted_masses, sig_rems_100_central[model_name], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
        else:
            plt.plot(sorted_masses, sig_rems_100_central[model_name], label=model_name.replace('High-level ', ''), linestyle=mylinestyles[model_name], color=mycolors[model_name])
    # Make a legend outside of the main plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Mass (GeV)')
    # plt.xlim([0.8,3.8])
    plt.ylabel('Expected Signal $S_{b_{100}}$ (Central Mass Region)')
    plt.title('Expected Signal vs Mass (Central Mass Region)')
    # plt.yscale('log')
    plt.show()


# %%
# Create LaTeX table with values from one of the metrics
# metric_name = 'sig_rems'
metric_name = 'roc_aucs'
metric = sig_rems if metric_name == 'sig_rems' else roc_aucs
print("\n" + "="*80)
print("LaTeX Table for Expected Signal Values")
print("="*80)

# Get the single-mass trained models in order
single_mass_models = []
for mass in sorted_masses:
    model_name = f'High-level {mass}-trained'
    if model_name in model_params:
        single_mass_models.append(model_name)

# Joint and parametrised models
# joint_models = ['High-level Joint NN', 'High-level Parametrised NN']
joint_models = ['High-level Joint NN', 'High-level Parametrised NN', 'High-level Joint NN (excl 0.9TeV)', 'High-level Parametrised NN (excl 0.9TeV)']

# Start building the LaTeX table
latex_table = []
latex_table.append("\\begin{table}[h!]")
latex_table.append("\\centering")
latex_table.append("\\caption{Expected Signal $S_{b_{200}}$ for Different Models Across Mass Points}")
latex_table.append("\\label{tab:expected_signals}")

# Create column specification
n_cols = len(sorted_masses) + 1  # +1 for model name column
col_spec = "l" + "c" * len(sorted_masses)
latex_table.append(f"\\begin{{tabular}}{{{col_spec}}}")
latex_table.append("\\hline")

# Create header row
header = "Model"
for mass in sorted_masses:
    header += f" & {mass} GeV"
header += " \\\\"
latex_table.append(header)
latex_table.append("\\hline")

# Find the best individual NN for each mass column
best_individual_values = []
best_individual_indices = []
for mass_idx in range(len(sorted_masses)):
    best_value = -1
    best_idx = -1
    for model_idx, model_name in enumerate(single_mass_models):
        if model_name in metric:
            value = metric[model_name][mass_idx]
            if value > best_value:
                best_value = value
                best_idx = model_idx
    best_individual_values.append(best_value)
    best_individual_indices.append(best_idx)

# Add single-mass trained models
for model_idx, model_name in enumerate(single_mass_models):
    if model_name in metric:
        # Clean up model name for display
        display_name = model_name.replace('High-level ', '').replace('-trained', ' TeV')
        row = display_name
        
        for mass_idx in range(len(sorted_masses)):
            value = metric[model_name][mass_idx]
            if model_idx == best_individual_indices[mass_idx]:
                # Bold this value as it's the best individual NN for this mass
                if metric_name == 'sig_rems':
                    row += f" & \\textbf{{{int(value):d}}}"
                else:
                    row += f" & \\textbf{{{value:.3f}}}"
            else:
                if metric_name == 'sig_rems':
                    row += f" & {int(value):d}"
                else:
                    row += f" & {value:.3f}"
        row += " \\\\"
        latex_table.append(row)

latex_table.append("\\hline")

# Add joint and parametrised models
for model_name in joint_models:
    if model_name in metric:
        display_name = model_name.replace('High-level ', '')
        row = display_name
        
        for mass_idx in range(len(sorted_masses)):
            value = metric[model_name][mass_idx]
            if metric_name == 'sig_rems':
                row += f" & {int(value):d}"
            else:
                row += f" & {value:.3f}"
        row += " \\\\"
        latex_table.append(row)

latex_table.append("\\hline")
latex_table.append("\\end{tabular}")
latex_table.append("\\end{table}")

# Print the LaTeX table
print("\nLaTeX Table Code:")
print("-" * 40)
for line in latex_table:
    print(line)

print("\n" + "="*80)
print("Table Summary:")
print("="*80)
print("- Single-mass trained models are shown first (10 rows)")
print("- Joint NN and Parametrised NN are shown last (2 rows)")
print("- Bold values indicate the best individual NN for each mass point")
print("- Values are Expected Signal S_b_200 rounded to 1 decimal place")

# %%
val_metrics_MCWts.reset_starts()
vms_MCWts[model_name] = vms_MCWts[model_name] = val_metrics_MCWts.compute_and_log(0, prefix="val_MC", step=0, log_level=log_level, save=False, commit=False, verbose=False)