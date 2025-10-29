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
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor, nn
import einops
import wandb
import torch.nn.functional as F
from models.models import ConfigurableNN
# from torchmetrics import Accuracy, AUC, ConfusionMatrix
# from torchmetrics import ConfusionMatrix
# from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import shutil
from metrics.highlevelmetrics import HEPMetrics, HEPLoss, init_wandb
from typing import List
from utils.utils import DSID_MASS_MAPPING
sorted_masses = sorted(list(DSID_MASS_MAPPING.values()))

# %%
# Save the current script to the saveDir so we know what the training script was



model_params = {
    'High-level Joint NN' : {
        'type': 'hl',
        # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-114052_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt86528.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level Parametrised NN' : {
        'type': 'hl',
        # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-114146_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt86528.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':True,
        'device':'cpu',
    },
    'High-level Joint NN (excl 0.9TeV)' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250630-183253_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83616.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level Parametrised NN (excl 0.9TeV)' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250630-183150_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83616.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':True,
        'device':'cpu',
    },
    'High-level 0.8-trained' : {
        'type': 'hl',
        # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt80650.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level 0.9-trained' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-114122_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt81500.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level 1.0-trained' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-175944_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt82100.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level 1.2-trained' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-180006_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt82750.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level 1.4-trained' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-180023_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83150.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level 1.6-trained' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-180158_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83400.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level 1.8-trained' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-195616_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83400.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level 2.0-trained' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-195637_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83600.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level 2.5-trained' : {
        'type': 'hl',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-195658_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83600.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
    'High-level 3.0-trained' : {
        'type': 'hl',
        # 'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090811_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt51616.pth',
        'filepath':'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250629-090928_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt83400.pth',
        'layers':[512,256,128],
        'PARAMETRISED_NN':False,
        'device':'cpu',
    },
}

vms = {}
vms_MCWts = {}
sig_rems = {}
sig_rems_100 = {}
sig_rems_central = {}
sig_rems_central_100 = {}
roc_aucs = {}
roc_aucs_central = {}


for model_name in model_params:

    if model_params[model_name]['type']=='hl':
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
        train_dataloader = ProportionalMemoryMappedDatasetHighLevel(
                        memmap_paths = memmap_paths_train,  # DSID to memmap path
                        N_Real_Vars=N_Real_Vars, # Will auto add 6 (for the y, w, dsid, mWh, mH, eventNumbers) inside the funciton
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
                        #  signal_reweights=np.array([3,3,3,1,1,1,1,1,1,1]),
                        #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
        )
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

        train_loader, val_loader = train_dataloader, val_dataloader

        criterion = HEPLoss(apply_correlation_penalty=False, alpha=1.0)
        # train_metrics = HEPMetrics(parametrised_nn=model_params[model_name]['PARAMETRISED_NN'], max_bkg_levels=[200], max_buffer_len=int(train_dataloader.get_total_samples()), channel=channel, total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[1000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
        # val_metrics = HEPMetrics(parametrised_nn=model_params[model_name]['PARAMETRISED_NN'], max_bkg_levels=[200], max_buffer_len=int(val_dataloader.get_total_samples()), channel=channel, total_weights_per_dsid=val_dataloader.abs_weight_sums, signal_acceptance_levels=[1000])
        # train_metrics_MCWts = HEPMetrics(parametrised_nn=model_params[model_name]['PARAMETRISED_NN'], max_bkg_levels=[200], max_buffer_len=int(train_dataloader.get_total_samples()), channel=channel, total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[1000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
        val_metrics_MCWts = HEPMetrics(parametrised_nn=model_params[model_name]['PARAMETRISED_NN'], max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), channel=channel, total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[1000])
        
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
        # sig_rems_central_100[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
    for k in range(len(sig_rems[model_name])):
        print(f"{sorted_masses[k]}: {sig_rems[model_name][k]:10.1f}", end=', ')
    print()
    for k in range(len(roc_aucs[model_name])):
        print(f"{sorted_masses[k]}: {roc_aucs[model_name][k]:10.1f}", end=', ')


# %%
sig_rems_100_central = {}
for model_name in model_params:
    sig_rems[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
    sig_rems_100[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow0_mHhigh10000000000.0'] for mass in sorted_masses]
    roc_aucs[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow0_mHhigh10000000'] for mass in sorted_masses]
    roc_aucs_central[model_name] = [vms_MCWts[model_name][f'val_MC/auc_qqbb_{mass}_mHlow95_mHhigh140'] for mass in sorted_masses]
    sig_rems_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/200_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
    sig_rems_100_central[model_name] = [vms_MCWts[model_name][f'val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/100_{mass}_mHlow95000.0_mHhigh140000.0'] for mass in sorted_masses]
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
mycolors = {'High-level Joint NN': 'tab:green', 'High-level Parametrised NN': 'tab:purple', 'High-level 0.8-trained': 'tab:red', 'High-level 1.2-trained': 'tab:blue', 'High-level 3.0-trained': 'tab:brown'}
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(5, 3))
# for k, model_name in enumerate(model_params):
for k, model_name in enumerate(['High-level Joint NN', 'High-level Parametrised NN', 'High-level 0.8-trained', 'High-level 1.2-trained', 'High-level 3.0-trained']):
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
mycolors = ['tab:green', 'tab:purple', 'tab:orange']
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(5, 3))
for k, model_name in enumerate(['High-level Joint NN', 'High-level Parametrised NN', 'High-level Individual NNs', 'High-level Joint NN (excl 0.9TeV)', 'High-level Parametrised NN (excl 0.9TeV)']):
    if model_name == 'High-level Individual NNs':
        sig_rems_individual = [sig_rems[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] for mass_n in range(len(sorted_masses))]
        plt.plot(sorted_masses, sig_rems_individual, label=model_name.replace('High-level ', ''), linestyle='-', color=mycolors[k])
    elif '(excl 0.9TeV)' in model_name:
        plt.plot(sorted_masses, sig_rems[model_name], label=model_name.replace('High-level ', ''), linestyle='--', color=mycolors[k-3])
    else:
        plt.plot(sorted_masses, sig_rems[model_name], label=model_name.replace('High-level ', ''), linestyle='-', color=mycolors[k])
# Make a legend outside of the main plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Mass (GeV)')
# plt.xlim([0.8,3.8])
plt.ylabel('Expected Signal $S_{b_{200}}$')
plt.title('Expected Signal vs Mass')
plt.show()


plt.rcParams['text.usetex'] = True
plt.figure(figsize=(5, 3))
for k, model_name in enumerate(['High-level Joint NN', 'High-level Parametrised NN', 'High-level Individual NNs', 'High-level Joint NN (excl 0.9TeV)', 'High-level Parametrised NN (excl 0.9TeV)']):
    if model_name == 'High-level Individual NNs':
        sig_rems_individual = [sig_rems_central[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] for mass_n in range(len(sorted_masses))]
        plt.plot(sorted_masses, sig_rems_individual, label=model_name.replace('High-level ', ''), linestyle='-', color=mycolors[k])
    elif '(excl 0.9TeV)' in model_name:
        plt.plot(sorted_masses, sig_rems_central[model_name], label=model_name.replace('High-level ', ''), linestyle='--', color=mycolors[k-3])
    else:
        plt.plot(sorted_masses, sig_rems_central[model_name], label=model_name.replace('High-level ', ''), linestyle='-', color=mycolors[k])
# Make a legend outside of the main plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Mass (GeV)')
# plt.xlim([0.8,3.8])
plt.ylabel('Expected Signal $S_{b_{200}}$ (Central Mass Region)')
plt.title('Expected Signal vs Mass (Central Mass Region)')
plt.show()

# Same for roc aucs
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(5, 3))
for k, model_name in enumerate(['High-level Joint NN', 'High-level Parametrised NN', 'High-level Individual NNs', 'High-level Joint NN (excl 0.9TeV)', 'High-level Parametrised NN (excl 0.9TeV)']):
    if model_name == 'High-level Individual NNs':
        roc_aucs_individual = [roc_aucs[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] for mass_n in range(len(sorted_masses))]
        plt.plot(sorted_masses, roc_aucs_individual, label=model_name.replace('High-level ', ''), linestyle='-', color=mycolors[k])
    elif '(excl 0.9TeV)' in model_name:
        plt.plot(sorted_masses, roc_aucs[model_name], label=model_name.replace('High-level ', ''), linestyle='--', color=mycolors[k-3])
    else:
        plt.plot(sorted_masses, roc_aucs[model_name], label=model_name.replace('High-level ', ''), linestyle='-', color=mycolors[k])
# Make a legend outside of the main plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Mass (GeV)')
# plt.xlim([0.8,3.8])
plt.ylabel('$S_{ROC}$')
plt.title('ROC AUC vs Mass')
plt.show()

plt.rcParams['text.usetex'] = True
plt.figure(figsize=(5, 3))
for k, model_name in enumerate(['High-level Joint NN', 'High-level Parametrised NN', 'High-level Individual NNs', 'High-level Joint NN (excl 0.9TeV)', 'High-level Parametrised NN (excl 0.9TeV)']):
    if model_name == 'High-level Individual NNs':
        roc_aucs_individual = [roc_aucs_central[f'High-level {sorted_masses[mass_n]}-trained'][mass_n] for mass_n in range(len(sorted_masses))]
        plt.plot(sorted_masses, roc_aucs_individual, label=model_name.replace('High-level ', ''), linestyle='-', color=mycolors[k])
    elif '(excl 0.9TeV)' in model_name:
        plt.plot(sorted_masses, roc_aucs_central[model_name], label=model_name.replace('High-level ', ''), linestyle='--', color=mycolors[k-3])
    else:
        plt.plot(sorted_masses, roc_aucs_central[model_name], label=model_name.replace('High-level ', ''), linestyle='-', color=mycolors[k])
# Make a legend outside of the main plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Mass (GeV)')
# plt.xlim([0.8,3.8])
plt.ylabel('$S_{ROC}$')
plt.title('ROC AUC vs Mass (Central Mass Region)')
plt.show()

# %%
# Create LaTeX table with values from one of the metrics
metric_name = 'sig_rems'
# metric_name = 'roc_aucs'
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
# joint_models = ['High-level Joint NN', 'High-level Parametrised NN', 'High-level Joint NN (excl 0.9TeV)', 'High-level Parametrised NN (excl 0.9TeV)']
joint_models = ['High-level Joint NN', 'High-level Parametrised NN', 'High-level Joint NN (excl 2.0TeV)', 'High-level Parametrised NN (excl 2.0TeV)']

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
