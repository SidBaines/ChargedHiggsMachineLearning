# %%
# BKG
from MyMetricsLowLevel import HEPMetrics as HepMetricsLL
import numpy as np
from MyMetricsHighLevel import HEPMetrics
import matplotlib.pyplot as plt
import os
import torch
nbins=20
bins=nbins
bins=np.arange(nbins+1)/nbins*4000
channel_labels = {'lvbb':1, 'qqbb':2}
BASE_DIR='/data/atlas/baines/TmpCommonModelResults/'
colors={'20250211-103234_lvbb':'r',
        '20250217-142417_qqbb':'r', 
        '20250219-232602':'b',
        '20250220-190937':'g',
        '20250220-212857':'gray',
        '20250221-191738':'m',
        '20250222-020838':'c',
        }
plt.figure(figsize=(8,3))
mHCut = 'mHCut'
modeltags = [
    # '20250221-191738',
    '20250222-020838',
    # '20250220-190937',
    # '20250220-212857',
    # '20250219-232602',
    '20250211-103234_lvbb',
    '20250217-142417_qqbb'
]
# modeltags = list(os.listdir(BASE_DIR))
for dir in modeltags:
    # Get the thresholds
    all_targets = torch.from_numpy(np.load(BASE_DIR + dir + '/targets.npy'))
    HighLevel=dir.endswith('bb')
    if HighLevel:
        channels = [dir[-4:]]
        DATA_PATH='/data/atlas/baines/tmp3_highLevel_MetCut_OldTruth_RemovedUncertainTruth/'
        weight_sums = {}
        for file_name in os.listdir(DATA_PATH):
            if ('shape' in file_name) or ('npy' in file_name):
                continue
            dsid = file_name[5:11]
            with open(DATA_PATH+file_name + '.shape', 'r') as f:
                metadata = f.read()
            _,_, sum_weights = metadata.split(',')
            weight_sums[int(dsid)] = float(sum_weights)
        metric_tracker = HEPMetrics(channel=channels[0], max_bkg_levels=[100, 200], max_buffer_len=len(all_targets), total_weights_per_dsid=weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
        metric_tracker.update(torch.from_numpy(np.load(BASE_DIR + dir + '/probs.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/targets.npy')), 
                              torch.from_numpy(np.load(BASE_DIR + dir + '/weights.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/mWh.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/dsids.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/mHs.npy')),
            )
    else:
        channels = ['lvbb', 'qqbb']
        # Need the the amounts of each signal total for the calculations
        DATA_PATH='/data/atlas/baines/tmp_SingleXbbSelected_WithRecoMasses_14_MetCut_OldTruth_RemovedUncertainTruth_WithTagInfo/'
        weight_sums = {}
        for file_name in os.listdir(DATA_PATH):
            if ('shape' in file_name) or ('npy' in file_name):
                continue
            dsid = file_name[5:11]
            with open(DATA_PATH+file_name + '.shape', 'r') as f:
                metadata = f.read()
            _,_, sum_weights = metadata.split(',')
            weight_sums[int(dsid)] = float(sum_weights)
        metric_tracker = HepMetricsLL(max_bkg_levels=[100, 200], max_buffer_len=len(all_targets), total_weights_per_dsid=weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
        metric_tracker.update(torch.from_numpy(np.load(BASE_DIR + dir + '/probs.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/targets.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/weights.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/mWh_qqbb.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/mWh_lvbb.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/dsids.npy')),
                              torch.from_numpy(np.load(BASE_DIR + dir + '/mHs.npy')),
            )
    print(f'Read in {dir} stats')
    metric_tracker.reset_starts()
    r2=metric_tracker.compute_signal_selection_metrics()
    thresholds = {}
    for k in r2.keys():
        if k[1]==510115: # Only need to look at the 
            if k[0] not in thresholds.keys():
                thresholds[k[0]] = {}
            mHcut_mapping  = {(0, 10000000000.0):'NoCut',(95000.0, 140000.0):'mHCut'}
            thresholds[k[0]][mHcut_mapping[k[2]]] = {}
            for channel in channels:
                try:
                    thresholds[k[0]][mHcut_mapping[k[2]]][channel] = r2[k][f'{channel}_bkg_threshold'].item()
                except:
                    assert(r2[k][f'{channel}_bkg_threshold']==1)
                    thresholds[k[0]][mHcut_mapping[k[2]]][channel]=1
    
    channel_labels = {'lvbb':1, 'qqbb':2}
    legsize=6
    if HighLevel:
        for channel in channels:
            plt.subplot(1,2,channel_labels[channel])
            bkg_sel = (metric_tracker.all_dsids < 500000) | (metric_tracker.all_dsids > 600000)
            density = True
            selectionBef = bkg_sel
            selectionAft = bkg_sel & (metric_tracker.all_probs[:,0]<thresholds[200]['NoCut'][channel])
            if mHCut=='mHCut':
                selectionAft = selectionAft & ((metric_tracker.all_mHs > 95e3) & (metric_tracker.all_mHs < 140e3))
            elif mHCut=='NoCut':
                pass
            else:
                assert(False)
            # plt.hist(metric_tracker.all_mWhs[selectionBef], weights=metric_tracker.all_weights[selectionBef], bins=bins, density=density, label='Before cut')
            plt.hist(metric_tracker.all_mWhs[selectionAft], weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=dir, color=colors[dir] )
            plt.xlabel('mWh [reco] / MeV')
            plt.legend(prop={'size':legsize})
    else:
        for channel in channel_labels.keys():
            this_sig_sel = metric_tracker.all_probs[:,channel_labels[channel]] > metric_tracker.all_probs[:,3-channel_labels[channel]]
            bkg_sel = (metric_tracker.all_dsids < 500000) | (metric_tracker.all_dsids > 600000)
            density = True
            plt.subplot(1,2,channel_labels[channel])
            selectionBef = bkg_sel & this_sig_sel
            selectionAft = bkg_sel & this_sig_sel & (metric_tracker.all_probs[:,0]<thresholds[200]['NoCut'][channel])
            if mHCut=='mHCut':
                selectionAft = selectionAft & ((metric_tracker.all_mHs > 95e3) & (metric_tracker.all_mHs < 140e3))
            elif mHCut=='NoCut':
                pass
            else:
                assert(False)
            if channel == 'lvbb':
                # plt.hist(metric_tracker.all_mWh_lvbb[selectionBef]*1e-3, weights=metric_tracker.all_weights[selectionBef], bins=bins, density=density, label='Before cut')
                plt.hist(metric_tracker.all_mWh_lvbb[selectionAft]*1e-3, weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=dir, color=colors[dir])
            elif channel == 'qqbb':
                # plt.hist(metric_tracker.all_mWh_qqbb[selectionBef]*1e-3, weights=metric_tracker.all_weights[selectionBef], bins=bins, density=density, label='Before cut')
                plt.hist(metric_tracker.all_mWh_qqbb[selectionAft]*1e-3, weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=dir, color=colors[dir])
            else:
                assert(False)
            plt.title(channel)
            plt.xlabel('mWh [reco] / MeV')
            plt.legend(prop={'size':legsize})
plt.suptitle('Density distributions of background')
plt.tight_layout()
plt.savefig(f'mWhDists2_{mHCut}.pdf')
plt.show()
plt.close()



# %%
# SIGNAL
from MyMetricsLowLevel import HEPMetrics as HepMetricsLL
import numpy as np
from MyMetricsHighLevel import HEPMetrics
import matplotlib.pyplot as plt
import os
import torch
density = False
nbins=60
bins=nbins
bins=np.arange(nbins+1)/nbins*3000
channel_labels = {'lvbb':1, 'qqbb':2}
BASE_DIR='/data/atlas/baines/TmpCommonModelResults/'
colors={'20250211-103234_lvbb':'r',
        '20250217-142417_qqbb':'r', 
        '20250219-232602':'b',
        '20250220-190937':'g',
        '20250220-212857':'gray',
        '20250221-191738':'m',
        '20250222-020838':'c',
        }
for sig_dsid in range(510115, 510125):
    plt.figure(figsize=(8,3))
    mHCut = 'mHCut'
    modeltags = [
        # '20250221-191738',
        '20250222-020838',
        # '20250220-190937',
        # '20250220-212857',
        # '20250219-232602',
        '20250211-103234_lvbb',
        '20250217-142417_qqbb'
    ]
    # modeltags = list(os.listdir(BASE_DIR))
    for dir in modeltags:
        # Get the thresholds
        all_targets = torch.from_numpy(np.load(BASE_DIR + dir + '/targets.npy'))
        HighLevel=dir.endswith('bb')
        if HighLevel:
            channels = [dir[-4:]]
            DATA_PATH='/data/atlas/baines/tmp3_highLevel_MetCut_OldTruth_RemovedUncertainTruth/'
            weight_sums = {}
            for file_name in os.listdir(DATA_PATH):
                if ('shape' in file_name) or ('npy' in file_name):
                    continue
                dsid = file_name[5:11]
                with open(DATA_PATH+file_name + '.shape', 'r') as f:
                    metadata = f.read()
                _,_, sum_weights = metadata.split(',')
                weight_sums[int(dsid)] = float(sum_weights)
            metric_tracker = HEPMetrics(channel=channels[0], max_bkg_levels=[100, 200], max_buffer_len=len(all_targets), total_weights_per_dsid=weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
            metric_tracker.update(torch.from_numpy(np.load(BASE_DIR + dir + '/probs.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/targets.npy')), 
                                torch.from_numpy(np.load(BASE_DIR + dir + '/weights.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/mWh.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/dsids.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/mHs.npy')),
                )
        else:
            channels = ['lvbb', 'qqbb']
            # Need the the amounts of each signal total for the calculations
            DATA_PATH='/data/atlas/baines/tmp_SingleXbbSelected_WithRecoMasses_14_MetCut_OldTruth_RemovedUncertainTruth_WithTagInfo/'
            weight_sums = {}
            for file_name in os.listdir(DATA_PATH):
                if ('shape' in file_name) or ('npy' in file_name):
                    continue
                dsid = file_name[5:11]
                with open(DATA_PATH+file_name + '.shape', 'r') as f:
                    metadata = f.read()
                _,_, sum_weights = metadata.split(',')
                weight_sums[int(dsid)] = float(sum_weights)
            metric_tracker = HepMetricsLL(max_bkg_levels=[100, 200], max_buffer_len=len(all_targets), total_weights_per_dsid=weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
            metric_tracker.update(torch.from_numpy(np.load(BASE_DIR + dir + '/probs.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/targets.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/weights.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/mWh_qqbb.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/mWh_lvbb.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/dsids.npy')),
                                torch.from_numpy(np.load(BASE_DIR + dir + '/mHs.npy')),
                )
        print(f'Read in {dir} stats')
        metric_tracker.reset_starts()
        r2=metric_tracker.compute_signal_selection_metrics()
        thresholds = {}
        for k in r2.keys():
            if k[1]==510115: # Only need to look at the 
                if k[0] not in thresholds.keys():
                    thresholds[k[0]] = {}
                mHcut_mapping  = {(0, 10000000000.0):'NoCut',(95000.0, 140000.0):'mHCut'}
                thresholds[k[0]][mHcut_mapping[k[2]]] = {}
                for channel in channels:
                    try:
                        thresholds[k[0]][mHcut_mapping[k[2]]][channel] = r2[k][f'{channel}_bkg_threshold'].item()
                    except:
                        assert(r2[k][f'{channel}_bkg_threshold']==1)
                        thresholds[k[0]][mHcut_mapping[k[2]]][channel]=1
        
        channel_labels = {'lvbb':1, 'qqbb':2}
        legsize=6
        if HighLevel:
            for channel in channels:
                plt.subplot(1,2,channel_labels[channel])
                sig_sel = (metric_tracker.all_dsids == sig_dsid)
                selectionBef = sig_sel
                selectionAft = sig_sel & (metric_tracker.all_probs[:,0]<thresholds[200]['NoCut'][channel])
                if mHCut=='mHCut':
                    selectionAft = selectionAft & ((metric_tracker.all_mHs > 95e3) & (metric_tracker.all_mHs < 140e3))
                elif mHCut=='NoCut':
                    pass
                else:
                    assert(False)
                # plt.hist(metric_tracker.all_mWhs[selectionBef], weights=metric_tracker.all_weights[selectionBef], bins=bins, density=density, label='Before cut')
                plt.hist(metric_tracker.all_mWhs[selectionAft], weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=dir, color=colors[dir] )
                plt.xlabel('mWh [reco] / MeV')
                plt.legend(prop={'size':legsize})
        else:
            for channel in channel_labels.keys():
                this_sig_sel = metric_tracker.all_probs[:,channel_labels[channel]] > metric_tracker.all_probs[:,3-channel_labels[channel]]
                sig_sel = (metric_tracker.all_dsids == sig_dsid)
                plt.subplot(1,2,channel_labels[channel])
                selectionBef = sig_sel & this_sig_sel
                selectionAft = sig_sel & this_sig_sel & (metric_tracker.all_probs[:,0]<thresholds[200]['NoCut'][channel])
                if mHCut=='mHCut':
                    selectionAft = selectionAft & ((metric_tracker.all_mHs > 95e3) & (metric_tracker.all_mHs < 140e3))
                elif mHCut=='NoCut':
                    pass
                else:
                    assert(False)
                if channel == 'lvbb':
                    # plt.hist(metric_tracker.all_mWh_lvbb[selectionBef]*1e-3, weights=metric_tracker.all_weights[selectionBef], bins=bins, density=density, label='Before cut')
                    plt.hist(metric_tracker.all_mWh_lvbb[selectionAft]*1e-3, weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=dir, color=colors[dir])
                elif channel == 'qqbb':
                    # plt.hist(metric_tracker.all_mWh_qqbb[selectionBef]*1e-3, weights=metric_tracker.all_weights[selectionBef], bins=bins, density=density, label='Before cut')
                    plt.hist(metric_tracker.all_mWh_qqbb[selectionAft]*1e-3, weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=dir, color=colors[dir])
                else:
                    assert(False)
                plt.title(channel)
                plt.xlabel('mWh [reco] / MeV')
                plt.legend(prop={'size':legsize})
    plt.suptitle('Density distributions of background')
    plt.tight_layout()
    plt.savefig(f'mWhDists2_{sig_dsid}_{mHCut}.pdf')
    plt.show()
    plt.close()

# %%
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing the ROOT files
ROOT_FILE_DIR='/data/atlas/HplusWh/20250115_SeparateLargeRJets_NominalWeights_extrainfo_fixed/'

# Define selections (example: based on `pt` and `eta` cuts)
selections = {
    "lvbb central-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_lvbb_Final_Combined"]>0.86),
    "qqbb central-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_qqbb_Final_Combined"]>0.395),
    "lvbb all-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & (events["NN_lvbb_Final_Combined"]>0.9666),
    "qqbb all-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & (events["NN_qqbb_Final_Combined"]>0.7145),
}

# Initialize histograms for each selection
hist_data = {key: [] for key in selections}

# Loop over ROOT files
for file in os.listdir(ROOT_FILE_DIR):
    if (file.endswith(".root")) and (not ("mc16_13TeV.510" in file)):
        pass
    else:
        if (not file.endswith("csv")):
            print(f'Skipping file {file}')
        continue
    print(f'Processing {file}')
    with uproot.open(ROOT_FILE_DIR+file) as f:
        tree = f["tree"]  # Change this to the correct TTree name
        # Convert the TTree into an Awkward array
        events = tree.arrays(["mVH_lvbb", "mVH_qqbb", "mH", "MET", "selection_category", "NN_lvbb_Final_Combined", "NN_qqbb_Final_Combined", "nTrkTagsOutside"], library="ak")  # Load necessary branches
        # Apply selections and store data
        for sel_name, sel_func in selections.items():
            selected_events = events[sel_func(events)]
            if "lvbb" in sel_name:
                hist_data[sel_name].extend(ak.to_numpy(selected_events["mVH_lvbb"]))  # Example: Histogram of 'pt'
            else:
                hist_data[sel_name].extend(ak.to_numpy(selected_events["mVH_qqbb"]))  # Example: Histogram of 'pt'

# Plot histograms
plt.figure(figsize=(10, 6))
for sel_name, data in hist_data.items():
    plt.hist(data, bins=50, alpha=0.5, label=sel_name, histtype='step', linewidth=2)

plt.xlabel("pT [GeV]")
plt.ylabel("Counts")
plt.legend()
plt.title("Histogram of pT with Different Selections")
plt.show()

# %%
