# %%
# BKG
from MyMetricsLowLevel import HEPMetrics as HepMetricsLL
import numpy as np
from MyMetricsHighLevel import HEPMetrics
import matplotlib.pyplot as plt
import os
import torch

import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os

DSID_MASS_MAPPING={510115:0.8, 510116:0.9, 510117:1.0, 510118:1.2, 510119:1.4, 510120:1.6, 510121:1.8, 510122:2.0, 510123:2.5, 510124:3.0}
density = False
nbins=30
bins=nbins
min_mass=500e3
bins=np.arange(nbins+1)/nbins*3000 + min_mass*1e-3
channel_labels = {'lvbb':1, 'qqbb':2}
BASE_DIR='/data/atlas/baines/TmpCommonModelResults/'
colors={'20250211-103234_lvbb':'gray',
        '20250217-142417_qqbb':'gray', 
        '20250219-232602':'b',
        '20250220-190937':'g',
        '20250220-212857':'gray',
        '20250221-191738':'m',
        '20250222-020838':'green',
        '20250222-121545':'black',
        '20250303-020724':'orange',
        '20250310-222454':'magenta',
        }
plt.figure(figsize=(8,3))
mHCut = 'mHCut'
modeltags = [
    # '20250221-191738',
    '20250222-020838',
    # '20250220-190937',
    # '20250220-212857',
    # '20250219-232602',
    # '20250222-121545',
    '20250310-222454',
    # '20250303-020724',
    '20250211-103234_lvbb',
    '20250217-142417_qqbb'
]
bkg_remaining = 200
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
            if ('shape' in file_name) or ('npy' in file_name) or (not (dir[-4:] in file_name)):
                continue
            dsid = file_name[5:11]
            with open(DATA_PATH+file_name + '.shape', 'r') as f:
                metadata = f.read()
            _,_, sum_weights = metadata.split(',')
            weight_sums[int(dsid)] = float(sum_weights)
        metric_tracker = HEPMetrics(channel=channels[0], max_bkg_levels=[bkg_remaining], max_buffer_len=len(all_targets), total_weights_per_dsid=weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
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
        if dir in ['20250310-222454']:
            DATA_PATH='/data/atlas/baines/20250310v1_AppliedRecoNN_WithRecoMasses_30_MetCut_RemovedUncertainTruth_WithTagInfo_KeepAllOldSelIncludingNegative/'
        weight_sums = {}
        for file_name in os.listdir(DATA_PATH):
            if ('shape' in file_name) or ('npy' in file_name):
                continue
            dsid = file_name[5:11]
            with open(DATA_PATH+file_name + '.shape', 'r') as f:
                metadata = f.read()
            _,_, sum_weights = metadata.split(',')
            weight_sums[int(dsid)] = float(sum_weights)
        metric_tracker = HepMetricsLL(max_bkg_levels=[bkg_remaining], max_buffer_len=len(all_targets), total_weights_per_dsid=weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
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
    if HighLevel:
        r2=metric_tracker.compute_signal_selection_metrics(min_mass=min_mass*1e-3)
    else:
        r2=metric_tracker.compute_signal_selection_metrics(min_mass=min_mass)
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
            min_mWh_mass_sel = (metric_tracker.all_mWhs>min_mass*1e-3)
            selectionBef = bkg_sel & min_mWh_mass_sel 
            selectionAft = bkg_sel & (metric_tracker.all_probs[:,0]<thresholds[bkg_remaining][mHCut][channel]) & min_mWh_mass_sel
            if mHCut=='mHCut':
                selectionAft = selectionAft & ((metric_tracker.all_mHs > 95e3) & (metric_tracker.all_mHs < 140e3))
            elif mHCut=='NoCut':
                pass
            else:
                assert(False)
            plt.hist(metric_tracker.all_mWhs[selectionBef], weights=metric_tracker.all_weights[selectionBef]*(metric_tracker.all_weights[selectionAft].sum()/metric_tracker.all_weights[selectionBef].sum()), bins=bins, density=density, label='Before cut (HighLevel)', histtype='step', color=colors[dir], alpha=1.0, linestyle='dashdot', linewidth=1)
            plt.hist(metric_tracker.all_mWhs[selectionAft], weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=dir, color=colors[dir], alpha=1.0)
            plt.xlabel('mWh [reco] / MeV')
            plt.legend(prop={'size':legsize})
    else:
        for channel in channel_labels.keys():
            this_sig_sel = metric_tracker.all_probs[:,channel_labels[channel]] > metric_tracker.all_probs[:,3-channel_labels[channel]]
            bkg_sel = (metric_tracker.all_dsids < 500000) | (metric_tracker.all_dsids > 600000)
            if channel=='lvbb':
                min_mWh_mass_sel = (metric_tracker.all_mWh_lvbb>min_mass)
            elif channel=='qqbb':
                min_mWh_mass_sel = (metric_tracker.all_mWh_qqbb>min_mass)
            else:
                assert(False)
            plt.subplot(1,2,channel_labels[channel])
            selectionBef = bkg_sel & this_sig_sel & min_mWh_mass_sel
            selectionAft = bkg_sel & this_sig_sel & (metric_tracker.all_probs[:,0]<thresholds[bkg_remaining][mHCut][channel]) & min_mWh_mass_sel
            if mHCut=='mHCut':
                selectionAft = selectionAft & ((metric_tracker.all_mHs > 95e3) & (metric_tracker.all_mHs < 140e3))
            elif mHCut=='NoCut':
                pass
            else:
                assert(False)
            if channel == 'lvbb':
                plt.hist(metric_tracker.all_mWh_lvbb[selectionBef]*1e-3, weights=metric_tracker.all_weights[selectionBef]*(metric_tracker.all_weights[selectionAft].sum()/metric_tracker.all_weights[selectionBef].sum()), bins=bins, density=density, label='Before cut (LowLevel)', histtype='step', color=colors[dir], alpha=1.0, linestyle='dashdot', linewidth=1)
                plt.hist(metric_tracker.all_mWh_lvbb[selectionAft]*1e-3, weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=dir, color=colors[dir])
            elif channel == 'qqbb':
                plt.hist(metric_tracker.all_mWh_qqbb[selectionBef]*1e-3, weights=metric_tracker.all_weights[selectionBef]*(metric_tracker.all_weights[selectionAft].sum()/metric_tracker.all_weights[selectionBef].sum()), bins=bins, density=density, label='Before cut (LowLevel)', histtype='step', color=colors[dir], alpha=1.0, linestyle='dashdot', linewidth=1)
                plt.hist(metric_tracker.all_mWh_qqbb[selectionAft]*1e-3, weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=dir, color=colors[dir])
            else:
                assert(False)
            plt.title(channel)
            plt.xlabel('mWh [reco] / MeV')
            plt.legend(prop={'size':legsize})

if 1:
    ROOT_FILE_DIR='/data/atlas/HplusWh/20250115_SeparateLargeRJets_NominalWeights_extrainfo_fixed/'

    # Define selections (example: based on `pt` and `eta` cuts)
    if mHCut == 'mHCut':
        selections = {
            "lvbb Original WithNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_lvbb_Final_Combined"]>0.86) & (events["mVH_lvbb"]>min_mass*1e-3),
            "qqbb Original WithNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_qqbb_Final_Combined"]>0.395) & (events["mVH_qqbb"]>min_mass*1e-3),
            "lvbb Original NoNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["mVH_lvbb"]>min_mass*1e-3),
            "qqbb Original NoNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["mVH_qqbb"]>min_mass*1e-3),
            # "lvbb central-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_lvbb_Final_Combined"]>0.86),
            # "qqbb central-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_qqbb_Final_Combined"]>0.395),
            # "lvbb all-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & (events["NN_lvbb_Final_Combined"]>0.9666),
            # "qqbb all-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & (events["NN_qqbb_Final_Combined"]>0.7145),
        }
    elif mHCut=='NoCut':
        selections = {
            "lvbb Original": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & (events["NN_lvbb_Final_Combined"]>0.9666),
            "qqbb Original": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & (events["NN_qqbb_Final_Combined"]>0.7145),
        }
    else:
        assert(False)

    if 1:
        # Now plot the hists from the OG networks
        # Initialize histograms for each selection
        hist_data = {key: [] for key in selections}
        weight_data = {key: [] for key in selections}
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
                tree = f["tree"] 
                events = tree.arrays(["mVH_lvbb", "mVH_qqbb", "mH", "MET", "selection_category", "NN_lvbb_Final_Combined", "NN_qqbb_Final_Combined", "nTrkTagsOutside", "eventWeight"], library="ak")  # Load necessary branches
                for sel_name, sel_func in selections.items():
                    selected_events = events[sel_func(events)]
                    if "lvbb" in sel_name:
                        hist_data[sel_name].extend(ak.to_numpy(selected_events["mVH_lvbb"]))  # Example: Histogram of 'pt'
                        weight_data[sel_name].extend(ak.to_numpy(selected_events["eventWeight"]))  # Example: Histogram of 'pt'
                    else:
                        hist_data[sel_name].extend(ak.to_numpy(selected_events["mVH_qqbb"]))  # Example: Histogram of 'pt'
                        weight_data[sel_name].extend(ak.to_numpy(selected_events["eventWeight"]))  # Example: Histogram of 'pt'
        for sel_name, data in hist_data.items():
            plt.subplot(1,2,channel_labels[sel_name[:4]])
            if 'WithNNCut' in sel_name:
                plt.hist(data, weights=weight_data[sel_name], bins=bins, density=density, alpha=1.0, label=sel_name, histtype='step', color='red')
            elif 'NoNNCut' in sel_name:
                plt.hist(data, weights=np.array(weight_data[sel_name])*(sum(weight_data[sel_name.replace('NoNNCut', 'WithNNCut')])/sum(weight_data[sel_name])), bins=bins, density=density, alpha=1.0, label=sel_name, histtype='step', color='red', linestyle='dashdot', linewidth=1)
            else:
                assert(False)
            plt.legend(prop={'size':legsize})

plt.suptitle('Density distributions of background')
plt.tight_layout()
plt.savefig(f'mWhDists13_{mHCut}.pdf')
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
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os


REQUIRE_CORRECT_TRUTH=True
density = False
mHCut = 'mHCut'
# mHCut = 'NoCut'
bkg_remaining = 200

total_processed_mHcut = {'lvbb': {0.8: 7311.162654252771,
                            0.9: 7943.843944228805,
                            1.0: 10808.736650021652,
                            1.2: 9649.011281344434,
                            1.4: 9998.691502663418,
                            1.6: 10109.722290197968,
                            1.8: 10378.003900802152,
                            2.0: 10284.156871981624,
                            2.5: 10081.426282512833,
                            3.0: 9607.916296221018},
                        'qqbb': {0.8: 3673.5881468187845,
                            0.9: 4290.121133322018,
                            1.0: 5426.8691054105575,
                            1.2: 5079.991653902178,
                            1.4: 5172.269811446534,
                            1.6: 5597.435663525405,
                            1.8: 5556.726140940655,
                            2.0: 5478.865115330267,
                            2.5: 5559.9747148226525,
                            3.0: 5395.088857592265
                            }
    }
total_processed = {'lvbb': {0.8: 10543.444565992937,
                                0.9: 11706.152042417945,
                                1.0: 15656.526521097336,
                                1.2: 14204.874236365351,
                                1.4: 14278.694467303585,
                                1.6: 14240.158800948562,
                                1.8: 14470.28450897462,
                                2.0: 14188.86082930099,
                                2.5: 13705.515385529301,
                                3.0: 13197.027573809039
                            },
                       'qqbb': {0.8: 5034.112435746156,
                                0.9: 5997.258905801877,
                                1.0: 7716.273578903217,
                                1.2: 7064.906963266285,
                                1.4: 7199.127888364572,
                                1.6: 7620.224158533539,
                                1.8: 7569.0606515802,
                                2.0: 7486.7352393128,
                                2.5: 7333.312492858596,
                                3.0: 7219.447452189326
                                }
        }

DSID_MASS_MAPPING={510115:0.8, 510116:0.9, 510117:1.0, 510118:1.2, 510119:1.4, 510120:1.6, 510121:1.8, 510122:2.0, 510123:2.5, 510124:3.0}
min_mass=500e3
nbins=30
bins=nbins
bins=np.arange(nbins+1)/nbins*3000 + min_mass*1e-3
channel_labels = {'lvbb':1, 'qqbb':2}
BASE_DIR='/data/atlas/baines/TmpCommonModelResults/'
colors={'20250211-103234_lvbb':'gray',
        '20250217-142417_qqbb':'gray', 
        '20250219-232602':'b',
        '20250220-190937':'g',
        '20250220-212857':'gray',
        '20250221-191738':'m',
        '20250222-020838':'green',
        '20250222-121545':'black',
        '20250303-020724':'orange',
        '20250310-222454':'magenta',
        }
for sig_dsid in range(510115, 510125):
    plt.figure(figsize=(8,3))
    modeltags = [
        # '20250221-191738',
        '20250222-020838',
        # '20250220-190937',
        # '20250220-212857',
        # '20250219-232602',
        # '20250222-121545',
        '20250310-222454',
        # '20250303-020724',
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
            metric_tracker = HEPMetrics(channel=channels[0], max_bkg_levels=[bkg_remaining], max_buffer_len=len(all_targets), total_weights_per_dsid=weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
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
            if dir in ['20250310-222454']:
                DATA_PATH='/data/atlas/baines/20250310v1_AppliedRecoNN_WithRecoMasses_30_MetCut_RemovedUncertainTruth_WithTagInfo_KeepAllOldSelIncludingNegative/'
            weight_sums = {}
            for file_name in os.listdir(DATA_PATH):
                if ('shape' in file_name) or ('npy' in file_name):
                    continue
                dsid = file_name[5:11]
                with open(DATA_PATH+file_name + '.shape', 'r') as f:
                    metadata = f.read()
                _,_, sum_weights = metadata.split(',')
                weight_sums[int(dsid)] = float(sum_weights)
            metric_tracker = HepMetricsLL(max_bkg_levels=[bkg_remaining], max_buffer_len=len(all_targets), total_weights_per_dsid=weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
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
        if HighLevel:
            r2=metric_tracker.compute_signal_selection_metrics(min_mass=min_mass*1e-3)
        else:
            r2=metric_tracker.compute_signal_selection_metrics(min_mass=min_mass)
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
                min_mWh_mass_sel = (metric_tracker.all_mWhs>min_mass*1e-3)
                if REQUIRE_CORRECT_TRUTH:
                    correct_truth = np.ones_like(min_mWh_mass_sel) # For these files, we already removed the combinatorial background
                else:
                    correct_truth = np.ones_like(min_mWh_mass_sel)
                selectionBef = sig_sel & min_mWh_mass_sel & correct_truth
                selectionAft = sig_sel & (metric_tracker.all_probs[:,0]<thresholds[bkg_remaining][mHCut][channel]) & min_mWh_mass_sel & correct_truth
                if mHCut=='mHCut':
                    selectionAft = selectionAft & ((metric_tracker.all_mHs > 95e3) & (metric_tracker.all_mHs < 140e3))
                elif mHCut=='NoCut':
                    pass
                else:
                    assert(False)
                # plt.hist(metric_tracker.all_mWhs[selectionBef], weights=metric_tracker.all_weights[selectionBef], bins=bins, density=density, label='Before cut')
                plt.hist(metric_tracker.all_mWhs[selectionAft], weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=f'{dir} ({metric_tracker.all_weights[selectionAft].sum()/total_processed_mHcut[channel][DSID_MASS_MAPPING[sig_dsid]]:.3f})', color=colors[dir] )
                plt.xlabel('mWh [reco] / MeV')
                plt.legend(prop={'size':legsize})
        else:
            for channel in channel_labels.keys():
                this_sig_sel = metric_tracker.all_probs[:,channel_labels[channel]] > metric_tracker.all_probs[:,3-channel_labels[channel]]
                sig_sel = (metric_tracker.all_dsids == sig_dsid)
                plt.subplot(1,2,channel_labels[channel])
                if channel=='lvbb':
                    min_mWh_mass_sel = (metric_tracker.all_mWh_lvbb>min_mass)
                elif channel=='qqbb':
                    min_mWh_mass_sel = (metric_tracker.all_mWh_qqbb>min_mass)
                else:
                    assert(False)
                if REQUIRE_CORRECT_TRUTH:
                    correct_truth = metric_tracker.all_targets == channel_labels[channel]
                else:
                    correct_truth = np.ones_like(min_mWh_mass_sel)
                selectionBef = sig_sel & correct_truth & this_sig_sel & min_mWh_mass_sel
                selectionAft = sig_sel & correct_truth & this_sig_sel & min_mWh_mass_sel & (metric_tracker.all_probs[:,0]<thresholds[bkg_remaining][mHCut][channel])
                if mHCut=='mHCut':
                    selectionAft = selectionAft & ((metric_tracker.all_mHs > 95e3) & (metric_tracker.all_mHs < 140e3))
                elif mHCut=='NoCut':
                    pass
                else:
                    assert(False)
                if channel == 'lvbb':
                    # plt.hist(metric_tracker.all_mWh_lvbb[selectionBef]*1e-3, weights=metric_tracker.all_weights[selectionBef], bins=bins, density=density, label='Before cut')
                    plt.hist(metric_tracker.all_mWh_lvbb[selectionAft]*1e-3, weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=f'{dir} ({metric_tracker.all_weights[selectionAft].sum()/total_processed_mHcut[channel][DSID_MASS_MAPPING[sig_dsid]]:.3f})', color=colors[dir])
                elif channel == 'qqbb':
                    # plt.hist(metric_tracker.all_mWh_qqbb[selectionBef]*1e-3, weights=metric_tracker.all_weights[selectionBef], bins=bins, density=density, label='Before cut')
                    plt.hist(metric_tracker.all_mWh_qqbb[selectionAft]*1e-3, weights=metric_tracker.all_weights[selectionAft], bins=bins, density=density, histtype='step', label=f'{dir} ({metric_tracker.all_weights[selectionAft].sum()/total_processed_mHcut[channel][DSID_MASS_MAPPING[sig_dsid]]:.3f})', color=colors[dir])
                else:
                    assert(False)
                plt.title(channel)
                plt.xlabel('mWh [reco] / MeV')
                plt.legend(prop={'size':legsize})
    
    if 1:
        ROOT_FILE_DIR='/data/atlas/HplusWh/20250115_SeparateLargeRJets_NominalWeights_extrainfo_fixed/'
        # Define selections (example: based on `pt` and `eta` cuts)
        if mHCut == 'mHCut':
            selections = {
                # "lvbb Original WithNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_lvbb_Final_Combined"]>0.86),
                # "qqbb Original WithNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_qqbb_Final_Combined"]>0.395),
            "lvbb Original WithNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_lvbb_Final_Combined"]>0.86) & (events["mVH_lvbb"]>min_mass*1e-3),
            "qqbb Original WithNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_qqbb_Final_Combined"]>0.395) & (events["mVH_qqbb"]>min_mass*1e-3),
            "lvbb Original WithNNCut WithCorrectTruth": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_lvbb_Final_Combined"]>0.86) & (events["mVH_lvbb"]>min_mass*1e-3) & (events["truth_W_decay_mode"]=="lvbb"),
            "qqbb Original WithNNCut WithCorrectTruth": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_qqbb_Final_Combined"]>0.395) & (events["mVH_qqbb"]>min_mass*1e-3) & (events["truth_W_decay_mode"]=="qqbb"),
                # "lvbb Original NoNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & ((events["mH"]>95)&(events["mH"]<140)),
                # "qqbb Original NoNNCut": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & ((events["mH"]>95)&(events["mH"]<140)),
                # "lvbb central-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_lvbb_Final_Combined"]>0.86),
                # "qqbb central-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & ((events["mH"]>95)&(events["mH"]<140)) & (events["NN_qqbb_Final_Combined"]>0.395),
                # "lvbb all-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & (events["NN_lvbb_Final_Combined"]>0.9666),
                # "qqbb all-mH all-btags high-NN-200bkg": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & (events["NN_qqbb_Final_Combined"]>0.7145),
            }
        elif mHCut=='NoCut':
            selections = {
                "lvbb Original": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & (events["NN_lvbb_Final_Combined"]>0.9666),
                "qqbb Original": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & (events["NN_qqbb_Final_Combined"]>0.7145),
                "lvbb Original WithCorrectTruth": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==0)|(events["selection_category"]==8)|(events["selection_category"]==10)) & (events["NN_lvbb_Final_Combined"]>0.9666) & (events["truth_W_decay_mode"]=="lvbb"),
                "qqbb Original WithCorrectTruth": lambda events: (events["MET"] > 30e3) & ((events["selection_category"]==3)|(events["selection_category"]==9)) & (events["NN_qqbb_Final_Combined"]>0.7145) & (events["truth_W_decay_mode"]=="qqbb"),
            }
        else:
            assert(False)

        if 1:
            # Now plot the hists from the OG networks
            # Initialize histograms for each selection
            hist_data = {key: [] for key in selections}
            weight_data = {key: [] for key in selections}
            # Loop over ROOT files
            for file in os.listdir(ROOT_FILE_DIR):
                if (file.endswith(".root")) and (f"mc16_13TeV.{sig_dsid}" in file):
                    pass
                else:
                    if (not file.endswith("csv")):
                        print(f'Skipping file {file}')
                    continue
                print(f'Processing {file}')
                with uproot.open(ROOT_FILE_DIR+file) as f:
                    tree = f["tree"] 
                    events = tree.arrays(["mVH_lvbb", "mVH_qqbb", "mH", "MET", "selection_category", "NN_lvbb_Final_Combined", "NN_qqbb_Final_Combined", "nTrkTagsOutside", "eventWeight", "truth_W_decay_mode"], library="ak")  # Load necessary branches
                    for sel_name, sel_func in selections.items():
                        selected_events = events[sel_func(events)]
                        if "lvbb" in sel_name:
                            hist_data[sel_name].extend(ak.to_numpy(selected_events["mVH_lvbb"]))  # Example: Histogram of 'pt'
                            weight_data[sel_name].extend(ak.to_numpy(selected_events["eventWeight"]))  # Example: Histogram of 'pt'
                        else:
                            hist_data[sel_name].extend(ak.to_numpy(selected_events["mVH_qqbb"]))  # Example: Histogram of 'pt'
                            weight_data[sel_name].extend(ak.to_numpy(selected_events["eventWeight"]))  # Example: Histogram of 'pt'
            for sel_name, data in hist_data.items():
                if REQUIRE_CORRECT_TRUTH and (not ("WithCorrectTruth" in sel_name)):
                    continue
                elif (not REQUIRE_CORRECT_TRUTH) and ("WithCorrectTruth" in sel_name):
                    continue
                plt.subplot(1,2,channel_labels[sel_name[:4]])
                if 'WithNNCut' in sel_name:
                    plt.hist(data, weights=weight_data[sel_name], bins=bins, density=density, alpha=1.0, label=f'{sel_name.replace(" WithCorrectTruth", "")} ({sum(weight_data[sel_name])/total_processed_mHcut[sel_name[:4]][DSID_MASS_MAPPING[sig_dsid]]:.3f})', histtype='step', color='red')
                elif 'NoNNCut' in sel_name:
                    plt.hist(data, weights=weight_data[sel_name], bins=bins, density=density, alpha=1.0, label=f'{sel_name.replace(" WithCorrectTruth", "")} ({sum(weight_data[sel_name])/total_processed_mHcut[sel_name[:4]][DSID_MASS_MAPPING[sig_dsid]]:.3f})', histtype='step', color='red', linestyle='dashdot', linewidth=1)
                else:
                    assert(False)
                plt.legend(prop={'size':legsize})

    plt.suptitle(f'{DSID_MASS_MAPPING[sig_dsid]}TeV Signal: Density distributions')
    plt.tight_layout()
    plt.savefig(f'mWhDists14_{sig_dsid}_{mHCut}{REQUIRE_CORRECT_TRUTH*"_CorrectTruthOnly"}.pdf')
    plt.show()
    plt.close()

# %%
# Directory containing the ROOT files
# %%
