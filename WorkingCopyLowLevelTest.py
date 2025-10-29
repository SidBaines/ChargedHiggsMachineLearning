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
# mpl.use('Agg')
# from utils import save_current_script#, decode_y_eval_to_info
from utils import decode_y_eval_to_info
from utils import weighted_correlation
from mynewdataloader import ProportionalMemoryMappedDataset
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
from MyMetricsLowLevel import HEPMetrics
from MyMetricsLowLevel2 import HEPMetrics2, HEPLoss
from tmpOldMetricsLowLevel import HEPMetrics3

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
SHUFFLE_OBJECTS = False
CONVERT_TO_PT_PHI_ETA_M = False
MET_CUT_ON = True
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
N_CTX = 7 # the six types of object, plus one for 'no object;. We need to hardcode this unfortunately
BIN_WRITE_TYPE=np.float32
max_n_objs = 14 # BE CAREFUL because this might change and if it does you ahve to rebinarise
N_Real_Vars = 4 # x, y, z, energy, d0val, dzval.  BE CAREFUL because this might change and if it does you ahve to rebinarise
types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}

# %%
# Set up stuff to read in data from bin file

batch_size = 64*32
DATA_PATH=f'/data/atlas/baines/tmp2_SingleXbbSelected_XbbTagged_WithRecoMasses_{max_n_objs}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired/'
DATA_PATH=f'/data/atlas/baines/tmptmp_SingleXbbSelected_XbbTagged_WithRecoMasses_{max_n_objs}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired/'
memmap_paths = {}
for file_name in os.listdir(DATA_PATH):
    if ('shape' in file_name) or ('npy' in file_name):
        continue
    dsid = file_name[5:11]
    # if dsid > 700339:
    #     continue
    memmap_paths[int(dsid)] = DATA_PATH+file_name
means = np.load(f'{DATA_PATH}mean.npy')[1:]
stds = np.load(f'{DATA_PATH}std.npy')[1:]
train_split = 1.0
train_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths,  # DSID to memmap path
                 max_n_objs=max_n_objs,
                 N_Real_Vars=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                 shuffle_batch=False,
                 train_split=train_split,
                #  means=means,
                #  stds=stds,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
val_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths,  # DSID to memmap path
                 max_n_objs=max_n_objs,
                 N_Real_Vars=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=False,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                 shuffle_batch=False,
                 train_split=train_split,
                #  means=means,
                #  stds=stds,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
# batch = next(train_dataloader)

# %%

class MyHookedTransformer(HookedTransformer):
    def __init__(self, cfg, mass_input_layer=2, mass_hidden_dim=256, **kwargs):
        super(MyHookedTransformer, self).__init__(cfg, **kwargs)
        self.hook_dict['hook_mytokens'] = HookPoint()
        self.hook_dict['hook_mytokens'].name = 'hook_mytokens'
        self.mod_dict['hook_mytokens'] = self.hook_dict['hook_mytokens']
        self.W_Embed = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_Embed, std=0.02)
        
    def forward(self, tokens: Float[Tensor, "batch object d_input"], token_types: Float[Tensor, "batch object"], **kwargs) -> Float[Tensor, "batch d_model"]:
        self.hook_dict['hook_mytokens'](tokens)
        expanded_W_E = self.W_Embed.unsqueeze(0).expand(token_types.shape[0], -1, -1, -1)
        expanded_types = token_types.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.W_Embed.shape[-2], self.W_Embed.shape[-1])
        W_E_selected = torch.gather(expanded_W_E, dim=1, index=expanded_types)
        output = einops.einsum(tokens, W_E_selected, "batch object d_input, batch object d_input d_model -> batch object d_model")
        if 'start_at_layer' in kwargs:
            raise NotImplementedError
        else:
            class_outs = super(MyHookedTransformer, self).forward(output, start_at_layer=0, **kwargs)
            class_outs = class_outs[:,0]
        return class_outs

# %%
# Create a new model
models = {}
fit_histories = {}
model_n = 0

# Create the model with the desired properties
model_cfg = HookedTransformerConfig(
    d_model=128,
    d_head=8,
    n_layers=8,
    n_heads=4,
    n_ctx=N_CTX, # Max number of types of object per event + 1 because we want a dummy row in the embedding matrix for non-existing particles
    d_vocab=N_Real_Vars, # Number of inputs per object
    d_vocab_out=N_TARGETS,  # 2 because we're doing binary classification
    d_mlp=256,
    attention_dir="bidirectional",  # defaults to "causal"
    act_fn="relu",
    use_attn_result=True,
    device=str(device),
    use_hook_tokens=True,
)

# models[model_n] = {'model' : Net(model_cfg).to(device), 'inputs' : inputs}
models[model_n] = {'model' : MyHookedTransformer(model_cfg).to(device)}
print(sum(p.numel() for p in models[model_n]['model'].parameters()))

# %%
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


models[model_n]['model'].reset_hooks(including_permanent=True)
models[model_n]['model'] = add_perma_hooks_to_mask_pad_tokens(models[model_n]['model'])


# %% Cell to load in old model if we want to BE CAREFUL if you don't want to overwrite
if 1: # Just an example of how to load one back in
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250131-112943_TrainingOutput/models/0/chkpt178200.pth" # d_model=128, d_head=8,    n_layers=8,    n_heads=4,    n_ctx=N_CTX,   d_vocab=N_Real_Vars,   d_vocab_out=N_TARGETS,  d_mlp=256,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250130-153911_TrainingOutput/models/0/chkpt178200.pth" # d_model=64
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250131-163426_TrainingOutput/models/0/chkpt134000.pth" # d_model=128
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250201-122944_TrainingOutput/models/0/chkpt329640.pth" # d_model=128
    loaded_state_dict = torch.load(modelfile, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    models[model_n]['model'].load_state_dict(loaded_state_dict)


# %%
model, train_loader, val_loader = models[model_n]['model'], train_dataloader, val_dataloader
criterion = HEPLoss()
global_step = 0


# %%
def myhist2d(xvals, 
             yvals,
             wvals=None,
             logx = True,
             logy = True,
             logz = False,
             nbins=100,
             xlabel='',
             ylabel='',
):
    if (wvals is None):
        wvals = np.ones_like(xvals)
    bins=[nbins,nbins]
    if logx:
        log_binsx = np.logspace(np.log10(min(xvals).item()), np.log10(max(xvals).item()), num=nbins)
        bins[0] = log_binsx
    if logy:
        log_binsy = np.logspace(np.log10(min(yvals).item()), np.log10(max(yvals).item()), num=nbins)
        bins[1] = log_binsy
    corrcoeff = weighted_correlation(torch.from_numpy(xvals), torch.from_numpy(yvals), torch.from_numpy(wvals)).item()
    plt.figure(figsize=(4.5,4))
    if logz:
        plt.hist2d(xvals, yvals, weights=wvals, norm=mpl.colors.LogNorm(), bins=bins)
    else:
        plt.hist2d(xvals, yvals, weights=wvals, bins=bins)
    # plt.hist2d(xvals, yvals, weights=wvals, bins=[log_binsx, log_binsy])
    plt.title(f'Weighted by abs(MC)\nCorrelation: {corrcoeff:5f}')
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()



# %% # Check how many samples/batches we have
if 0:
    train_loader._reset_indices()
    samps = 0
    for i in range(len(train_loader)+1):
        batch = next(train_loader)
        samps += len(batch['y'])
        if ((i%100)==0):
            print(i)
    print(f'Batches: {i}')
    print(f'Samples: {samps}')
    print(train_loader.get_total_samples())





# %% # Check weights on the whole dataset
# train_metrics_MCWts2 = HEPMetrics2(max_buffer_len=int(train_loader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[], max_bkg_levels=[200]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
if 0:
    train_loader._reset_indices()
    sweights=0
    sweights2=0
    for i in range(len(train_loader)+1):
    # for _ in range(100):
        batch = next(train_loader)
        dsid = 510121
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        sweights += w[dsids==dsid].sum().item()
        sweights2 += MCWts[dsids==dsid].sum().item()
        if ((i%100)==0):
            print(i)
        
# %% # Check on the whole dataset
train_metrics_MCWts2 = HEPMetrics2(max_buffer_len=int(train_loader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[], max_bkg_levels=[200]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
if 1:
    train_loader._reset_indices()
    all_ys = []
    all_MCWts = []
    all_mHs = []
    all_preds = []
    for i in range(len(train_loader)+1):
    # for i in range(int((len(train_loader)+1)/2)):
    # for _ in range(100):
        if ((i%100)==0):
            print(i)
        batch = next(train_loader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        y, w, mqq, mlv, MCWts, dsids, mHs = y.cpu(), w.cpu(), mqq.cpu(), mlv.cpu(), MCWts.cpu(), dsids.cpu(), mHs.cpu()
        preds = model(x, types).detach().cpu() # Make sure to detach or it keeps it and fills up GPU memory!!
        train_metrics_MCWts2.update(preds, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
        all_ys.append(y)
        all_MCWts.append(MCWts)
        all_mHs.append(mHs)
        all_preds.append(preds)

# %%
ww=train_metrics_MCWts2.all_weights
dd=train_metrics_MCWts2.all_dsids






# %%
for k in train_metrics_MCWts2.processed_weight_sums_per_dsid.keys():
    print(f'{k}: {train_metrics_MCWts2.processed_weight_sums_per_dsid[k]:.5f}\t{train_metrics_MCWts2.total_weights_per_dsid[k]:.5f}')
train_metrics_MCWts2.reset_starts()
res22 = train_metrics_MCWts2.compute_and_log(0, prefix="train_MC", step=global_step, log_level=3, save=False)
for k in res22.keys():
    # print(f'{k}: {res22[k]:10.5f} ({res2[k]:10.5f})')
    print(f'{k}: {res22[k]:10.5f}')

assert(False)
# %% # Calculate some of the metrics
train_metrics_MCWts = HEPMetrics(total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[], max_bkg_levels=[200]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
train_metrics_MCWts2 = HEPMetrics2(max_buffer_len=int(train_loader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[], max_bkg_levels=[200]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
train_metrics_MCWts3 = HEPMetrics3(total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
if 1:
    train_loader._reset_indices()
    all_ys = []
    all_MCWts = []
    all_mHs = []
    all_preds = []
    for i in range(len(train_loader)+1):
        if ((i%100)==0):
            print(i)
        batch = next(train_loader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        y, w, mqq, mlv, MCWts, dsids, mHs = y.cpu(), w.cpu(), mqq.cpu(), mlv.cpu(), MCWts.cpu(), dsids.cpu(), mHs.cpu()
        preds = model(x, types).detach().cpu() # Make sure to detach or it keeps it and fills up GPU memory!!
        train_metrics_MCWts.update(preds, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
        train_metrics_MCWts2.update(preds, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
        train_metrics_MCWts3.update(preds, y.argmax(dim=-1), MCWts, mqq, mlv, dsids)
        all_ys.append(y)
        all_MCWts.append(MCWts)
        all_mHs.append(mHs)
        all_preds.append(preds)
    # y = torch.concatenate(all_ys, dim=0)
    # MCWts = torch.concatenate(all_MCWts, dim=0)
    # mHs = torch.concatenate(all_mHs, dim=0)
    # preds = torch.concatenate(all_preds, dim=0)

res = train_metrics_MCWts.compute_and_log(0, prefix="train_MC", step=global_step, log_level=3, save=False)
res2 = train_metrics_MCWts2.compute_and_log(0, prefix="train_MC", step=global_step, log_level=3, save=False)
res3 = train_metrics_MCWts3.compute_and_log(0, prefix="train_MC", step=global_step, log_level=3, save=False)


channel='lvbb'
mass=3.0
for r in [res, res2, res3]:#, resbak, res2bak, res3bak]:
    try:
        print(r[f'train_MC_ByMassAcceptance_FixedBkg/sig_{channel}_expected/200_{mass}'])
    except:
        print(r[f'train_MC_ByMassAcceptance_FixedBkg/sig_{channel}_expected/200_{mass}_mHlow0_mHhigh10000000000.0'])


# %%
res = train_metrics_MCWts.compute_and_log(0, prefix="train_MC", step=global_step, log_level=3, save=False)
train_metrics_MCWts2.reset_starts()
res2 = train_metrics_MCWts2.compute_and_log(0, prefix="train_MC", step=global_step, log_level=3, save=False)
res3 = train_metrics_MCWts3.compute_and_log(0, prefix="train_MC", step=global_step, log_level=3, save=False)
channel='lvbb'
mass=3.0
for r in [res, res2, res3]:#, resbak, res2bak, res3bak]:
    try:
        print(r[f'train_MC_ByMassAcceptance_FixedBkg/sig_{channel}_expected/200_{mass}'])
    except:
        print(r[f'train_MC_ByMassAcceptance_FixedBkg/sig_{channel}_expected/200_{mass}_mHlow0_mHhigh10000000000.0'])

# %%
import math
DSID_MASS_MAPPING = {510115:0.8, 510116:0.9, 510117:1.0, 510118:1.2, 510119:1.4, 510120:1.6, 510121:1.8, 510122:2.0, 510123:2.5, 510124:3.0}
for k in train_metrics_MCWts2.processed_weight_sums_per_dsid.keys():
    if math.isclose(train_metrics_MCWts.by_mass_signal_selection.processed_weight_sums_per_dsid[k], train_metrics_MCWts2.processed_weight_sums_per_dsid[k]):
        print(f'YES {k}: {train_metrics_MCWts.by_mass_signal_selection.processed_weight_sums_per_dsid[k]:10.5f} ~= {train_metrics_MCWts2.processed_weight_sums_per_dsid[k]:10.5f}')
    else:
        print(f'NO! {k}: {train_metrics_MCWts.by_mass_signal_selection.processed_weight_sums_per_dsid[k]:10.5f} ~= {train_metrics_MCWts2.processed_weight_sums_per_dsid[k]:10.5f}')

# %%
resbak = {k:res[k] for k in res.keys()}
res2bak = {k:res2[k] for k in res2.keys()}
res3bak = {k:res3[k] for k in res3.keys()}

# %%
for r in [res, res2, res3, resbak, res2bak, res3bak]:
    print(f'-----------------------------')
    print(f'-----------------------------')
    for k in r.keys():
        if 'acc' in k:
            print(f'{k}:   {r[k]}')

# %%
import math
for k in res2.keys():
    if k in res.keys():
        if math.isclose(res[k], res2[k]):
            print(f"GOOD! {k}: {res[k]} = {res2[k]}")
        else:
            print(f"BAD! {k}: {res[k]} != {res2[k]}")
    else:
        print(f"{k} not in res")
print('----------------------------------------------------')
print('----------------------------------------------------')
print('----------------------------------------------------')
for k in res2.keys():
    if '_mHlow0' in k:
        k3 = k.split('_mHlow0')[0]
        if k3 in res3.keys():
            if math.isclose(res2[k], res3[k3]):
                print(f"GOOD! {k}: {res3[k3]} = {res2[k]}")
            else:
                print(f"BAD! {k}: {res3[k3]} != {res2[k]}")
        else:
            print(f"{k3} not in res")

# %%
plt.figure(figsize=(9,6))
probs = F.softmax(preds,dim=-1)
nbins=40
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,i*3+j+1)
        batch_sel = (y.argmax(dim=-1)==i) & (mHs.cpu()>95e3) & (mHs.cpu()<140e3)
        batch_sel = (y.argmax(dim=-1)==i)
        plt.hist(probs[batch_sel,j], weights=MCWts[batch_sel], bins=np.arange(nbins+1)/nbins)
        # plt.yscale('log')
        plt.title(f"Truth {i}, pred {j}")
plt.tight_layout()
plt.show()
# %%
if 1:
    res = train_metrics_MCWts.compute_and_log(0, prefix="train_MC", step=global_step, log_level=3, save=False)


# %%
if 1:
    train_loader._reset_indices()
    batch = next(train_loader)
    x, y, w, types, dsids, _, mlv, MCWts, mHs = batch.values()
    y, w, mlv, MCWts, dsids = y.cpu(), w.cpu(), mlv.cpu(), MCWts.cpu(), dsids.cpu()
    preds = model(x, types).cpu()
    probs = F.softmax(preds, dim=-1)
    from utils import weighted_correlation
    true_bkg_mask = (y.argmax(dim=-1)) == 0 # Separate signal and background
    qqbb_rej_mask = probs[:,2] < probs[:,1]
    bins=100
    corrcoeff = weighted_correlation(mlv[true_bkg_mask & qqbb_rej_mask], 1-probs[true_bkg_mask & qqbb_rej_mask, 0], w[true_bkg_mask & qqbb_rej_mask])
    corrcoeff2 = weighted_correlation(mlv[true_bkg_mask & qqbb_rej_mask], 1-probs[true_bkg_mask & qqbb_rej_mask, 0], MCWts[true_bkg_mask & qqbb_rej_mask])
    plt.figure(figsize=(5,5))
    plt.hist2d(mlv[true_bkg_mask & qqbb_rej_mask].detach().cpu().numpy()*1e-6, 1-probs[true_bkg_mask & qqbb_rej_mask, 0].detach().cpu().numpy(), weights=w[true_bkg_mask & qqbb_rej_mask].detach().cpu().numpy(), norm=mpl.colors.LogNorm(), bins=bins)
    plt.title(corrcoeff)
    plt.colorbar()
    plt.show()
    plt.figure(figsize=(5,5))
    plt.hist2d(mlv[true_bkg_mask & qqbb_rej_mask].detach().cpu().numpy()*1e-6, 1-probs[true_bkg_mask & qqbb_rej_mask, 0].detach().cpu().numpy(), weights=MCWts[true_bkg_mask & qqbb_rej_mask].detach().cpu().numpy(), norm=mpl.colors.LogNorm(), bins=bins)
    plt.title(corrcoeff2)
    plt.colorbar()
    plt.show()

# %% Same with log x bins, and more than one batch
if 1: # Normally have this enableda as it calculates the data, just off if we want to quickly remake the plot for changing title etc.
    channel = 'lvbb'
    train_loader._reset_indices()
    xvals = []
    yvals = []
    wvals = []
    MCwvals = []
    for _ in range(100):
        batch = next(train_loader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        y, w, mqq, mlv, MCWts = y.cpu(), w.cpu(), mqq.cpu(), mlv.cpu(), MCWts.cpu()
        preds = model(x, types).detach().cpu() # Make sure to detach or it keeps it and fills up GPU memory!!
        probs = F.softmax(preds, dim=-1)
        true_bkg_mask = (y.argmax(dim=-1)) == 0 # Separate signal and background
        if channel == 'lvbb':
            other_sig_rej_mask = probs[:,2] < probs[:,1]
            xvals.append(mlv[true_bkg_mask & other_sig_rej_mask].cpu()*1e-6)
        elif channel == 'qqbb':
            other_sig_rej_mask = probs[:,1] < probs[:,2]
            xvals.append(mqq[true_bkg_mask & other_sig_rej_mask].cpu()*1e-6)
        else:
            raise NotImplementedError
        yvals.append(1-probs[true_bkg_mask & other_sig_rej_mask, 0])
        wvals.append(w[true_bkg_mask & other_sig_rej_mask])
        MCwvals.append(MCWts[true_bkg_mask & other_sig_rej_mask])
    xvals = torch.cat(xvals)
    yvals = torch.cat(yvals)
    wvals = torch.cat(wvals)
    MCwvals = torch.cat(MCwvals)
logx = True
logy = True
logz = False
nbins=100
bins=[nbins,nbins]
if logx:
    log_binsx = np.logspace(np.log10(min(xvals.cpu()).item()), np.log10(max(xvals.cpu()).item()), num=nbins)
    bins[0] = log_binsx
if logy:
    log_binsy = np.logspace(np.log10(min(yvals.cpu()).item()), np.log10(max(yvals.cpu()).item()), num=nbins)
    bins[1] = log_binsy
corrcoeff = weighted_correlation(xvals, yvals, wvals).item()
corrcoeff2 = weighted_correlation(xvals, yvals, MCwvals).item()
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
if logz:
    plt.hist2d(xvals.detach().cpu().numpy(), yvals.detach().cpu().numpy(), weights=wvals.detach().cpu().numpy(), norm=mpl.colors.LogNorm(), bins=bins)
else:
    plt.hist2d(xvals.detach().cpu().numpy(), yvals.detach().cpu().numpy(), weights=wvals.detach().cpu().numpy(), bins=bins)
# plt.hist2d(xvals.detach().cpu().numpy(), yvals.detach().cpu().numpy(), weights=wvals.detach().cpu().numpy(), bins=[log_binsx, log_binsy])
plt.title(f'Weighted by abs(MC)\nCorrelation: {corrcoeff:5f}')
if logx:
    plt.xscale('log')
if logy:
    plt.yscale('log')
plt.colorbar()
plt.xlabel(f'Reconstructed mWh ({channel})')
plt.ylabel(f'1-P(bkg)')
plt.subplot(1,2,2)
if logz:
    plt.hist2d(xvals.detach().cpu().numpy(), yvals.detach().cpu().numpy(), weights=MCwvals.detach().cpu().numpy(), norm=mpl.colors.LogNorm(), bins=bins)
else:
    plt.hist2d(xvals.detach().cpu().numpy(), yvals.detach().cpu().numpy(), weights=MCwvals.detach().cpu().numpy(), bins=bins)
# plt.hist2d(xvals.detach().cpu().numpy(), yvals.detach().cpu().numpy(), weights=wvals2.detach().cpu().numpy(), bins=[log_binsx, log_binsy])
if logx:
    plt.xscale('log')
if logy:
    plt.yscale('log')
plt.title(f'Weighted by MC\nCorrelation: {corrcoeff2:5f}')
plt.xlabel(f'Reconstructed mWh ({channel})')
plt.ylabel(f'1-P(bkg)')
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
plt.figure()
for i in [0, 0.01, 0.05, 0.1, 0.5, 0.9]:
    _,_,_=plt.hist(xvals[yvals>i], weights=MCwvals[yvals>i], bins=np.arange(101)/100*2, histtype='step', label=f'{i:.2f} ({sum(MCwvals[yvals>i]):10.2f})',density=True)
    # _,_,_=plt.hist(xvals[yvals>i], bins=np.arange(101)/100*2, histtype='step', label=f'{i:.2f}')#,density=True)
plt.legend()
plt.show()
plt.close('all')

# %%
_,_,_=plt.hist(yvals,bins=np.arange(nbins+1)/nbins, weights=wvals, density=True)
_,_,_=plt.hist(yvals,bins=np.arange(nbins+1)/nbins, weights=MCwvals, histtype='step', density=True)


# %%
def myhist2d(xvals, 
             yvals,
             wvals=None,
             logx = True,
             logy = True,
             logz = False,
             nbins=100,
             xlabel='',
             ylabel='',
):
    if (wvals is None):
        wvals = np.ones_like(xvals)
    bins=[nbins,nbins]
    if logx:
        log_binsx = np.logspace(np.log10(min(xvals).item()), np.log10(max(xvals).item()), num=nbins)
        bins[0] = log_binsx
    if logy:
        log_binsy = np.logspace(np.log10(min(yvals).item()), np.log10(max(yvals).item()), num=nbins)
        bins[1] = log_binsy
    corrcoeff = weighted_correlation(torch.from_numpy(xvals), torch.from_numpy(yvals), torch.from_numpy(wvals)).item()
    plt.figure(figsize=(4.5,4))
    if logz:
        plt.hist2d(xvals, yvals, weights=wvals, norm=mpl.colors.LogNorm(vmin=1e-4), bins=bins)
    else:
        plt.hist2d(xvals, yvals, weights=wvals, bins=bins)
    # plt.hist2d(xvals, yvals, weights=wvals, bins=[log_binsx, log_binsy])
    plt.title(f'Weighted by abs(MC)\nCorrelation: {corrcoeff:5f}')
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
# %%
