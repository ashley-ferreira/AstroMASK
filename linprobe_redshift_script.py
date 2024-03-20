# a very quick and dirty well to test if AttentiveProbing works better than 
# previous methods. conclusion: it does but still not great performance.

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import itertools
import h5py
import os
import glob                                                
import wget
import torch
import umap
import sys
dataloader_path = '/arc/home/ashley/SSL/git/dark3d/src/models/training_framework/'
sys.path.insert(0, dataloader_path)
import dataloaders
canfar_data_path = '/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/'

from functools import partial
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping
from timm.layers import AttentionPoolLatent

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score
from models_mae import mae_vit_large_patch16

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Times'],
    "font.size": 10})

#### FUNCTIONS USED ####
class AttentiveProbing(pl.LightningModule):
    def __init__(
        self, n_inputs: int, alpha=0.001, l1_ratio=0.9, learning_rate=0.01, max_epochs=500, 
        num_heads=12, mlp_ratio=4, x_mean=0, x_std=1, device=torch.device('cpu')):
        super().__init__()
        '''
        A "linear probing"-esque approach using an Attention block instead of a single linear layer.
        This is designed to be fit like scikit-learn's ElasticNet, utilizing L1 and L2 regularization in
        the same way.
        Fitting is accomplished with pytorch_lightning to allow for GPU training.
        '''

        # Elastic Params
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.loss_fn = torch.nn.MSELoss()

        # Normalization params
        self.x_mean = x_mean
        self.x_std = x_std

        # NN layers
        self.attn_pool = AttentionPoolLatent(n_inputs,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
        self.output_layer = torch.nn.Linear(n_inputs, 1)

        # Initialize weights
        self._init_weights()
        self.to(device)
        self.dev = device

        # Trianer with early stopping
        early_stop_callback = EarlyStopping(
           monitor='val_loss',
           min_delta=0.0001,  # Minimum change to qualify as an improvement
           patience=50,  # How many epochs to wait before stopping
           verbose=False,
           mode='min'
        )
        self.trainer = pl.Trainer(max_epochs=max_epochs,
                                  callbacks=[early_stop_callback])
        
        self.save_hyperparameters()

        self.train_log = []
        self.val_log = []

    def _init_weights(self):
        # Initialize AttentionPoolLatent layers
        for m in self.attn_pool.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
        # Initialize output layer
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.constant_(self.output_layer.bias, 0)

    def norm_inputs(self, x):
        return (x - self.x_mean) / self.x_std
        

    def forward(self, x):
        x = x.to(self.dev)
        x = self.norm_inputs(x)
        return self.output_layer(self.attn_pool(x))

    def predict(self, x):
        '''Predict from numpy array.'''
        return self.forward(x).cpu().detach().numpy().flatten()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Define the CosineAnnealingLR scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=0),
            'name': 'cosine_annealing_lr',
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

    def l1_reg(self):
        l1_norm = sum(p.abs().sum() for p in self.parameters() if p.requires_grad)
        return self.alpha * self.l1_ratio * l1_norm
    
    def l2_reg(self):
        l2_norm = sum(p.pow(2).sum() for p in self.parameters() if p.requires_grad)
        return self.alpha * (1 - self.l1_ratio) * l2_norm

    def calc_loss(self, batch):
        x, y = batch
        x, y = x.to(self.dev), y.to(self.dev)
        y_hat = self(x)
        mse_loss = self.loss_fn(y_hat, y.view_as(y_hat))
        l1_penalty = self.l1_reg() * self.alpha * self.l1_ratio
        l2_penalty = self.l2_reg() * self.alpha * (1 - self.l1_ratio) * 0.5
        total_loss = mse_loss + l1_penalty + l2_penalty

        return total_loss
    
    def training_step(self, batch, batch_idx):
        total_loss = self.calc_loss(batch)
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_log.append(total_loss.detach().cpu().numpy())
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = self.calc_loss(batch)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_log.append(total_loss.detach().cpu().numpy())
        return total_loss

    def fit(self,  dataloader_train, dataloader_test):
        self.trainer.fit(self, dataloader_train, dataloader_test)

def build_dataloader(X, y, batch_size=None):
    '''Convert numpy arrays X and y to a DataLoader.'''
    if batch_size is None:
        batch_size = X.shape[0]
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y[:, np.newaxis]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def plot_convergence(train_loss, val_loss, y_lim=(0,1)):
    val_loss = val_loss[:-1]
    # Number of epochs is the length of val_loss
    epochs = len(val_loss)
    # Number of batches per epoch
    batches_per_epoch = len(train_loss) // epochs
    
    # Calculating average training loss per epoch
    avg_train_loss = [sum(train_loss[i:i+batches_per_epoch]) / batches_per_epoch for i in range(0, len(train_loss), batches_per_epoch)]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(avg_train_loss, label='Train (avg per epoch)')
    ax.plot(val_loss, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    #ax.set_ylim(*y_lim)
    ax.set_ylim(0, val_loss[-1]*4)
    plt.legend()
    fig.savefig('covergence.png')

def plot_z_scatter(fig, ax, z_pred, z_true, cax=None, snr=None, snr_max=20,
                   y_lims=(0,1), cmap='ocean_r', fontsize=12):
    
    # Plot
    if snr is not None:
        scatter = ax.scatter(z_true, z_pred, c=snr, cmap=cmap, s=3, vmin=0, vmax=snr_max)
    else:
        scatter = ax.scatter(z_true, z_pred, c=cmap, s=3)
        
    # Axis params
    ax.set_xlabel('Spectroscopic Redshift', size=fontsize)
    ax.set_ylabel('Predicted Redshift', size=fontsize)
    ax.plot([0,2],[0,2], linewidth=1, c='black', linestyle='--')
    ax.set_xlim(*y_lims)
    ax.set_ylim(*y_lims)
        
    # Colorbar
    if snr is not None:
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label('[S/N]', size=fontsize)

    ax.grid(alpha=0.2)

def plot_predictions(y_train, y_pred_train, y_test, y_pred_test):

    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    # Create a figure
    fig = plt.figure(figsize=(8,4))
    
    # Define a GridSpec layout
    gs = gridspec.GridSpec(1, 2, figure=fig)
        
    # Plot one-to-one
    ax1 = fig.add_subplot(gs[0,0])
    plot_z_scatter(fig, ax1, y_pred_train, y_train, 
                   y_lims=(0.0,4.1), cmap='#ff7f00', fontsize=12)
                   #y_lims=(0.0,7.1), cmap='#ff7f00', fontsize=12)
    ax1.set_title('Training Set', size=15)
    ax1.annotate(f'$R^2$ = {r2_train:.3f}',
                (0.1,0.9), size=12, xycoords='axes fraction', 
                bbox=bbox_props)

    ax2 = fig.add_subplot(gs[0,1])
    plot_z_scatter(fig, ax2, y_pred_test, y_test, 
                   y_lims=(0.0,4.1), cmap='#4daf4a', fontsize=12)
                   #y_lims=(0.0,7.1), cmap='#4daf4a', fontsize=12)
    ax2.set_title('Validation Set', size=15)
    ax2.annotate(f'$R^2$ = {r2_test:.3f}',
                (0.1,0.9), size=12, xycoords='axes fraction', 
                bbox=bbox_props)
    ax2.set_ylabel('')
    ax2.set_yticklabels([])
    plt.tight_layout()
    fig.savefig('predictions.png')

#### PREPARE MAE PRETRAINED MODEL ####
mae_model = mae_vit_large_patch16(norm_method='min_max')
checkpoint_path = "/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/mae/output_dir/to_save/checkpoint-60.pth"
checkpoint = torch.load(checkpoint_path)
mae_model.load_state_dict(checkpoint['model'])

device = torch.device('cuda')
mae_model.to(device)
mae_model.eval()

#### PREPARED LABELS ####
# read in redshift files
matches = pd.read_csv('spencer_redshifts.csv').sample(frac = 1) # random shuffle is applied to table

# filter out the extremely high and extremely low redshift entries
matches = matches[matches['zspec']>0.1] #0.1]
matches = matches[matches['zspec']<2.5] #6]
match_zspec = np.array(matches['zspec'])

#### PREPARED INPUT DATASET ####
data_set = list(range(int(1024*1.8)))
cutouts = []
redshifts = []
index_count = 0
for i, row in matches.iterrows():
    print(index_count) 
    if index_count <= data_set[-1]:
        redshifts.append(row['zspec'])
        file_num = int(row['file_num']) 
        filename = f'cutout.stacks.ugriz.lsb.200x200.{file_num}.10000.h5'
        f = h5py.File(canfar_data_path + filename,'r')
        index = int(row['index'])
        image = f['images'][index]
        cutouts.append(dataloaders.utils.crop_center(image, cropx=64, cropy=64))
    else:
        break
    index_count +=1 

#### PREDICT REPRESENTATION ####
cutouts = torch.from_numpy(np.array(cutouts)).float().to(device, non_blocking=True)
with torch.no_grad():
    with torch.cuda.amp.autocast():
        latent_representation, _, _ = mae_model.forward_encoder(cutouts, mask_ratio=0.0)
        
del cutouts # now we don't need the cutouts anymore 

#### PREPARE INPUT REPRESENTATIONS ####
val_size = 256
train_set = data_set[:len(data_set)-val_size]
valid_set = data_set[len(data_set)-val_size:len(data_set)-1]

match_zspec = torch.from_numpy(match_zspec).to(device).float()
latent_representation = latent_representation.to(device).float()

print(latent_representation.shape) # (num_samples, numb_patches, embed_dim)

dataloader_train = build_dataloader(latent_representation[train_set], match_zspec[train_set], batch_size=64)
dataloader_test = build_dataloader(latent_representation[valid_set], match_zspec[valid_set])

#### PREPARE AND TRAIN LIN PROBE MODEL ####
elasticnet_model = AttentiveProbing(
    n_inputs=latent_representation[train_set].shape[2],
    alpha=0.01,
    l1_ratio=0.5,
    learning_rate=0.0005,
    max_epochs=300,
    num_heads=2, 
    mlp_ratio=2,
    device=device,
)
elasticnet_model.fit(dataloader_train, dataloader_test)

#### PREDICT REDSHIFTS AND PLOT RESULTS ####
device = torch.device('cpu') 
latent_representation = latent_representation.to(device)
match_zspec = match_zspec.to(device)
elasticnet_model.dev = device

y_pred_valid = elasticnet_model.predict(latent_representation[valid_set])
y_pred_train = elasticnet_model.predict(latent_representation[train_set])

y_train = match_zspec[train_set]
y_valid = match_zspec[valid_set]

plot_predictions(y_train, y_pred_train, y_valid, y_pred_valid)
plot_convergence(elasticnet_model.train_log, elasticnet_model.val_log, y_lim=(0,0.1))