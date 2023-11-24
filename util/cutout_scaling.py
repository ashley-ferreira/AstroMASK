import numpy as np
import torch
from astropy.stats import mad_std
import torch
import datetime
import sys
sys.path.append('/arc/home/ashley/SSL/git/dark3d/src/models/training_framework/dataloaders/')
import dataloaders
import timm.optim.optim_factory as optim_factory
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler

# move these into args if using long term
data_path = '/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/'
norm_method = None 
norm_args = None
date_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

overall_means = []
overall_stds = []

def entire_dataset_args(loader):
    '''
    returns std and mean per channel of entire dataset 

    for current training set this is 

    channel 1:
    std = 
    mean = 

    channel 2:
    std = 
    mean = 

    channel 3:
    std = 
    mean = 

    channel 4:
    std = 
    mean = 

    channel 5:
    std = 
    mean = 

    thats why overall_means and overall_std are set like that
    '''
    #batch_means = []
    #batch_stds = []
    for data in loader: 
        mean = data.mean(axis=0) 
        meansq = (data**2).mean(axis=0)
    
    std = torch.sqrt(meansq - mean**2, axis=0) # or average stds of batches?
    print(mean.shape)
    print(std.shape)
    return mean, std

'''
code used to call the above
'''
n_cutouts_train = 5*10000
    
dataset_indices = list(range(n_cutouts_train))
np.random.shuffle(dataset_indices)
frac = 0.3 # decrease when we have more data
val_split_index = int(np.floor(frac * n_cutouts_train))
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx) 

kwargs_train = {
        'n_cutouts': 5*10000,
        'bands': ['u', 'g', 'r', 'i', 'z'],
        'cutout_size': 64,
        'cutouts_per_file': 10000,
        'h5_directory': f'{data_path}/valid2/'
    }
    
train_dataset = dataloaders.SpencerHDF5ReaderDataset(**kwargs_train)
train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1024, sampler=train_sampler, num_workers=4)
entire_dataset_args(train_loader)

def min_max(cutout):
    #normalize so that min pixel value 
    #across all channels is 0 and max is 1
    min_overall = torch.min(cutout)
    max_overall = torch.max(cutout)
    normed_cutout = (cutout - min_overall) / (max_overall - min_overall)
    kwargs = {'min_overall': min_overall, 'max_overall': max_overall}
    return normed_cutout, kwargs

def inv_min_max(cutout, args):
    #normalize so that min pixel value 
    #across all channels is 0 and max is 1
    min_overall = args['min_overall']
    max_overall = args['max_overall']
    original_cutout = cutout * (max_overall - min_overall) + min_overall
    return original_cutout

def standardize(cutout, args):
    '''
    args['mean'] and args['std'] should be over whole dataset to preseve color
    '''
    normed_cutout = (cutout - args['mean'])/args['std']
    return normed_cutout

def inv_standardize(cutout, args):
    '''
    args['mean'] and args['std'] should be over whole dataset to preseve color
    '''
    original_cutout = cutout*args['std'] + args['mean']
    return original_cutout

def paper_scale(cutouts, args):
    '''
    args['mean'] and args['std'] should be over whole dataset to preseve color

    currently not using find_mad_std() as it only does norm per channel

    taken from paper: https://arxiv.org/abs/2308.07962
    specifically github: https://github.com/LSSTISSC/Tidalsaurus/tree/main
    '''
    def find_mad_std():
        """
        NOT USING RIGHT NOW - not tensor converted either

        Find the median absolute deviation of each channel to use for normalisationg and augmentations

        Parameters
        ----------
        dataset: Tensorflow Dataset
            Dataset to be sampled from
        bands: list of str
            List of channels of the image e.g. ['g', 'r', 'i']

        Returns
        -------
        scaling: List of floats of length len(bands)
            Scaling factor by band/channel
        """
        # add median absolute deviation for each band to an array
        scaling = []
        for i in range(5):
            sigma = mad_std(cutouts[..., i].flatten())
            scaling.append(sigma)
        
        return scaling

    def paper_normalisation(img, scale):
        """
        Normalises images
        example: element in dataset
        scale: array of median absolute deviation for each band

        Parameters
        ----------
        example: Dataset item
            Element in dataset
        scale: List of floats of length len(bands)
            Scaling factor by band/channel

        Returns
        -------
        img: numpy array
            normalised image
        """
        img = torch.asinh(img / torch.Tensor(scale) / 3.)
        # We return the normalised images
        return img

    scale = args['std']

    for img in cutouts:
        img = paper_normalisation(img, scale)

    return cutouts

def inv_paper_scale(cutout, args):
    def inv_paper_normalisation(img, scale):
        img = torch.sinh(img) *3. * torch.Tensor(scale)
        return img

    for img in cutout:
        img = inv_paper_normalisation(img, scale=args['std'])

    return cutout

def normalize(cutout, method=None):

    if method == None:
        return cutout, None

    elif method == 'min_max':
        return min_max(cutout)
    
    elif method == 'standard':
        return standardize(cutout, args={'mean': overall_means, 'std': overall_stds}) # over entire dataset
    
    elif method == 'paper_norm':
        return paper_scale(cutout, args={'std': overall_stds})
        

def inv_normalize(cutout, method=None, args=None):

    if method == None:
        return cutout

    elif method == 'min_max':
        return inv_min_max(cutout, args)
    
    elif method == 'standard':
        return inv_standardize(cutout, args={'mean': overall_means, 'std': overall_stds}) # over entire dataset
    
    elif method == 'paper_norm':
        return inv_paper_scale(cutout, args={'std': overall_stds})
        