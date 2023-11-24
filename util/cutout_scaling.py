import numpy as np
import torch
from astropy.visualization import interval, ZScaleInterval

def unit_sphere_norm(pointcloud): # check this works correctly if needed
    norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
    norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
    return norm_pointcloud


def basic_norm(pointcloud): 
    '''
    normalized 0 to 1 with respect to whole thing
    hopefully not too big outliers? checked in notebook 4
    '''
    norm_pointcloud = pointcloud.copy() # i know this isn't ideal with how to work with arrays

    norm_pointcloud[:,0] = normalize_eq2(pointcloud[:,0], x_bounds[0], x_bounds[1]) 
    norm_pointcloud[:,1] = normalize_eq2(pointcloud[:,1], y_bounds[0], y_bounds[1])
    norm_pointcloud[:,2] = normalize_eq2(pointcloud[:,2], z_bounds[0], z_bounds[1])

    return norm_pointcloud


def relative_unit_norm(pointcloud):
    max_sp_x, min_sp_x = np.max(pointcloud[:,0]), np.min(pointcloud[:,0]) 
    max_sp_y, min_sp_y = np.max(pointcloud[:,1]), np.min(pointcloud[:,1])
    max_sp_z, min_sp_z = np.max(pointcloud[:,2]), np.min(pointcloud[:,2])

    #return basic_norm(pointcloud, x_bounds=(min_sp_x, max_sp_x), y_bounds=(min_sp_y, max_sp_y), z_bounds=(min_sp_z, max_sp_z))
    return basic_norm(pointcloud, z_bounds=(min_sp_z, max_sp_z))


def basic_standard(pointcloud):
    '''
    should work well even if we have noise
    '''
    norm_pointcloud = pointcloud.copy() 

    norm_pointcloud[:,0] = standardize_eq(pointcloud[:,0], pointcloud[:,0])
    norm_pointcloud[:,1] = standardize_eq(pointcloud[:,1], pointcloud[:,1])
    norm_pointcloud[:,2] = standardize_eq(pointcloud[:,2], pointcloud[:,2])

    return norm_pointcloud
'''
def min_max(cutout):
    #normalize so that min pixel value 
    #across all channels is 0 and max is 1
    min_overall = np.min(cutout)
    max_overall = np.max(cutout)
    normed_cutout = (cutout - min_overall) / (max_overall - min_overall)
    kwargs = {'min_overall': min_overall, 'max_overall': max_overall}
    
    return normed_cutout, kwargs #--> how to store this to go back later?? do it batch-wise? or not part of normal transforms. can then define it in there later.
'''
def min_max(cutout):
    #normalize so that min pixel value 
    #across all channels is 0 and max is 1
    min_overall = torch.min(cutout)
    max_overall = torch.max(cutout)
    normed_cutout = (cutout - min_overall) / (max_overall - min_overall)
    kwargs = {'min_overall': min_overall, 'max_overall': max_overall}
    return normed_cutout, kwargs

def inv_min_max(cutout, kwargs):
    #normalize so that min pixel value 
    #across all channels is 0 and max is 1
    min_overall = kwargs['min_overall']
    max_overall = kwargs['max_overall']
    original_cutout = cutout * (max_overall - min_overall) + min_overall
    return original_cutout

def zscale(cutout):
    (z1, z2) = zscale.get_limits(cutout)
    normer = interval.ManualInterval(z1,z2)
    return nomer(cutout)

def inv_zscale(cutout):
    #(z1, z2) = zscale.get_limits(cutout)
    #original_cutout = cutout * (z2 - z1) + z1
    #return original_cutout
    pass

class Normalize(object):
    ''' 
    NOT READY TO USE YET
    '''
    def __call__(self, cutout):
        
        if method == 'None':
            return cutout

        elif method == 'min_max':
            return min_max(cutout)
        
        elif method == 'zscale':
            return zscale(cutout)
        
        elif method == 'standard':
            return standard(cutout)
        
        elif method == 'paper_norm':
            return paper_norm(cutout)
        
        
class inv_Normalize(object):
    ''' 
    NOT READY TO USE YET
    '''
    def __call__(self, cutout):
        
        if method == 'None':
            return cutout

        elif method == 'min_max':
            return inv_min_max(cutout)
        
        elif method == 'zscale':
            return inv_zscale(cutout)
        
        elif method == 'standard':
            return inv_standard(cutout)
        
        elif method == 'paper_norm':
            return inv_paper_norm(cutout)
        
            
# make inverse stuff too