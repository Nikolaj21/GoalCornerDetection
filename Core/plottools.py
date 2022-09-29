import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers import to_numpy

def target_to_keypoints(target_dict):
    return target_dict['keypoints'][:,:2]
def batch_target_to_keypoints(batch_target_dict):
    batch_keypoints = []
    for item in batch_target_dict['keypoints']:
        batch_keypoints.append(item[:,:2])
    return batch_keypoints
    
# # plot images and corresponding keypoints for a subset of the batch
def plot_batch_keypoints(batch,figsize=(10,10)):
    imgs,targets = batch
    keypoints = batch_target_to_keypoints(targets)

    # loop to plot all images and their keypoints
    if len(imgs) >= 9:
        fig,axes = plt.subplots(ncols=3,nrows=3,figsize=figsize)
        # choose random subset of images and keypoints
        choices = torch.randperm(len(imgs))[:9]
        imgs = imgs[choices]
        keypoints = [keypoints[i] for i in choices]
    else:
        fig,axes = plt.subplots(ncols=len(imgs),nrows=1,figsize=figsize)

    for i,ax in enumerate(np.ravel(axes)):
        ax.imshow(to_numpy(imgs[i]))
        ax.plot(*keypoints[i].T,'r.')
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])