import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers import to_numpy
import cv2

def target_to_keypoints(target_dict):
    return target_dict['keypoints'][0,:,:2]

def batch_target_to_keypoints(batch_targets):

    batch_keypoints = []
    for target_dict in batch_targets:
        keypoints = target_dict['keypoints']
        batch_keypoints.append(keypoints[:,:2])
    return batch_keypoints
    
# # plot images and corresponding keypoints for a subset of the batch
def plot_batch_keypoints(batch,figsize=(10,10)):
    # imgs and targets is a list of tensors and list of dicts, respectively
    imgs,targets = batch
    keypoints = batch_target_to_keypoints(targets)
    calibrationquality = [target_dict['calibration_quality'] for target_dict in targets]
    # loop to plot all images and their keypoints
    if len(imgs) >= 9:
        fig,axes = plt.subplots(ncols=3,nrows=3,figsize=figsize)
        # choose random subset of images and keypoints
        choices = torch.randperm(len(imgs))[:9]
        imgs = [imgs[i] for i in choices]
        keypoints = [keypoints[i] for i in choices]
        calibrationquality = [calibrationquality[i] for i in choices]
        # calibrationquality = calibrationquality[choices]
    else:
        fig,axes = plt.subplots(ncols=len(imgs),nrows=1,figsize=figsize)

    for i,ax in enumerate(np.ravel(axes)):
        ax.imshow(to_numpy(imgs[i]))
        ax.plot(*keypoints[i].T,'r.')
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax.set_title(f'Quality: {calibrationquality[i].detach().cpu():.2f}')



keypoints_classes_ids2names = {0: 'top-left', 1: 'bot-left', 2: 'top-right', 3: 'bot-right'}

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(20,10))
        plt.imshow(image)

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)