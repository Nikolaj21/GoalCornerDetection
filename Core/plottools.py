import torch
import numpy as np
import matplotlib.pyplot as plt
from Core.helpers import im_to_numpy,to_numpy
import cv2
import torchvision
import os

def target_to_keypoints(target_dict):
    return target_dict['keypoints'][0,:,:2]

def batch_target_to_keypoints(batch_targets):

    batch_keypoints = []
    for target_dict in batch_targets:
        keypoints = target_dict['keypoints']
        for kps in keypoints:
            batch_keypoints.append(kps[:,:2])
    return batch_keypoints
    
# # plot images and corresponding keypoints for a subset of the batch
def plot_batch_keypoints(batch,figsize=(10,10), show_qualityfactor=False):
    # imgs and targets is a list of tensors and list of dicts, respectively
    imgs,targets = batch
    keypoints = batch_target_to_keypoints(targets)
    calibrationquality = [target_dict.get('calibration_quality') for target_dict in targets]
    # loop to plot all images and their keypoints
    if len(imgs) >= 9:
        fig,axes = plt.subplots(ncols=3,nrows=3,figsize=figsize)
        # choose random subset of images and keypoints
        choices = torch.randperm(len(imgs))[:9]
        imgs = [imgs[i] for i in choices]
        keypoints = [keypoints[i] for i in choices]
        calibrationquality = [calibrationquality[i] for i in choices if show_qualityfactor]
    else:
        fig,axes = plt.subplots(ncols=len(imgs),nrows=1,figsize=figsize)

    for i,ax in enumerate(np.ravel(axes)):
        ax.imshow(im_to_numpy(imgs[i]))
        for kp in keypoints[i]:
            ax.plot()
        ax.plot(*keypoints[i].T,'r.')
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if show_qualityfactor:
            ax.set_title(f'Quality: {calibrationquality[i].detach().cpu():.2f}')

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None, title_old='Original Results', title_new='New results',pred_score=None):

    fontsize = 18
    keypoints_classes_ids2names = {0: 'top-left', 1: 'top-right', 2: 'bot-left', 3: 'bot-right'}

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

        keypoints_classes_ids2names_original= {0: 'top-left', 1: 'top-right', 2: 'bot-left', 3: 'bot-right'}
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names_original[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 40))

        ax[0].imshow(image_original)
        ax[0].set_title(title_old, fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title(f'{title_new}, Prediction confidence: {pred_score:.4f}', fontsize=fontsize)

def visualize_results(model, images, targets, device, plotdim, score_thresh=0.7, iou_thresh=0.3, opaqueness=0.4):
    '''
    Plots the results, i.e. bounding boxes and keypoints for a given model and list of images. Also plots the ground-truth keypoints.
    Args:
        plotdim: A tuple of two ints containing the dimensions for plotting in the form (H,W). HxW should be <= number of images and targets!
        opaqueness: A factor between 0 and 1 that decides how unsee-through the keypoints are. Lower value means it is more see-through.
        score_thresh: A threshold for the confidence score on the output from the model, so we only plot boxes where the model has a high confidence.
        iou_thresh: A threshold for performing NMS on the resulting predictions after doing the score thresholding.
    '''

    pred_images,scores_images = get_prediction_images(model,images,targets,device,score_thresh=score_thresh,iou_thresh=iou_thresh,opaqueness=opaqueness,return_scores=True)

    print(f'Scores for images\n')
    for id,scores in zip(pred_images.keys(),scores_images):
        print(f"scores image {id}: {scores}")

    _,axes = plt.subplots(plotdim[0],plotdim[1], figsize=(40,40), layout="constrained")
    for i,(im_id,pred_image) in enumerate(pred_images.items()):
        axes.ravel()[i].imshow(pred_image)
        axes.ravel()[i].set_title(f'Image ID: {im_id}')

    return

def get_prediction_images(model,images,targets,device,score_thresh=0.7,iou_thresh=0.3,opaqueness=0.4,return_scores=False):
    """
    Takes model, data and list of image_ids and returns
    list of prediction images for the model of type
    data_loader: pytorch dataloader
    """
    model.eval()
    model.to(device)
    images = list(image.to(device) for image in images)
    # outputs will be a list of dict of len == len(images)
    with torch.no_grad():
        outputs = model(images)
    pred_images = {}
    for image,target,output in zip(images,targets,outputs):
        id = target['image_id']
        keypoints_gt = [[list(map(int, kp[:2])) for kp in kps] for kps in to_numpy(target['keypoints'])]
        bboxes,keypoints = NMS(output,score_thresh=score_thresh,iou_thresh=iou_thresh)
        pred_image = make_pred_image(image, bboxes, keypoints, keypoints_gt, opaqueness=opaqueness)
        pred_images[id] = pred_image

    if return_scores:
        scores_images = [to_numpy(out['scores']) for out in outputs]
        return pred_images,scores_images
    else:
        return pred_images
    
def make_pred_image(image, bboxes, keypoints, keypointsGT, opaqueness=0.4):
    """
    Adds bboxes and keypoints to a single image
    alpha is the opaqueness of the keypoints, lower value means more see-through (must be between 0 and 1)
    returns np.array for the image
    """
    image = (im_to_numpy(image) * 255).astype(np.uint8)
    keypoints_classes_ids2names = {0: 'TL', 1: 'TR', 2: 'BL', 3: 'BR'}

    # put all bounding boxes in the image
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    # put all ground-truth keypoints in the image
    for kps in keypointsGT:
        for idx, kp in enumerate(kps):
            overlay = image.copy()
            cv2.circle(overlay, tuple(kp), 5, (0,0,255), 10)
            image = cv2.addWeighted(overlay, opaqueness, image, 1 - opaqueness, 0)
    # put all predicted keypoints in the image
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            overlay = image.copy()
            cv2.circle(overlay, tuple(kp), 5, (255,0,0), 10)
            image = cv2.addWeighted(overlay, opaqueness, image, 1 - opaqueness, 0)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
    return image

def NMS(output,score_thresh=0.7,iou_thresh=0.3):
    """
    Perform non-maximum suppresion on a single output
    output: dict containing the prediction results for image.
        keys (minimum): keypoints, scores, boxes
    returns the updated bboxes and keypoints that passed nms
    """
    scores = to_numpy(output['scores'])

    # Indexes of boxes with scores > score_thresh
    high_scores_idxs = np.where(scores > score_thresh)[0].tolist()
    # if there are no predictions above the threshold, take the best one, i.e. index 0
    high_scores_idxs = high_scores_idxs if len(high_scores_idxs) >= 1 else [0]

    post_nms_idxs = torchvision.ops.nms(output['boxes'][high_scores_idxs], output['scores'][high_scores_idxs], iou_thresh).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=iou_thresh)

    keypoints = []
    for kps in to_numpy(output['keypoints'][high_scores_idxs][post_nms_idxs]):
        keypoints.append([list(map(int, kp[:2])) for kp in kps])

    bboxes = []
    for bbox in to_numpy(output['boxes'][high_scores_idxs][post_nms_idxs]):
        bboxes.append(list(map(int, bbox.tolist())))
    return bboxes,keypoints

def plot_loss(loss_dict, save_folder, epochs):
        '''
        plot losses from a loss dict
        '''
        ticks_step = np.floor(epochs/10)+1
        ############## train/val loss per epoch
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        x = np.arange(epochs)
        ax.plot(x,loss_dict['train']['all_mean'], label='training loss')
        ax.plot(x,loss_dict['val']['all_mean'], label='validation loss')
        ax.legend()
        ax.set_title('Loss curve')
        ax.set_xticks(np.arange(0,epochs+1,ticks_step))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        fig.savefig(os.path.join(save_folder,'loss_all_epochs.png'))

        ############## train/val keypoint loss per epoch
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        x = np.arange(epochs)
        ax.plot(x,loss_dict['train']['keypoint_mean'], label='training loss')
        ax.plot(x,loss_dict['val']['keypoint_mean'], label='validation loss')
        ax.legend()
        # fig.suptitle('Keypoint loss', fontweight ="bold")
        ax.set_title('Keypoint loss curve')
        ax.set_xticks(np.arange(0,epochs+1,ticks_step))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        fig.savefig(os.path.join(save_folder,'loss_keypoint_epochs.png'))

        ############## train loss per step
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        steps = len(loss_dict['train']['all'])
        x = np.arange(steps)
        ax.plot(x,loss_dict['train']['all'], label='training loss')
        ax.legend()
        ax.set_title('Loss curve')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        fig.savefig(os.path.join(save_folder,'loss_all_steps.png'))

