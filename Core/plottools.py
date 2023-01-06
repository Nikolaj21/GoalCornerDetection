import torch
import numpy as np
import matplotlib.pyplot as plt
from Core.helpers import im_to_numpy,to_numpy
import cv2
import torchvision
import os
from collections import defaultdict

def target_to_keypoints(target_dict):
    return target_dict['keypoints'][0,:,:2]

def batch_target_to_keypoints(batch_targets):
    batch_keypoints = []
    for target_dict in batch_targets:
        keypoints = target_dict['keypoints']
        for kps in keypoints:
            batch_keypoints.append(kps[:,:2])
    return batch_keypoints
    
# plot images and corresponding keypoints for a subset of the batch
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

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None, title_old='Results 1', title_new='Results 2',pred_score=None):

    fontsize = 18
    # keypoints_classes_ids2names = {0: 'TL', 1: 'TR', 2: 'BL', 3: 'BR'}
    image = make_GT_image(image, bboxes, keypoints, opaqueness=0.4)


    # for bbox in bboxes:
    #     start_point = (bbox[0], bbox[1])
    #     end_point = (bbox[2], bbox[3])
    #     image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    # for kps in keypoints:
    #     for idx, kp in enumerate(kps):
    #         image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
    #         image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(20,10))
        plt.imshow(image)

    else:
        image_original = make_GT_image(image_original, bboxes_original, keypoints_original, opaqueness=0.4)

        # for bbox in bboxes_original:
        #     start_point = (bbox[0], bbox[1])
        #     end_point = (bbox[2], bbox[3])
        #     image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)

        # keypoints_classes_ids2names_original= {0: 'top-left', 1: 'top-right', 2: 'bot-left', 3: 'bot-right'}
        # for kps in keypoints_original:
        #     for idx, kp in enumerate(kps):
        #         image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
        #         image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names_original[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 40))

        ax[0].imshow(image_original)
        ax[0].set_title(title_old, fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title(f'{title_new}, Prediction confidence: {pred_score:.4f}', fontsize=fontsize)

def visualize_results(model, images, targets, device, num_objects, figsize=(40,30), show_axis=False, score_thresh=0.7, iou_thresh=0.3, opaqueness=0.5, return_scores=True):
    '''
    Plots the results, i.e. bounding boxes and keypoints for a given model and list of images. Also plots the ground-truth keypoints.
    Args:
        plotdim: A tuple of two ints containing the dimensions for plotting in the form (H,W). HxW should be <= number of images and targets!
        opaqueness: A factor between 0 and 1 that decides how unsee-through the keypoints are. Lower value means it is more see-through.
        score_thresh: A threshold for the confidence score on the output from the model, so we only plot boxes where the model has a high confidence.
        iou_thresh: A threshold for performing NMS on the resulting predictions after doing the score thresholding.
    '''

    pred_images,scores_images = get_prediction_images(model,images,targets,device,num_objects=num_objects,score_thresh=score_thresh,iou_thresh=iou_thresh,opaqueness=opaqueness, return_scores=return_scores)
    if return_scores:
        print(f'Scores for images\n')
        for id,scores in zip(pred_images.keys(),scores_images):
            print(f"scores image {id}: {scores}")

    # n = len(images)
    # # Makes the plot dimensions in such a way that the images fit into a grid, that is as square as possible
    # plotdim = round(np.sqrt(n)), int(np.ceil(np.sqrt(n))) 

    # _,axes = plt.subplots(plotdim[0],plotdim[1], figsize=(40,40), layout="constrained")
    # if plotdim == (1,1):
    #     for im_id,pred_image in pred_images.items():
    #         axes.imshow(pred_image)
    #         axes.set_title(f'Image ID: {im_id}')
    # else:
    #     for i,(im_id,pred_image) in enumerate(pred_images.items()):
    #         axes.ravel()[i].imshow(pred_image)
    #         axes.ravel()[i].set_title(f'Image ID: {im_id}')
    
    im_ids, pred_ims = zip(*pred_images.items())
    visualize_images(images=pred_ims,
                    figtitle=f'Prediction Results',
                    subplottitles=[f'Image ID: {im_id}' for im_id in im_ids],
                    figsize=figsize,
                    show_axis=show_axis)
    
    return
    

def get_prediction_images(model, images, targets, device, num_objects, opaqueness=0.5, return_scores=False, score_thresh=0.7,iou_thresh=0.3):
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
        id = target['image_id'].item()
        # scores = to_numpy(output['scores'])
        keypoints_gt = kps_to_cv2(target['keypoints'])
        keypoints_ua = kps_to_cv2(target['keypoints_ua'])
        #bboxes,keypoints = NMS(output,score_thresh=score_thresh,iou_thresh=iou_thresh, num_objects=num_objects)
        bboxes,keypoints,labels,scores = filter_preds(output,num_objects=num_objects)
        pred_image = make_pred_image(image, bboxes=bboxes, keypoints=keypoints, keypointsGT=keypoints_gt, keypointsUA=keypoints_ua, opaqueness=opaqueness, scores=scores, labels=labels, return_scores=return_scores)
        pred_images[id] = pred_image

    if return_scores:
        scores_images = [to_numpy(out['scores']) for out in outputs]
        return pred_images,scores_images
    else:
        return pred_images
    
def make_pred_image(image, bboxes=None, keypoints=None, keypointsGT=None, keypointsUA=None, opaqueness=0.5, scores=None, labels=None, return_scores=False):
    """
    Adds bboxes and keypoints to a single image
    alpha is the opaqueness of the keypoints, lower value means more see-through (must be between 0 and 1)
    returns np.array for the image
    """
    # keypoints_classes_ids2names = {1: 'TL', 2: 'TR', 3: 'BL', 4: 'BR'}
    label_to_color =  {1: (255,0,0), 2: (0,255,0), 3: (0,150,255), 4: (255,255,0)}
    # convert data to correct type for using with cv2
    image,bboxes,keypoints = data_to_cv2(image,bboxes,keypoints)
    if bboxes:
        # put all bounding boxes in the image
        for i,bbox in enumerate(bboxes):
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
            # add prediction confidence for each box
            image = cv2.putText(image.copy(), f"L:{labels[i]} C:{round(scores[i],3)}", start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, label_to_color[labels[i]], 2, cv2.LINE_AA)
    if keypointsGT:
        # put all ground-truth keypoints in the image
        for kpsGT in keypointsGT:
            for kp_gt in kpsGT:
                overlay = image.copy()
                overlay = cv2.circle(img=overlay, center=tuple(kp_gt), radius=10, color=(0,0,255), thickness=cv2.FILLED)#, thickness=10)
                image = cv2.addWeighted(overlay, opaqueness, image, 1 - opaqueness, 0)
                # image = cv2.circle(img=image.copy(),center=tuple(kp_gt), radius=10, color=(0,0,255), thickness=1)#cv2.FILLED)

    if keypointsUA:
        # put all user annotated (ua) keypoints in the image
        for kpsUA in keypointsUA:
            for kp_ua in kpsUA:
                overlay = image.copy()
                overlay = cv2.circle(img=overlay, center=tuple(kp_ua), radius=10, color=(255,140,0), thickness=cv2.FILLED)#, thickness=5)
                image = cv2.addWeighted(overlay, opaqueness, image, 1 - opaqueness, 0)
                # add circle around GT which has radius set to radial distance between gt and ua
                # radius = int(np.linalg.norm(np.subtract(kp_ua[:2],kp_gt[:2])))
                # image = cv2.circle(img=image.copy(), center=tuple(kp_gt), radius=radius, color=(255,140,0), thickness=1)
    if keypoints:
        # put all predicted keypoints in the image
        for kps in keypoints:
            for kp in kps:
                overlay = image.copy()
                overlay = cv2.circle(img=overlay, center=tuple(kp), radius=10, color=(255,0,0), thickness=cv2.FILLED)#thickness=5)
                image = cv2.addWeighted(overlay, opaqueness, image, 1 - opaqueness, 0)
                # FIXME temporarily removed while testing 4corner method
                # image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
    return image

def NMS(output,score_thresh=0.7,iou_thresh=0.3, num_objects=4):
    """
    Perform non-maximum suppresion on a single output
    Args:
        output: dict containing the prediction results for image.
            keys (minimum): keypoints, scores, boxes
    Returns: the updated bboxes and keypoints that passed nms
    """
    all_scores = to_numpy(output['scores'])
    # Indexes of boxes with scores > score_thresh
    high_scores_idxs = np.where(all_scores > score_thresh)[0]
    # perform non-maximum suppresion on the predictions with high score, so there are no overlapping predicitions
    post_nms_idxs = torchvision.ops.nms(output['boxes'][high_scores_idxs], output['scores'][high_scores_idxs], iou_thresh).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=iou_thresh)

    keypoints = output['keypoints'][post_nms_idxs]
    bboxes = output['boxes'][post_nms_idxs]
    labels = output['labels'][post_nms_idxs]
    scores = output['scores'][post_nms_idxs]

    return bboxes,keypoints,labels,scores

def filter_preds(output, num_objects):
    """
    Filter predictions from a single output, so only the object with the highest confidence in each class remains.
    num_objects: the number of object classes that there should be in every image.
    output: dict containing the prediction results for image.
        keys (minimum): keypoints, scores, boxes
    returns the updated bboxes and keypoints that passed the filtering
    """
    bboxes, keypoints, labels, scores = [],[],[],[]
    label_to_dts = defaultdict(list)
    # make a dictionary for the dts in every output that save the label, keypoints and scores, for later sorting
    for label,kps,box,score in zip(output['labels'],output['keypoints'],output['boxes'],output['scores']):
        label_to_dts[label.item()].append((box,kps,score.item()))
    # compare the gt and dt of every object with the same label, taking only the highest scored one
    all_labels = range(1,num_objects+1)
    for label in all_labels:
        # get the obj_dt for this label (obj_dt may not exist)
        obj_dt = label_to_dts.get(label)
        # if there are any predictions with this label
        if not obj_dt == None:
            # take the set of keypoints with the highest score
            obj_dt = sorted(obj_dt, key=lambda tup_box_kp_and_score: tup_box_kp_and_score[2], reverse=True)[0]
            bboxes.append(obj_dt[0])
            keypoints.append(obj_dt[1])
            labels.append(label)
            scores.append(obj_dt[2])


    return bboxes,keypoints,labels,scores

def data_to_cv2(image,bboxes,keypoints):
    image = (im_to_numpy(image) * 255).astype(np.uint8)
    bboxes = [list(map(int, bbox.tolist())) for bbox in bboxes]
    keypoints = kps_to_cv2(keypoints)#[[list(map(int, kp[:2])) for kp in kps] for kps in keypoints]
    return image,bboxes,keypoints

def kps_to_cv2(keypoints):
    keypoints = [[list(map(int, kp[:2])) for kp in kps] for kps in keypoints]
    return keypoints

def make_GT_image(image, bboxes, keypointsGT, opaqueness=0.4):
    """
    Adds ground-truth bboxes and keypoints to a single image
    alpha is the opaqueness of the keypoints, lower value means more see-through (must be between 0 and 1)
    returns np.array for the image
    """
    # keypoints_classes_ids2names = {0: 'TL', 1: 'TR', 2: 'BL', 3: 'BR'}
    # convert data to correct type for using with cv2
    image,bboxes,keypointsGT = data_to_cv2(image,bboxes,keypointsGT)
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
            # image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
    return image

def crop_image(image,crop_regions):
    """
    Crop multiple areas in an image.
    Args:
        Image: np.array of an image with shape H,W,C
        crop_regions: an array_like with shape (C,4), where C is the number of crop regions to make in the image and each crop region consists of (x_min,y_min,x_max,y_max) representing the area to be cropped in the image.
    Returns:
        A list of np.array's that each represent a cropped part of the image
    """
    image_crops = [image[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2], :]
                   for crop_region in crop_regions]
    return image_crops
 
def crop_images(images,crop_regions_images):
    """
    Crop multiple areas of mulitple images. There can be a variable number of crop_regions for every image
    Args:
        Images: A list of np.array of images each with shape H,W,C. Sizes of each image can vary.
        crop_regions: A list of array_like with shape (C,4), where C is the number of crop regions to make in the given image (variable) and each crop region consists of (x_min,y_min,x_max,y_max) representing the area to be cropped in the image.
    Returns:
        A list of lists of np.array's that each represent a cropped part of a given image
    """
    assert len(images) == len(crop_regions_images), f'length of images ({len(images)}) and length of crop_regions_images ({len(crop_regions_images)}) must be the same!'

    images_crops = []
    for image,crop_regions in zip(images,crop_regions_images):
        image_crops = crop_image(image,crop_regions=crop_regions)
        images_crops.append(image_crops)
    return images_crops

def visualize_images(images,figtitle=None,subplottitles=None,figsize=(20,20), show_axis=False):
    n = len(images)
    # Makes the plot dimensions in such a way that the images fit into a grid, that is as square as possible
    plotdim = round(np.sqrt(n)), int(np.ceil(np.sqrt(n)))
    if subplottitles is None:
        subplottitles = [None for _ in range(n)]
    fig,axes = plt.subplots(plotdim[0],plotdim[1], figsize=figsize, sharex=True, gridspec_kw=dict(hspace=0.05,wspace=0.05)) #layout="constrained",
    fig.suptitle(figtitle)
    if plotdim == (1,1):
        for image in images:
            axes.imshow(image)
            axes.axis(show_axis)
    else:
        for i,image in enumerate(images):
            axes.ravel()[i].imshow(image)
            axes.ravel()[i].axis(show_axis)
            axes.ravel()[i].set_title(subplottitles[i])

def visualize_cropped_results(model, images, targets, device, num_objects, figsize=(10,10), opaqueness=0.5):
    model.eval()
    model.to(device)
    images = list(image.to(device) for image in images)
    # outputs will be a list of dict of len == len(images)
    with torch.no_grad():
        outputs = model(images)
    images_crops = []
    im_ids = []
    labels_images = []
    scores_images = []
    # number of pixels to crop the pred_boxes by, so you can't see the drawn bounding boxes
    ce = 2
    for image,target,output in zip(images,targets,outputs):
        keypoints_gt = kps_to_cv2(target['keypoints'])
        keypoints_ua = kps_to_cv2(target['keypoints_ua'])
        im_id = target['image_id'].item()
    
        bboxes,keypoints,labels,scores = filter_preds(output,num_objects=num_objects)
        pred_image = make_pred_image(image, bboxes=bboxes, keypoints=keypoints, keypointsGT=keypoints_gt,
                                     keypointsUA=keypoints_ua, opaqueness=opaqueness, scores=scores,
                                     labels=labels, return_scores=False)
        # create crop_regions, making it a bit smaller so the boxes drawn on the image don't show
        crop_regions = np.stack([to_numpy(box,as_int=True)+np.array([ce,ce,-ce,-ce]) for box in bboxes])
        image_crops = crop_image(image=pred_image,crop_regions=crop_regions)
        
        images_crops.append(image_crops)
        im_ids.append(im_id)
        labels_images.append(labels)
        scores_images.append(scores)
        
    for idx in range(len(images)):
        visualize_images(images=images_crops[idx],
                         figtitle=f'Image ID: {im_ids[idx]}',
                         subplottitles=[f'label: {label}, score: {np.round(conf,3)}'
                                        for label,conf in zip(labels_images[idx],scores_images[idx])],
                         figsize=figsize)

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


