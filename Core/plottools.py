import torch
import numpy as np
import matplotlib.pyplot as plt
from Core.helpers import im_to_numpy,to_numpy
import cv2
import torchvision

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


def visualize_results(model, images, device, plotdim, image_original=None, bboxes_original=None, keypoints_original=None):
    '''
    Plots the results, i.e. bounding boxes and keypoints for a given model batch.
    '''
    images = list(image.to(device) for image in images)

    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model(images)

    print(f'Scores for batch\n')
    for i,out in enumerate(output):
        print(f"scores item {i}: {to_numpy(out['scores'])}")

    images = [(im_to_numpy(image) * 255).astype(np.uint8) for image in images]
    scores_images = [to_numpy(outputsingle['scores']) for outputsingle in output]


    high_scores_idxs = [np.where(scores > 0.8)[0].tolist() for scores in scores_images] # Indexes of boxes with scores > 0.8
    post_nms_idxs = [torchvision.ops.nms(outputsingle['boxes'][high_scores_idxs[i]], outputsingle['scores'][high_scores_idxs[i]], 0.3).cpu().numpy()
                    for i, outputsingle in enumerate(output)] # Indexes of boxes left after applying NMS (iou_threshold=0.3)

    # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
    # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
    # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

    keypoints_images = [[[list(map(int, kp[:2])) for kp in kps]
                        for kps in to_numpy(outputsingle['keypoints'][high_scores_idxs[i]][post_nms_idxs[i]])]
                        for i, outputsingle in enumerate(output)]

    bboxes_images = [[list(map(int, bbox.tolist()))
                        for bbox in to_numpy(outputsingle['boxes'][high_scores_idxs[i]][post_nms_idxs[i]])]
                        for i, outputsingle in enumerate(output)]
    fontsize = 18
    keypoints_classes_ids2names = {0: 'top-left', 1: 'top-right', 2: 'bot-left', 3: 'bot-right'}
    

    # loop through all images and add bboxes and keypoints to them
    for i, (bboxes,keypoints) in enumerate(zip(bboxes_images,keypoints_images)):
        # loop through all bounding boxes in every image
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            images[i] = cv2.rectangle(images[i].copy(), start_point, end_point, (0,255,0), 2)
        # loop through all keypoints in every image
        for kps in keypoints:
            for idx, kp in enumerate(kps):
                images[i] = cv2.circle(images[i].copy(), tuple(kp), 5, (255,0,0), 10)
                images[i] = cv2.putText(images[i].copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        _,axes = plt.subplots(plotdim[0],plotdim[1], figsize=(40,40), layout="constrained")
        for i,ax in enumerate(axes.ravel()):
            ax.imshow(images[i])

    # else:
    #     for bbox in bboxes_original:
    #         start_point = (bbox[0], bbox[1])
    #         end_point = (bbox[2], bbox[3])
    #         image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)

    #     keypoints_classes_ids2names_original= {0: 'top-left', 1: 'top-right', 2: 'bot-left', 3: 'bot-right'}
    #     for kps in keypoints_original:
    #         for idx, kp in enumerate(kps):
    #             image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
    #             image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names_original[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    #     f, ax = plt.subplots(plotdim[0], plotdim[1], figsize=(40, 40))

    #     ax[0].imshow(image_original)
    #     ax[0].set_title('Original results', fontsize=fontsize)

    #     ax[1].imshow(image)
    #     ax[1].set_title('New results', fontsize=fontsize)

