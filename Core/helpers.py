import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler,Subset
from torch.utils.data import DataLoader,random_split
from Core.torchhelpers.utils import collate_fn
import albumentations as A # Library for augmentations
from tqdm.notebook import tqdm
import time
import datetime
from collections import deque
import torchvision
import cv2
import wandb

# helper / utility functions

def im_to_numpy(tensor):
    '''
    Converts an image tensor to numpy. Mainly for plotting purposes
    '''
    if torch.is_tensor(tensor):
        return tensor.permute(1,2,0).detach().cpu().numpy()
    else:
        print('could not convert to np array. Check type of input')
        pass


def to_numpy(tensor):
    '''
    Convert torch tensor to numpy array and make sure it's detached and on cpu
    '''
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        print('could not convert to np array. Check type of input')
        pass


def to_torch(nparray):
    '''
    Converts a np.ndarray to a torch tensor
    '''
    if isinstance(nparray,np.ndarray):
        return torch.from_numpy(nparray)
    else:
        print('could not convert to torch tensor. Check type of input')
        return nparray


def split_data_train_test(DatasetClass_train,DatasetClass_val,validation_split=0.25,batch_size=1, data_amount=1, num_workers=0, shuffle_dataset=False, shuffle_dataset_seed=None, shuffle_epoch=False, shuffle_epoch_seed=0, pin_memory=False):
    '''
    Function that splits data from dataset class into a train and validation set

    validation_split: the percentage of the data that should be in the validation set
    '''
    assert len(DatasetClass_train) == len(DatasetClass_val), f'Size of DatasetClass for train ({len(DatasetClass_train)}) and val ({len(DatasetClass_val)}) is different, check input'
    # Creating data indices for training and validation splits:
    dataset_size = int(np.floor(len(DatasetClass_train)*data_amount))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(shuffle_dataset_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)
    train_subset = Subset(DatasetClass_train,train_indices)
    val_subset = Subset(DatasetClass_val,val_indices)
    
    if shuffle_epoch:
        torch.manual_seed(shuffle_epoch_seed)
    train_loader = DataLoader(train_subset,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
    validation_loader = DataLoader(val_subset,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
    # train_loader = DataLoader(DatasetClass_train,
    #                             batch_size=batch_size, 
    #                             sampler=train_sampler,
    #                             collate_fn=collate_fn,
    #                             num_workers=num_workers,
    #                             pin_memory=pin_memory)
    # validation_loader = DataLoader(DatasetClass_val,
    #                                 batch_size=batch_size,
    #                                 sampler=val_sampler,
    #                                 collate_fn=collate_fn,
    #                                 num_workers=num_workers,
    #                                 pin_memory=pin_memory)
    # print(f'###################\nTotal size of dataset: {dataset_size}\nTrain data --> Size: {len(train_loader.sampler)}, batch size: {train_loader.batch_size}\nValidation data --> Size: {len(validation_loader.sampler)}, batch size: {validation_loader.batch_size}')
    print(f'###################\nTotal size of dataset: {dataset_size}\nTrain data --> Size: {len(train_loader.dataset)}, batch size: {train_loader.batch_size}\nValidation data --> Size: {len(validation_loader.dataset)}, batch size: {validation_loader.batch_size}')

    return train_loader,validation_loader


def train_transform():
    '''
    Makes data augmentation transformations
    '''
    return A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.Blur(blur_limit=10, p=0.5),
        A.Rotate(limit=3,p=0.5)
        ],
        keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )


def test_num_workers(data,batch_size, data_amount=1, pin_memory = False):
    """
    Check the time of running dataloader with different values of num_workers.
    Make sure data_amount is between 0 and 1
    """
    _,data = random_split(data, [int(np.ceil((1-data_amount)*len(data))), int(np.floor(data_amount*len(data)))])
    print('pin_memory is', pin_memory)
    
    for num_workers in range(1, 20): 
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        start = time.time()
        # Simulate running epochs
        for _ in range(1):
            for i, data in tqdm(enumerate(dataloader),total=len(dataloader)):
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

def find_pixelerror(model,data_loader,device):
    """
    Find distance (error) between ground truth and predictions in pixels, for all corners together and individually
    """
    N_keypoints = 4
    model.eval()
    cpu_device = torch.device("cpu")
    # save pixelerrors as a deque list, which is faster at appending than a normal list
    pixelerrors_all, pixelerrors_TL, pixelerrors_TR, pixelerrors_BL, pixelerrors_BR = deque(), deque(), deque(), deque(), deque()
    print(f'Finding pixelerror for all predictions...')
    start_time = time.time()
    # Run through all images and get the pixel distance (error) between predictions and ground-truth
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        # outputs will be a list of dict of len == batch_size
        with torch.no_grad():
            outputs = model(images)
        # move outputs to cpu
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # extract the euclidean distance (in pixels) between every ground-truth and detection keypoint in the batch, and threshold them to return the matches
        distances = [np.linalg.norm(dt[:2]-gt[:2])
        for target, output in zip(targets, outputs)
        for obj_gt,obj_dt in zip(target['keypoints'],output['keypoints'])
        for gt, dt in zip(obj_gt,obj_dt)]
        # add pixelerrors for batch to list
        pixelerrors_all.extend(distances)
        # add the pixelerrors in each corner, using the fact that the corners show up in set intervals of 4 (for N_keypoints=4)
        pixelerrors_TL.extend(distances[0::N_keypoints])
        pixelerrors_TR.extend(distances[1::N_keypoints])
        pixelerrors_BL.extend(distances[2::N_keypoints])
        pixelerrors_BR.extend(distances[3::N_keypoints])

    pixelerrors = {
        "all":pixelerrors_all,
        "TL":pixelerrors_TL,
        "TR":pixelerrors_TR,
        "BL": pixelerrors_BL,
        "BR": pixelerrors_BR
        }
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total time: {total_time_str}')
    return pixelerrors

def eval_PCK(model, data_loader, device, thresholds=[50]):
    """
    Run PCK evaluation on model output for given thresholds
    thresholds: iterable of thresholds to calculate PCK for
    """
    N_keypoints = 4
    # calculate pixel error between ground-truth and predictions for all corners, TL, TR, BL and BR (total of 5 lists (deques))
    pixelerrors = find_pixelerror(model,data_loader,device)

    # count the number of correctly classified keypoints according to every threshold
    print(f'Running PCK evaluation on all thresholds...')
    start_time = time.time()
    PCK = {
        key:{threshold: np.count_nonzero([error < threshold for error in errors]) / len(errors) for threshold in thresholds}
        for key,errors in pixelerrors.items()
        }

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total time: {total_time_str} ({total_time/len(thresholds):.4f} s / threshold)')
    return PCK

def get_prediction_images(model,data_loader,device):
    """
    Takes model, data and list of image_ids and returns
    list of prediction images for the model of type
    data_loader: pytorch dataloader
    """
    model.eval()
    model.to(device)
    # FIXME
    images,targets = next(iter(data_loader))
    images = list(image.to(device) for image in images)
    image_ids = [target['image_id'] for target in targets]
    # images = list(dataset.__getitem__(idx)[0].to(device) for idx in image_ids)
    # outputs will be a list of dict of len == len(image_ids)
    with torch.no_grad():
        outputs = model(images)
    pred_images = {}
    for id,image,output in zip(image_ids,images,outputs):
        bboxes,keypoints = NMS(output,score_thresh=0.7,iou_thresh=0.3)
        pred_image = make_pred_image(image, bboxes, keypoints)
        pred_images[id] = pred_image
    return pred_images

def make_pred_image(image, bboxes, keypoints):
    """
    Adds bboxes and keypoints to the image
    returns np.array for the image
    """
    image = (im_to_numpy(image) * 255).astype(np.uint8)
    keypoints_classes_ids2names = {0: 'top-left', 1: 'top-right', 2: 'bot-left', 3: 'bot-right'}

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
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

    high_scores_idxs = np.where(scores > score_thresh)[0].tolist() # Indexes of boxes with scores > score_thresh
    post_nms_idxs = torchvision.ops.nms(output['boxes'][high_scores_idxs], output['scores'][high_scores_idxs], iou_thresh).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=iou_thresh)

    keypoints = []
    for kps in to_numpy(output['keypoints'][high_scores_idxs][post_nms_idxs]):
        keypoints.append([list(map(int, kp[:2])) for kp in kps])

    bboxes = []
    for bbox in to_numpy(output['boxes'][high_scores_idxs][post_nms_idxs]):
        bboxes.append(list(map(int, bbox.tolist())))
    return bboxes,keypoints

def make_PCK_plot_objects(PCK,thresholds):
    PCK_plot_objects = {}
    for pck_type,pck_values in PCK.items():
        data = [[x,y] for x,y in zip(thresholds,pck_values.values())]
        table = wandb.Table(data=data, columns = ["Threshold", "PCK"])
        # wandb.log({f"PCK_{pck_type}" : wandb.plot.line(table, "Threshold", "PCK", title=f"PCK_{pck_type} Curve")})
        PCK_plot_objects[f"PCK_{pck_type} Curve"] = wandb.plot.line(table, "Threshold", "PCK", title=f"PCK_{pck_type} Curve")
    return PCK_plot_objects
