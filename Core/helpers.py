import torch
import numpy as np
from torch.utils.data import DataLoader,random_split, SubsetRandomSampler, Subset
from Core.torchhelpers.utils import collate_fn
from tqdm import tqdm
import time
import datetime
# from collections import deque
import wandb
from collections import defaultdict

# helper / utility functions

def im_to_numpy(tensor):
    '''
    Converts an image tensor to numpy. Mainly for plotting purposes
    '''
    if torch.is_tensor(tensor):
        return tensor.permute(1,2,0).detach().cpu().numpy()
    else:
        print('could not convert to np array. Check type of input')
        return tensor

def to_numpy(tensor, as_int=False):
    '''
    Convert torch tensor to numpy array and make sure it's detached and on cpu
    '''
    if torch.is_tensor(tensor):
        if as_int:
            return np.array(tensor.detach().cpu(),int)
        else:
            return tensor.detach().cpu().numpy()
    else:
        print('could not convert to np array. Check type of input')
        return tensor

def to_torch(nparray):
    '''
    Converts a np.ndarray to a torch tensor
    '''
    if isinstance(nparray,np.ndarray):
        return torch.from_numpy(nparray)
    else:
        print('could not convert to torch tensor. Check type of input')
        return nparray

def split_data_train_test(DatasetClass_train, DatasetClass_val, validation_split=0.25, batch_size=1, data_amount=1, num_workers=0, shuffle_dataset=False, shuffle_dataset_seed=-1, shuffle_epoch=False, shuffle_epoch_seed=-1, pin_memory=False, collate_fn=collate_fn):
    '''
    Function that splits data from dataset class into a train and validation set

    validation_split: the percentage of the data that should be in the validation set
    shuffle_dataset_seed: seed for shuffling the dataset into training and validation set. if -1, no seed is set.
    shuffle_epoch_seed: seed for shuffling the data in the dataloader at every epoch. if -1, no seed is set.
    '''
    assert len(DatasetClass_train) == len(DatasetClass_val), f'Size of DatasetClass for train ({len(DatasetClass_train)}) and val ({len(DatasetClass_val)}) is different, check input'
    # Creating data indices for training and validation splits:
    dataset_size = int(np.floor(len(DatasetClass_train)*data_amount))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # if true and seed is set, shuffle the dataset randomly into a train and val set
    if shuffle_dataset:
        if shuffle_dataset_seed != -1:
            np.random.seed(shuffle_dataset_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Use pytorch subset function to sample subset of data indices for each dataloader
    train_subset = Subset(DatasetClass_train,train_indices)
    val_subset = Subset(DatasetClass_val,val_indices)
    
    # if true and seed is set, shuffle data in dataloaders at every epoch
    if shuffle_epoch and shuffle_epoch_seed != -1:
            torch.manual_seed(shuffle_epoch_seed)

    train_loader = DataLoader(train_subset,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    shuffle=shuffle_epoch)
    validation_loader = DataLoader(val_subset,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    shuffle=False)
    print(f'###################\nTotal size of dataset: {dataset_size}\nTrain data --> Size: {len(train_loader.dataset)}, batch size: {train_loader.batch_size}\nValidation data --> Size: {len(validation_loader.dataset)}, batch size: {validation_loader.batch_size}')

    return train_loader,validation_loader

def test_num_workers(data, batch_size, data_amount=1, pin_memory = False):
    """
    Check the time of running dataloader with different values of num_workers.
    Make sure data_amount is between 0 and 1
    """
    _,data = random_split(data, [int(np.ceil((1-data_amount)*len(data))), int(np.floor(data_amount*len(data)))])
    print('pin_memory is', pin_memory)
    
    for num_workers in range(0, 20):
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
        start = time.time()
        # Simulate running epochs
        for _ in range(1):
            for i, data in tqdm(enumerate(dataloader),total=len(dataloader)):
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

def find_pixelerror(model, data_loader, device, num_objects):
    """
    Find distance (error) between ground truth and predictions in pixels, for all corners together and individually
    Parameters:
        model: a neural network made using pytorch
        data_loader: a pytorch dataloader object
        device: device on which to run data through model. Either torch.device('cuda') or torch.device('cpu')
        num_objects: the number of objects in every gt image. (only supports 1 or 4)
    Returns:
        pixelerrors: a dict of all the pixel errors for every point in different categories.
    """
    assert num_objects==1 or num_objects==4,f"num_objects should either be 1 or 4, but {num_objects} was given!"
    model.eval()
    model.to(device)
    cpu_device = torch.device("cpu")
    # save pixelerrors as a deque list, which is faster at appending than a normal list
    pixelerrors_all = []
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
        # extract the euclidean distance error (in pixels) between every ground-truth and detection keypoint in the batch. Also return image_ids and labels for every distance measure for reference
        for target, output in zip(targets, outputs):
            label_to_gts = {}
            label_to_dts = defaultdict(list)
            # make a dictionary for the gts and dts in every target and output that save the label, keypoints and scores, for later sorting
            for label,kp in zip(target['labels'],target['keypoints']):
                label_to_gts[label.item()] = kp
            for label,kp,score in zip(output['labels'],output['keypoints'],output['scores']):
                label_to_dts[label.item()].append((kp,score.item()))
            # compare the gt and dt of every object with the same label, taking only the highest scored one
            for label in range(1,num_objects+1):
                # get the obj_gt and obj_dt for this label (obj_dt may not exist)
                obj_gt = label_to_gts[label]
                obj_dt = label_to_dts.get(label)
                # if there are any predictions with this label
                if not obj_dt == None:
                    # take the set of keypoints with the highest score
                    obj_dt = sorted(obj_dt, key=lambda tup_kp_and_score: tup_kp_and_score[1], reverse=True)[0][0]
                    # find the distance between every gt and gt for this label, and add to list of distances, along with the image_id
                    for gt,dt in zip(obj_gt,obj_dt):
                        pixelerrors_all.append((target['image_id'].item(), label, np.linalg.norm(dt[:2]-gt[:2])))
    if num_objects == 1:
        num_keypoints = 4
        pixelerrors_TL = pixelerrors_all[0::num_keypoints]
        pixelerrors_TR = pixelerrors_all[1::num_keypoints]
        pixelerrors_BL = pixelerrors_all[2::num_keypoints]
        pixelerrors_BR = pixelerrors_all[3::num_keypoints]
    elif num_objects == 4:
        TL_label,TR_label,BL_label,BR_label = 1,2,3,4
        pixelerrors_TL = [pixelerrors_all[i] for i,(_,label,_) in enumerate(pixelerrors_all) if label==TL_label]
        pixelerrors_TR = [pixelerrors_all[i] for i,(_,label,_) in enumerate(pixelerrors_all) if label==TR_label]
        pixelerrors_BL = [pixelerrors_all[i] for i,(_,label,_) in enumerate(pixelerrors_all) if label==BL_label]
        pixelerrors_BR = [pixelerrors_all[i] for i,(_,label,_) in enumerate(pixelerrors_all) if label==BR_label]

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

def eval_PCK(model, data_loader, device, thresholds, num_objects):
    """
    Run PCK evaluation on model output for given thresholds
    thresholds: iterable of thresholds to calculate PCK for
    num_object: number of object classes there are in each ground truth image
    """
    # calculate pixel error between ground-truth and predictions for all corners, TL, TR, BL and BR (total of 5 lists (deques))
    pixelerrors = find_pixelerror(model,data_loader,device,num_objects=num_objects)

    # count the number of correctly classified keypoints according to every threshold
    print(f'Running PCK evaluation on all thresholds...')
    start_time = time.time()
    # HACK hardcoded the number of total keypoints for every category in keypoint r-cnn with 4 keypoints to detect in every image
    N_ims = len(data_loader.dataset.indices)
    total_keypoints = {'all':N_ims*4, 'TL':N_ims, 'TR':N_ims, 'BL':N_ims, 'BR':N_ims}
    PCK = {
        key:{threshold: np.count_nonzero([error < threshold for _,_,error in errors]) / total_keypoints[key] for threshold in thresholds}
        for key,errors in pixelerrors.items()
        }

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total time: {total_time_str} ({total_time/len(thresholds):.4f} s / threshold)')
    return PCK,pixelerrors
