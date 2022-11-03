import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader,random_split
from Core.torchhelpers.utils import collate_fn
import albumentations as A # Library for augmentations
from tqdm.notebook import tqdm
import time

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

def split_data_train_test(DatasetClass,validation_split=0.25,batch_size=1, shuffle_dataset=False, shuffle_seed=None,data_amount=1, num_workers=0, pin_memory=False):
    '''
    Function that splits data from dataset class into a train and validation set

    validation_split: the percentage of the data that should be in the validation set
    '''

    # Creating data indices for training and validation splits:
    dataset_size = int(np.floor(len(DatasetClass)*data_amount))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(shuffle_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(DatasetClass,
                                batch_size=batch_size, 
                                sampler=train_sampler,
                                collate_fn=collate_fn,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    validation_loader = DataLoader(DatasetClass,
                                    batch_size=batch_size,
                                    sampler=val_sampler,
                                    collate_fn=collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
    print(f'###################\nTotal size of dataset: {dataset_size}\nTrain data --> Size: {len(train_loader.sampler)}, batch size: {train_loader.batch_size}\nValidation data --> Size: {len(validation_loader.sampler)}, batch size: {validation_loader.batch_size}')
    return train_loader,validation_loader

def train_transform():
    '''
    Makes data augmentation transformations
    '''
    return A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.Blur(blur_limit=10, p=0.5),
        A.Rotate(limit=5,p=0.5)
        ],
        keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )

def test_num_workers(data,batch_size, data_amount=1, pin_memory = True):
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
