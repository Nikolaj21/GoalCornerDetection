import glob
import cv2
import torch
from torch.utils.data import Dataset
import json
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

# data loader for the data before it was re-annotated
class GoalCalibrationDatasetOLD(Dataset):
    def __init__(self,datapath, transforms=None, filter_data=False):
        # list of paths to the images
        self.img_list = sorted(glob.glob(datapath + '/*/*.jpg'))
        # list of paths to the annotation data
        self.calibration_list = sorted(glob.glob(datapath + '/*/*.txt'))
        # list of paths to the re-annoated data
        self.reannotate_list = sorted(glob.glob(datapath + '/FootballGoalCorners/AnnotationFiles/*.json'))
        self.transforms = transforms

        if filter_data:
            # self.img_list_filtered, self.annotation_list_filtered = filter_data(self.img_list,self.annotation_list)
            ### remove image paths from list if they aren't of category 'free kick'
            img_list_filtered = []
            annotation_list_filtered = []
            for idx, path in enumerate(self.reannotate_list):
                annotation_json = json.load(open(path,'r',encoding='latin'))
                # if annotation is from a valid freekick image
                if annotation_json['Annotations']['0'][-1]['Attributes']['calibration_type'] == 'freekick':
                    # add only annotation and image paths of type 'freekick'
                    annotation_list_filtered.append(self.calibration_list[idx])
                    img_list_filtered.append(self.img_list[idx])

            print(f'All images: {len(self.img_list)}')
            # print(f'All annotations: {len(self.annotation_list)}')
            print(f'Filtered images: {len(img_list_filtered)}')
            # print(f'Filtered annotations: {len(self.annotation_list_filtered)}')
            ############################
            self.img_list = img_list_filtered
            self.calibration_list = annotation_list_filtered

    def  __len__(self):
        return len(self.calibration_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        calibration_path = self.calibration_list[idx]
        # shape of most imgs is (3120, 4208, 3), but some examples are different, e.g. (1920, 2560, 3)
        img = cv2.imread(img_path)
        # cv2 loads image in b,g,r. Return it to r,g,b
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load keypoint annotations (and other relevant metrics)
        calibration_json = json.load(open(calibration_path,'r',encoding='latin'))
        keypoints = calibration_json['GoalCalibrationPoints']
        calibration_quality = calibration_json['CalibrationQuality']

        # make bounding boxes
        ##########################
        expand_x = 0.05 # expand boxes by this percentage
        expand_y = 0.02
        box_topleft = np.min(keypoints,axis=0)
        # print(f'box top left: {box_topleft}')
        box_topleft_expand = box_topleft - np.array([img.shape[1]*expand_y,img.shape[0]*expand_x])
        box_topleft_keepinIm = np.array([val if val > 0 else 1 for val in box_topleft_expand])

        box_botright = np.max(keypoints,axis=0)
        box_botright_expand = box_botright + np.array([img.shape[1]*expand_y,img.shape[0]*expand_x])
        shapeperm = np.transpose(img,axes=(1,0,2)).shape
        box_botright_keepinIm = np.array([val if val <= shapeperm[i] else shapeperm[i]-1 for i,val in enumerate(box_botright_expand)])

        bboxes = np.concatenate((box_topleft_keepinIm,box_botright_keepinIm))
        bboxes = torch.tensor(bboxes)[None,:]
        ##########################

        # change format of keypoints from [x,y] -> [x,y,visibility] where visibility=0 means the keypoint is not visible
        for kpt in keypoints:
            kpt.append(1)

        # convert image to tensor
        img_tensor = F.to_tensor(img)
        # convert keypoints to tensor
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

        


        # bboxes = torch.cat((keypoints_tensor[0][:2],keypoints_tensor[3][:2]))[None,:]#.unsqueeze(0), # needs to be added for using keypoint r-cnn. I made temporary boxes now that just use the keypoints top-left and bot-right. dims Needs to be (N,4), hence the unsqueeze
        target_dict = {
            'boxes': bboxes,
            'labels': torch.tensor([1 for _ in bboxes], dtype=torch.int64), # class label hard-coded to 1 always, as we are only interested in the football goals
            'image_id': torch.tensor([idx]), # save id of image for reference
            'keypoints': keypoints_tensor[None,:],
            'calibration_quality': torch.tensor(calibration_quality),
            'area': (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0]),
            'iscrowd': torch.zeros(len(bboxes), dtype=torch.int64)
        }
        if self.transforms:
            img_tensor,target_dict = self.transforms(img_tensor,target_dict)


        return img_tensor,target_dict





## data loader for the newly annotated dataset
class GoalCalibrationDataset(Dataset):
    def __init__(self,datapath, transforms=None):

        self.img_list = sorted(glob.glob(datapath + '/*/*.jpg'))
        self.annotation_list = sorted(glob.glob(datapath + '/FootballGoalCorners/AnnotationFiles/*.json'))

        ### remove image paths from list if they aren't of category 'free kick'
        self.img_list_filtered = []
        self.annotation_list_filtered = []
        for path in self.annotation_list:
            annotation_json = json.load(open(path,'r',encoding='latin'))
            if annotation_json['Annotations']['0'][-1]['Attributes']['calibration_type'] == 'freekick':
                # add only annotation paths of type 'freekick'
                self.annotation_list_filtered.append(path)
                # make correct path for images
                splitpath = path.split('/')
                basepath = '/'.join(splitpath[:-3])
                img_name = splitpath[-1].split('.')[0]
                temp_img_path = os.path.join(basepath, img_name, img_name + '.jpg')
                # add image paths that are of type 'freekick'
                self.img_list_filtered.append(temp_img_path)
        print(f'All images: {len(self.img_list)}')
        # print(f'All annotations: {len(self.annotation_list)}')
        print(f'Filtered images: {len(self.img_list_filtered)}')
        # print(f'Filtered annotations: {len(self.annotation_list_filtered)}')
        ############################

        self.transforms = transforms

    def  __len__(self):
        return len(self.annotation_list_filtered)

    def __getitem__(self, idx):
        img_path = self.img_list_filtered[idx]
        annotation_path = self.annotation_list_filtered[idx]
        # shape of most imgs is (3120, 4208, 3), but some examples are different, e.g. (1920, 2560, 3)
        img = cv2.imread(img_path)
        # cv2 loads image in b,g,r. Return it to r,g,b
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load keypoint annotations (and other relevant metrics)
        annotation_json = json.load(open(annotation_path,'r',encoding='latin'))
        keypoints = [[corner['CenterU'],corner['CenterV']] for corner in annotation_json['Annotations']['0'][:4]]
        box_json = annotation_json['Annotations']['0'][4]
        Cu,Cv,w,h = box_json['CenterU'], box_json['CenterV'], box_json['Width'], box_json['Height']
        bboxes = torch.tensor([[(2*Cu-w)/2, (2*Cv-h)/2, w+(2*Cu-w)/2, h+(2*Cv-h)/2]])
        radii = torch.tensor([corner['Radius'] for corner in annotation_json['Annotations']['0'][:4]])

        # change format of keypoints from [x,y] -> [x,y,visibility] where visibility=0 means the keypoint is not visible
        for kpt in keypoints:
            kpt.append(1)

        if self.transforms: # not fixed yet per 24-10-2022
            pass

        # convert image to tensor
        img_tensor = F.to_tensor(img)
        # convert keypoints to tensor
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

        target_dict = {
            'boxes': bboxes,
            'labels': torch.tensor([1 for _ in bboxes], dtype=torch.int64), # class label hard-coded to 1 always, as we are only interested in the football goals
            'image_id': torch.tensor([idx]), # save id of image for reference
            'keypoints': keypoints_tensor[None,:],
            'radii': radii,
            'area': (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0]),
            'iscrowd': torch.zeros(len(bboxes), dtype=torch.int64)
        }
        if self.transforms:
            img_tensor,target_dict = self.transforms(img_tensor,target_dict)


        return img_tensor,target_dict
    
    
############################################################
############################################################
## data loader with data augmentation
class GoalCalibrationDatasetAUG(Dataset):
    def __init__(self,datapath, transforms=None, istrain=False):

        self.img_list = sorted(glob.glob(datapath + '/*/*.jpg'))
        self.annotation_list = sorted(glob.glob(datapath + '/FootballGoalCorners/AnnotationFiles/*.json'))
        ### remove image paths from list if they aren't of category 'free kick'
        self.img_list_filtered, self.annotation_list_filtered = filter_data(self.img_list,self.annotation_list)
        self.istrain = istrain
        self.transforms = transforms
        

    def  __len__(self):
        return len(self.annotation_list_filtered)

    def __getitem__(self, idx):
        img_path = self.img_list_filtered[idx]
        annotation_path = self.annotation_list_filtered[idx]

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) 
        
        # load keypoint annotations (and other relevant metrics)
        annotation_json = json.load(open(annotation_path,'r',encoding='latin'))
        keypoints_original = [[corner['CenterU'],corner['CenterV']] for corner in annotation_json['Annotations']['0'][:4]]
        box_json = annotation_json['Annotations']['0'][4]
        Cu,Cv,w,h = box_json['CenterU'], box_json['CenterV'], box_json['Width'], box_json['Height']
        bboxes_original = [[(2*Cu-w)/2, (2*Cv-h)/2, w+(2*Cu-w)/2, h+(2*Cv-h)/2]]
        radii = torch.tensor([corner['Radius'] for corner in annotation_json['Annotations']['0'][:4]])

        # All objects are goal
        bboxes_labels_original = ['goal' for _ in bboxes_original]
        if self.istrain:
            if self.transforms:

                # Apply augmentations
                transformed = self.transforms(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original)
                img = transformed['image']
                bboxes = transformed['bboxes']
                keypoints = np.array(transformed['keypoints']).tolist()
        
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original 

        # change format of keypoints from [x,y] -> [x,y,visibility] where visibility=0 means the keypoint is not visible
        for kpt in keypoints:
            kpt.append(1)

        # convert image to tensor
        img_tensor = F.to_tensor(img)
        # convert keypoints to tensor
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        # convert bboxes to tensor
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

        target_dict = {
            'boxes': bboxes_tensor,
            'labels': torch.tensor([1 for _ in bboxes_tensor], dtype=torch.int64), # class label hard-coded to 1 always, as we are only interested in the football goals
            'image_id': torch.tensor((idx)), # save id of image for reference
            'keypoints': keypoints_tensor[None,:],
            'radii': radii,
            'area': (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) * (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]),
            'iscrowd': torch.zeros(len(bboxes), dtype=torch.int64)
        }

        return img_tensor,target_dict

def filter_data(img_list,annotation_list):
    ### remove image paths from list if they aren't of category 'free kick'
    img_list_filtered = []
    annotation_list_filtered = []
    for path in annotation_list:
        annotation_json = json.load(open(path,'r',encoding='latin'))
        if annotation_json['Annotations']['0'][-1]['Attributes']['calibration_type'] == 'freekick':
            # add only annotation paths of type 'freekick'
            annotation_list_filtered.append(path)
            # make correct path for images
            splitpath = path.split('/')
            basepath = '/'.join(splitpath[:-3])
            img_name = splitpath[-1].split('.')[0]
            temp_img_path = os.path.join(basepath, img_name, img_name + '.jpg')
            # add image paths that are of type 'freekick'
            img_list_filtered.append(temp_img_path)
    print(f'All images: {len(img_list)}')
    print(f'Filtered images: {len(img_list_filtered)}')
    return img_list_filtered, annotation_list_filtered