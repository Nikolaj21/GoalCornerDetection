import glob
import cv2
import torch
from torch.utils.data import Dataset
import json
from torchvision.transforms import functional as F
import torchvision.transforms as T
import numpy as np
import os
from PIL import Image
import math
from pathlib import Path

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



# ## data loader for the newly annotated dataset
# class GoalCalibrationDataset(Dataset):
#     def __init__(self,datapath, transforms=None):

#         self.img_list = sorted(glob.glob(datapath + '/*/*.jpg'))
#         self.annotation_list = sorted(glob.glob(datapath + '/FootballGoalCorners/AnnotationFiles/*.json'))

#         ### remove image paths from list if they aren't of category 'free kick'
#         self.img_list_filtered = []
#         self.annotation_list_filtered = []
#         for path in self.annotation_list:
#             annotation_json = json.load(open(path,'r',encoding='latin'))
#             if annotation_json['Annotations']['0'][-1]['Attributes']['calibration_type'] == 'freekick':
#                 # add only annotation paths of type 'freekick'
#                 self.annotation_list_filtered.append(path)
#                 # make correct path for images
#                 splitpath = path.split('/')
#                 basepath = '/'.join(splitpath[:-3])
#                 img_name = splitpath[-1].split('.')[0]
#                 temp_img_path = os.path.join(basepath, img_name, img_name + '.jpg')
#                 # add image paths that are of type 'freekick'
#                 self.img_list_filtered.append(temp_img_path)
#         print(f'All images: {len(self.img_list)}')
#         # print(f'All annotations: {len(self.annotation_list)}')
#         print(f'Filtered images: {len(self.img_list_filtered)}')
#         # print(f'Filtered annotations: {len(self.annotation_list_filtered)}')
#         ############################

#         self.transforms = transforms

#     def  __len__(self):
#         return len(self.annotation_list_filtered)

#     def __getitem__(self, idx):
#         img_path = self.img_list_filtered[idx]
#         annotation_path = self.annotation_list_filtered[idx]
#         # shape of most imgs is (3120, 4208, 3), but some examples are different, e.g. (1920, 2560, 3)
#         img = cv2.imread(img_path)
#         # cv2 loads image in b,g,r. Return it to r,g,b
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # load keypoint annotations (and other relevant metrics)
#         annotation_json = json.load(open(annotation_path,'r',encoding='latin'))
#         keypoints = [[corner['CenterU'],corner['CenterV']] for corner in annotation_json['Annotations']['0'][:4]]
#         box_json = annotation_json['Annotations']['0'][4]
#         Cu,Cv,w,h = box_json['CenterU'], box_json['CenterV'], box_json['Width'], box_json['Height']
#         bboxes = torch.tensor([[(2*Cu-w)/2, (2*Cv-h)/2, w+(2*Cu-w)/2, h+(2*Cv-h)/2]])
#         radii = torch.tensor([corner['Radius'] for corner in annotation_json['Annotations']['0'][:4]])

#         # change format of keypoints from [x,y] -> [x,y,visibility] where visibility=0 means the keypoint is not visible
#         for kpt in keypoints:
#             kpt.append(1)

#         if self.transforms: # not fixed yet per 24-10-2022
#             pass

#         # convert image to tensor
#         img_tensor = F.to_tensor(img)
#         # convert keypoints to tensor
#         keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

#         target_dict = {
#             'boxes': bboxes,
#             'labels': torch.tensor([1 for _ in bboxes], dtype=torch.int64), # class label hard-coded to 1 always, as we are only interested in the football goals
#             'image_id': torch.tensor([idx]), # save id of image for reference
#             'keypoints': keypoints_tensor[None,:],
#             'radii': radii,
#             'area': (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0]),
#             'iscrowd': torch.zeros(len(bboxes), dtype=torch.int64)
#         }
#         if self.transforms:
#             img_tensor,target_dict = self.transforms(img_tensor,target_dict)


#         return img_tensor,target_dict
    
    
############################################################
############################################################
## data loader with data augmentation
class GoalCalibrationDatasetAUG(Dataset):
    def __init__(self,datapath, transforms=None, istrain=False):

        self.img_list = sorted(glob.glob(str(Path(datapath + '/*/*.jpg'))))
        self.annotation_list = sorted(glob.glob(str(Path(datapath + '/FootballGoalCorners/AnnotationFiles/*.json'))))
        ### remove image paths from list if they aren't of category 'free kick'
        self.img_list_filtered, self.annotation_list_filtered,self.old_idxs = filter_data(self.img_list,self.annotation_list)
        self.istrain = istrain
        self.transforms = transforms
        

    def  __len__(self):
        return len(self.annotation_list_filtered)

    def __getitem__(self, idx):
        img_path = self.img_list_filtered[idx]
        annotation_path = self.annotation_list_filtered[idx]

        # cv2 loads image in shape H,W,C
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

        # convert image to tensor of shape C,H,W
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
            'iscrowd': torch.zeros(len(bboxes), dtype=torch.int64),
            'old_image_id': torch.tensor((self.old_idxs[idx]))
        }

        return img_tensor,target_dict

def filter_data(img_list,annotation_list):
    ### remove image paths from list if they aren't of category 'free kick'
    img_list_filtered = []
    annotation_list_filtered = []
    old_idxs = []
    for idx,path in enumerate(annotation_list):
        annotation_json = json.load(open(path,'r',encoding='latin'))
        if annotation_json['Annotations']['0'][-1]['Attributes']['calibration_type'] == 'freekick':
            # save the old image id for reference
            old_idxs.append(idx)
            # add only annotation paths of type 'freekick'
            annotation_list_filtered.append(path)
            # # make correct path for images
            # splitpath = path.split('/')
            # basepath = '/'.join(splitpath[:-3])
            # img_name = splitpath[-1].split('.')[0]
            # temp_img_path = os.path.join(basepath, img_name, img_name + '.jpg')
            # # add image paths that are of type 'freekick'
            # img_list_filtered.append(temp_img_path)
            img_list_filtered.append(img_list[idx])
    print(f'All images: {len(img_list)}')
    print(f'Filtered images: {len(img_list_filtered)}')
    return img_list_filtered, annotation_list_filtered,old_idxs


############################################################
############################################################
## data loader for making a bounding box around every corner
class GoalCalibrationDataset4boxes(Dataset):
    def __init__(self,datapath, transforms=None, istrain=False):

        self.img_list = sorted(glob.glob(str(Path(datapath + '/*/*.jpg'))))
        self.annotation_list = sorted(glob.glob(str(Path(datapath + '/FootballGoalCorners/AnnotationFiles/*.json'))))
        ### remove image paths from list if they aren't of category 'free kick'
        self.img_list_filtered, self.annotation_list_filtered,self.old_idxs = filter_data(self.img_list,self.annotation_list)
        self.istrain = istrain
        self.transforms = transforms
        

    def  __len__(self):
        return len(self.annotation_list_filtered)

    def __getitem__(self, idx):
        img_path = self.img_list_filtered[idx]
        annotation_path = self.annotation_list_filtered[idx]

        # cv2 loads image in shape H,W,C
        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) 
        
        # load keypoint annotations (and other relevant metrics)
        annotation_json = json.load(open(annotation_path,'r',encoding='latin'))
        keypoints_original = [[corner['CenterU'],corner['CenterV']] for corner in annotation_json['Annotations']['0'][:4]]
        radii = torch.tensor([corner['Radius'] for corner in annotation_json['Annotations']['0'][:4]])

        # make bounding boxes
        ##########################
        expand_x = 0.05 # expand boxes by this percentage (in decimal)
        expand_y = 0.05
        boxes_topleft_expand = keypoints_original - np.array([img_original.shape[1]*expand_x,img_original.shape[0]*expand_y])
        box_topleft_keepinIm = np.array([[val if val > 0 else 1 for val in point] for point in boxes_topleft_expand])
        # boxes_topleft_expand
        box_topleft_keepinIm
        boxes_botright_expand = keypoints_original + np.array([img_original.shape[1]*expand_x,img_original.shape[0]*expand_y])
        # shapeperm = np.transpose(img,axes=(1,0,2)).shape
        maxsizes = (img_original.shape[1],img_original.shape[0])
        box_botright_keepinIm = np.array([[val if val <= maxsizes[i] else maxsizes[i]-1 for i,val in enumerate(point)] for point in boxes_botright_expand])

        bboxes_original = np.concatenate((box_topleft_keepinIm,box_botright_keepinIm),axis=1)
        ##########################

        # Each object is a corner of the goal
        bboxes_labels_original = ['TL','TR','BL','BR']
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
            'labels': torch.tensor([i for i in range(1,len(bboxes_tensor)+1)], dtype=torch.int64), # class label hard-coded to 1,2,3 or 4, one for each corner TL,TR,BL,BR
            'image_id': torch.tensor((idx)), # save id of image for reference
            'keypoints': keypoints_tensor[:,None,:],
            'radii': radii,
            'area': (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) * (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]),
            'iscrowd': torch.zeros(len(bboxes), dtype=torch.int64),
            'old_image_id': torch.tensor((self.old_idxs[idx]))
        }

        return img_tensor,target_dict


############################################################
############################################################
## data loader for heatmap regression
class GoalCalibrationDatasetHeatmap(Dataset):
    def __init__(self,datapath, transforms=None, istrain=False):
        self.num_keypoints = 4
        self.img_list = sorted(glob.glob(str(Path(datapath + '/*/*.jpg'))))
        self.annotation_list = sorted(glob.glob(str(Path(datapath + '/FootballGoalCorners/AnnotationFiles/*.json'))))
        ### remove image paths from list if they aren't of category 'free kick'
        self.img_list_filtered, self.annotation_list_filtered,self.old_idxs = filter_data(self.img_list,self.annotation_list)
        self.istrain = istrain
        self.transforms = transforms
        

    def  __len__(self):
        return len(self.annotation_list_filtered)

    def __getitem__(self, idx):
        img_path = self.img_list_filtered[idx]
        annotation_path = self.annotation_list_filtered[idx]

        # img has shape H,W,C
        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        # print(f'img_original shape: {img_original.shape}')
        
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

        # convert image to tensor. Gets shape C,H,W
        img_tensor = F.to_tensor(img)
        # resize to 256x256
        img_tensor = T.Resize(size=(256,256))(img_tensor)
        # convert keypoints to tensor
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        # convert bboxes to tensor
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        
        # Make ground truth heatmaps
        # heatmaps expect shape in form K,H,W, i.e. number of keypoints,height,width
        heatmap = np.zeros((self.num_keypoints,64,64))#img_tensor.shape[1],img_tensor.shape[1]))
        # print(f'initial heatmap shape: {heatmap.shape}')
        for i in range(self.num_keypoints):
            heatmap[i] = draw_gaussian(heatmap[i], keypoints[i], 1)


        target_dict = {
            'image': img_tensor,
            'heatmap': torch.tensor(heatmap, dtype=torch.float32),
            'landmarks': keypoints_tensor,#[None,:],
            'labels': torch.tensor([1 for _ in bboxes_tensor], dtype=torch.int64), # class label hard-coded to 1 always, as we are only interested in the football goals
            'image_id': torch.tensor((idx)), # save id of image for reference
            'keypoints': keypoints_tensor[None,:],
            'radii': radii,
            'area': (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) * (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]),
            'iscrowd': torch.zeros(len(bboxes), dtype=torch.int64),
            'old_image_id': torch.tensor((self.old_idxs[idx]))
        }

        return target_dict



def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss

def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [np.floor(np.floor(point[0]) - 3 * sigma),
          np.floor(np.floor(point[1]) - 3 * sigma)]
    br = [np.floor(np.floor(point[0]) + 3 * sigma),
          np.floor(np.floor(point[1]) + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    correct = False
    while not correct:
        try:
            image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
            ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
            correct = True
        except:
            print('img_x: {}, img_y: {}, g_x:{}, g_y:{}, point:{}, g_shape:{}, ul:{}, br:{}'.format(img_x, img_y, g_x, g_y, point, g.shape, ul, br))
            ul = [np.floor(np.floor(point[0]) - 3 * sigma),
                np.floor(np.floor(point[1]) - 3 * sigma)]
            br = [np.floor(np.floor(point[0]) + 3 * sigma),
                np.floor(np.floor(point[1]) + 3 * sigma)]
            g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
                int(max(1, ul[0])) + int(max(1, -ul[0]))]
            g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
                int(max(1, ul[1])) + int(max(1, -ul[1]))]
            img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
            img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
            pass
    image[image > 1] = 1
    return image