import glob
import cv2
import torch
from torch.utils.data import Dataset
import json
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import numpy as np

class GoalCalibrationDataset(Dataset):
    def __init__(self,datapath, transforms=None):
        self.img_list = sorted(glob.glob(datapath + '/*/*.jpg'))
        self.calibration_list = sorted(glob.glob(datapath + '/*/*.txt'))
        self.transforms = transforms

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
        session_type = calibration_json.get('SessionType')
        
        # change order of keypoints so the top left point is first and bottom right point is last
        # keypoints = sorted(keypoints, key=lambda x: np.sum(x))

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

        # if self.transforms:
        #     img = cv2.resize(img,(4208,3120)) # cv2 shape order is opposite (W,H) instead of normal (H,W)

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