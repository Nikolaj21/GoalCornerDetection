import glob
import cv2
import torch
from torch.utils.data import Dataset
import json
from torchvision.transforms import functional as F


class GoalCalibrationDataset(Dataset):
    def __init__(self,datapath, transform=None):
        self.img_list = sorted(glob.glob(datapath + '/*/*.jpg'))
        self.calibration_list = sorted(glob.glob(datapath + '/*/*.txt'))
        self.transform = transform

    def  __len__(self):
        return len(self.calibration_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        calibration_path = self.calibration_list[idx]
        img = cv2.imread(img_path)
        # cv2 loads image in b,g,r. Return it to r,g,b
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load keypoint annotations
        calibration_json = json.load(open(calibration_path,'r',encoding='latin'))
        keypoints = calibration_json['GoalCalibrationPoints']
        
        # change format of keypoints from [x,y] -> [x,y,visibility] where visibility=0 means the keypoint is not visible
        for kpt in keypoints:
            kpt.append(1)
        # placeholder for adding transformations to data later
        if self.transform:
            pass

        # convert image to tensor
        img_tensor = F.to_tensor(img)
        # convert keypoints to tensor
        keypoints_tensor = torch.tensor(keypoints)

        target_dict = {}
        target_dict['boxes'] = torch.tensor([]) # needs to be added for using keypoint r-cnn
        target_dict['labels'] = torch.tensor([1]) # class label hard-coded to 1 always, as we are only interested in the football goals
        target_dict['keypoints'] = keypoints_tensor
        target_dict['image_id'] = torch.tensor([idx])
        
        return img_tensor,target_dict