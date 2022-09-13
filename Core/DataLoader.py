import glob
import cv2
import torch
from torch.utils.data import Dataset
import json

class GoalCalibrationDataset(Dataset):
    def __init__(self,datapath):
        self.img_list = glob.glob(datapath + '/*/*.jpg')
        self.calibration_list = glob.glob(datapath + '/*/*.txt')
    def  __len__(self):
        return len(self.calibration_list)
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        calibration_path = self.calibration_list[idx]
        img = cv2.imread(img_path)
        # cv2 loads image in b,g,r. Return it to r,g,b
        b, g, r = cv2.split(img)
        img = cv2.merge([r,g,b])
        calibration_json = json.load(open(calibration_path,'r',encoding='latin'))
        calibrationpts = calibration_json['GoalCalibrationPoints']
        # convert image to tensor
        img_tensor = torch.from_numpy(img)
        # change channel order from Width,Height,Channels to Channels,Width,Height
        img_tensor = img_tensor.permute(2, 0, 1)
        calibrationpts_tensor = torch.tensor(calibrationpts)
        return img_tensor,calibrationpts_tensor