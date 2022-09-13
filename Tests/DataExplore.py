# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:04:11 2022

@author: Nikolaj
"""

import os
from util import ROOT_DIR, CODE_DIR
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np

datapath = os.path.join(ROOT_DIR,'Trackman contact stuff\RawCalibrationResult.json')
picpath = os.path.join(ROOT_DIR,'Trackman contact stuff\calibration83c79cfa-24ea-4cec-8244-1f90b71bb281.jpg')

im = plt.imread(picpath)

def formatData(path):
    with open(path,'r') as f:
        data = json.load(f)
    data['SiteCalibrationInput'] = json.loads(data['SiteCalibrationInput'])
    data['SiteCalibrationOutput'] = json.loads(data['SiteCalibrationOutput'])
    data['ImageMetadata'] = json.loads(data['ImageMetadata'])
    return data
    
data = formatData(datapath)
with open(os.path.join(CODE_DIR,'first_data_explore\RawCalibration-fixed.json'),'w') as f:
    json.dump(data,f,indent=4)


calpts = np.asarray(data['GoalCalibrationPoints'])

#%%
_,ax = plt.subplots(1,1,figsize=(50,25))
ax.imshow(im)
ax.plot(*calpts.T,'r.',markersize=5)
plt.show()

