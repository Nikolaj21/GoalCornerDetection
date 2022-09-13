# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:04:11 2022

@author: Nikolaj
"""

import os
from util import ROOT_DIR, CODE_DIR
import json

datapath = os.path.join(ROOT_DIR,'Trackman contact stuff\RawCalibrationResult.json')
picpath = os.path.join(ROOT_DIR,'Trackman contact stuff\calibration83c79cfa-24ea-4cec-8244-1f90b71bb281.jpg')

# dataraw = json.loads(datapath)

with open(datapath,'r') as f:
    data = json.load(f)

data['SiteCalibrationInput'] = json.loads(data['SiteCalibrationInput'])
data['SiteCalibrationOutput'] = json.loads(data['SiteCalibrationOutput'])

with open(os.path.join(CODE_DIR,'first_data_explore\RawCalibration-fixed.json'),'w') as f:
    json.dump(f)
