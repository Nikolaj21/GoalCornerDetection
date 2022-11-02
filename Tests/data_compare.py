### This file is meant for comparing the old data annotations to the new data

#%% load packages and data
import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Core/torchhelpers')
import matplotlib.pyplot as plt
import numpy as np
from Core.DataLoader import GoalCalibrationDataset,GoalCalibrationDatasetOLD
from utils import DATA_DIR
from torch.utils.data import DataLoader
from Core.torchhelpers.utils import collate_fn
from Core.helpers import to_numpy
from tqdm.notebook import tqdm
import pandas as pd
#%%
GoalDataold = GoalCalibrationDatasetOLD(DATA_DIR)
GoalData = GoalCalibrationDataset(DATA_DIR)

torchloaderold = DataLoader(GoalDataold, batch_size=2,collate_fn=collate_fn)
torchloader = DataLoader(GoalData, batch_size=2,collate_fn=collate_fn)

#%% load annotation points

# keypoints_dict = {to_numpy(target['image_id'])[0]: target['keypoints']
# for images,targets in tqdm(torchloader)
# for target in targets}

# #%% load old annotation points
# keypointsold_dict = {to_numpy(target['image_id'])[0]: target['keypoints']
# for images,targets in tqdm(torchloaderold)
# for target in targets}


#%%
import pandas as pd
keypoints_data = [{'image_id':to_numpy(target['image_id'])[0], 'keypoints': to_numpy(target['keypoints'])}
                for _,targets in tqdm(torchloader)
                for target in targets]

torchloader.dataset.annotation_list_filtered
#%%
df = pd.DataFrame.from_dict(keypoints_data)

#%%
keypointsold_data = [to_numpy(target['keypoints'])
                    for _,targets in tqdm(torchloaderold)
                    for target in targets]
# df['keypoints_old'] = keypointsold_data

#%%
for images,targets in torchloader.:
    for target in targets:
        print(to_numpy(target['keypoints']))
        print(to_numpy(target['image_id'])[0])
    # print(targets['keypoints'])
    break