### This file is meant for comparing the old data annotations to the new data

#%% load packages and data
import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
import matplotlib.pyplot as plt
import numpy as np
from Core.DataLoader import GoalCalibrationDataset,GoalCalibrationDatasetOLD
from utils import DATA_DIR
from torch.utils.data import DataLoader
from Core.torchhelpers.utils import collate_fn
from Core.helpers import to_numpy
from tqdm import tqdm
import pandas as pd

#%%
GoalDataold = GoalCalibrationDatasetOLD(DATA_DIR,filter_data=True)
GoalData = GoalCalibrationDataset(DATA_DIR)

torchloaderold = DataLoader(GoalDataold, batch_size=2,collate_fn=collate_fn)
torchloader = DataLoader(GoalData, batch_size=2,collate_fn=collate_fn)

#%% load annotation points

keypoints_dict = {to_numpy(target['image_id'])[0]: target['keypoints']
for _,targets in tqdm(torchloader)
for target in targets}

#%% load old annotation points
keypointsold_dict = {to_numpy(target['image_id'])[0]: target['keypoints']
for _,targets in tqdm(torchloaderold)
for target in targets}

#%% Do comparison of old keypoints and new keypoints

distances = [np.linalg.norm(kp[:2]-kp_old[:2])
            for (idold,kpsold),(id,kps) in tqdm(zip(keypointsold_dict.items(),keypoints_dict.items()))
            for obj_old,obj in zip(kpsold,kps)
            for kp_old, kp in zip(obj_old,obj)]

distances_TL = distances[0::4]
distances_TR = distances[1::4]
distances_BL = distances[2::4]
distances_BR = distances[3::4]

#%%
mean_all = np.mean(distances)
mean_TL = np.mean(distances_TL)
mean_TR = np.mean(distances_TR)
mean_BL = np.mean(distances_BL)
mean_BR = np.mean(distances_BR)

print(f'mean all: {mean_all}')
print(f'mean TL: {mean_TL}')
print(f'mean TR: {mean_TR}')
print(f'mean BL: {mean_BL}')
print(f'mean BR: {mean_BR}')

# outputs from running above
# mean all: 37.15945816040039
# mean TL: 23.0086669921875
# mean TR: 45.796661376953125
# mean BL: 42.918609619140625
# mean BR: 36.91389465332031
#%%

# #%%
# import pandas as pd
# keypoints_data = [{'image_id':to_numpy(target['image_id'])[0], 'keypoints': to_numpy(target['keypoints'])}
#                 for _,targets in tqdm(torchloader)
#                 for target in targets]

# torchloader.dataset.annotation_list_filtered
# #%%
# df = pd.DataFrame.from_dict(keypoints_data)

# #%%
# keypointsold_data = [to_numpy(target['keypoints'])
#                     for _,targets in tqdm(torchloaderold)
#                     for target in targets]
# # df['keypoints_old'] = keypointsold_data

# #%%
# for images,targets in torchloader:
#     for target in targets:
#         print(to_numpy(target['keypoints']))
#         print(to_numpy(target['image_id'])[0])
#     # print(targets['keypoints'])
    # break
# %%
