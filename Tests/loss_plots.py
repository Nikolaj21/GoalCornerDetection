

#%%
import matplotlib.pyplot as plt
import numpy as np
import json

with open(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Models/kp_rcnn_v2_50epochs/losses.json','r') as file:
    loss_dict = json.load(file)
# %%
fig,ax = plt.subplots(1,1,figsize=(5,5))
x = np.arange(20)
ax.plot(x,loss_dict['loss_keypoint_train_mean'], label='training loss')
ax.plot(x,loss_dict['loss_keypoint_val_mean'], label='validation loss')
ax.legend()
# fig.suptitle('Keypoint loss', fontweight ="bold")
ax.set_title('keypoint loss')
ax.set_xticks(np.arange(2,21,2))
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

# %%
fig,ax = plt.subplots(1,1,figsize=(5,5))
x = np.arange(20)+1
ax.plot(x,loss_dict['loss_all_train_mean'], label='training loss')
ax.plot(x,loss_dict['loss_all_val_mean'], label='validation loss')
ax.legend()
ax.set_title('loss curve')
ax.set_xticks(np.arange(2,21,2))
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

#%% 
