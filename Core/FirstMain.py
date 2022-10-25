import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Core/torchhelpers')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import torch
import numpy as np
import matplotlib.pyplot as plt
from Core.helpers import to_torch,im_to_numpy,split_data_train_test
from Core.DataLoader import GoalCalibrationDatasetNEW
from Core.plottools import target_to_keypoints,batch_target_to_keypoints,plot_batch_keypoints,visualize
from utils import DATA_DIR

from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchsummary import summary
# https://github.com/pytorch/vision/tree/main/references/detection
from Core.torchhelpers import transforms, utils, engine, train
from Core.torchhelpers.utils import collate_fn
from Core.torchhelpers.engine import train_one_epoch, evaluate
from torchvision.models.detection.rpn import AnchorGenerator

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print(f'Running on {device}')

    # initialize an instance of the dataloader
    GoalData = GoalCalibrationDatasetNEW(DATA_DIR,transforms=None)
    # put dataloader into pytorch dataloader function. Batch size chosen to be 14 as it 854 is divisible by this
    train_loader,validation_loader = split_data_train_test(GoalData,validation_split=0.25,batch_size=4,shuffle_dataset=True,shuffle_seed=None,data_amount=1)

    # Setting hyper-parameters
    num_classes = 2 # 1 class (goal) + background
    anchor_generator = AnchorGenerator(sizes=(64, 128, 256, 512, 1024), aspect_ratios=(1.0, 2.0, 2.5, 3.0, 4.0))
    
    model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=num_classes, num_keypoints=4,rpn_anchor_generator=anchor_generator)
    model.to(device)
    print(f'Model moved to device: {device}')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    num_epochs = 10

    # gpu test
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, validation_loader, device)

    torch.save(model.state_dict(), r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Models/model_states/first_train/kp_rcnn_weights_10epochs_newdata_bs4.pth')